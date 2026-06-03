"""Evaluate a BC-transformer checkpoint under three conditions (10 eps each).

Conditions:
  1. no_obstacle     — standard task BDDL, env.reset() only (no fixed init state)
  2. train_obstacle  — obstacle BDDL, same (x,y) sampling as mjpl augmentation training
  3. bowl_plate_obstacle — obstacle BDDL, pillar sampled on bowl→plate segment (not arm path)

Outputs under data/research/bc/eval/<eval_run_id>/:
  PLAN.md, eval.log, results.csv, per-condition videos/
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
import tyro

_LIBERO = pathlib.Path(__file__).resolve().parents[3] / "third_party/libero"
_LIBERO_EXAMPLES = pathlib.Path(__file__).resolve().parents[1]
if str(_LIBERO) not in sys.path:
    sys.path.insert(0, str(_LIBERO))
if str(_LIBERO_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_LIBERO_EXAMPLES))

import obstacle  # noqa: F401

from common import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    DualCameraRecorder,
    check_obstacle_collision,
    check_task_success,
    make_env,
    obstacle_bddl_path,
    set_obstacle_pose,
    task_bddl_path,
)
from libero.lifelong.datasets import get_dataset  # noqa: E402
from libero.lifelong.models import get_policy_class  # noqa: E402
from libero.lifelong.utils import get_task_embs, safe_device  # noqa: E402
from train_bc import TASK_LANGUAGE, _load_cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAIN_OBSTACLE_X = -0.36
TRAIN_OBSTACLE_Z = 0.98
TRAIN_OBSTACLE_Y_LO = 0.08
TRAIN_OBSTACLE_Y_HI = 0.20

CONDITIONS = ("no_obstacle", "train_obstacle", "bowl_plate_obstacle")


def overlay_step(frames: list[np.ndarray]) -> list[np.ndarray]:
    from PIL import Image, ImageDraw

    out = []
    for i, fr in enumerate(frames):
        img = Image.fromarray(fr)
        ImageDraw.Draw(img).text((2, 1), f"step {i}", fill=(255, 0, 0))
        out.append(np.asarray(img))
    return out


@dataclasses.dataclass
class Args:
    checkpoint: str = "data/research/bc/runs/latest/bc_transformer_best.pth"
    train_config: str = "data/research/bc/runs/latest/config.json"
    dataset: str = "data/research/mjpl/obstacle_demo_success.hdf5"
    eval_dir: str = ""
    num_episodes: int = 10
    max_steps: int = 300
    num_steps_wait: int = 10
    resolution: int = 128
    seed: int = 0
    device: str = "cuda"
    train_eval_seed: int = 0
    conditions: str = ""  # comma-separated subset; default = all three


def _conditions_list(args: Args) -> tuple[str, ...]:
    if not args.conditions.strip():
        return CONDITIONS
    chosen = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    bad = [c for c in chosen if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; choose from {CONDITIONS}")
    return chosen


def _make_eval_dir(eval_dir: str) -> pathlib.Path:
    if eval_dir:
        root = pathlib.Path(eval_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = pathlib.Path("data/research/bc/eval") / f"eval_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    latest = pathlib.Path("data/research/bc/eval/latest")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(root.resolve(), target_is_directory=True)
    return root


def _write_plan(root: pathlib.Path, args: Args) -> None:
    text = f"""# BC evaluation plan

**Checkpoint:** `{args.checkpoint}`  
**Episodes per condition:** {args.num_episodes}  
**Max steps:** {args.max_steps} (auto-fail / timeout if not successful)  
**Success detector:** LIBERO `check_ontop` with 7cm XY tolerance (`base_object_states.py`)  
**Init:** `env.reset()` only — **no** `set_init_state` (matches `main.py`, not LIBERO lifelong eval).

## Conditions

| ID | Folder | BDDL | Obstacle |
|----|--------|------|----------|
| `no_obstacle` | `no_obstacle/videos/` | Standard spatial task-1 | None |
| `train_obstacle` | `train_obstacle/videos/` | `obstacle.bddl` | x={TRAIN_OBSTACLE_X}, z={TRAIN_OBSTACLE_Z}, y ~ Uniform[{TRAIN_OBSTACLE_Y_LO}, {TRAIN_OBSTACLE_Y_HI}] via `rng(train_seed+ep)` |
| `bowl_plate_obstacle` | `bowl_plate_obstacle/videos/` | `obstacle.bddl` | On bowl→plate segment (t∈[0.2,0.8] in XY), same z |

## Tail log

```bash
tail -f {root.resolve()}/eval.log
```
"""
    (root / "PLAN.md").write_text(text)


def _train_obstacle_xyz(episode: int, seed: int) -> tuple[float, float, float]:
    oy = float(np.random.default_rng(seed + episode).uniform(TRAIN_OBSTACLE_Y_LO, TRAIN_OBSTACLE_Y_HI))
    return TRAIN_OBSTACLE_X, oy, TRAIN_OBSTACLE_Z


def _object_xy(env, obj_name: str) -> np.ndarray:
    body = env.env.objects_dict[obj_name].root_body
    return env.sim.data.get_body_xpos(body)[:2].copy()


def _bowl_plate_obstacle_xyz(env, rng: np.random.Generator) -> tuple[float, float, float]:
    bowl = _object_xy(env, "akita_black_bowl_1")
    plate = _object_xy(env, "plate_1")
    t = float(rng.uniform(0.2, 0.8))
    xy = (1.0 - t) * bowl + t * plate
    return float(xy[0]), float(xy[1]), TRAIN_OBSTACLE_Z


def _get_obs(env):
    if hasattr(env.env, "_get_observations"):
        return env.env._get_observations()
    return env._get_observations()


def _obs_to_policy_batch(obs, task_emb: torch.Tensor, cfg) -> dict:
    data = {"obs": {}, "task_emb": task_emb.unsqueeze(0)}
    for modality_list in cfg.data.obs.modality.values():
        for obs_name in modality_list:
            key = cfg.data.obs_key_mapping[obs_name]
            tensor = ObsUtils.process_obs(
                torch.from_numpy(obs[key]), obs_key=obs_name
            ).float()
            data["obs"][obs_name] = tensor.unsqueeze(0)
    return TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))


def _load_policy(args: Args):
    cfg = _load_cfg()
    if pathlib.Path(args.train_config).exists():
        with open(args.train_config) as f:
            train_cfg = json.load(f)
        cfg.seed = train_cfg.get("seed", args.seed)
    cfg.device = args.device if torch.cuda.is_available() else "cpu"

    _, shape_meta = get_dataset(
        dataset_path=str(pathlib.Path(args.dataset).resolve()),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
        frame_stack=cfg.data.frame_stack,
        hdf5_cache_mode="low_dim",
    )
    task_embs = get_task_embs(cfg, [TASK_LANGUAGE])
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    cfg.shape_meta = shape_meta

    policy = get_policy_class(cfg.policy.policy_type)(cfg, shape_meta)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    policy.load_state_dict(ckpt["state_dict"])
    policy = safe_device(policy, cfg.device)
    policy.eval()
    return cfg, policy, task_embs[0]


def _rollout_episode(
    env,
    policy,
    task_emb: torch.Tensor,
    cfg,
    *,
    max_steps: int,
    num_steps_wait: int,
    obstacle_xyz: tuple[float, float, float] | None,
    video_dir: pathlib.Path,
    video_prefix: str,
    skip_reset: bool = False,
) -> tuple[bool, bool, int]:
    """reset → settle → optional obstacle → policy rollout."""
    if not skip_reset:
        env.reset()
    obs = _get_obs(env)
    if num_steps_wait > 0:
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    if obstacle_xyz is not None:
        set_obstacle_pose(env, x=obstacle_xyz[0], y=obstacle_xyz[1], z=obstacle_xyz[2])
        obs = _get_obs(env)

    policy.reset()
    recorder = DualCameraRecorder()
    collision = False
    success = False
    steps = 0

    for _ in range(max_steps):
        recorder.append(obs)
        steps += 1
        data = _obs_to_policy_batch(obs, task_emb, cfg)
        action = policy.get_action(data)[0]
        obs, _, done, _ = env.step(action.tolist())
        if check_obstacle_collision(env):
            collision = True
        if done or check_task_success(env):
            success = bool(done) or check_task_success(env)
            if success:
                break

    import imageio

    agent_path = video_dir / f"{video_prefix}_agent.mp4"
    wrist_path = video_dir / f"{video_prefix}_wrist.mp4"
    imageio.mimwrite(agent_path, overlay_step(recorder.agent_frames), fps=20)
    imageio.mimwrite(wrist_path, overlay_step(recorder.wrist_frames), fps=20)
    return success, collision, steps


def _obstacle_xyz_for_episode(
    condition: str,
    env,
    ep: int,
    args: Args,
    rng: np.random.Generator,
    num_steps_wait: int,
) -> tuple[float, float, float] | None:
    if condition == "no_obstacle":
        return None
    if condition == "train_obstacle":
        return _train_obstacle_xyz(ep, args.train_eval_seed)
    # bowl_plate: env already reset+settled by caller before reading bodies
    return _bowl_plate_obstacle_xyz(env, rng)


def _run_condition(
    condition: str,
    args: Args,
    cfg,
    policy,
    task_emb: torch.Tensor,
    root: pathlib.Path,
    writer: csv.DictWriter,
    csv_file,
) -> dict:
    use_obstacle = condition != "no_obstacle"
    bddl = obstacle_bddl_path() if use_obstacle else task_bddl_path()
    env = make_env(bddl, resolution=args.resolution, seed=args.seed)
    video_dir = root / condition / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + hash(condition) % 10000)

    n_success = 0
    n_collision = 0

    for ep in range(args.num_episodes):
        t0 = time.perf_counter()

        obstacle_xyz = None
        if condition == "bowl_plate_obstacle":
            env.reset()
            for _ in range(args.num_steps_wait):
                env.step(LIBERO_DUMMY_ACTION)
            obstacle_xyz = _bowl_plate_obstacle_xyz(env, rng)
            success, collision, steps = _rollout_episode(
                env, policy, task_emb, cfg,
                max_steps=args.max_steps,
                num_steps_wait=0,
                obstacle_xyz=obstacle_xyz,
                video_dir=video_dir,
                video_prefix=f"ep_{ep:02d}",
                skip_reset=True,
            )
        else:
            obstacle_xyz = _obstacle_xyz_for_episode(condition, env, ep, args, rng, args.num_steps_wait)
            success, collision, steps = _rollout_episode(
                env, policy, task_emb, cfg,
                max_steps=args.max_steps,
                num_steps_wait=args.num_steps_wait,
                obstacle_xyz=obstacle_xyz,
                video_dir=video_dir,
                video_prefix=f"ep_{ep:02d}",
            )

        if success:
            n_success += 1
        if collision:
            n_collision += 1

        ox = oy = oz = ""
        if obstacle_xyz is not None:
            ox, oy, oz = (f"{obstacle_xyz[0]:.3f}", f"{obstacle_xyz[1]:.3f}", f"{obstacle_xyz[2]:.3f}")

        status = "ok" if success else ("timeout" if steps >= args.max_steps else "fail")
        wall = time.perf_counter() - t0
        writer.writerow({
            "condition": condition,
            "episode": ep,
            "success": int(success),
            "steps": steps,
            "wall_time_s": f"{wall:.1f}",
            "collision": int(collision),
            "obstacle_x": ox,
            "obstacle_y": oy,
            "obstacle_z": oz,
            "status": status,
        })
        csv_file.flush()
        log.info("[%s] ep %d: success=%s steps=%d collision=%s obs=(%s,%s,%s)",
                 condition, ep, success, steps, collision, ox, oy, oz)

    env.close()
    return {
        "success_rate": n_success / max(args.num_episodes, 1),
        "collision_rate": n_collision / max(args.num_episodes, 1),
        "episodes": args.num_episodes,
    }


def main(args: Args) -> None:
    root = _make_eval_dir(args.eval_dir)
    log_path = root / "eval.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    _write_plan(root, args)

    log.info("eval_dir=%s", root)
    log.info("tail -f %s", log_path.resolve())
    print(f"\n>>> Eval log: tail -f {log_path.resolve()}\n", flush=True)

    cfg, policy, task_emb = _load_policy(args)
    csv_path = root / "results.csv"
    fieldnames = [
        "condition", "episode", "success", "steps", "wall_time_s", "collision",
        "obstacle_x", "obstacle_y", "obstacle_z", "status",
    ]
    summary = {"checkpoint": str(pathlib.Path(args.checkpoint).resolve()), "conditions": {}}

    run_conditions = _conditions_list(args)
    log.info("conditions=%s  episodes_per_condition=%d", run_conditions, args.num_episodes)

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for condition in run_conditions:
            log.info("=== condition: %s ===", condition)
            summary["conditions"][condition] = _run_condition(
                condition, args, cfg, policy, task_emb, root, writer, cf,
            )

    summary_path = root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("wrote %s", summary_path)
    for c in run_conditions:
        s = summary["conditions"][c]
        log.info(
            "%s: success %.0f%% (%d/%d) collision %.0f%%",
            c, 100 * s["success_rate"],
            int(s["success_rate"] * s["episodes"]), s["episodes"],
            100 * s["collision_rate"],
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
