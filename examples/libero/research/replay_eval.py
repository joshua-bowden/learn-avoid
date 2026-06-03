"""Replay the original demonstrations (plain, no obstacle), tracking the same
metrics as the augmentation run (success/steps/wall-time/collision/status) to a
live-updating CSV, and saving per-episode videos with a step-counter overlay.
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import sys
import time

import h5py
import numpy as np
import tyro

_LIBERO_EXAMPLES = pathlib.Path(__file__).resolve().parents[1]
if str(_LIBERO_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_LIBERO_EXAMPLES))

from common import (  # noqa: E402
    TASK_ID, TASK_SUITE, DualCameraRecorder, check_obstacle_collision,
    check_task_success, demo_hdf5_path, load_episode_from_hdf5, make_env,
    obstacle_bddl_path, reset_env_to_demo, sorted_demo_keys, task_bddl_path,
)


@dataclasses.dataclass
class Args:
    demo_file: str = ""
    num_episodes: int = 50
    max_episodes: int = 0
    resolution: int = 128
    video_out_path: str = "data/research/replay/videos"
    csv_path: str = "data/research/replay/results.csv"
    use_obstacle_bddl: bool = False
    seed: int = 0
    max_steps: int = 300


def overlay_step(frames: list[np.ndarray]) -> list[np.ndarray]:
    from PIL import Image, ImageDraw
    out = []
    for i, fr in enumerate(frames):
        img = Image.fromarray(fr)
        ImageDraw.Draw(img).text((2, 1), f"step {i}", fill=(255, 0, 0))
        out.append(np.asarray(img))
    return out


def main(cfg: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    demo_path = demo_hdf5_path(cfg.demo_file or None)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file missing: {demo_path}")

    video_dir = pathlib.Path(cfg.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    bddl = obstacle_bddl_path() if cfg.use_obstacle_bddl else task_bddl_path(TASK_SUITE, TASK_ID)
    env = make_env(bddl, resolution=cfg.resolution, seed=cfg.seed)

    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["episode", "success", "steps", "wall_time_s", "collision", "status"])
    csv_f.flush()

    import imageio

    with h5py.File(demo_path, "r") as f:
        demo_keys = sorted_demo_keys(f)[: cfg.num_episodes]
        if cfg.max_episodes > 0:
            demo_keys = demo_keys[: cfg.max_episodes]
        n_eps = len(demo_keys)
        n_success = 0

        for ep_idx, ep_key in enumerate(demo_keys):
            t0 = time.perf_counter()
            model_xml, states, actions = load_episode_from_hdf5(f, ep_key)
            reset_env_to_demo(env, model_xml, states[0])

            cameras = DualCameraRecorder()
            done = False
            steps = 0
            collided = False
            for action in actions:
                if steps >= cfg.max_steps:
                    break
                obs, _, done, _ = env.step(action.tolist())
                cameras.append(obs)
                steps += 1
                if check_obstacle_collision(env):
                    collided = True
                if done:
                    break

            timed_out = steps >= cfg.max_steps and not done
            success = (bool(done) or check_task_success(env)) and not timed_out
            status = "timeout" if timed_out else "ok"
            n_success += int(success)
            wall = time.perf_counter() - t0

            imageio.mimwrite(video_dir / f"ep_{ep_idx:02d}_agent.mp4", overlay_step(cameras.agent_frames), fps=20)
            imageio.mimwrite(video_dir / f"ep_{ep_idx:02d}_wrist.mp4", overlay_step(cameras.wrist_frames), fps=20)

            writer.writerow([ep_idx, int(success), steps, f"{wall:.1f}", int(collided), status])
            csv_f.flush()
            logging.info("replay ep %d/%d: success=%s steps=%d time=%.1fs status=%s",
                         ep_idx, n_eps - 1, success, steps, wall, status)

    csv_f.close()
    env.close()
    logging.info("REPLAY DONE: %d/%d success. CSV=%s", n_success, n_eps, csv_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
