"""Minimal single-task BC-transformer training using LIBERO's policy + dataset stack.

Hyperparameters (lr, policy arch, obs modalities, etc.) are loaded from LIBERO's
Hydra configs under third_party/libero/libero/configs/. CLI args override only
epochs, batch_size, num_workers, and paths. Training uses ONE HDF5 / ONE task —
not the full 10-task LIBERO-Spatial suite loop in lifelong/main.py.
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

import robomimic.utils.tensor_utils as TensorUtils
import torch
import tyro
import yaml
from easydict import EasyDict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

_LIBERO = pathlib.Path(__file__).resolve().parents[3] / "third_party/libero"
if str(_LIBERO) not in sys.path:
    sys.path.insert(0, str(_LIBERO))

from libero.lifelong.datasets import SequenceVLDataset, get_dataset  # noqa: E402
from libero.lifelong.models import get_policy_class  # noqa: E402
from libero.lifelong.utils import (  # noqa: E402
    control_seed,
    get_task_embs,
    safe_device,
    torch_save_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TASK_LANGUAGE = "pick up the black bowl next to the ramekin and place it on the plate"
RUNS_ROOT = pathlib.Path("data/research/bc/runs")


@dataclasses.dataclass
class Args:
    dataset: str = "data/research/mjpl/obstacle_demo_success.hdf5"
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 0
    device: str = "cuda"
    run_dir: str = ""  # default: auto data/research/bc/runs/run_YYYYMMDD_HHMMSS


def _load_cfg() -> EasyDict:
    cfg_dir = _LIBERO / "libero/configs"
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        hydra_cfg = compose(
            config_name="config",
            overrides=["lifelong=single_task", "policy=bc_transformer_policy"],
        )
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    return EasyDict(yaml.safe_load(yaml_config))


def _make_run_dir(run_dir: str) -> pathlib.Path:
    if run_dir:
        rd = pathlib.Path(run_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rd = RUNS_ROOT / f"run_{stamp}"
    rd.mkdir(parents=True, exist_ok=True)
    latest = RUNS_ROOT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(rd.resolve(), target_is_directory=True)
    return rd


def _append_run_index(row: dict) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    index = RUNS_ROOT / "index.csv"
    write_header = not index.exists()
    with open(index, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _save_run_config(run_dir: pathlib.Path, args: Args, cfg: EasyDict, dataset_path: pathlib.Path) -> None:
    snap = {
        "run_dir": str(run_dir.resolve()),
        "dataset": str(dataset_path),
        "task_language": TASK_LANGUAGE,
        "single_task": True,
        "note": "Trains on one HDF5 only; not the 10-task lifelong/main.py loop.",
        "cli": dataclasses.asdict(args),
        "libero_config_sources": {
            "root": str(_LIBERO / "libero/configs/config.yaml"),
            "policy": "policy/bc_transformer_policy.yaml",
            "train": "train/default.yaml",
            "optimizer": "train/optimizer/adam_w.yaml",
            "data": "data/default.yaml",
        },
        "train": {
            "n_epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "grad_clip": cfg.train.grad_clip,
            "use_augmentation": cfg.train.use_augmentation,
            "optimizer": cfg.train.optimizer.kwargs,
            "scheduler": cfg.train.scheduler,  # loaded but not used in this script
        },
        "data": {
            "seq_len": cfg.data.seq_len,
            "img_h": cfg.data.img_h,
            "img_w": cfg.data.img_w,
            "obs_modality": cfg.data.obs.modality,
        },
        "policy": {
            "policy_type": cfg.policy.policy_type,
            "transformer_num_layers": cfg.policy.transformer_num_layers,
            "transformer_num_heads": cfg.policy.transformer_num_heads,
            "transformer_mlp_hidden_size": cfg.policy.transformer_mlp_hidden_size,
            "transformer_dropout": cfg.policy.transformer_dropout,
            "transformer_max_seq_len": cfg.policy.transformer_max_seq_len,
        },
        "seed": args.seed,
        "device": args.device,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(snap, f, indent=2)


def main(args: Args) -> None:
    run_dir = _make_run_dir(args.run_dir)
    log_path = run_dir / "train.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info("run_dir=%s", run_dir)
    log.info("tail -f %s", log_path.resolve())
    print(f"\n>>> Training log: tail -f {log_path.resolve()}\n", flush=True)

    dataset_path = pathlib.Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"dataset not found: {dataset_path}")

    cfg = _load_cfg()
    cfg.benchmark_name = "LIBERO_SPATIAL"
    cfg.lifelong.algo = "SingleTask"
    cfg.eval.eval = False
    cfg.seed = args.seed
    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.train.n_epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.experiment_dir = str(run_dir.resolve())
    cfg.experiment_name = run_dir.name

    _save_run_config(run_dir, args, cfg, dataset_path)
    control_seed(cfg.seed)
    log.info("device=%s dataset=%s", cfg.device, dataset_path)

    seq_ds, shape_meta = get_dataset(
        dataset_path=str(dataset_path),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
        frame_stack=cfg.data.frame_stack,
        hdf5_cache_mode="all",
    )
    log.info("sequences=%d demos=%d", seq_ds.total_num_sequences, seq_ds.n_demos)

    task_embs = get_task_embs(cfg, [TASK_LANGUAGE])
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    cfg.shape_meta = shape_meta
    dataset = SequenceVLDataset(seq_ds, task_embs[0])

    policy = get_policy_class(cfg.policy.policy_type)(cfg, shape_meta)
    policy = safe_device(policy, cfg.device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.train.optimizer.kwargs.lr,
        betas=tuple(cfg.train.optimizer.kwargs.betas),
        weight_decay=cfg.train.optimizer.kwargs.weight_decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        sampler=RandomSampler(dataset),
        persistent_workers=cfg.train.num_workers > 0,
    )

    def _to_device(data):
        return TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))

    if cfg.device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = policy.compute_loss(_to_device(next(iter(loader))))
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        log.info("GPU mem probe: peak=%.0f MiB / %.0f MiB total (batch=%d)", peak_mb, total_mb, cfg.train.batch_size)

    ckpt = run_dir / "bc_transformer_best.pth"
    metrics_path = run_dir / "metrics.csv"
    best_loss = float("inf")
    t0 = time.perf_counter()

    with open(metrics_path, "w", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow(["epoch", "loss", "epoch_time_s", "is_best"])

        for epoch in range(1, args.epochs + 1):
            policy.train()
            epoch_loss = 0.0
            t_ep = time.perf_counter()
            for data in loader:
                optimizer.zero_grad()
                loss = policy.compute_loss(_to_device(data))
                loss.backward()
                if cfg.train.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(len(loader), 1)
            elapsed = time.perf_counter() - t_ep
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss
                torch_save_model(policy, str(ckpt), cfg=cfg)
                log.info("epoch %3d/%d  loss=%.4f  time=%.1fs  *best*", epoch, args.epochs, epoch_loss, elapsed)
            else:
                log.info("epoch %3d/%d  loss=%.4f  time=%.1fs", epoch, args.epochs, epoch_loss, elapsed)
            mw.writerow([epoch, f"{epoch_loss:.6f}", f"{elapsed:.1f}", int(is_best)])
            mf.flush()

    wall = time.perf_counter() - t0
    summary = {
        "best_loss": best_loss,
        "wall_time_s": wall,
        "checkpoint": str(ckpt),
        "metrics_csv": str(metrics_path),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _append_run_index({
        "run_id": run_dir.name,
        "started_utc": run_dir.name.replace("run_", ""),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "demos": seq_ds.n_demos,
        "best_loss": f"{best_loss:.6f}",
        "wall_time_min": f"{wall / 60:.1f}",
        "checkpoint": str(ckpt),
    })
    log.info("done in %.1f min  best_loss=%.4f  checkpoint=%s", wall / 60, best_loss, ckpt)


if __name__ == "__main__":
    main(tyro.cli(Args))
