"""Run the mjpl augmentation in parallel shards, then merge into one HDF5 + CSV.

Splits the demos across N worker processes (each handling a contiguous slice),
each writing its own shard HDF5/CSV into a shared video dir (global episode names).
After all workers finish, shard HDF5s are merged into one file and shard CSVs are
concatenated/sorted into the final results CSV.

Merge is best-effort: failed or truncated worker outputs are skipped and the
final CSV/HDF5 are built from whatever completed successfully.
"""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import pathlib
import subprocess
import sys
import time
from collections import Counter

import h5py
import tyro

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]


@dataclasses.dataclass
class Args:
    num_episodes: int = 50
    workers: int = 5
    resolution: int = 128
    seed: int = 0
    out_dir: str = "data/research/mjpl"


def _merge_hdf5_shards(procs, shard_dir: pathlib.Path, final_hdf5: pathlib.Path) -> tuple[int, int, list[str]]:
    """Return (copied_demos, total_samples, skipped_shard_messages)."""
    copied = 0
    total = 0
    skipped: list[str] = []
    with h5py.File(final_hdf5, "w") as fout:
        grp = fout.create_group("data")
        for i, *_ in procs:
            shard = shard_dir / f"shard_{i}.hdf5"
            if not shard.exists():
                skipped.append(f"shard_{i}: missing")
                continue
            try:
                with h5py.File(shard, "r") as fin:
                    if "data" not in fin:
                        skipped.append(f"shard_{i}: no data group")
                        continue
                    for k, v in fin["data"].attrs.items():
                        if k not in grp.attrs:
                            grp.attrs[k] = v
                    for name in fin["data"].keys():
                        fin.copy(fin[f"data/{name}"], grp, name=name)
                        copied += 1
                        total += int(fin[f"data/{name}"].attrs.get("num_samples", 0))
            except OSError as e:
                skipped.append(f"shard_{i}: unreadable ({e})")
        grp.attrs["num_demos"] = copied
        grp.attrs["total"] = total
    return copied, total, skipped


def _merge_csv_shards(procs, shard_dir: pathlib.Path, final_csv: pathlib.Path) -> tuple[list[list[str]], list[str]]:
    """Return (rows, skipped_shard_messages)."""
    rows: list[list[str]] = []
    header = None
    skipped: list[str] = []
    for i, *_ in procs:
        c = shard_dir / f"shard_{i}.csv"
        if not c.exists():
            skipped.append(f"shard_{i}: missing csv")
            continue
        try:
            with open(c) as fh:
                r = list(csv.reader(fh))
        except OSError as e:
            skipped.append(f"shard_{i}: unreadable csv ({e})")
            continue
        if len(r) <= 1:
            skipped.append(f"shard_{i}: empty csv")
            continue
        if header is None:
            header = r[0]
        rows.extend(r[1:])
    rows.sort(key=lambda x: int(x[0]))
    with open(final_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(header)
        w.writerows(rows)
    return rows, skipped


def main(cfg: Args) -> None:
    out_dir = pathlib.Path(cfg.out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    final_hdf5 = out_dir / "obstacle_demo.hdf5"
    final_csv = out_dir / "results.csv"

    chunk = math.ceil(cfg.num_episodes / cfg.workers)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["PYTHONPATH"] = os.pathsep.join([
        env.get("PYTHONPATH", ""),
        str(_REPO / "third_party/libero"),
        str(_REPO / "examples/libero"),
        str(_REPO / "examples/libero/research"),
    ])

    procs = []
    for i in range(cfg.workers):
        start = i * chunk
        if start >= cfg.num_episodes:
            break
        cnt = min(chunk, cfg.num_episodes - start)
        out = shard_dir / f"shard_{i}.hdf5"
        csvp = shard_dir / f"shard_{i}.csv"
        logp = shard_dir / f"shard_{i}.log"
        cmd = [sys.executable, str(_HERE.parent / "mjpl_augment.py"),
               "--num_episodes", str(cfg.num_episodes),
               "--start_episode", str(start), "--max_episodes", str(cnt),
               "--resolution", str(cfg.resolution), "--seed", str(cfg.seed),
               "--output_file", str(out), "--csv_path", str(csvp),
               "--video_out_path", str(video_dir)]
        lf = open(logp, "w")
        procs.append((i, start, cnt, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf))
        print(f"launched worker {i}: episodes [{start}, {start + cnt})  pid={procs[-1][3].pid}", flush=True)

    t0 = time.time()
    failed_workers = []
    for i, start, cnt, p, lf in procs:
        p.wait()
        lf.close()
        rc = p.returncode
        tag = "ok" if rc == 0 else f"FAILED rc={rc}"
        if rc != 0:
            failed_workers.append(i)
        print(f"worker {i} done ({tag}) at {time.time() - t0:.0f}s", flush=True)

    copied, total_samples, hdf5_skipped = _merge_hdf5_shards(procs, shard_dir, final_hdf5)
    rows, csv_skipped = _merge_csv_shards(procs, shard_dir, final_csv)

    n_succ = sum(1 for r in rows if r[3] == "1")
    print(f"\nMERGED: {copied} demos ({total_samples} samples) -> {final_hdf5}", flush=True)
    print(f"success {n_succ}/{len(rows)}  CSV={final_csv}", flush=True)
    print("status counts:", dict(Counter(r[7] for r in rows)), flush=True)
    if failed_workers:
        print(f"failed workers: {failed_workers}", flush=True)
    all_skipped = hdf5_skipped + csv_skipped
    if all_skipped:
        print("skipped shards:", all_skipped, flush=True)
    if len(rows) < cfg.num_episodes:
        done = {int(r[0]) for r in rows}
        missing = [i for i in range(cfg.num_episodes) if i not in done]
        print(f"missing episodes ({len(missing)}): {missing}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
