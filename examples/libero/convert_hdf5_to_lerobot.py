"""
Convert Libero demo HDF5s to LeRobot format. Expects structure: data/demo_0, demo_1, ... with
obs/agentview_rgb, obs/eye_in_hand_rgb, robot_states (9,), actions (7,). Output to
$HF_LEROBOT_HOME/<output_repo_id>; use data.repo_id and data.dataset_root in your training config.

Usage:
  uv run examples/libero/convert_hdf5_to_lerobot.py --data-dir path/to/file.hdf5
  uv run examples/libero/convert_hdf5_to_lerobot.py --data-dir path/to/folder_of_hdf5s
"""

import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

TARGET_IMAGE_SHAPE = (256, 256, 3)


def _output_repo_id_from_data_dir(data_dir: str) -> str:
    """Derive output repo id from --data-dir (same as input path, under local/)."""
    p = Path(data_dir).resolve()
    if p.is_file():
        name = p.stem
        parent = p.parent.name
        repo_name = f"{parent}_{name}" if parent and parent != "." else name
    else:
        repo_name = p.name or "lerobot_dataset"
    return f"local/{repo_name}"


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 (H, W, 3) for LeRobot."""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.ascontiguousarray(img)


def _resize_image(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize to (target_height, target_width, 3)."""
    h, w = target_hw[0], target_hw[1]
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _collect_hdf5_paths(data_dir: str) -> list[Path]:
    """Return list of HDF5 paths: single file if data_dir is a .hdf5 file, else all .hdf5 in the folder."""
    p = Path(data_dir).resolve()
    if p.is_file():
        if p.suffix.lower() != ".hdf5":
            raise ValueError(f"Single file given but not .hdf5: {p}")
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {p}")
    paths = sorted(p.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files found in {p}")
    return paths


def _task_from_filename(path: Path) -> str:
    """Derive task string from filename, e.g. ..._put_the_black_bowl_..._demo.hdf5 -> put the black bowl ..."""
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    # Drop leading scene prefix like KITCHEN_SCENE10_
    if "_" in stem:
        parts = stem.split("_")
        for i, p in enumerate(parts):
            if p and p[0].islower():
                return " ".join(parts[i:])
    print(f"Task description: {stem.replace('_', ' ')}")
    return stem.replace("_", " ")


def _process_hdf5_file(f: h5py.File, dataset: LeRobotDataset, path: Path) -> int:
    """Load data/demo_* from Libero demo HDF5 and append to dataset. Returns number of demos added."""
    if "data" not in f:
        raise ValueError(f"Expected 'data' group at root (Libero demo format); got keys {list(f.keys())}")
    data = f["data"]
    demo_keys = sorted(
        [k for k in data.keys() if k.startswith("demo_")],
        key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
    )
    if not demo_keys:
        raise ValueError(f"No data/demo_* groups in {path}")
    task_desc = _task_from_filename(path)

    for key in demo_keys:
        demo = data[key]
        images = np.array(demo["obs/agentview_rgb"])
        wrist_images = np.array(demo["obs/eye_in_hand_rgb"])
        robot_states = np.array(demo["robot_states"], dtype=np.float32)
        actions = np.array(demo["actions"], dtype=np.float32)
        # LeRobot state is (8,); libero robot_states is (T, 9) — use first 8
        states = robot_states[:, :8]

        n = len(actions)
        if n != len(states) or n != len(images) or n != len(wrist_images):
            raise ValueError(
                f"{path.name} {key}: length mismatch actions={n} states={len(states)} "
                f"images={len(images)} wrist_images={len(wrist_images)}"
            )

        for i in range(n):
            img = _ensure_uint8_rgb(images[i])
            img = _resize_image(img, (TARGET_IMAGE_SHAPE[0], TARGET_IMAGE_SHAPE[1]))
            wrist = _ensure_uint8_rgb(wrist_images[i])
            wrist = _resize_image(wrist, (TARGET_IMAGE_SHAPE[0], TARGET_IMAGE_SHAPE[1]))
            dataset.add_frame(
                {
                    "image": img,
                    "wrist_image": wrist,
                    "state": states[i],
                    "actions": actions[i],
                    "task": task_desc,
                }
            )
        dataset.save_episode()
    return len(demo_keys)


def main(
    data_dir: str,
    *,
    overwrite: bool = True,
):
    """Convert Libero demo HDF5(s) to LeRobot format."""
    output_repo_id = _output_repo_id_from_data_dir(data_dir)
    root = Path(HF_LEROBOT_HOME).resolve()
    output_path = root / output_repo_id
    print(f"HF_LEROBOT_HOME={root}")
    print(f"output_repo_id={output_repo_id}")
    print(f"output_path={output_path}")
    hdf5_paths = _collect_hdf5_paths(data_dir)
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace.")

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": TARGET_IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": TARGET_IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            n = _process_hdf5_file(f, dataset, path)
            total_episodes += n
            if not n:
                raise ValueError(f"No data/demo_* groups found in {path}")

    print(f"Converted {total_episodes} episodes from {len(hdf5_paths)} file(s) to {output_path}")
    print(
        "Use with compute_norm_stats by using a config with "
        f"data.repo_id={output_repo_id!r} and data.dataset_root={str(output_path)!r}"
    )


if __name__ == "__main__":
    tyro.cli(main)
