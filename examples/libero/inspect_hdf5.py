# run 'uv run  examples/libero/inspect_hdf5.py data/libero/videos/training_data_libero_goal.hdf5'

import h5py
import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

def extract_hdf5_data(hdf5_path):
    """
    Extracts HDF5 data into a timestamped folder in the same directory as the input file.
    Structure:
      [hdf5_dir]/[filename]_extracted_[timestamp]/
        task_name/
          episode_0/
            agent_images/
            wrist_images/
            data.csv
            metadata.json
    """
    file_path = Path(hdf5_path).resolve()
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    # Generate output directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{file_path.stem}_extracted_{timestamp}"
    output_root = file_path.parent / output_dir_name
    
    print(f"Processing: {file_path}")
    print(f"Extracting to: {output_root}")

    with h5py.File(file_path, "r") as f:
        # Iterate over Task Groups
        # The structure is Group(Task) -> Group(Episode) -> Datasets
        tasks = list(f.keys())
        
        for task_name in tqdm(tasks, desc="Extracting Tasks"):
            
            # Iterate over Episodes within the task
            for ep_id in f[task_name].keys():
                ep_group = f[task_name][ep_id]
                
                # Create Output Directory
                # Structure: root / task_name / episode_id
                save_dir = output_root / task_name / ep_id
                img_dir = save_dir / "agent_images"
                wrist_dir = save_dir / "wrist_images"
                
                img_dir.mkdir(parents=True, exist_ok=True)
                wrist_dir.mkdir(parents=True, exist_ok=True)

                # --- 1. Extract Metadata ---
                metadata = {}
                # Extract prompt from attributes
                if "language_instruction" in ep_group.attrs:
                    metadata["language_instruction"] = ep_group.attrs["language_instruction"]
                
                # Save metadata
                with open(save_dir / "metadata.json", "w") as jf:
                    json.dump(metadata, jf, indent=4)

                # --- 2. Extract Images ---
                # HDF5 handles gzip decompression automatically when slicing
                if "observation/image" in ep_group:
                    images = ep_group["observation/image"][:]
                    for idx, img in enumerate(images):
                        # Convert RGB (from Libero/HDF5) to BGR (for OpenCV)
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(img_dir / f"{idx:05d}.png"), bgr_img)
                
                if "observation/wrist_image" in ep_group:
                    wrist_images = ep_group["observation/wrist_image"][:]
                    for idx, img in enumerate(wrist_images):
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(wrist_dir / f"{idx:05d}.png"), bgr_img)

                # --- 3. Extract Robot State / Numerical Data ---
                data_dict = {}
                
                # Scalar/1D Data
                scalar_keys = ["timestamp", "done"]
                for key in scalar_keys:
                    if key in ep_group:
                        data_dict[key] = ep_group[key][:]

                # Vector/2D Data (flatten to columns)
                # Matches the keys defined in your DataCollector class
                vector_keys = [
                    "robot_state/joint_positions",
                    "robot_state/gripper_state",
                    "robot_state/eef_position",
                    "robot_state/eef_quaternion"
                ]

                for key in vector_keys:
                    if key in ep_group:
                        arr = ep_group[key][:]
                        # If shape is (T, D), create D columns
                        if len(arr.shape) > 1:
                            dim = arr.shape[1]
                            short_name = key.split("/")[-1]
                            for d in range(dim):
                                col_name = f"{short_name}_{d}"
                                data_dict[col_name] = arr[:, d]
                        else:
                            # Fallback if somehow 1D
                            short_name = key.split("/")[-1]
                            data_dict[short_name] = arr[:]

                # Save to CSV
                try:
                    df = pd.DataFrame(data_dict)
                    df.to_csv(save_dir / "trajectory_data.csv", index_label="step")
                except Exception as e:
                    print(f"Error saving CSV for {task_name}/{ep_id}: {e}")

    print(f"\nSuccess! Data extracted to:\n{output_root}")


class HDF5Loader:
    """
    Helper class to load data directly in Python code (no extraction).
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def get_episode_keys(self):
        keys = []
        with h5py.File(self.file_path, 'r') as f:
            for task in f.keys():
                for ep in f[task].keys():
                    keys.append((task, ep))
        return keys

    def load_episode(self, task_name, episode_id):
        data = {}
        with h5py.File(self.file_path, 'r') as f:
            grp = f[task_name][episode_id]
            data['prompt'] = grp.attrs.get("language_instruction", "")
            
            def visit_func(name, node):
                if isinstance(node, h5py.Dataset):
                    data[name] = node[:]
            grp.visititems(visit_func)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HDF5 training data to a timestamped folder.")
    parser.add_argument("path", type=str, help="Path to the input .hdf5 file")
    
    args = parser.parse_args()
    
    extract_hdf5_data(args.path)