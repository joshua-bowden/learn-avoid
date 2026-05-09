## pi0.5 LoRA Finetune on LIBERO (single task, local data)

LoRA finetuning on top of the pi0.5 libero checkpoint with a local dataset.

### 1. Download LIBERO demonstrations
```
uv run --group rlds examples/libero/download_modified_libero_rlds.py --output-dir ~/openpi/data/modified_libero_rlds
```

### 2. Convert to LeRobot format
```
uv run --group rlds examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir ~/openpi/data/modified_libero_rlds \
  --task-filter "turn on the stove and put the moka pot on it" \
  --repo-id libero_10/stove_moka
```
Output: `~/openpi/data/lerobot/libero_10/stove_moka`

`--task-filter` does exact match on episode `language_instruction`. Find all possible instructions at openpi/third_party/libero/libero/libero/bddl_files/libero_10/tasks_info.txt, etc 
`--repo-id` sets the output folder name. Omit it to default based on task-filter.

If you change `--repo-id`, update `PI05_LORA_LOCAL_REPO_ID` in `src/openpi/training/config.py` to match.

### 3. Tune training config

Edit the `pi05_lora_local` block in `src/openpi/training/config.py`:
- `num_train_steps` (default 5000)
- `peak_lr` (default 2.5e-6)
- `save_interval` (default 1000 steps) for checkpoints; also see max_to_keep in src/openpi/training/checkpoints.py

### 4. Norm stats

The config already loads norm stats from the original libero finetune checkpoint (`gs://openpi-assets/checkpoints/pi05_libero/assets`). 

Seems that the original norm stats should be good; you could recompute norm stats based on only the specific task using:
uv run scripts/compute_norm_stats.py --config-name pi05_libero


### 5. Run finetuning
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lora_local --exp-name=my_experiment --overwrite
```

### 6. Eval

Follow the setup in `examples/libero/README.md` ("Without Docker"). Then:

Terminal 1 — serve your finetuned checkpoint:
```
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/pi05_lora_local/my_experiment
```

Terminal 2 — run the LIBERO sim (change suite name and task id in code to run only the chosen task):
```
python examples/libero/main.py
```
