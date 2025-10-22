# LeRobot Version Compatibility Notes

## Current Environment

- **LeRobot Version**: 0.3.3
- **Python Version**: 3.10
- **Environment**: `/home/daniel/Desktop/envs/py310`

## API Differences: 0.3.x vs 0.4.x+

### 1. Import Path Changes

| Version | Import Statement |
|---------|-----------------|
| **0.3.x** (current) | `from lerobot.datasets.lerobot_dataset import LeRobotDataset` |
| 0.4.x+ | `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset` |

**Why**: LeRobot 0.3.x does not have the `common` submodule. It was introduced in v0.4.0+.

### 2. Loading Local Datasets

#### ❌ Incorrect (causes `HFValidationError`)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# This will fail: './lerobot_data' is not a valid HF repo_id
ds = LeRobotDataset('./lerobot_data')
```

**Error**: `HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.'`

#### ✅ Correct (works in 0.3.x)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# Provide both a valid repo_id AND local root path
ds = LeRobotDataset(
    repo_id='local/pika_dataset',  # Valid HF format identifier
    root=str(Path('./lerobot_data').resolve())  # Absolute local path
)
```

**Why**: LeRobotDataset requires a valid Hugging Face repo_id format (alphanumeric, `-`, `_`, `.` only). The `root` parameter tells it to load from a local directory instead of downloading from the Hub.

### 3. Training Commands

#### LeRobot CLI (Recommended)

```bash
lerobot train \
  --dataset-repo-id local/pika_dataset \
  --root ./lerobot_data \
  --policy-name diffusion \
  --output-dir ./outputs \
  --num-epochs 100 \
  --batch-size 8
```

#### Python Script (Helper)

```bash
python train.py \
  --dataset-repo-id local/pika_dataset \
  --dataset-root ./lerobot_data \
  --policy diffusion \
  --num-epochs 100
```

## Upgrade Path

If you want to use LeRobot 0.4.0+ (with LeRobotDataset v3.0):

```bash
# Upgrade LeRobot
pip install --upgrade lerobot

# Migrate dataset format (if needed)
lerobot migrate-dataset \
  --from-version 2.1 \
  --to-version 3.0 \
  --dataset-path ./lerobot_data
```

**Note**: LeRobot 0.4.0+ introduces streaming datasets, better performance, and improved API consistency.

## Quick Reference

### Dataset Loading (0.3.x)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# Load local dataset
ds = LeRobotDataset(
    repo_id='local/pika_dataset',
    root=str(Path('./lerobot_data').resolve())
)

# Access dataset info
print(f"Episodes: {ds.meta.info['total_episodes']}")
print(f"Frames: {ds.meta.info['total_frames']}")
print(f"FPS: {ds.meta.info['fps']}")

# Iterate through frames
for i in range(len(ds)):
    frame = ds[i]
    observation = frame['observation']
    action = frame['action']
```

### Policy Modules (0.3.x)

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPC
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'lerobot.common'`

**Solution**: You're using code for LeRobot 0.4.x+ on version 0.3.x. Change import:
```python
# Change this:
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# To this:
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

### Issue: `HFValidationError: Repo id must use alphanumeric chars`

**Solution**: Use valid repo_id format with `root` parameter:
```python
# Instead of:
ds = LeRobotDataset('./lerobot_data')

# Use:
ds = LeRobotDataset(repo_id='local/pika_dataset', root='./lerobot_data')
```

### Issue: Training command not found

**Solution**: Ensure LeRobot CLI is accessible:
```bash
which lerobot

# If not found, try:
python -m lerobot.scripts.train --help

# Or reinstall:
pip install --force-reinstall lerobot
```

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobotDataset v3.0 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [LeRobot Training Guide](https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot)
