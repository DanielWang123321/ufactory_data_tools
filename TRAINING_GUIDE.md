# Training Guide - LeRobot 0.4.1

Quick reference for training robot policies.

---

## Installation

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt  # installs lerobot==0.4.1
```

**Important**: 0.4.1 expects datasets in codebase version `v3.0`. Reconvert or upgrade old datasets before training.

---

## Quick Start

### Test Run (100 steps, ~1 min)

```bash
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot_data \
    --dataset.root=./lerobot_data \
    --policy.type=act \
    --policy.push_to_hub=false \
    --steps=100 \
    --batch_size=2 \
    --output_dir=./outputs/test
```

### Full Training (10,000 steps, ~2-3 hrs)

```bash
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot_data \
    --dataset.root=./lerobot_data \
    --policy.type=act \
    --policy.push_to_hub=false \
    --steps=10000 \
    --batch_size=8 \
    --optimizer.lr=1e-4 \
    --eval_freq=1000 \
    --save_freq=5000 \
    --output_dir=./outputs/training
```

---

## Core Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| `--dataset.repo_id` | Dataset ID | `lerobot_data` |
| `--dataset.root` | Dataset path | `./lerobot_data` |
| `--policy.type` | Policy algorithm | `act`, `diffusion`, `tdmpc` |
| `--policy.push_to_hub` | Upload to Hub | `false` |
| `--steps` | Training steps | `10000` |
| `--batch_size` | Batch size | `8` (4 for 8GB GPU) |
| `--optimizer.lr` | Learning rate | `1e-4` |
| `--eval_freq` | Eval interval | `1000` |
| `--save_freq` | Save interval | `5000` |
| `--output_dir` | Output directory | `./outputs/training` |

---

## Policies

### ACT (Recommended)
Best for robot manipulation with smooth trajectories.

```bash
--policy.type=act --steps=10000 --batch_size=8
```

### Diffusion
Best for complex, multi-modal tasks.

```bash
--policy.type=diffusion --steps=20000 --batch_size=16
```

### TDMPC
Best for limited data (model-based).

```bash
--policy.type=tdmpc --steps=15000 --batch_size=32
```

---

## GPU Settings

| GPU | Batch Size |
|-----|------------|
| 8GB (4060 Ti) | `4-8` |
| 16GB+ (3090) | `16` |

**Out of memory?** Reduce batch size: `--batch_size=2`

---

## Output Structure

```
outputs/training/
├── checkpoints/
│   ├── 005000/pretrained_model/
│   └── 010000/pretrained_model/
├── config.json
└── logs/
```

---

## Monitoring

```bash
# View logs
tail -f outputs/training/logs/train.log

# GPU usage
watch -n 1 nvidia-smi
```

**Key metrics**:
- `train/loss`: Should decrease
- `train/grad_norm`: Should be stable

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "Repository Not Found" | Add `--policy.push_to_hub=false` |
| CUDA out of memory | Reduce batch size: `--batch_size=4` |
| Dataset version mismatch | Ensure LeRobot 0.4.1 is installed and dataset was exported in v3.0 format |
| `ForwardCompatibilityError` | Run `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --root ./lerobot_data --repo-id <id>` or rerun `hdf5_to_lerobot.py` |

---

## Best Practices

1. Start with 100-step test run
2. Monitor GPU usage (keep below 90%)
3. Save checkpoints frequently
4. Use descriptive output directories
5. Backup important models

---

## Next Steps

After training:
1. Evaluate model on validation data
2. Deploy policy for robot control
3. Collect more data if needed
4. Iterate and improve

See [LeRobot docs](https://github.com/huggingface/lerobot) for deployment.
