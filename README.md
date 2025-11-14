# UFactory Data Tools

Multi-sensor data capture and LeRobot training toolkit for Pika Sense robots.

**Pipeline**: Data Capture -> HDF5 -> LeRobot v3.0 -> Training

---

## Quick Start

```bash
# 1. Install dependencies (includes lerobot==0.4.1)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/agilexrobotics/pika_sdk.git@master

# 2. Serial port permission (first time only)
sudo usermod -a -G dialout $USER  # Logout/login required

# 3. Capture data
python data_capture.py --episode 0 --fps 30
```

Press `Ctrl+C` to stop. Data saved to `captured_data/episode0/`

---

## Data Pipeline

### 1. Data Capture

```bash
python data_capture.py --episode 0 --fps 30
```

Output: `captured_data/episode0/` (RealSense, fisheye, depth, gripper, pose)

### 2. Convert to HDF5

```bash
python data_to_hdf5.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --episodeName episode0 \
    --useIndex True
```

Output: `captured_data/episode0/data.hdf5`

### 3. Convert to LeRobot

```bash
python hdf5_to_lerobot.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --datasetName my_dataset \
    --targetDir ./lerobot_data \
    --instruction "pick and place"
```

Output: `lerobot_data/` (codebase v3.0 format required by LeRobot 0.4.1+).

### 4. Verify Dataset (Optional)

```bash
python verify_dataset.py
```

The verifier checks that `codebase_version == "v3.0"`, prints key tensor shapes, and fails fast if reconversion or the official v2.1->v3.0 upgrade script is required.

---

## Migrating Legacy Datasets

Already have a LeRobot v2.1 dataset? Upgrade it in-place (or in a copy) before training:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id <org-or-local>/<dataset_name> \
    --root ./lerobot_data \
    --push-to-hub false
```

The conversion script rewrites parquet chunks and concatenates mp4s; ensure you have enough free disk space. If you only kept the raw HDF5 files, simply rerun `hdf5_to_lerobot.py` with this repository.

---

## Training

### Test Run (100 steps)

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

### Full Training (10,000 steps)

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

**GPU Settings**:
- 8GB GPU: `--batch_size=4`
- 16GB+ GPU: `--batch_size=8` or higher

**Policies**: `act`, `diffusion`, `tdmpc` (ACT recommended for manipulation)

See `TRAINING_GUIDE.md` for detailed parameters.

---

## Key Arguments

### data_capture.py
- `--episode`: Episode number
- `--fps`: Frame rate (default: 30)

### data_to_hdf5.py
- `--type`: Robot type (`single_pika`)
- `--datasetDir`: Dataset directory
- `--episodeName`: Episode name
- `--useIndex`: Use index mode (`True` recommended)

### hdf5_to_lerobot.py
- `--type`: Robot type (`single_pika`)
- `--datasetDir`: HDF5 directory
- `--targetDir`: Output directory
- `--datasetName`: Dataset/Hub repo id stored in metadata
- `--instruction`: Task label stored in each episode (`"null"` if unused)

### verify_dataset.py
- `--dataset-root`: Dataset path (default: `./lerobot_data`)
- `--dataset-repo-id`: Repo ID used for loading (default: `local/pika_dataset`)

---

## Requirements

- Ubuntu 20.04+
- Python 3.10+
- LeRobot 0.4.1+ (dataset codebase v3.0)
- NVIDIA GPU with CUDA (recommended for training)
- Additional disk space for chunked parquet/mp4 data (see notes below)

---

## Files

| File | Description |
|------|-------------|
| `data_capture.py` | Multi-sensor synchronized capture |
| `data_to_hdf5.py` | HDF5 converter |
| `hdf5_to_lerobot.py` | LeRobot v3.0 converter |
| `verify_dataset.py` | Dataset validation |
| `single_pika_data_params.yaml` | Sensor configuration |

---

## Notes on Storage Layout

LeRobot 0.4.1 writes data in chunked parquet/mp4 files (`data/chunk-XXX/file-YYY.parquet`, `videos/<camera>/chunk-XXX/file-YYY.mp4`). This improves streaming but temporarily duplicates video frames during encoding. Adjust the `DatasetConfig` in `hdf5_to_lerobot.py` if you need to tweak chunk counts, parallel writers, or disable video export.


## References

- [Pika SDK](https://github.com/agilexrobotics/pika_sdk)
- [LeRobot](https://github.com/huggingface/lerobot)
