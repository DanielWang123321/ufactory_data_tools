# UFactory Data Tools

Multi-sensor data capture and LeRobot training toolkit for Pika Sense robots.

**Pipeline**: Data Capture → HDF5 → LeRobot → Training

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install lerobot==0.3.3
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
    --targetDir ./lerobot_data
```

Output: `lerobot_data/` (v2.1 format, compatible with LeRobot 0.3.3)

### 4. Verify Dataset (Optional)

```bash
python verify_dataset.py
```

Output: Dataset statistics and structure

---

## Training

### Test Run (100 steps)

```bash
python -m lerobot.scripts.train \
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
python -m lerobot.scripts.train \
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

### verify_dataset.py
- `--dataset-root`: Dataset path (default: `./lerobot_data`)

---

## Requirements

- Ubuntu 20.04+
- Python 3.10+
- LeRobot 0.3.3 (do NOT use 0.3.4)
- NVIDIA GPU with CUDA (recommended for training)

---

## Files

| File | Description |
|------|-------------|
| `data_capture.py` | Multi-sensor synchronized capture |
| `data_to_hdf5.py` | HDF5 converter |
| `hdf5_to_lerobot.py` | LeRobot v2.1 converter |
| `verify_dataset.py` | Dataset validation |
| `single_pika_data_params.yaml` | Sensor configuration |


## References

- [Pika SDK](https://github.com/agilexrobotics/pika_sdk)
- [LeRobot](https://github.com/huggingface/lerobot)
