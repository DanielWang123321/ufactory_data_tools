# UFactory Data Tools

Multi-sensor synchronized data capture and conversion toolkit for Pika Sense (Python, no ROS required)

**Complete Pipeline**: Data Capture → HDF5 → LeRobot Format → Model Training

**LeRobot Version**: 0.3.x compatible

---

## Quick Start

```bash
# 1. Install dependencies
conda create -y -n pika python=3.10
conda activate pika
pip install -r requirements.txt
pip install git+https://github.com/agilexrobotics/pika_sdk.git@master

# 2. Serial port permission (first time)
sudo usermod -a -G dialout $USER  # Logout and login required

# 3. Capture data
python data_capture.py --episode 0 --fps 30
```

Press `Ctrl+C` to stop. Data saved to `captured_data/episode0/`

---

## Data Pipeline

### Step 1: Data Capture

```bash
python data_capture.py --episode 0 --fps 30
```

**Output Structure:**
```
captured_data/episode0/
├── camera/color/pikaDepthCamera/    # RealSense D405
├── camera/color/pikaFisheyeCamera/  # Fisheye camera
├── camera/depth/pikaDepthCamera/    # Depth images
├── gripper/encoder/pika/            # Gripper distance (mm)
├── localization/pose/pika/          # 6DOF pose (x,y,z, roll,pitch,yaw)
└── statistics.json
```

### Step 2: Convert to HDF5

```bash
python data_to_hdf5.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --episodeName episode0 \
    --useIndex True
```

**Output:** `captured_data/episode0/data.hdf5`

### Step 3: Convert to LeRobot Format

```bash
python hdf5_to_lerobot.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --datasetName my_dataset \
    --targetDir ./lerobot_data
```

**Output:** LeRobot dataset with videos (MP4) and data (Parquet)

### Step 4: Train Policy (Optional)

```bash
# Verify dataset
python train.py

# Start training (using LeRobot CLI)
lerobot train \
    --dataset-repo-id local/pika_dataset \
    --root ./lerobot_data \
    --policy-name diffusion \
    --output-dir ./outputs \
    --num-epochs 100 \
    --batch-size 8 \
    --device cuda
```

**Output:** Trained policy checkpoint in `./outputs/`

---

## Command Line Arguments

### data_capture.py
- `--config`: YAML config file (default: `./single_pika_data_params.yaml`)
- `--output`: Output directory (default: `./captured_data`)
- `--episode`: Episode number (default: `0`)
- `--fps`: Capture frame rate (default: `30`)

### data_to_hdf5.py
- `--type`: Robot type (`single_pika`)
- `--datasetDir`: Dataset root directory
- `--episodeName`: Episode name (e.g., `episode0`)
- `--useIndex`: Use index mode (recommended: `True`)

### hdf5_to_lerobot.py
- `--type`: Robot type (`single_pika`)
- `--datasetDir`: Directory containing HDF5 files
- `--datasetName`: LeRobot dataset name
- `--targetDir`: Output directory

### train.py
- `--dataset-repo-id`: Dataset repo ID (default: `local/pika_dataset`)
- `--dataset-root`: Local dataset path (default: `./lerobot_data`)
- `--output-dir`: Output directory (default: `./outputs`)
- `--policy`: Policy type (`diffusion`, `act`, `tdmpc`)
- `--batch-size`: Training batch size (default: `8`)
- `--num-epochs`: Number of epochs (default: `100`)
- `--device`: Device (`cuda` or `cpu`)

---

## Data Format

### Sensor Data

**Gripper** (`gripper/encoder/pika/*.json`):
```json
{"distance": 99.00}
```

**Pose** (`localization/pose/pika/*.json`):
```json
{"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
```

### LeRobot Format

**observation.state** (7-dim vector):
```
[gripper_distance, x, y, z, roll, pitch, yaw]
```

**observation.images**:
- `pikaDepthCamera`: RGB image (3, 480, 640)
- `pikaFisheyeCamera`: RGB image (3, 480, 640)

**action** (7-dim vector): Same as observation.state (teleoperation data)

---

## Training Details

<details>
<summary><b>Click to expand training guide</b></summary>

### Prerequisites

```bash
# Ensure LeRobot is installed in your environment
python -c "import lerobot; print(f'LeRobot {lerobot.__version__}')"

# Should output: LeRobot 0.3.x
```

### Load Dataset (LeRobot 0.3.x)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# Load local dataset (requires valid repo_id + root path)
ds = LeRobotDataset(
    repo_id='local/pika_dataset',
    root=str(Path('./lerobot_data').resolve())
)

print(f"Episodes: {ds.meta.info['total_episodes']}")
print(f"Frames: {ds.meta.info['total_frames']}")
```

### Training Commands

**Quick Test (3 epochs)**:
```bash
lerobot train \
  --dataset-repo-id local/pika_dataset \
  --root ./lerobot_data \
  --policy-name diffusion \
  --output-dir ./outputs/test \
  --num-epochs 3 \
  --batch-size 2 \
  --device cuda
```

**Full Training**:
```bash
lerobot train \
  --dataset-repo-id local/pika_dataset \
  --root ./lerobot_data \
  --policy-name diffusion \
  --output-dir ./outputs \
  --num-epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cuda \
  --eval-freq 10
```

**With Weights & Biases**:
```bash
lerobot train \
  --dataset-repo-id local/pika_dataset \
  --root ./lerobot_data \
  --policy-name diffusion \
  --output-dir ./outputs \
  --wandb-project pika-training \
  --wandb-entity your-username
```

### Available Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| `diffusion` | Diffusion Policy | Smooth trajectories (recommended) |
| `act` | Action Chunking Transformer | Sequential tasks |
| `tdmpc` | TD-MPC | Model-based RL |

### Data Requirements

- **Minimum**: 10+ episodes for meaningful training
- **Recommended**: 50-100 episodes for robust policies
- **Current dataset**: 1 episode (validation only)

</details>

---

## Troubleshooting

### ModuleNotFoundError: No module named 'lerobot.common'
**Cause**: Using code for LeRobot 0.4.x+ on version 0.3.x

**Solution**: Use correct import path
```python
# Change this (0.4.x+):
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# To this (0.3.x):
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

### HFValidationError: Repo id must use alphanumeric chars
**Cause**: Invalid repo_id format (e.g., `./lerobot_data`)

**Solution**: Use valid repo_id + root parameter
```python
# Wrong:
ds = LeRobotDataset('./lerobot_data')

# Correct:
ds = LeRobotDataset(repo_id='local/pika_dataset', root='./lerobot_data')
```

### ModuleNotFoundError: lerobot
```bash
pip uninstall lerobot -y
pip install lerobot>=0.3.0
```

### Camera not found
```bash
lsusb | grep Intel       # Check RealSense
ls /dev/video*           # Check Dexcin (should be /dev/video6)
```

### Serial port permission error
```bash
sudo usermod -a -G dialout $USER
# Logout and login to apply
```

### Low FPS / Missed frames
- Use USB 3.0 port
- Lower FPS: `--fps 20`
- Close other CPU-intensive programs

---

## Example Workflow

```bash
# 1. Capture 3 episodes
python data_capture.py --episode 0 --fps 30
python data_capture.py --episode 1 --fps 30
python data_capture.py --episode 2 --fps 30

# 2. Convert to HDF5
for i in 0 1 2; do
    python data_to_hdf5.py \
        --type single_pika \
        --datasetDir ./captured_data \
        --episodeName episode${i} \
        --useIndex True
done

# 3. Convert to LeRobot
python hdf5_to_lerobot.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --datasetName pika_demo \
    --targetDir ./lerobot_data

# 4. Train policy (optional)
lerobot train \
    --dataset-repo-id local/pika_dataset \
    --root ./lerobot_data \
    --policy-name diffusion \
    --output-dir ./outputs \
    --num-epochs 100
```

---

## System Requirements

- **OS**: Ubuntu 20.04+
- **Python**: 3.10+
- **RAM**: 16GB+ (recommended)
- **Hardware**: RealSense D405, Dexcin Fisheye, Pika Sense, Vive Tracker

---

## Files

| File | Description | Size |
|------|-------------|------|
| `data_capture.py` | Data capture with master clock sync | 21KB |
| `data_to_hdf5.py` | HDF5 format converter | 31KB |
| `hdf5_to_lerobot.py` | LeRobot format converter | 16KB |
| `train.py` | Training helper script | 3.9KB |
| `single_pika_data_params.yaml` | Sensor configuration | 501B |
| `requirements.txt` | Python dependencies | 773B |
| `README.md` | Complete documentation | - |

---

## Features

✅ **Master Clock Sync** - All sensors share unified timestamp (< 0.001ms error)  
✅ **Data Integrity** - 100% frame alignment guarantee  
✅ **ROS-Free** - Pure Python implementation  
✅ **Auto Indexing** - Automatic sync.txt generation  
✅ **Complete Pipeline** - From capture to training  

---

## References

- **Pika ROS**: https://github.com/agilexrobotics/pika_ros
- **Pika SDK**: https://github.com/agilexrobotics/pika_sdk
- **Ufactory Teleop**: https://github.com/xArm-Developer/ufactory_teleop
- **LeRobot**: https://github.com/huggingface/lerobot
