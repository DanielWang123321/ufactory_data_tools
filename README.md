# UFactory Data Tools

Multi-sensor synchronized data capture and conversion toolkit for Pika Sense (Python, no ROS required)

**Complete Pipeline**: Data Capture → HDF5 → LeRobot Format → Model Training

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

## Troubleshooting

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
# Capture 3 episodes
python data_capture.py --episode 0 --fps 30
python data_capture.py --episode 1 --fps 30
python data_capture.py --episode 2 --fps 30

# Convert to HDF5
for i in 0 1 2; do
    python data_to_hdf5.py \
        --type single_pika \
        --datasetDir ./captured_data \
        --episodeName episode${i} \
        --useIndex True
done

# Convert to LeRobot
python hdf5_to_lerobot.py \
    --type single_pika \
    --datasetDir ./captured_data \
    --datasetName pika_demo \
    --targetDir ./lerobot_data
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
| `single_pika_data_params.yaml` | Sensor configuration | 1.8KB |
| `requirements.txt` | Python dependencies | 773B |
| `README.md` | This document | - |

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
