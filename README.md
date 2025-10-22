# UFactory Data Tools

Multi-sensor synchronized data capture and conversion toolkit for Pika Sense (Python, no ROS required)

**Complete Pipeline**: Data Capture → HDF5 → LeRobot Format

**LeRobot Version**: 0.3.x compatible

---

## Quick Start

```bash
# 1. Install dependencies
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

### Step 4: Verify Dataset (Optional)

```bash
# Verify LeRobot dataset is correctly formatted
python train.py
```

**Output:** Dataset validation report (episodes, frames, observations)

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

---


## System Requirements

- **OS**: Ubuntu 20.04/22.04/24.04
- **Python**: 3.10+
- **Hardware**: Pika Sense, Ufactory Robots(Lite6, xArm 5/6/7 or 850)

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


## References

- **Pika ROS**: https://github.com/agilexrobotics/pika_ros
- **Pika SDK**: https://github.com/agilexrobotics/pika_sdk
- **Ufactory Teleop**: https://github.com/xArm-Developer/ufactory_teleop
- **LeRobot**: https://github.com/huggingface/lerobot
