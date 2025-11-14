"""
Convert UFactory/Pika Sense HDF5 data to the LeRobot dataset v3.0 format required by lerobot==0.4.1.

Example usage:
python hdf5_to_lerobot.py --type single_pika --datasetDir ./captured_data --datasetName local/pika --targetDir ./lerobot_data
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import cv2
import os
import argparse
import yaml


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 8
    video_backend: str | None = None
    chunks_size: int | None = None
    data_files_size_in_mb: int | None = None
    video_files_size_in_mb: int | None = None
    batch_encoding_size: int = 1


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    args,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    states = []
    actions = []
    
    # Support for arm-based robots (ALOHA)
    for i in range(len(args.armJointStateNames)):
        if "puppet" in args.armJointStateNames[i]:
            for j in range(args.armJointStateDims[i]):
                states += [f'arm.jointStatePosition.{args.armJointStateNames[i]}.joint{j}']
        if "master" in args.armJointStateNames[i]:
            for j in range(args.armJointStateDims[i]):
                actions += [f'arm.jointStatePosition.{args.armJointStateNames[i]}.joint{j}']

    for i in range(len(args.armEndPoseNames)):
        if "puppet" in args.armEndPoseNames[i]:
            for j in range(args.armEndPoseDims[i]):
                states += [f'arm.endPose.{args.armEndPoseNames[i]}.joint{j}']
        if "master" in args.armEndPoseNames[i]:
            for j in range(args.armEndPoseDims[i]):
                actions += [f'arm.endPose.{args.armEndPoseNames[i]}.joint{j}']
    
    # Support for single_pika (gripper + localization)
    for gripper_name in args.gripperEncoderNames:
        states += [f'gripper.encoderDistance.{gripper_name}']
    
    for localization_name in args.localizationPoseNames:
        states += [f'localization.pose.{localization_name}.x']
        states += [f'localization.pose.{localization_name}.y']
        states += [f'localization.pose.{localization_name}.z']
        states += [f'localization.pose.{localization_name}.roll']
        states += [f'localization.pose.{localization_name}.pitch']
        states += [f'localization.pose.{localization_name}.yaw']
    
    # For single_pika, action = state (teleoperation data)
    if len(args.armJointStateNames) == 0 and len(args.armEndPoseNames) == 0:
        actions = states.copy()

    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": (len(states),),
            "names": [
                states,
            ],
        },
        "action": {
            "dtype": "float64",
            "shape": (len(actions),),
            "names": [
                actions,
            ],
        }
    }

    for camera in args.cameraColorNames:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }
    # for camera in args.cameraDepthNames:
    #     features[f"observation.depths.{camera}"] = {
    #         # "dtype": mode,
    #         # "shape": (1, 480, 640),
    #         # "names": [
    #         #     "channels",
    #         #     "height",
    #         #     "width",
    #         # ],
    #         "dtype": "uint16",
    #         "shape": (480, 640)
    #     }
    if args.useCameraPointCloud:
        for camera in args.cameraPointCloudNames:
            features[f"observation.pointClouds.{camera}"] = {
                "dtype": "float64",
                "shape": ((args.pointNum * 6),)
            }

    dataset = LeRobotDataset.create(
        repo_id=args.datasetName,
        root=args.targetDir,
        fps=args.fps,
        robot_type=args.robotType,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        batch_encoding_size=dataset_config.batch_encoding_size,
    )
    if (
        dataset_config.chunks_size is not None
        or dataset_config.data_files_size_in_mb is not None
        or dataset_config.video_files_size_in_mb is not None
    ):
        dataset.meta.update_chunk_settings(
            chunks_size=dataset_config.chunks_size,
            data_files_size_in_mb=dataset_config.data_files_size_in_mb,
            video_files_size_in_mb=dataset_config.video_files_size_in_mb,
        )
    return dataset


def load_episode_data(
    args,
    episode_path: Path,
):
    with h5py.File(episode_path, "r") as episode:
        try:
            # Support for arm-based robots (ALOHA)
            if len(args.armJointStateNames) > 0 or len(args.armEndPoseNames) > 0:
                states = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/jointStatePosition/{name}"][()] for name in args.armJointStateNames if "puppet" in name] + \
                        [episode[f"arm/endPose/{name}"][()] for name in args.armEndPoseNames if "puppet" in name], axis=1
                    )
                )
                actions = torch.from_numpy(
                    np.concatenate(
                        [episode[f"arm/jointStatePosition/{name}"][()] for name in args.armJointStateNames if "master" in name] + \
                        [episode[f"arm/endPose/{name}"][()] for name in args.armEndPoseNames if "master" in name], axis=1
                    )
                )
            # Support for single_pika (gripper + localization)
            else:
                state_components = []
                
                # Add gripper distance
                for gripper_name in args.gripperEncoderNames:
                    gripper_data = episode[f"gripper/encoderDistance/{gripper_name}"][()]
                    state_components.append(gripper_data.reshape(-1, 1))  # (N, 1)
                
                # Add localization pose (x,y,z, roll,pitch,yaw)
                for localization_name in args.localizationPoseNames:
                    pose_data = episode[f"localization/pose/{localization_name}"][()]  # (N, 6)
                    state_components.append(pose_data)
                
                states = torch.from_numpy(np.concatenate(state_components, axis=1))  # (N, 7)
                actions = states.clone()  # For teleoperation data, action = state
            colors = {}
            for camera in args.cameraColorNames:
                colors[camera] = []
                for i in range(episode[f'camera/color/{camera}'].shape[0]):
                    colors[camera].append(cv2.cvtColor(cv2.imread(
                        os.path.join(str(episode_path.resolve())[:-9], episode[f'camera/color/{camera}'][i].decode('utf-8')),
                        cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
                colors[camera] = colors[camera]
            depths = {}
            # for camera in args.cameraDepthNames:
            #     depths[camera] = []
            #     for i in range(episode[f'camera/depth/{camera}'].shape[0]):
            #         depths[camera].append(cv2.imread(
            #             os.path.join(str(episode_path.resolve())[:-9], episode[f'camera/depth/{camera}'][i].decode('utf-8')),
            #             cv2.IMREAD_UNCHANGED))
            pointclouds = {}
            if args.useCameraPointCloud:
                for camera in args.cameraPointCloudNames:
                    pointclouds[camera] = []
                    for i in range(episode[f'camera/pointCloud/{camera}'].shape[0]):
                        pointclouds[camera].append(np.load(
                            os.path.join(str(episode_path.resolve())[:-9], episode[f'camera/color/{camera}'][i].decode('utf-8'))))
            return colors, depths, pointclouds, states, actions
        except:
            return None, None, None, None, None

def populate_dataset(
    args,
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
) -> LeRobotDataset:
    error_file = []
    task_label = "null"
    if task is not None:
        stripped = str(task).strip()
        if stripped:
            task_label = stripped
    episodes = range(len(hdf5_files))
    for ep_idx in tqdm.tqdm(episodes):
        episode_path = hdf5_files[ep_idx]

        colors, depths, pointclouds, states, actions = load_episode_data(args, episode_path)
        if colors is not None:
            num_frames = states.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": states[i],
                    "action": actions[i],
                    "task": task_label,
                }
                for camera, color in colors.items():
                    frame[f"observation.images.{camera}"] = color[i]
                # for camera, depth in depths.items():
                #     frame[f"observation.depths.{camera}"] = depth[i]
                if args.useCameraPointCloud:
                    for camera, pointcloud in pointclouds.items():
                        frame[f"observation.pointClouds.{camera}"] = pointcloud[i]
                dataset.add_frame(frame)
                frame = None

            dataset.save_episode()
        else:
            error_file.append(episode_path)
    
    # Print processing summary
    if len(error_file) == 0:
        print(f"[OK] Successfully processed all {len(hdf5_files)} episode(s)")
    else:
        print(f"[WARN] Failed to process {len(error_file)} episode(s):")
        for path in error_file:
            print(f"  - {path}")
    
    return dataset


def process(
    args,
    push_to_hub: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    dataset_dir = Path(args.datasetDir)
    if not dataset_dir.exists():
        raise ValueError("dataset_dir does not exist")
    if Path(args.targetDir).exists():
        shutil.rmtree(Path(args.targetDir))

    hdf5_files = sorted(dataset_dir.glob("**/data.hdf5"))

    dataset = create_empty_dataset(
        args,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        args,
        dataset,
        hdf5_files,
        task=args.instruction,
    )

    if push_to_hub:
        dataset.push_to_hub()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', action='store', type=str, help='datasetDir',
                        default="/home/agilex/data", required=False)
    parser.add_argument('--datasetName', action='store', type=str, help='datasetName',
                        default="data", required=False)
    parser.add_argument('--type', action='store', type=str, help='type',
                        default="aloha", required=False)
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default="null", required=False)
    parser.add_argument('--targetDir', action='store', type=str, help='targetDir',
                        default="/home/agilex/data", required=False)
    parser.add_argument('--robotType', action='store', type=str, help='robotType',
                        default="cobot_magic", required=False)
    parser.add_argument('--fps', action='store', type=int, help='fps',
                        default=30, required=False)
    parser.add_argument('--cameraColorNames', action='store', type=str, help='cameraColorNames',
                        default=[], required=False)
    parser.add_argument('--cameraDepthNames', action='store', type=str, help='cameraDepthNames',
                        default=[], required=False)
    parser.add_argument('--cameraPointCloudNames', action='store', type=str, help='cameraPointCloudNames',
                        default=[], required=False)
    parser.add_argument('--useCameraPointCloud', action='store', type=bool, help='useCameraPointCloud',
                        default=False, required=False)
    parser.add_argument('--pointNum', action='store', type=int, help='point_num',
                        default=5000, required=False)
    parser.add_argument('--armJointStateNames', action='store', type=str, help='armJointStateNames',
                        default=[], required=False)
    parser.add_argument('--armJointStateDims', action='store', type=int, help='armJointStateDims',
                        default=[], required=False)
    parser.add_argument('--armEndPoseNames', action='store', type=str, help='armEndPoseNames',
                        default=[], required=False)
    parser.add_argument('--armEndPoseDims', action='store', type=int, help='armEndPoseDims',
                        default=[], required=False)
    parser.add_argument('--localizationPoseNames', action='store', type=str, help='localizationPoseNames',
                        default=[], required=False)
    parser.add_argument('--gripperEncoderNames', action='store', type=str, help='gripperEncoderNames',
                        default=[], required=False)
    parser.add_argument('--imu9AxisNames', action='store', type=str, help='imu9AxisNames',
                        default=[], required=False)
    parser.add_argument('--lidarPointCloudNames', action='store', type=str, help='lidarPointCloudNames',
                        default=[], required=False)
    parser.add_argument('--robotBaseVelNames', action='store', type=str, help='robotBaseVelNames',
                        default=[], required=False)
    parser.add_argument('--liftMotorNames', action='store', type=str, help='liftMotorNames',
                        default=[], required=False)
    args = parser.parse_args()

    with open(f'./{args.type}_data_params.yaml', 'r') as file:
        yaml_data = yaml.safe_load(file)
        args.cameraColorNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('color', {}).get('names', [])
        args.cameraDepthNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('depth', {}).get('names', [])
        args.cameraPointCloudNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('camera', {}).get('pointCloud', {}).get('names', [])
        args.armJointStateNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('arm', {}).get('jointState', {}).get('names', [])
        args.armEndPoseNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('arm', {}).get('endPose', {}).get('names', [])
        args.localizationPoseNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('localization', {}).get('pose', {}).get('names', [])
        args.gripperEncoderNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('gripper', {}).get('encoder', {}).get('names', [])
        args.imu9AxisNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('imu', {}).get('9axis', {}).get('names', [])
        args.lidarPointCloudNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('lidar', {}).get('pointCloud', {}).get('names', [])
        args.robotBaseVelNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('robotBase', {}).get('vel', {}).get('names', [])
        args.liftMotorNames = yaml_data.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {}).get('lift', {}).get('motor', {}).get('names', [])
        args.armJointStateDims = [7 for _ in range(len(args.armJointStateNames))]
        args.armEndPoseDims = [7 for _ in range(len(args.armEndPoseNames))]
    return args


def main():
    args = get_arguments()
    process(args)


if __name__ == "__main__":
    main()
