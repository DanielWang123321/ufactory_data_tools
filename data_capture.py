#!/usr/bin/env python3
"""
时间同步数据采集系统
使用主时钟同步所有传感器，确保时间戳完美对齐
"""
import os
import time
import json
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
from threading import Thread, Lock, Event
import signal
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class V4L2Camera:
    """V4L2相机驱动"""
    
    def __init__(self, device_id=0, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.available = False
        self.latest_frame = None
        self.lock = Lock()
    
    def start(self):
        """启动相机"""
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Failed to open V4L2 camera {self.device_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 测试读取
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"V4L2 camera {self.device_id} cannot capture frames")
                self.cap.release()
                return False
            
            self.available = True
            print(f"✓ V4L2 camera {self.device_id} started: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Failed to start V4L2 camera {self.device_id}: {e}")
            return False
    
    def get_latest(self):
        """获取最新帧（非阻塞）"""
        if not self.available or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame.copy()
                return frame
            return self.latest_frame
        except Exception as e:
            print(f"V4L2 capture error: {e}")
            return self.latest_frame
    
    def stop(self):
        """停止相机"""
        if self.cap:
            self.cap.release()
            self.available = False


class RealSenseCamera:
    """RealSense相机驱动"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.available = False
        self.latest_color = None
        self.latest_depth = None
        self.lock = Lock()
        
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            print("Warning: pyrealsense2 not installed.")
            self.rs = None
    
    def start(self):
        """启动相机"""
        if not self.rs:
            return False
            
        try:
            ctx = self.rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                print("No RealSense device connected")
                return False
            
            self.pipeline = self.rs.pipeline()
            config = self.rs.config()
            
            config.enable_stream(self.rs.stream.color, self.width, self.height, 
                               self.rs.format.bgr8, self.fps)
            config.enable_stream(self.rs.stream.depth, self.width, self.height, 
                               self.rs.format.z16, self.fps)
            
            self.profile = self.pipeline.start(config)
            
            # 预热
            for _ in range(10):
                self.pipeline.wait_for_frames()
            
            self.available = True
            print("✓ RealSense camera started")
            return True
            
        except Exception as e:
            print(f"Failed to start RealSense: {e}")
            return False
    
    def get_latest(self):
        """获取最新帧（非阻塞）"""
        if not self.available or not self.pipeline:
            return None, None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            align = self.rs.align(self.rs.stream.color)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if color_frame and depth_frame:
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                with self.lock:
                    self.latest_color = color_image.copy()
                    self.latest_depth = depth_image.copy()
                
                return color_image, depth_image
            
            # 返回上一帧
            return self.latest_color, self.latest_depth
            
        except Exception as e:
            return self.latest_color, self.latest_depth
    
    def stop(self):
        """停止相机"""
        if self.pipeline and self.available:
            self.pipeline.stop()
            self.available = False


class PikaSenseReader:
    """Pika Sense数据读取器（gripper + pose）"""
    
    def __init__(self):
        self.sense = None
        self.available = False
        self.target_device = None
        self.latest_gripper = {'distance': 0.0}
        self.latest_pose = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.lock = Lock()
    
    def start(self):
        """启动Pika Sense"""
        try:
            from pika.sense import Sense
        except ImportError:
            print("Pika SDK not available")
            return False
        
        try:
            self.sense = Sense()
            if not self.sense.connect():
                print("Failed to connect to Sense device")
                return False
            
            print("✓ Sense device connected")
            
            # 初始化Vive Tracker
            tracker = self.sense.get_vive_tracker()
            if tracker:
                print("✓ Vive Tracker initialized")
                time.sleep(2)
                
                devices = self.sense.get_tracker_devices()
                if devices:
                    for device in devices:
                        if device.startswith('WM'):
                            self.target_device = device
                            break
                    
                    if self.target_device is None:
                        self.target_device = devices[0]
                    
                    print(f"✓ Tracking device: {self.target_device}")
                else:
                    print("⚠ No Vive Tracker devices detected")
            else:
                print("⚠ Vive Tracker initialization failed")
            
            self.available = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Sense: {e}")
            return False
    
    def get_latest(self):
        """获取最新的gripper和pose数据"""
        if not self.available or not self.sense:
            return self.latest_gripper.copy(), self.latest_pose.copy()
        
        gripper_data = self.latest_gripper.copy()
        pose_data = self.latest_pose.copy()
        
        # 读取gripper
        try:
            distance = self.sense.get_gripper_distance()
            if distance is not None:
                gripper_data = {
                    'distance': float(distance)
                }
                with self.lock:
                    self.latest_gripper = gripper_data.copy()
        except:
            pass
        
        # 读取pose
        if self.target_device:
            try:
                pose_obj = self.sense.get_pose(self.target_device)
                if pose_obj:
                    # 四元数转欧拉角（与C++版本一致）
                    qx, qy, qz, qw = pose_obj.rotation
                    roll, pitch, yaw = self._quaternion_to_euler(qx, qy, qz, qw)
                    
                    pose_data = {
                        'x': float(pose_obj.position[0] * 1000),
                        'y': float(pose_obj.position[1] * 1000),
                        'z': float(pose_obj.position[2] * 1000),
                        'roll': float(roll),
                        'pitch': float(pitch),
                        'yaw': float(yaw)
                    }
                    with self.lock:
                        self.latest_pose = pose_data.copy()
            except:
                pass
        
        return gripper_data, pose_data
    
    def _quaternion_to_euler(self, qx, qy, qz, qw):
        """四元数转欧拉角（RPY），与tf2::Matrix3x3::getRPY()一致"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def stop(self):
        """停止Sense"""
        if self.sense and self.available:
            self.sense.disconnect()
            self.available = False


class SynchronizedDataCapture:
    """主时钟同步数据采集"""
    
    def __init__(self, config_path: str, output_dir: str, episode_index: int = 0, fps: int = 30):
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config.get('/**', {}).get('ros__parameters', {}).get('dataInfo', {})
        
        self.fps = fps
        self.interval = 1.0 / fps
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / f"episode{episode_index}"
        
        # 创建输出目录
        self._create_directories()
        
        # 传感器
        self.fisheye_camera = None
        self.depth_camera = None
        self.sense_reader = None
        
        # 控制
        self.stop_event = Event()
        self.frame_count = 0
        self.start_time = None
        
        # 统计
        self.stats = {
            'fisheye': 0,
            'depth_color': 0,
            'depth_depth': 0,
            'gripper': 0,
            'pose': 0,
            'missed_frames': 0
        }
    
    def _create_directories(self):
        """创建输出目录"""
        dirs = [
            self.episode_dir / 'camera' / 'color' / 'pikaFisheyeCamera',
            self.episode_dir / 'camera' / 'color' / 'pikaDepthCamera',
            self.episode_dir / 'camera' / 'depth' / 'pikaDepthCamera',
            self.episode_dir / 'gripper' / 'encoder' / 'pika',
            self.episode_dir / 'localization' / 'pose' / 'pika'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def start_sensors(self):
        """启动所有传感器"""
        print("Starting sensors...")
        print("="*60)
        
        # 启动鱼眼相机
        self.fisheye_camera = V4L2Camera(device_id=6, width=640, height=480, fps=self.fps)
        self.fisheye_camera.start()
        
        # 启动深度相机
        self.depth_camera = RealSenseCamera(width=640, height=480, fps=self.fps)
        self.depth_camera.start()
        
        # 启动Sense
        self.sense_reader = PikaSenseReader()
        self.sense_reader.start()
        
        print("="*60)
        print(f"All sensors started. Target FPS: {self.fps}Hz")
        print()
    
    def stop_sensors(self):
        """停止所有传感器"""
        print("\nStopping sensors...")
        if self.fisheye_camera:
            self.fisheye_camera.stop()
        if self.depth_camera:
            self.depth_camera.stop()
        if self.sense_reader:
            self.sense_reader.stop()
    
    def capture_synchronized_frame(self, timestamp: float):
        """采集一帧同步数据"""
        ts_str = f"{timestamp:.6f}"
        frame_data = {}
        
        # 采集鱼眼相机
        if self.fisheye_camera and self.fisheye_camera.available:
            color = self.fisheye_camera.get_latest()
            if color is not None:
                path = self.episode_dir / 'camera' / 'color' / 'pikaFisheyeCamera' / f"{ts_str}.jpg"
                cv2.imwrite(str(path), color)
                frame_data['fisheye'] = True
                self.stats['fisheye'] += 1
        
        # 采集深度相机
        if self.depth_camera and self.depth_camera.available:
            color, depth = self.depth_camera.get_latest()
            if color is not None:
                path = self.episode_dir / 'camera' / 'color' / 'pikaDepthCamera' / f"{ts_str}.jpg"
                cv2.imwrite(str(path), color)
                frame_data['depth_color'] = True
                self.stats['depth_color'] += 1
            
            if depth is not None:
                path = self.episode_dir / 'camera' / 'depth' / 'pikaDepthCamera' / f"{ts_str}.png"
                cv2.imwrite(str(path), depth)
                frame_data['depth_depth'] = True
                self.stats['depth_depth'] += 1
        
        # 采集Sense数据
        if self.sense_reader and self.sense_reader.available:
            gripper, pose = self.sense_reader.get_latest()
            
            # 保存gripper
            path = self.episode_dir / 'gripper' / 'encoder' / 'pika' / f"{ts_str}.json"
            with open(path, 'w') as f:
                json.dump(gripper, f, indent=2)
            frame_data['gripper'] = gripper
            self.stats['gripper'] += 1
            
            # 保存pose
            path = self.episode_dir / 'localization' / 'pose' / 'pika' / f"{ts_str}.json"
            with open(path, 'w') as f:
                json.dump(pose, f, indent=2)
            frame_data['pose'] = pose
            self.stats['pose'] += 1
        
        return frame_data
    
    def run(self):
        """运行主同步采集循环"""
        print("Starting synchronized capture loop")
        print(f"Press Ctrl+C to stop...")
        print()
        
        self.start_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                cycle_start = time.time()
                
                # 使用当前时刻作为统一时间戳
                unified_timestamp = cycle_start
                
                # 同步采集所有传感器
                frame_data = self.capture_synchronized_frame(unified_timestamp)
                
                self.frame_count += 1
                
                # 每10帧打印一次状态
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n{'='*80}")
                    print(f"Frame: {self.frame_count} | Time: {elapsed:.1f}s | FPS: {actual_fps:.1f}")
                    print(f"Stats: fisheye={self.stats['fisheye']} | "
                          f"depth_color={self.stats['depth_color']} | "
                          f"depth_depth={self.stats['depth_depth']}")
                    print(f"       gripper={self.stats['gripper']} | "
                          f"pose={self.stats['pose']}")
                    
                    # 显示最新数据
                    if 'gripper' in frame_data:
                        g = frame_data['gripper']
                        print(f"Latest: gripper=(dist={g['distance']:.2f})")
                    if 'pose' in frame_data:
                        p = frame_data['pose']
                        print(f"        pose=(x={p['x']:.1f}, y={p['y']:.1f}, z={p['z']:.1f}, "
                              f"roll={p['roll']:.3f}, pitch={p['pitch']:.3f}, yaw={p['yaw']:.3f})")
                    
                    print(f"{'='*80}", flush=True)
                
                # 精确定时控制
                elapsed = time.time() - cycle_start
                if elapsed < self.interval:
                    time.sleep(self.interval - elapsed)
                else:
                    self.stats['missed_frames'] += 1
                    
        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user")
        finally:
            self.stop()
    
    def _generate_sync_files(self):
        """生成 sync.txt 文件（用于兼容 data_to_hdf5.py）"""
        print("\nGenerating sync.txt files...")
        
        # 定义需要生成 sync.txt 的目录和对应的文件扩展名
        sync_targets = [
            (self.episode_dir / 'camera' / 'color' / 'pikaFisheyeCamera', '.jpg'),
            (self.episode_dir / 'camera' / 'color' / 'pikaDepthCamera', '.jpg'),
            (self.episode_dir / 'camera' / 'depth' / 'pikaDepthCamera', '.png'),
            (self.episode_dir / 'gripper' / 'encoder' / 'pika', '.json'),
            (self.episode_dir / 'localization' / 'pose' / 'pika', '.json')
        ]
        
        for data_dir, file_ext in sync_targets:
            if not data_dir.exists():
                continue
            
            # 收集所有数据文件
            files = []
            for f in data_dir.iterdir():
                if f.is_file() and f.suffix == file_ext and f.name != 'sync.txt':
                    files.append(f.name)
            
            # 按文件名（时间戳）排序
            files.sort()
            
            # 写入 sync.txt
            if files:
                sync_file = data_dir / 'sync.txt'
                with open(sync_file, 'w') as f:
                    for filename in files:
                        f.write(f"{filename}\n")
                print(f"  ✓ {data_dir.relative_to(self.episode_dir)}/sync.txt ({len(files)} files)")
        
        print("Sync files generated successfully!")
    
    def stop(self):
        """停止采集"""
        self.stop_event.set()
        self.stop_sensors()
        
        # 生成 sync.txt 文件
        self._generate_sync_files()
        
        # 保存统计信息
        duration = time.time() - self.start_time if self.start_time else 0
        stats_file = self.episode_dir / 'statistics.json'
        
        final_stats = {
            'total_frames': self.frame_count,
            'duration_seconds': duration,
            'average_fps': self.frame_count / duration if duration > 0 else 0,
            'missed_frames': self.stats['missed_frames'],
            'sensor_counts': {
                'fisheye': self.stats['fisheye'],
                'depth_color': self.stats['depth_color'],
                'depth_depth': self.stats['depth_depth'],
                'gripper': self.stats['gripper'],
                'pose': self.stats['pose']
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print("\n\nCapture finished!")
        print(f"Total frames: {self.frame_count}")
        print(f"Duration: {duration:.2f}s")
        print(f"Average FPS: {final_stats['average_fps']:.2f}")
        print(f"Missed frames: {self.stats['missed_frames']}")
        print(f"Data saved to: {self.episode_dir}")


def main():
    parser = argparse.ArgumentParser(description='Synchronized Data Capture System')
    parser.add_argument('--config', type=str, 
                       default='./single_pika_data_params.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default='./captured_data',
                       help='Output directory')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    
    args = parser.parse_args()
    
    # 创建并运行采集器
    capture = SynchronizedDataCapture(
        config_path=args.config,
        output_dir=args.output,
        episode_index=args.episode,
        fps=args.fps
    )
    
    capture.start_sensors()
    capture.run()


if __name__ == '__main__':
    main()
