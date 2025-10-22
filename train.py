#!/usr/bin/env python3
"""
LeRobot Dataset Verification Script for Pika Sense

This script verifies that the LeRobot dataset is correctly formatted.
Compatible with LeRobot 0.3.x
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Verify LeRobot dataset for Pika Sense')
    parser.add_argument('--dataset-repo-id', type=str, default='local/pika_dataset',
                        help='Dataset repository ID')
    parser.add_argument('--dataset-root', type=str, default='./lerobot_data',
                        help='Path to local LeRobot dataset directory')
    
    args = parser.parse_args()
    
    # Verify dataset exists
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"✗ Error: Dataset directory not found: {dataset_root}")
        return
    
    print("=" * 70)
    print("LeRobot Dataset Verification")
    print("=" * 70)
    print(f"Dataset repo ID: {args.dataset_repo_id}")
    print(f"Dataset root: {dataset_root}")
    print("=" * 70)
    
    # Load and verify dataset
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        print("\nLoading dataset...")
        ds = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=str(dataset_root)
        )
        
        # Display dataset info
        print(f"\n✓ Dataset loaded successfully!")
        print(f"\nDataset Statistics:")
        print(f"  - Episodes: {ds.meta.info['total_episodes']}")
        print(f"  - Frames: {ds.meta.info['total_frames']}")
        print(f"  - FPS: {ds.fps}")
        
        # Display first frame info
        if len(ds) > 0:
            sample = ds[0]
            print(f"\nFirst Frame Structure:")
            print(f"  - Keys: {list(sample.keys())}")
            
            # Display action shape
            if 'action' in sample:
                print(f"\nAction:")
                print(f"  - Shape: {sample['action'].shape}")
            
            # Display image observations
            image_keys = [k for k in sample.keys() if 'observation.images' in k]
            if image_keys:
                print(f"\nImage Observations:")
                for key in image_keys:
                    camera_name = key.split('.')[-1]
                    print(f"  - {camera_name}: {sample[key].shape}")
            
            # Display state observation
            if 'observation.state' in sample:
                print(f"\nState Observation:")
                print(f"  - Shape: {sample['observation.state'].shape}")
                print(f"  - Values: {sample['observation.state'].tolist()}")
        
        print("\n" + "=" * 70)
        print("✓ Verification Complete - Dataset is valid!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Failed to load dataset!")
        print(f"Error: {e}")
        print("\nPlease check:")
        print("  1. Dataset directory exists and contains data")
        print("  2. Dataset was converted correctly using hdf5_to_lerobot.py")
        print("  3. LeRobot is installed: pip install lerobot>=0.3.0")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
