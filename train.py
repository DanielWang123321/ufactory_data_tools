#!/usr/bin/env python3
"""
LeRobot Training Script for Pika Sense Dataset

This script trains a diffusion policy model on the captured Pika Sense data.
Compatible with LeRobot 0.3.x
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train LeRobot policy on Pika Sense data')
    parser.add_argument('--dataset-repo-id', type=str, default='local/pika_dataset',
                        help='Dataset repository ID (use valid HF format)')
    parser.add_argument('--dataset-root', type=str, default='./lerobot_data',
                        help='Path to local LeRobot dataset directory')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save training outputs')
    parser.add_argument('--policy', type=str, default='diffusion',
                        choices=['diffusion', 'act', 'tdmpc'],
                        help='Policy type to train')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for training')
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluation frequency (epochs)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Verify dataset exists
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"Error: Dataset directory not found: {dataset_root}")
        return
    
    print("=" * 70)
    print("LeRobot Training Configuration (LeRobot 0.3.x)")
    print("=" * 70)
    print(f"Dataset repo ID: {args.dataset_repo_id}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Policy type: {args.policy}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Verify dataset can be loaded
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print("\nVerifying dataset...")
        ds = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=str(dataset_root)
        )
        print(f"✓ Dataset loaded: {ds.meta.info['total_episodes']} episodes, "
              f"{ds.meta.info['total_frames']} frames")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    print("\n" + "=" * 70)
    print("TRAINING COMMAND")
    print("=" * 70)
    print("Use LeRobot CLI for full training features:\n")
    print(f"lerobot train \\")
    print(f"  --dataset-repo-id {args.dataset_repo_id} \\")
    print(f"  --root {dataset_root} \\")
    print(f"  --policy-name {args.policy} \\")
    print(f"  --output-dir {args.output_dir} \\")
    print(f"  --batch-size {args.batch_size} \\")
    print(f"  --num-epochs {args.num_epochs} \\")
    print(f"  --lr {args.lr} \\")
    print(f"  --device {args.device} \\")
    print(f"  --eval-freq {args.eval_freq}")
    
    print("\n" + "=" * 70)
    print("ADDITIONAL OPTIONS")
    print("=" * 70)
    print("- Add W&B logging: --wandb-project <name> --wandb-entity <user>")
    print("- Add checkpointing: --save-freq 25")
    print("- Customize policy: See lerobot/configs/policies.py")
    print("\nFor more options: lerobot train --help")
    print("=" * 70)


if __name__ == "__main__":
    main()
