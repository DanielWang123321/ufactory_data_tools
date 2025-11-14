#!/usr/bin/env python3
"""
LeRobot Dataset Verification Script for Pika Sense

Validates that a dataset exported from this repository matches the LeRobot v3.0 format
required by lerobot==0.4.1.
"""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify LeRobot dataset for Pika Sense")
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="local/pika_dataset",
        help="Dataset repository ID stored in metadata",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./lerobot_data",
        help="Path to local LeRobot dataset directory",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_root}")
        return 1

    print("=" * 70)
    print("LeRobot Dataset Verification")
    print("=" * 70)
    print(f"Dataset repo ID: {args.dataset_repo_id}")
    print(f"Dataset root: {dataset_root}")
    print("=" * 70)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print("\nLoading dataset...")
        ds = LeRobotDataset(repo_id=args.dataset_repo_id, root=str(dataset_root))
        codebase_version = ds.meta.info.get("codebase_version", "unknown")
        print(f"\nDataset codebase version: {codebase_version}")
        if codebase_version != "v3.0":
            print("\n[ERROR] Dataset is not in v3.0 format.")
            print(
                "Reconvert your HDF5 data with hdf5_to_lerobot.py, or upgrade the"
                " legacy LeRobot dataset via:"
            )
            print(
                "  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30"
                " --root ./lerobot_data --repo-id <id>"
            )
            return 1

        print("\n[OK] Dataset loaded successfully!")
        print("\nDataset Statistics:")
        print(f"  - Episodes: {ds.meta.info['total_episodes']}")
        print(f"  - Frames: {ds.meta.info['total_frames']}")
        print(f"  - FPS: {ds.fps}")

        if len(ds) > 0:
            sample = ds[0]
            print("\nFirst Frame Structure:")
            print(f"  - Keys: {list(sample.keys())}")

            if "action" in sample:
                print("\nAction:")
                print(f"  - Shape: {sample['action'].shape}")

            image_keys = [k for k in sample.keys() if "observation.images" in k]
            if image_keys:
                print("\nImage Observations:")
                for key in image_keys:
                    camera_name = key.split(".")[-1]
                    print(f"  - {camera_name}: {sample[key].shape}")

            if "observation.state" in sample:
                print("\nState Observation:")
                print(f"  - Shape: {sample['observation.state'].shape}")
                print(f"  - Values: {sample['observation.state'].tolist()}")

        print("\n" + "=" * 70)
        print("[OK] Verification Complete - Dataset is valid!")
        print("=" * 70)

    except Exception as exc:  # pragma: no cover - sanity script
        print("\n[ERROR] Failed to load dataset!")
        print(f"Error: {exc}")
        print("\nPlease check:")
        print("  1. Dataset directory exists and contains data")
        print("  2. Dataset was converted correctly using hdf5_to_lerobot.py (v3.0 format)")
        print(
            "  3. Legacy LeRobot datasets were upgraded via"
            " lerobot.datasets.v30.convert_dataset_v21_to_v30"
        )
        print("  4. LeRobot 0.4.1 is installed: pip install lerobot==0.4.1")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
