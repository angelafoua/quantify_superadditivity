#!/usr/bin/env python3
"""Validate dataset loading and visualize samples.

This script tests that a dataset has been downloaded and prepared correctly,
prints statistics about the data, and saves sample images.

Usage
-----
    # Validate DomainNet-126 (local folder; never downloaded)
    python scripts/validate_dataset.py --dataset domainnet126 --data_dir data/DomainNet-126

    # Validate CIFAR-100 with sample images
    python scripts/validate_dataset.py --dataset cifar100 --output_dir outputs/samples

    # Validate EMNIST
    python scripts/validate_dataset.py --dataset emnist --data_dir data/emnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.datasets.dataset_loader import DatasetLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate dataset loading and visualize samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="domainnet126",
        help="Dataset to validate (default: domainnet126).",
        choices=["cifar100", "cifar10", "emnist", "domainnet126", "inaturalist",
                 "pathmnist", "speech_commands"],
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Data directory (default: data/<dataset>; "
             "data/DomainNet-126 for domainnet126).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for sample images. If None, no images are saved.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of sample images to save (default: 5).",
    )
    parser.add_argument(
        "--no_download", action="store_true",
        help="Do not download the dataset if missing.",
    )

    args = parser.parse_args()

    # Infer data_dir if not provided
    if args.data_dir is None:
        if args.dataset == "domainnet126":
            args.data_dir = "data/DomainNet-126"
        else:
            args.data_dir = f"data/{args.dataset}"

    print(f"\n🔄 Loading {args.dataset}...")
    print(f"   Data directory: {args.data_dir}")

    try:
        loader = DatasetLoader(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            download=not args.no_download,
        )
        train_ds, test_ds = loader.load()
        print("✅ Dataset loaded successfully!\n")

        # Validate and print statistics
        loader.validate_and_visualize(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
