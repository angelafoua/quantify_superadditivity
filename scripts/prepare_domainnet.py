#!/usr/bin/env python3
"""Download DomainNet clipart and organise it into train/test ImageFolder layout.

Usage
-----
    # Auto-download:
    python scripts/prepare_domainnet.py --data_dir data

    # If you already downloaded the files manually from https://ai.bu.edu/DomainNet/:
    python scripts/prepare_domainnet.py --data_dir data \\
        --local_zip /path/to/clipart.zip \\
        --local_train_txt /path/to/clipart_train.txt \\
        --local_test_txt /path/to/clipart_test.txt

After completion, the directory layout is::

    data_dir/clipart/train/<class_name>/image.jpg
    data_dir/clipart/test/<class_name>/image.jpg
    data_dir/staging_clipart/          (raw downloads; safe to delete)

The same logic runs automatically when a ``DatasetLoader`` for DomainNet is
created with ``download=True`` and the data is missing. All of it lives in
``superadditivity/datasets/domainnet_prep.py``.

Official dataset page: https://ai.bu.edu/DomainNet/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as ``python scripts/prepare_domainnet.py`` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.datasets.domainnet_prep import (  # noqa: E402
    STAGING_DIRNAME,
    prepare_domainnet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare DomainNet clipart for training.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help=(
            "Root data directory. Output goes to <data_dir>/clipart/ and "
            "must match the data_dir given to DatasetLoader (default: data)."
        ),
    )
    parser.add_argument("--local_zip", type=str, default=None,
                        help="Path to a local clipart.zip (skips that download).")
    parser.add_argument("--local_train_txt", type=str, default=None,
                        help="Path to a local clipart_train.txt.")
    parser.add_argument("--local_test_txt", type=str, default=None,
                        help="Path to a local clipart_test.txt.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if data_dir/clipart is already prepared.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = prepare_domainnet(
        args.data_dir,
        local_zip=args.local_zip,
        local_train_txt=args.local_train_txt,
        local_test_txt=args.local_test_txt,
        force=args.force,
    )
    print(
        f"\nReady: {out}\n"
        "You can delete the staging downloads to save space:\n"
        f"  rm -rf {Path(args.data_dir) / STAGING_DIRNAME}"
    )


if __name__ == "__main__":
    main()
