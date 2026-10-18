#!/usr/bin/env python3
"""Download DomainNet clipart and organise it into train/test ImageFolder layout.

Usage
-----
    python scripts/prepare_domainnet.py --data_dir data/domainnet

The script downloads the clipart images and the official train/test split
files, then arranges them into::

    data_dir/clipart/train/<class_name>/image.jpg
    data_dir/clipart/test/<class_name>/image.jpg

Once complete, ``DatasetLoader("domainnet", data_dir)`` loads the real data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

CLIPART_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.tar"
TRAIN_SPLIT_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/clipart_train.txt"
TEST_SPLIT_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/clipart_test.txt"


def download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress indicator."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def organise_split(
    split_file: Path, raw_dir: Path, out_dir: Path,
) -> int:
    """Move images listed in *split_file* into ImageFolder layout under *out_dir*.

    Each line in the split file has the format ``<relative_path> <label_int>``.
    The class name is taken from the first subdirectory in the relative path.

    Returns the number of images moved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]  # e.g. clipart/aircraft_carrier/sketch_0.jpg
            # Strip leading domain folder if present
            path_parts = Path(rel_path).parts
            if path_parts[0] == "clipart":
                class_name = path_parts[1]
                src = raw_dir / Path(*path_parts[1:])
            else:
                class_name = path_parts[0]
                src = raw_dir / rel_path

            if not src.exists():
                # Try without the domain prefix
                alt = raw_dir / Path(*path_parts)
                if alt.exists():
                    src = alt
                else:
                    continue

            dst_dir = out_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare DomainNet clipart for training."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/domainnet",
        help="Root data directory (default: data/domainnet).",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    download_dir = data_dir / "_downloads"

    # 1. Download the tarball and split files
    print("Step 1/3: Downloading clipart archive and split files ...")
    tar_path = download_dir / "clipart.tar"
    train_split = download_dir / "clipart_train.txt"
    test_split = download_dir / "clipart_test.txt"

    download(CLIPART_URL, tar_path)
    download(TRAIN_SPLIT_URL, train_split)
    download(TEST_SPLIT_URL, test_split)

    # 2. Extract
    raw_dir = download_dir / "clipart_raw"
    if not raw_dir.exists():
        print("Step 2/3: Extracting archive ...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=str(raw_dir))
        print(f"  Extracted to {raw_dir}")
    else:
        print("Step 2/3: Already extracted.")

    # The tarball extracts into a 'clipart/' subfolder
    extracted = raw_dir / "clipart"
    if not extracted.exists():
        # Might extract flat
        extracted = raw_dir

    # 3. Organise into ImageFolder layout
    print("Step 3/3: Organising into train/test ImageFolder layout ...")
    out_train = data_dir / "clipart" / "train"
    out_test = data_dir / "clipart" / "test"

    n_train = organise_split(train_split, extracted, out_train)
    n_test = organise_split(test_split, extracted, out_test)

    n_train_classes = sum(1 for d in out_train.iterdir() if d.is_dir())
    n_test_classes = sum(1 for d in out_test.iterdir() if d.is_dir())

    print(f"\nDone!  train: {n_train} images in {n_train_classes} classes")
    print(f"       test:  {n_test} images in {n_test_classes} classes")
    print(f"       Path:  {data_dir / 'clipart'}")
    print(
        "\nYou can now safely delete the downloads folder to save space:\n"
        f"  rm -rf {download_dir}"
    )


if __name__ == "__main__":
    main()
