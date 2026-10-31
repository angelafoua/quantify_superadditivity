#!/usr/bin/env python3
"""Download DomainNet clipart and organise it into train/test ImageFolder layout.

Usage
-----
    # Auto-download (uses the official cleaned .zip from ai.bu.edu):
    python scripts/prepare_domainnet.py --data_dir data/domainnet

    # If you already downloaded the files manually from https://ai.bu.edu/DomainNet/
    # place them in a directory and point the script at it:
    python scripts/prepare_domainnet.py --data_dir data/domainnet \\
        --local_zip /path/to/clipart.zip \\
        --local_train_txt /path/to/clipart_train.txt \\
        --local_test_txt /path/to/clipart_test.txt

The script downloads the recommended cleaned .zip from https://ai.bu.edu/DomainNet/.

After completion, the directory layout is::

    data_dir/clipart/train/<class_name>/image.jpg
    data_dir/clipart/test/<class_name>/image.jpg

Once complete, ``DatasetLoader("domainnet", data_dir)`` loads the real data.

Official dataset page: https://ai.bu.edu/DomainNet/
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

CLIPART_ZIP_URL = (
    "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"
)
TRAIN_SPLIT_URL = (
    "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/clipart_train.txt"
)
TEST_SPLIT_URL = (
    "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/clipart_test.txt"
)


def download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress indicator."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    """Extract the .zip archive to *raw_dir*."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix != ".zip":
        raise ValueError(
            f"Expected .zip archive, got {archive_path.suffix}. "
            f"Download clipart.zip from https://ai.bu.edu/DomainNet/"
        )
    print(f"  Extracting zip archive to {raw_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=str(raw_dir))
    print(f"  Extracted to {raw_dir}")


def find_image_root(raw_dir: Path) -> Path:
    """Return the directory that contains the class-level subdirectories.

    The .zip extracts as: raw_dir/clipart/<class>/<image>
    This function returns the directory one level above the class folders.
    """
    # .zip layout: raw_dir/clipart/<class>/
    candidate = raw_dir / "clipart"
    if candidate.is_dir():
        # Verify it contains class folders with images
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if subdirs and any(
            f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for d in subdirs
            for f in d.iterdir()
            if f.is_file()
        ):
            return candidate

    raise FileNotFoundError(
        f"Could not find image directory in {raw_dir}. "
        f"Expected structure: {raw_dir}/clipart/<class>/<image>. "
        f"Verify the archive extracted correctly."
    )


def organise_split(
    split_file: Path, image_root: Path, out_dir: Path,
) -> int:
    """Copy images listed in *split_file* into ImageFolder layout under *out_dir*.

    Each line in the split file has the format ``<relative_path> <label_int>``.
    Paths in the split file are relative to the extraction root, e.g.
    ``clipart/aircraft_carrier/image.jpg``.

    The function builds a ``(class_name, filename)`` index from the actual
    files under *image_root* and matches split-file entries against that
    index, so it is robust to any leading path components in the split file
    that differ from the extracted layout.

    Returns the number of images copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup: (class_name, filename) -> full Path
    index: dict[tuple[str, str], Path] = {}
    for class_dir in image_root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[(class_dir.name, img_file.name)] = img_file

    count = 0
    missing = 0

    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rel_path = line.split()[0]  # e.g. clipart/aircraft_carrier/sketch_0.jpg
            path_parts = Path(rel_path).parts

            # Resolve class_name and filename regardless of leading path components.
            # Split files may prefix paths with 'clipart/' or omit it entirely.
            if len(path_parts) < 2:
                missing += 1
                continue
            if path_parts[0] == "clipart" and len(path_parts) >= 3:
                class_name = path_parts[1]
                filename = path_parts[2]
            else:
                class_name = path_parts[-2]
                filename = path_parts[-1]

            src = index.get((class_name, filename))
            if src is None:
                missing += 1
                continue

            dst_dir = out_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / filename
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
            count += 1

    if missing:
        print(f"  Warning: {missing} images listed in split file were not found.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare DomainNet clipart for training.\n\n"
            "By default the script auto-downloads the official cleaned .zip "
            "(~1.3 GB) from ai.bu.edu/DomainNet/ together with the train/test "
            "split .txt files.\n\n"
            "If you have already downloaded the files manually, pass their "
            "local paths via --local_zip / --local_train_txt / --local_test_txt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/domainnet",
        help="Root data directory (default: data/domainnet).",
    )
    parser.add_argument(
        "--local_zip", type=str, default=None,
        help=(
            "Path to a locally downloaded clipart.zip to use instead "
            "of downloading. Skips the network download for the image archive."
        ),
    )
    parser.add_argument(
        "--local_train_txt", type=str, default=None,
        help="Path to a locally downloaded clipart_train.txt.",
    )
    parser.add_argument(
        "--local_test_txt", type=str, default=None,
        help="Path to a locally downloaded clipart_test.txt.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    download_dir = data_dir / "_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — obtain the archive and split files
    # ------------------------------------------------------------------ #
    print("Step 1/3: Obtaining clipart archive and split files ...")

    # Image archive (.zip from official site)
    if args.local_zip:
        archive_path = Path(args.local_zip)
        print(f"  Using local archive: {archive_path}")
    else:
        archive_path = download_dir / "clipart.zip"
        download(CLIPART_ZIP_URL, archive_path)

    # Train / test split files
    if args.local_train_txt:
        train_split = Path(args.local_train_txt)
        print(f"  Using local train split: {train_split}")
    else:
        train_split = download_dir / "clipart_train.txt"
        download(TRAIN_SPLIT_URL, train_split)

    if args.local_test_txt:
        test_split = Path(args.local_test_txt)
        print(f"  Using local test split: {test_split}")
    else:
        test_split = download_dir / "clipart_test.txt"
        download(TEST_SPLIT_URL, test_split)

    # ------------------------------------------------------------------ #
    # Step 2 — extract
    # ------------------------------------------------------------------ #
    raw_dir = download_dir / "clipart_raw"
    if not raw_dir.exists():
        print("Step 2/3: Extracting archive ...")
        extract_archive(archive_path, raw_dir)
    else:
        print("Step 2/3: Already extracted, skipping.")

    image_root = find_image_root(raw_dir)
    print(f"  Image root: {image_root}")

    # ------------------------------------------------------------------ #
    # Step 3 — organise into ImageFolder layout
    # ------------------------------------------------------------------ #
    print("Step 3/3: Organising into train/test ImageFolder layout ...")
    out_train = data_dir / "clipart" / "train"
    out_test = data_dir / "clipart" / "test"

    n_train = organise_split(train_split, image_root, out_train)
    n_test = organise_split(test_split, image_root, out_test)

    n_train_classes = sum(1 for d in out_train.iterdir() if d.is_dir())
    n_test_classes = sum(1 for d in out_test.iterdir() if d.is_dir())

    print(f"\nDone!")
    print(f"  train : {n_train:,} images in {n_train_classes} classes")
    print(f"  test  : {n_test:,} images in {n_test_classes} classes")
    print(f"  Path  : {data_dir / 'clipart'}")
    print(
        "\nYou can now safely delete the downloads folder to save space:\n"
        f"  rm -rf {download_dir}"
    )


if __name__ == "__main__":
    main()
