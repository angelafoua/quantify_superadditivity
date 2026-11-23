#!/usr/bin/env python3
"""Download DomainNet clipart and organise it into train/test ImageFolder layout.

Usage
-----
    # Auto-download (uses the official cleaned .zip from ai.bu.edu):
    python scripts/prepare_domainnet.py --data_dir data

    # If you already downloaded the files manually from https://ai.bu.edu/DomainNet/
    # place them in a directory and point the script at it:
    python scripts/prepare_domainnet.py --data_dir data \\
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
    """Download *url* to *dest*.

    The file is written to a ``.part`` temporary path and renamed only after
    the transfer completes, so an interrupted download never leaves a
    truncated file that later runs would mistake for a finished one.
    """
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)
    print(f"  Saved to {dest}")


_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def split_file_problem(path: Path) -> str | None:
    """Return a description of what is wrong with a split file, or None if OK.

    A valid DomainNet split file has lines of the form
    ``clipart/<class>/<image>.jpg <label>``. A common failure is an HTML
    error page (proxy/captive portal/redirect) saved in its place.
    """
    try:
        with open(path, errors="replace") as f:
            head = [ln.strip() for ln in f.readlines()[:5] if ln.strip()]
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    if not head:
        return f"{path} is empty"
    for line in head:
        parts = line.split()
        if (
            len(parts) != 2
            or not parts[1].lstrip("-").isdigit()
            or Path(parts[0]).suffix.lower() not in _IMG_SUFFIXES
        ):
            return f"{path} does not look like a split file; first line: {line[:120]!r}"
    return None


def archive_problem(path: Path) -> str | None:
    """Return a description of what is wrong with the archive, or None if OK."""
    if not zipfile.is_zipfile(path):
        return f"{path} is not a valid zip archive (size {path.stat().st_size:,} bytes)"
    return None


def fetch_validated(url: str, dest: Path, check) -> None:
    """Download *url* to *dest*, discarding and retrying once if it is invalid.

    Handles stale or corrupt files cached by an earlier failed run.
    """
    for attempt in (1, 2):
        download(url, dest)
        problem = check(dest)
        if problem is None:
            return
        print(f"  Invalid download ({problem}); removing it.")
        dest.unlink(missing_ok=True)
        if attempt == 2:
            raise RuntimeError(
                f"Downloaded file from {url} is invalid: {problem}. "
                f"Download it manually from https://ai.bu.edu/DomainNet/ and "
                f"pass it with --local_zip / --local_train_txt / --local_test_txt."
            )


def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    """Extract the .zip archive to *raw_dir*.

    Extracts into a temporary sibling directory and renames it into place on
    success, so an interrupted extraction is retried on the next run instead
    of being treated as complete.
    """
    suffix = archive_path.suffix.lower()
    if suffix != ".zip":
        raise ValueError(
            f"Expected .zip archive, got {archive_path.suffix}. "
            f"Download clipart.zip from https://ai.bu.edu/DomainNet/"
        )
    tmp_dir = raw_dir.with_name(raw_dir.name + ".part")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    print(f"  Extracting zip archive to {raw_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=str(tmp_dir))
    tmp_dir.replace(raw_dir)
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
            "(~1.3 GB) from csr.bu.edu (the VisDA 2019 mirror of DomainNet) "
            "together with the train/test "
            "split .txt files.\n\n"
            "If you have already downloaded the files manually, pass their "
            "local paths via --local_zip / --local_train_txt / --local_test_txt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help=(
            "Root data directory. Output goes to <data_dir>/clipart/ and "
            "must match the data_dir given to DatasetLoader (default: data)."
        ),
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
        fetch_validated(CLIPART_ZIP_URL, archive_path, archive_problem)

    # Train / test split files
    if args.local_train_txt:
        train_split = Path(args.local_train_txt)
        print(f"  Using local train split: {train_split}")
    else:
        train_split = download_dir / "clipart_train.txt"
        fetch_validated(TRAIN_SPLIT_URL, train_split, split_file_problem)

    if args.local_test_txt:
        test_split = Path(args.local_test_txt)
        print(f"  Using local test split: {test_split}")
    else:
        test_split = download_dir / "clipart_test.txt"
        fetch_validated(TEST_SPLIT_URL, test_split, split_file_problem)

    # Local files are not re-downloaded, so validate them too.
    for problem in (
        archive_problem(archive_path),
        split_file_problem(train_split),
        split_file_problem(test_split),
    ):
        if problem:
            raise RuntimeError(problem)

    # ------------------------------------------------------------------ #
    # Step 2 — extract
    # ------------------------------------------------------------------ #
    raw_dir = download_dir / "clipart_raw"
    if raw_dir.exists():
        try:
            image_root = find_image_root(raw_dir)
            print("Step 2/3: Already extracted, skipping.")
        except FileNotFoundError:
            print("Step 2/3: Existing extraction is incomplete; redoing it.")
            shutil.rmtree(raw_dir)
    if not raw_dir.exists():
        print("Step 2/3: Extracting archive ...")
        extract_archive(archive_path, raw_dir)
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

    if n_train == 0 or n_test == 0:
        # Do not leave empty split directories behind: they make the loader
        # fail later with a confusing "no class folder" error.
        shutil.rmtree(data_dir / "clipart", ignore_errors=True)
        sample_classes = sorted(d.name for d in image_root.iterdir() if d.is_dir())[:3]
        with open(train_split, errors="replace") as f:
            sample_lines = [ln.strip() for ln in f.readlines()[:2]]
        raise RuntimeError(
            f"No images were copied (train={n_train}, test={n_test}). "
            f"Split-file lines such as {sample_lines} did not match files "
            f"under {image_root} (e.g. classes {sample_classes}). "
            f"Check that the archive and split files belong together."
        )

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
