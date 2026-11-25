"""Download DomainNet clipart and organise it into train/test ImageFolder layout.

This is the library behind both ``scripts/prepare_domainnet.py`` and
``DatasetLoader(..., download=True)`` for DomainNet. Everything lives under a
single ``data_dir``::

    data_dir/staging_clipart/           raw downloads + extracted archive (disposable)
    data_dir/clipart/train/<class>/*.jpg
    data_dir/clipart/test/<class>/*.jpg

``data_dir/clipart`` is only ever created by an atomic rename once it is
complete, so its presence with class folders in both splits means "ready".

Official dataset page: https://ai.bu.edu/DomainNet/
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

STAGING_DIRNAME = "staging_clipart"
CLIPART_DIRNAME = "clipart"

_BASE = "https://csr.bu.edu/ftp/visda/2019/multi-source"
CLIPART_ZIP_URLS: Sequence[str] = (f"{_BASE}/groundtruth/clipart.zip",)
# Two historical locations for the split files; the first one that returns a
# valid file wins.
TRAIN_SPLIT_URLS: Sequence[str] = (
    f"{_BASE}/domainnet/txt/clipart_train.txt",
    f"{_BASE}/txt/clipart_train.txt",
)
TEST_SPLIT_URLS: Sequence[str] = (
    f"{_BASE}/domainnet/txt/clipart_test.txt",
    f"{_BASE}/txt/clipart_test.txt",
)

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
_TIMEOUT_S = 120


# --------------------------------------------------------------------------- #
# State check
# --------------------------------------------------------------------------- #

def is_prepared(data_dir: str | Path) -> bool:
    """Return True if ``data_dir/clipart/{train,test}`` both contain class folders."""
    root = Path(data_dir) / CLIPART_DIRNAME
    return all(
        any(p.is_dir() for p in (root / split).glob("*"))
        for split in ("train", "test")
    )


# --------------------------------------------------------------------------- #
# Download + validation
# --------------------------------------------------------------------------- #

def download(url: str, dest: Path) -> None:
    """Download *url* to *dest*.

    Written to a ``.part`` file and renamed only after a complete transfer, so
    an interrupted download never leaves a truncated file that later runs
    would mistake for a finished one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    logger.info("  Downloading %s ...", url)
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
        expected = int(response.headers.get("Content-Length", 0)) or None
        with tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    if expected is not None and tmp.stat().st_size != expected:
        tmp.unlink(missing_ok=True)
        raise OSError(f"Incomplete download from {url}")
    tmp.replace(dest)
    logger.info("  Saved to %s", dest)


def split_file_problem(path: Path) -> Optional[str]:
    """Return what is wrong with a split file, or None if it looks valid.

    A valid DomainNet split file has lines ``clipart/<class>/<image>.jpg <label>``.
    A common failure is an HTML error page saved in its place.
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


def archive_problem(path: Path) -> Optional[str]:
    """Return what is wrong with the archive, or None if it is a valid zip."""
    if not zipfile.is_zipfile(path):
        return f"{path} is not a valid zip archive ({path.stat().st_size:,} bytes)"
    return None


def fetch_validated(
    urls: Sequence[str],
    dest: Path,
    check: Callable[[Path], Optional[str]],
) -> None:
    """Make *dest* a valid file, trying each URL in turn.

    A cached *dest* that fails *check* (stale or corrupt, e.g. from an earlier
    failed run) is discarded first. Network errors and invalid downloads move
    on to the next URL; if all fail, a RuntimeError lists every failure.
    """
    if dest.exists():
        problem = check(dest)
        if problem is None:
            logger.info("  Already exists: %s", dest)
            return
        logger.warning("  Cached file invalid (%s); removing it.", problem)
        dest.unlink()

    failures = []
    for url in urls:
        try:
            download(url, dest)
        except OSError as exc:  # URLError and HTTPError are OSError subclasses
            failures.append(f"{url}: {exc}")
            continue
        problem = check(dest)
        if problem is None:
            return
        dest.unlink(missing_ok=True)
        failures.append(f"{url}: {problem}")

    raise RuntimeError(
        "Could not obtain a valid " + dest.name + ":\n  " + "\n  ".join(failures)
        + "\nDownload it manually from https://ai.bu.edu/DomainNet/ and pass it "
        "with --local_zip / --local_train_txt / --local_test_txt."
    )


# --------------------------------------------------------------------------- #
# Extract + organise
# --------------------------------------------------------------------------- #

def extract_archive(archive_path: Path, raw_dir: Path) -> None:
    """Extract the .zip archive to *raw_dir*.

    Extracts into a temporary sibling directory and renames it into place on
    success, so an interrupted extraction is retried rather than reused.
    """
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(
            f"Expected .zip archive, got {archive_path.suffix}. "
            f"Download clipart.zip from https://ai.bu.edu/DomainNet/"
        )
    tmp_dir = raw_dir.with_name(raw_dir.name + ".part")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    logger.info("  Extracting zip archive to %s ...", raw_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=str(tmp_dir))
    tmp_dir.replace(raw_dir)


def find_image_root(raw_dir: Path) -> Path:
    """Return ``raw_dir/clipart`` after checking it holds class folders with images."""
    candidate = raw_dir / "clipart"
    if candidate.is_dir():
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if subdirs and any(
            f.suffix.lower() in _IMG_SUFFIXES
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


def organise_split(split_file: Path, image_root: Path, out_dir: Path) -> int:
    """Copy images listed in *split_file* into ImageFolder layout under *out_dir*.

    Each line is ``<relative_path> <label_int>`` (e.g.
    ``clipart/aircraft_carrier/img.jpg 0``). Entries are matched by
    ``(class_name, filename)`` against the extracted files, so leading path
    components that differ from the extracted layout are tolerated.

    Returns the number of images copied.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[tuple[str, str], Path] = {}
    for class_dir in image_root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in _IMG_SUFFIXES:
                index[(class_dir.name, img_file.name)] = img_file

    count = 0
    missing = 0
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path_parts = Path(line.split()[0]).parts
            if len(path_parts) < 2:
                missing += 1
                continue
            if path_parts[0] == "clipart" and len(path_parts) >= 3:
                class_name, filename = path_parts[1], path_parts[2]
            else:
                class_name, filename = path_parts[-2], path_parts[-1]

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
        logger.warning("  %d images listed in split file were not found.", missing)
    return count


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def prepare_domainnet(
    data_dir: str | Path,
    local_zip: Optional[str | Path] = None,
    local_train_txt: Optional[str | Path] = None,
    local_test_txt: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Download (or use local files for) DomainNet clipart and organise it.

    Parameters
    ----------
    data_dir:
        Root data directory. Output goes to ``<data_dir>/clipart``.
    local_zip, local_train_txt, local_test_txt:
        Optional local copies to use instead of downloading.
    force:
        Rebuild even if ``<data_dir>/clipart`` is already prepared.

    Returns
    -------
    Path
        ``<data_dir>/clipart``.

    Raises
    ------
    RuntimeError
        If the downloads are invalid or no images could be matched.
    """
    data_dir = Path(data_dir)
    final_dir = data_dir / CLIPART_DIRNAME
    if is_prepared(data_dir) and not force:
        logger.info("DomainNet already prepared at %s", final_dir)
        return final_dir

    staging = data_dir / STAGING_DIRNAME
    staging.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: obtain archive and split files ----
    logger.info("Step 1/3: Obtaining clipart archive and split files ...")
    if local_zip:
        archive_path = Path(local_zip)
        logger.info("  Using local archive: %s", archive_path)
    else:
        archive_path = staging / "clipart.zip"
        fetch_validated(CLIPART_ZIP_URLS, archive_path, archive_problem)

    if local_train_txt:
        train_split = Path(local_train_txt)
        logger.info("  Using local train split: %s", train_split)
    else:
        train_split = staging / "clipart_train.txt"
        fetch_validated(TRAIN_SPLIT_URLS, train_split, split_file_problem)

    if local_test_txt:
        test_split = Path(local_test_txt)
        logger.info("  Using local test split: %s", test_split)
    else:
        test_split = staging / "clipart_test.txt"
        fetch_validated(TEST_SPLIT_URLS, test_split, split_file_problem)

    # Local files are not re-downloaded, so validate them too.
    for problem in (
        archive_problem(archive_path),
        split_file_problem(train_split),
        split_file_problem(test_split),
    ):
        if problem:
            raise RuntimeError(problem)

    # ---- Step 2: extract ----
    raw_dir = staging / "clipart_raw"
    image_root: Optional[Path] = None
    if raw_dir.exists():
        try:
            image_root = find_image_root(raw_dir)
            logger.info("Step 2/3: Already extracted, skipping.")
        except FileNotFoundError:
            logger.warning("Step 2/3: Existing extraction is incomplete; redoing it.")
            shutil.rmtree(raw_dir)
    if image_root is None:
        logger.info("Step 2/3: Extracting archive ...")
        extract_archive(archive_path, raw_dir)
        image_root = find_image_root(raw_dir)
    logger.info("  Image root: %s", image_root)

    # ---- Step 3: organise into a temp dir, then rename into place ----
    logger.info("Step 3/3: Organising into train/test ImageFolder layout ...")
    build_dir = data_dir / (CLIPART_DIRNAME + ".part")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    n_train = organise_split(train_split, image_root, build_dir / "train")
    n_test = organise_split(test_split, image_root, build_dir / "test")

    if n_train == 0 or n_test == 0:
        shutil.rmtree(build_dir, ignore_errors=True)
        sample_classes = sorted(d.name for d in image_root.iterdir() if d.is_dir())[:3]
        with open(train_split, errors="replace") as f:
            sample_lines = [ln.strip() for ln in f.readlines()[:2]]
        raise RuntimeError(
            f"No images were copied (train={n_train}, test={n_test}). "
            f"Split-file lines such as {sample_lines} did not match files "
            f"under {image_root} (e.g. classes {sample_classes}). "
            f"Check that the archive and split files belong together."
        )

    # Only reached when the existing dir is absent, incomplete, or force=True.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    build_dir.replace(final_dir)

    n_train_classes = sum(1 for d in (final_dir / "train").iterdir() if d.is_dir())
    n_test_classes = sum(1 for d in (final_dir / "test").iterdir() if d.is_dir())
    logger.info("Done: train %s images / %d classes, test %s images / %d classes at %s",
                f"{n_train:,}", n_train_classes, f"{n_test:,}", n_test_classes, final_dir)
    return final_dir
