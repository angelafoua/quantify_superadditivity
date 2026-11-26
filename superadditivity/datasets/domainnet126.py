"""DomainNet-126 (clipart domain), loaded from a local folder and its list file.

The data is never downloaded. It must already be on disk, in a folder named
``DomainNet-126`` with this layout::

    DomainNet-126/
    ├── clipart/            # one image folder per class
    ├── clipart_list.txt    # one line per image: "<path> <label>"
    └── ...                 # painting/, real/, sketch/ and their lists (ignored)

Only ``clipart/`` and ``clipart_list.txt`` are used. Nothing is ever written
into the folder and no image is copied.

List-file format
----------------
Each non-empty line holds an image path, optionally followed by an integer
label, e.g. ``clipart/aircraft_carrier/aircraft_carrier_001.jpg 0``. The path
may be relative to ``DomainNet-126/`` (starting with ``clipart/``) or relative
to ``clipart/`` (starting with the class folder). The class name is always
taken from the image's parent folder; the list file's own labels are only
compared against the alphabetical order and reported, never used.

Labels and split
----------------
* Labels are ``0..125`` in alphabetical order of class name.
* The train/test split is a deterministic, stratified 70/30 split per class,
  seeded by :data:`SPLIT_SEED` only (independent of ``run_seed``,
  ``graph_seed`` and the probe seed). Every class has at least one image in
  each split.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: Seed for the train/test split. Fixed so every run sees the same split.
SPLIT_SEED: int = 2026
#: Fraction of each class assigned to the test split.
TEST_FRACTION: float = 0.3
#: Number of classes DomainNet-126 must have.
NUM_CLASSES: int = 126
#: The only domain used.
DOMAIN: str = "clipart"
LIST_FILE: str = f"{DOMAIN}_list.txt"

# One parsed list-file line: (path relative to DomainNet-126/, class name,
# label from the list file or None).
Entry = Tuple[str, str, Optional[int]]


def _missing_data_error(root: Path, what: Path) -> FileNotFoundError:
    """Build the error raised when the local DomainNet-126 data is missing."""
    return FileNotFoundError(
        f"DomainNet-126 not found: {what} does not exist. The DomainNet-126 "
        f"folder is expected at {root} (config: data.data_dir), containing "
        f"'{DOMAIN}/' and '{LIST_FILE}'. It cannot be downloaded "
        f"automatically; place it there by hand, or point data.data_dir at "
        f"the folder (e.g. data.data_dir=/content/drive/MyDrive/DomainNet-126)."
    )


def check_present(root: str | Path) -> None:
    """Raise :class:`FileNotFoundError` unless the DomainNet-126 data is present.

    Parameters
    ----------
    root:
        Path to the ``DomainNet-126`` folder.
    """
    root = Path(root)
    for p in (root, root / DOMAIN, root / LIST_FILE):
        if not p.exists():
            raise _missing_data_error(root, p)


def _normalise_path(raw: str) -> PurePosixPath:
    """Return ``raw`` as a path relative to ``DomainNet-126/``.

    Accepts paths written relative to ``DomainNet-126/`` (``clipart/<cls>/x``),
    relative to ``clipart/`` (``<cls>/x``), or with any prefix before a
    ``clipart`` component (``.../DomainNet-126/clipart/<cls>/x``).
    """
    parts = [p for p in raw.replace("\\", "/").split("/") if p not in ("", ".")]
    if DOMAIN in parts[:-1]:
        parts = parts[parts.index(DOMAIN):]
    else:
        parts = [DOMAIN, *parts]
    return PurePosixPath(*parts)


def parse_list_file(list_path: str | Path) -> List[Entry]:
    """Parse ``clipart_list.txt`` into ``(relative path, class name, label)``.

    Parameters
    ----------
    list_path:
        Path to the list file.

    Returns
    -------
    list of (str, str, int or None)
        Paths are relative to the ``DomainNet-126`` folder and start with
        ``clipart/``. The class name is the image's parent folder.

    Raises
    ------
    ValueError
        If a line has no ``<class>/<file>`` structure, or the file is empty.
    """
    entries: List[Entry] = []
    with open(list_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.rsplit(maxsplit=1)
            label: Optional[int] = None
            path_str = line
            if len(tokens) == 2:
                try:
                    label = int(tokens[1])
                    path_str = tokens[0]
                except ValueError:
                    pass  # no trailing label; the whole line is the path
            rel = _normalise_path(path_str)
            if len(rel.parts) < 3:
                raise ValueError(
                    f"{list_path}:{lineno}: cannot find a class folder in "
                    f"path {path_str!r} (expected '{DOMAIN}/<class>/<file>' "
                    f"or '<class>/<file>')."
                )
            entries.append((str(rel), rel.parts[-2], label))
    if not entries:
        raise ValueError(f"{list_path} contains no entries.")
    return entries


def _validate(root: Path, entries: Sequence[Entry]) -> List[str]:
    """Check files exist and classes match ``clipart/``; return sorted classes."""
    missing = [p for p, _, _ in entries if not (root / p).is_file()]
    if missing:
        examples = "\n  ".join(str(root / p) for p in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} of {len(entries)} images listed in "
            f"{root / LIST_FILE} are missing, e.g.:\n  {examples}"
        )

    classes = sorted({c for _, c, _ in entries})
    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"DomainNet-126 must have exactly {NUM_CLASSES} classes, but "
            f"{root / LIST_FILE} lists {len(classes)}."
        )

    folders = sorted(d.name for d in (root / DOMAIN).iterdir() if d.is_dir())
    if folders != classes:
        only_list = sorted(set(classes) - set(folders))[:5]
        only_dir = sorted(set(folders) - set(classes))[:5]
        raise ValueError(
            f"Classes in {LIST_FILE} do not match the subfolders of "
            f"{root / DOMAIN}: only in list file {only_list}, "
            f"only as folders {only_dir}."
        )
    return classes


def _report_label_agreement(
    entries: Sequence[Entry], class_to_idx: Dict[str, int],
) -> bool:
    """Log whether the list file's labels match alphabetical order."""
    labelled = [(c, lab) for _, c, lab in entries if lab is not None]
    if not labelled:
        logger.info("%s has no labels; using alphabetical labels.", LIST_FILE)
        return False
    disagree = sum(1 for c, lab in labelled if class_to_idx[c] != lab)
    if disagree == 0:
        logger.info(
            "%s labels agree with alphabetical class order (%d entries).",
            LIST_FILE, len(labelled),
        )
        return True
    logger.warning(
        "%s labels disagree with alphabetical class order for %d of %d "
        "entries; they are ignored and alphabetical labels 0-%d are used.",
        LIST_FILE, disagree, len(labelled), NUM_CLASSES - 1,
    )
    return False


def split_entries(
    entries: Sequence[Entry], classes: Sequence[str],
) -> Tuple[List[int], List[int]]:
    """Deterministic stratified 70/30 split of ``entries`` per class.

    Entries are sorted by path, grouped by class (in sorted class order) and
    each class is shuffled with one ``numpy.random.RandomState(SPLIT_SEED)``.

    Parameters
    ----------
    entries:
        Parsed list-file entries.
    classes:
        Sorted class names.

    Returns
    -------
    (train_indices, test_indices)
        Indices into ``sorted(entries, key=path)``.

    Raises
    ------
    ValueError
        If a class has fewer than two images (it cannot appear in both splits).
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i][0])
    by_class: Dict[str, List[int]] = {c: [] for c in classes}
    for pos, i in enumerate(order):
        by_class[entries[i][1]].append(pos)

    rng = np.random.RandomState(SPLIT_SEED)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for c in classes:
        idx = np.array(by_class[c])
        if len(idx) < 2:
            raise ValueError(
                f"Class {c!r} has {len(idx)} image(s); at least 2 are needed "
                f"for a train/test split."
            )
        rng.shuffle(idx)
        n_test = min(max(1, math.floor(TEST_FRACTION * len(idx) + 0.5)), len(idx) - 1)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return sorted(train_idx), sorted(test_idx)


class DomainNet126Dataset(Dataset):
    """Image dataset over a list of ``(absolute path, label)`` samples.

    Images are loaded from disk on demand and converted to RGB.

    Parameters
    ----------
    samples:
        ``[(image_path, label), ...]``.
    classes:
        Sorted class names; ``classes[label]`` is a sample's class.
    transform:
        Optional transform applied to each PIL image.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        classes: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.classes = list(classes)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.targets: List[int] = [int(lab) for _, lab in samples]
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Return ``(image, label)`` for ``index``."""
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_domainnet126(
    root: str | Path,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
) -> Tuple[DomainNet126Dataset, DomainNet126Dataset]:
    """Build the DomainNet-126 clipart train and test datasets.

    Parameters
    ----------
    root:
        Path to the ``DomainNet-126`` folder (config ``data.data_dir``).
    train_transform, test_transform:
        Transforms for the train and test splits.

    Returns
    -------
    (train_dataset, test_dataset)

    Raises
    ------
    FileNotFoundError
        If the folder, ``clipart/``, ``clipart_list.txt`` or any listed image
        is missing.
    ValueError
        If the list file does not describe exactly 126 classes matching the
        subfolders of ``clipart/``.
    """
    root = Path(root)
    check_present(root)

    entries = parse_list_file(root / LIST_FILE)
    classes = _validate(root, entries)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    _report_label_agreement(entries, class_to_idx)

    train_idx, test_idx = split_entries(entries, classes)
    ordered = sorted(entries, key=lambda e: e[0])

    def _samples(idx: List[int]) -> List[Tuple[str, int]]:
        return [(str(root / ordered[i][0]), class_to_idx[ordered[i][1]]) for i in idx]

    train_ds = DomainNet126Dataset(_samples(train_idx), classes, train_transform)
    test_ds = DomainNet126Dataset(_samples(test_idx), classes, test_transform)

    test_counts = np.bincount(test_ds.targets, minlength=len(classes))
    logger.info(
        "DomainNet-126 clipart: %d classes, %d train / %d test images "
        "(split seed %d); smallest per-class test count %d.",
        len(classes), len(train_ds), len(test_ds), SPLIT_SEED,
        int(test_counts.min()),
    )
    return train_ds, test_ds
