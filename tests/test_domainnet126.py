"""DomainNet-126 list-based loader — synthetic local folder, no network, CPU only."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch
from PIL import Image

from superadditivity.datasets import domainnet126 as dn
from superadditivity.datasets.dataset_loader import DATASET_SPECS, DatasetLoader
from superadditivity.datasets.semantic_partitioner import SemanticPartitioner

N_CLASSES = 126


def _class_names() -> List[str]:
    # Deliberately not created in alphabetical order, so label assignment is
    # tested against sorting rather than creation order.
    return [f"class_{i:03d}" for i in reversed(range(N_CLASSES))]


def _make_root(
    tmp_path: Path,
    n_classes: int = N_CLASSES,
    per_class=lambda i: 2 + i % 3,
    relative_to_clipart: bool = False,
    shuffled_labels: bool = False,
) -> Path:
    """Create a tiny ``DomainNet-126``-shaped folder and return its path."""
    root = tmp_path / "DomainNet-126"
    names = _class_names()[:n_classes]
    sorted_names = sorted(names)
    lines: List[str] = []
    for name in names:
        i = sorted_names.index(name)
        d = root / "clipart" / name
        d.mkdir(parents=True)
        for k in range(per_class(i)):
            fname = f"{name}_{k:03d}.jpg"
            Image.new("RGB", (4, 4), color=(i % 256, k * 40, 0)).save(d / fname)
            rel = f"{name}/{fname}" if relative_to_clipart else f"clipart/{name}/{fname}"
            label = (i + 1) % n_classes if shuffled_labels else i
            lines.append(f"{rel} {label}")
    for other in ("painting", "real", "sketch"):
        (root / other).mkdir()
        (root / f"{other}_list.txt").write_text("")
    (root / "clipart_list.txt").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A shared 126-class synthetic DomainNet-126 folder."""
    return _make_root(tmp_path_factory.mktemp("dn126"))


# --------------------------------------------------------------------------- #
# Parsing and labels
# --------------------------------------------------------------------------- #

def test_parse_list_file(root):
    entries = dn.parse_list_file(root / "clipart_list.txt")
    assert len(entries) == sum(2 + i % 3 for i in range(N_CLASSES))
    path, cls, label = entries[0]
    assert path.startswith("clipart/") and path.split("/")[1] == cls
    assert isinstance(label, int)


def test_parse_paths_relative_to_clipart(tmp_path):
    r = _make_root(tmp_path, relative_to_clipart=True)
    entries = dn.parse_list_file(r / "clipart_list.txt")
    assert all(p.startswith("clipart/") for p, _, _ in entries)
    train, test = dn.build_domainnet126(r)
    assert len(train) + len(test) == len(entries)


def test_alphabetical_labels(root):
    train, test = dn.build_domainnet126(root)
    assert train.classes == sorted(_class_names())
    assert train.class_to_idx == {c: i for i, c in enumerate(sorted(_class_names()))}
    for ds in (train, test):
        for (path, label), t in zip(ds.samples, ds.targets):
            assert Path(path).parent.name == ds.classes[label]
            assert t == label
            assert isinstance(t, int)


def test_label_agreement_reported(root, tmp_path, caplog):
    entries = dn.parse_list_file(root / "clipart_list.txt")
    cti = {c: i for i, c in enumerate(sorted(_class_names()))}
    assert dn._report_label_agreement(entries, cti) is True

    bad = _make_root(tmp_path, shuffled_labels=True)
    with caplog.at_level("WARNING"):
        train, _ = dn.build_domainnet126(bad)
    assert "disagree" in caplog.text
    # Labels still come from alphabetical order, not the list file.
    assert train.classes[train.targets[0]] == Path(train.samples[0][0]).parent.name


def test_getitem_loads_rgb(root):
    train, _ = dn.build_domainnet126(root, train_transform=lambda im: im.size)
    img_size, label = train[0]
    assert img_size == (4, 4) and 0 <= label < N_CLASSES
    raw, _ = dn.build_domainnet126(root)
    assert raw[0][0].mode == "RGB"


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

def test_split_deterministic(root):
    a_train, a_test = dn.build_domainnet126(root)
    b_train, b_test = dn.build_domainnet126(root)
    assert a_train.samples == b_train.samples
    assert a_test.samples == b_test.samples

    entries = dn.parse_list_file(root / "clipart_list.txt")
    classes = sorted({c for _, c, _ in entries})
    assert dn.split_entries(entries, classes) == dn.split_entries(list(reversed(entries)), classes)


def test_split_stratified_and_disjoint(root):
    train, test = dn.build_domainnet126(root)
    train_paths = {p for p, _ in train.samples}
    test_paths = {p for p, _ in test.samples}
    assert not train_paths & test_paths
    assert len(train_paths) + len(test_paths) == sum(2 + i % 3 for i in range(N_CLASSES))
    assert set(train.targets) == set(range(N_CLASSES))
    assert set(test.targets) == set(range(N_CLASSES))
    tr = torch.bincount(torch.tensor(train.targets), minlength=N_CLASSES)
    te = torch.bincount(torch.tensor(test.targets), minlength=N_CLASSES)
    for c in range(N_CLASSES):
        n = int(tr[c] + te[c])
        assert int(te[c]) == min(max(1, int(0.3 * n + 0.5)), n - 1)


def test_split_70_30_on_larger_class(tmp_path):
    r = _make_root(tmp_path, per_class=lambda i: 20 if i == 0 else 2)
    train, test = dn.build_domainnet126(r)
    assert test.targets.count(0) == 6 and train.targets.count(0) == 14


def test_split_seed_constant():
    assert dn.SPLIT_SEED == 2026


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

def test_missing_listed_file_raises(tmp_path):
    r = _make_root(tmp_path)
    victim = next((r / "clipart" / "class_000").iterdir())
    victim.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        dn.build_domainnet126(r)


def test_wrong_class_count_raises(tmp_path):
    r = _make_root(tmp_path, n_classes=5)
    with pytest.raises(ValueError, match="exactly 126 classes"):
        dn.build_domainnet126(r)


def test_class_folders_must_match(tmp_path):
    r = _make_root(tmp_path)
    (r / "clipart" / "extra_folder").mkdir()
    with pytest.raises(ValueError, match="do not match"):
        dn.build_domainnet126(r)


def test_missing_list_file_raises(tmp_path):
    r = _make_root(tmp_path)
    (r / "clipart_list.txt").unlink()
    with pytest.raises(FileNotFoundError, match="clipart_list.txt"):
        dn.build_domainnet126(r)


@pytest.mark.parametrize("download", [False, True])
def test_missing_folder_raises(tmp_path, download, caplog):
    missing = tmp_path / "DomainNet-126"
    loader = DatasetLoader("domainnet126", missing, download=download)
    with caplog.at_level("WARNING"):
        with pytest.raises(FileNotFoundError, match="place it there by hand"):
            loader.load()
    if download:
        assert "cannot be downloaded automatically" in caplog.text


# --------------------------------------------------------------------------- #
# Spec, clusters, loader and partitioner integration
# --------------------------------------------------------------------------- #

def test_clusters_cover_all_classes_once():
    spec = DATASET_SPECS["domainnet126"]
    assert spec.num_classes == 126 and spec.image_size == 64 and spec.in_channels == 3
    assert spec.n_semantic_clusters == 4
    clusters = spec.semantic_clusters
    assert [len(clusters[k]) for k in sorted(clusters)] == [31, 31, 32, 32]
    flat = [c for k in sorted(clusters) for c in clusters[k]]
    assert sorted(flat) == list(range(126)) and len(flat) == 126
    assert all(isinstance(c, int) for c in flat)


def test_loader_end_to_end(root):
    loader = DatasetLoader("domainnet126", root, download=False, probe_size=126)
    train, test = loader.load()
    x, y = train[0]
    assert tuple(x.shape) == (3, 64, 64)
    x, y = test[0]
    assert tuple(x.shape) == (3, 64, 64)
    probe = loader.get_probe_set()
    assert len(probe) == 126


def test_targets_work_with_semantic_partitioner(root):
    train, _ = dn.build_domainnet126(root)
    clusters = DATASET_SPECS["domainnet126"].semantic_clusters
    communities = {k: [2 * k, 2 * k + 1] for k in range(4)}
    for mode, alpha in (("iid", 1.0), ("dirichlet_semantic", 0.5)):
        part = SemanticPartitioner(
            train, clusters, communities, alpha=alpha, mode=mode,
            run_seed=0, num_clients=8,
        ).partition()
        all_idx = sorted(i for idx in part.values() for i in idx)
        assert all_idx == list(range(len(train)))
        if mode == "dirichlet_semantic":
            for comm, clients in communities.items():
                allowed = set(clusters[comm])
                for c in clients:
                    assert {train.targets[i] for i in part[c]} <= allowed
