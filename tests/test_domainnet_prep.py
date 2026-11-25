"""DomainNet preparation and ``download`` handling — synthetic data, no network."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from superadditivity.datasets import domainnet_prep
from superadditivity.datasets.dataset_loader import DatasetLoader
from superadditivity.datasets.domainnet_prep import (
    fetch_validated,
    is_prepared,
    prepare_domainnet,
    split_file_problem,
)

CLASSES = ("cat", "dog")


@pytest.fixture
def local_files(tmp_path: Path) -> dict[str, Path]:
    """A tiny clipart.zip plus matching split files (2 classes, 2 images each)."""
    src = tmp_path / "src"
    for c in CLASSES:
        (src / "clipart" / c).mkdir(parents=True)
        for i in (1, 2):
            (src / "clipart" / c / f"img{i}.jpg").write_bytes(b"x")
    zip_path = tmp_path / "clipart.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted((src / "clipart").rglob("*.jpg")):
            zf.write(f, f.relative_to(src))
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text("".join(f"clipart/{c}/img1.jpg {i}\n" for i, c in enumerate(CLASSES)))
    test.write_text("".join(f"clipart/{c}/img2.jpg {i}\n" for i, c in enumerate(CLASSES)))
    return {"zip": zip_path, "train": train, "test": test}


def _prepare(data_dir: Path, files: dict[str, Path], **kw) -> Path:
    return prepare_domainnet(
        data_dir, local_zip=files["zip"],
        local_train_txt=files["train"], local_test_txt=files["test"], **kw,
    )


def _fake_prepared_layout(data_dir: Path) -> None:
    for split in ("train", "test"):
        for c in CLASSES:
            d = data_dir / "clipart" / split / c
            d.mkdir(parents=True)
            (d / "img.jpg").write_bytes(b"x")


# --------------------------------------------------------------------------- #
# prepare_domainnet
# --------------------------------------------------------------------------- #

def test_prepare_builds_imagefolder_layout(tmp_path, local_files):
    data_dir = tmp_path / "data"
    out = _prepare(data_dir, local_files)

    assert out == data_dir / "clipart"
    assert is_prepared(data_dir)
    assert sorted(p.name for p in (out / "train").iterdir()) == list(CLASSES)
    assert (out / "train" / "cat" / "img1.jpg").exists()
    assert (out / "test" / "cat" / "img2.jpg").exists()
    # train and test are disjoint
    assert not (out / "train" / "cat" / "img2.jpg").exists()
    # no leftovers from the atomic build
    assert not (data_dir / "clipart.part").exists()


def test_prepare_is_noop_when_ready_unless_forced(tmp_path, local_files):
    data_dir = tmp_path / "data"
    _prepare(data_dir, local_files)
    marker = data_dir / "clipart" / "train" / "cat" / "marker.jpg"
    marker.write_bytes(b"m")

    _prepare(data_dir, local_files)
    assert marker.exists()  # untouched

    _prepare(data_dir, local_files, force=True)
    assert not marker.exists()  # rebuilt
    assert is_prepared(data_dir)


def test_html_split_file_is_rejected(tmp_path, local_files):
    local_files["train"].write_text("<html><body>Not Found</body></html>\n")
    with pytest.raises(RuntimeError, match="split file"):
        _prepare(tmp_path / "data", local_files)
    assert not (tmp_path / "data" / "clipart").exists()


def test_mismatched_split_copies_nothing_and_leaves_no_dirs(tmp_path, local_files):
    local_files["train"].write_text("clipart/zebra/a.jpg 0\n")
    local_files["test"].write_text("clipart/zebra/b.jpg 0\n")
    data_dir = tmp_path / "data"
    with pytest.raises(RuntimeError, match="No images were copied"):
        _prepare(data_dir, local_files)
    assert not (data_dir / "clipart").exists()
    assert not (data_dir / "clipart.part").exists()


def test_stale_incomplete_clipart_dir_is_replaced(tmp_path, local_files):
    data_dir = tmp_path / "data"
    (data_dir / "clipart" / "train").mkdir(parents=True)  # empty, as after a failed run
    (data_dir / "clipart" / "test").mkdir()
    assert not is_prepared(data_dir)
    _prepare(data_dir, local_files)
    assert is_prepared(data_dir)


def test_split_file_problem_accepts_valid_and_rejects_html(tmp_path):
    good = tmp_path / "g.txt"
    good.write_text("clipart/cat/a.jpg 0\n")
    bad = tmp_path / "b.txt"
    bad.write_text("<html>\n")
    assert split_file_problem(good) is None
    assert split_file_problem(bad) is not None


# --------------------------------------------------------------------------- #
# fetch_validated
# --------------------------------------------------------------------------- #

def test_fetch_validated_falls_back_to_next_url(tmp_path, monkeypatch):
    calls = []

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_text("<html>" if "first" in url else "clipart/cat/a.jpg 0\n")

    monkeypatch.setattr(domainnet_prep, "download", fake_download)
    dest = tmp_path / "split.txt"
    fetch_validated(["https://x/first", "https://x/second"], dest, split_file_problem)
    assert calls == ["https://x/first", "https://x/second"]
    assert split_file_problem(dest) is None


def test_fetch_validated_reports_every_failure(tmp_path, monkeypatch):
    def fake_download(url: str, dest: Path) -> None:
        if "net" in url:
            raise OSError("unreachable")
        dest.write_text("<html>")

    monkeypatch.setattr(domainnet_prep, "download", fake_download)
    with pytest.raises(RuntimeError) as exc:
        fetch_validated(
            ["https://net/a", "https://x/b"], tmp_path / "s.txt", split_file_problem,
        )
    assert "unreachable" in str(exc.value) and "https://x/b" in str(exc.value)
    assert not (tmp_path / "s.txt").exists()


def test_fetch_validated_discards_invalid_cached_file(tmp_path, monkeypatch):
    dest = tmp_path / "s.txt"
    dest.write_text("<html>")  # stale cache from an earlier bad run

    def fake_download(url: str, d: Path) -> None:
        d.write_text("clipart/cat/a.jpg 0\n")

    monkeypatch.setattr(domainnet_prep, "download", fake_download)
    fetch_validated(["https://x/ok"], dest, split_file_problem)
    assert split_file_problem(dest) is None


# --------------------------------------------------------------------------- #
# DatasetLoader honours `download`
# --------------------------------------------------------------------------- #

def test_loader_download_false_raises_when_missing(tmp_path):
    loader = DatasetLoader("domainnet", tmp_path, download=False)
    with pytest.raises(FileNotFoundError, match="download=True"):
        loader.load()


def test_loader_download_false_rejects_partial_dir(tmp_path):
    (tmp_path / "clipart" / "train" / "cat").mkdir(parents=True)  # test/ missing
    loader = DatasetLoader("domainnet", tmp_path, download=False)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_loader_download_true_prepares_when_missing(tmp_path, monkeypatch):
    calls = []

    def fake_prepare(data_dir):
        calls.append(Path(data_dir))
        _fake_prepared_layout(Path(data_dir))

    monkeypatch.setattr(
        "superadditivity.datasets.dataset_loader.prepare_domainnet", fake_prepare,
    )
    train, test = DatasetLoader("domainnet", tmp_path, download=True).load()
    assert calls == [tmp_path]
    assert train.classes == list(CLASSES) and len(test) == len(CLASSES)


def test_loader_download_true_skips_when_already_prepared(tmp_path, monkeypatch):
    _fake_prepared_layout(tmp_path)

    def boom(data_dir):
        raise AssertionError("must not download when data is present")

    monkeypatch.setattr("superadditivity.datasets.dataset_loader.prepare_domainnet", boom)
    train, _ = DatasetLoader("domainnet", tmp_path, download=True).load()
    assert len(train) == len(CLASSES)
