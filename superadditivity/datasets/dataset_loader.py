"""Unified multi-dataset loader.

Handles downloading, normalization, probe-set extraction, and semantic
clustering for all benchmark datasets used in this project:
CIFAR-100, CIFAR-10, EMNIST, DomainNet, iNaturalist, PathMNIST,
and Google Speech Commands.

Each dataset is described by a ``DatasetSpec`` that stores its metadata
(num_classes, image_size, in_channels, normalization stats, semantic
cluster definitions). Nothing is hardcoded outside these specs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
import torchvision.transforms as T

from superadditivity.utils.seed import PROBE_SEED, seed_worker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Metadata for a single dataset — no data, just the spec."""

    name: str
    num_classes: int
    image_size: int
    in_channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    semantic_clusters: Optional[Dict[int, List[Any]]] = None
    n_semantic_clusters: int = 4


# CIFAR-100 superclass mapping (needed for semantic cluster resolution)
SUPERCLASS_TO_FINE: Dict[str, List[str]] = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": [
        "apple", "mushroom", "orange", "pear", "sweet_pepper",
    ],
    "household_electrical_devices": [
        "clock", "keyboard", "lamp", "telephone", "television",
    ],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": [
        "bridge", "castle", "house", "road", "skyscraper",
    ],
    "large_natural_outdoor_scenes": [
        "cloud", "forest", "mountain", "plain", "sea",
    ],
    "large_omnivores_and_herbivores": [
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    ],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": [
        "crab", "lobster", "snail", "spider", "worm",
    ],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


# ---------------------------------------------------------------------------
# Registry of all supported datasets
# ---------------------------------------------------------------------------

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar100": DatasetSpec(
        name="cifar100",
        num_classes=100,
        image_size=32,
        in_channels=3,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        semantic_clusters={
            0: [  # Animals
                "aquatic_mammals", "fish", "insects",
                "large_carnivores", "reptiles",
            ],
            1: [  # Artifacts
                "vehicles_1", "vehicles_2",
                "household_electrical_devices",
                "household_furniture", "food_containers",
            ],
            2: [  # Nature / Structures
                "flowers", "fruit_and_vegetables", "trees",
                "large_natural_outdoor_scenes",
                "large_man-made_outdoor_things",
            ],
            3: [  # Mammals / People
                "people", "medium_mammals", "small_mammals",
                "large_omnivores_and_herbivores",
                "non-insect_invertebrates",
            ],
        },
        n_semantic_clusters=4,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        image_size=32,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        semantic_clusters={
            0: [2, 3, 4, 5, 6, 7],   # Animals
            1: [0, 1, 8, 9],          # Vehicles
        },
        n_semantic_clusters=2,
    ),
    "emnist": DatasetSpec(
        name="emnist",
        num_classes=62,
        image_size=32,
        in_channels=1,
        mean=(0.1751,),
        std=(0.3332,),
        semantic_clusters={
            0: list(range(0, 10)),    # Digits 0-9
            1: list(range(10, 36)),   # Uppercase A-Z
            2: list(range(36, 62)),   # Lowercase a-z
        },
        n_semantic_clusters=3,
    ),
    "domainnet": DatasetSpec(
        name="domainnet",
        num_classes=345,
        image_size=64,
        in_channels=3,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        semantic_clusters={
            0: list(range(0, 86)),
            1: list(range(86, 172)),
            2: list(range(172, 258)),
            3: list(range(258, 345)),
        },
        n_semantic_clusters=4,
    ),
    "inaturalist": DatasetSpec(
        name="inaturalist",
        num_classes=200,
        image_size=64,
        in_channels=3,
        mean=(0.466, 0.480, 0.374),
        std=(0.237, 0.231, 0.252),
        semantic_clusters={
            0: list(range(0, 50)),    # Animalia subset
            1: list(range(50, 100)),  # Plantae subset
            2: list(range(100, 150)), # Fungi subset
            3: list(range(150, 200)), # Mixed/Other
        },
        n_semantic_clusters=4,
    ),
    "pathmnist": DatasetSpec(
        name="pathmnist",
        num_classes=9,
        image_size=32,
        in_channels=3,
        mean=(0.7406, 0.5331, 0.7059),
        std=(0.1279, 0.1606, 0.1191),
        semantic_clusters={
            0: [0, 1, 2],       # Tissue types group A
            1: [3, 4, 5],       # Tissue types group B
            2: [6, 7, 8],       # Tissue types group C
        },
        n_semantic_clusters=3,
    ),
    "speech_commands": DatasetSpec(
        name="speech_commands",
        num_classes=35,
        image_size=64,
        in_channels=1,
        mean=(0.0,),
        std=(1.0,),
        semantic_clusters={
            0: list(range(0, 10)),    # Digits
            1: list(range(10, 18)),   # Directional/action
            2: list(range(18, 26)),   # Action commands
            3: list(range(26, 35)),   # Binary/misc
        },
        n_semantic_clusters=4,
    ),
}

_SUPPORTED_DATASETS = frozenset(DATASET_SPECS.keys())


class DatasetLoader:
    """Unified loader for all supported datasets.

    Parameters
    ----------
    dataset_name:
        One of the keys in ``DATASET_SPECS``.
    data_dir:
        Root directory for dataset downloads / caching.
    probe_size:
        Number of samples in the stratified probe set.
    probe_seed:
        Fixed seed for probe-set selection (default 999, never change).
    download:
        Whether to download the dataset if not present.
    batch_size:
        Default batch size for data loaders.
    num_workers:
        Number of data-loading worker processes.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str | Path,
        probe_size: int = 1000,
        probe_seed: int = PROBE_SEED,
        download: bool = True,
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> None:
        dataset_name = dataset_name.lower()
        if dataset_name not in _SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}. "
                f"Choose from {sorted(_SUPPORTED_DATASETS)}."
            )
        self.dataset_name = dataset_name
        self.spec = DATASET_SPECS[dataset_name]
        self.data_dir = Path(data_dir)
        self.probe_size = probe_size
        self.probe_seed = probe_seed
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._probe_set: Optional[Subset] = None

        logger.info(
            "DatasetLoader initialised: dataset=%s, data_dir=%s, "
            "image_size=%d, in_channels=%d",
            self.dataset_name, self.data_dir,
            self.spec.image_size, self.spec.in_channels,
        )

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def get_transforms(self, train: bool = True) -> T.Compose:
        """Return dataset-specific transforms.

        Parameters
        ----------
        train:
            If ``True``, include data augmentation (random crop + flip).
            If ``False``, only normalize.

        Returns
        -------
        torchvision.transforms.Compose
        """
        spec = self.spec
        size = spec.image_size
        mean = spec.mean
        std = spec.std

        if self.dataset_name in ("cifar100", "cifar10"):
            if train:
                return T.Compose([
                    T.RandomCrop(size, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        if self.dataset_name == "emnist":
            # 28x28 grayscale -> resize to target size
            if train:
                return T.Compose([
                    T.Resize(size),
                    T.RandomCrop(size, padding=4),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.Resize(size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        if self.dataset_name == "domainnet":
            if train:
                return T.Compose([
                    T.Resize((size, size)),
                    T.RandomCrop(size, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        if self.dataset_name == "inaturalist":
            if train:
                return T.Compose([
                    T.Resize((size, size)),
                    T.RandomCrop(size, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        if self.dataset_name == "pathmnist":
            if train:
                return T.Compose([
                    T.Resize((size, size)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        if self.dataset_name == "speech_commands":
            # Audio -> mel-spectrogram; transforms are handled in load()
            return T.Compose([T.ToTensor()])

        raise ValueError(f"No transforms defined for {self.dataset_name}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> Tuple[Dataset, Dataset]:
        """Download (if needed) and load train and test datasets.

        Returns
        -------
        (train_dataset, test_dataset)
        """
        train_tf = self.get_transforms(train=True)
        test_tf = self.get_transforms(train=False)

        if self.dataset_name == "cifar100":
            self._train_dataset = torchvision.datasets.CIFAR100(
                root=str(self.data_dir), train=True,
                transform=train_tf, download=self.download,
            )
            self._test_dataset = torchvision.datasets.CIFAR100(
                root=str(self.data_dir), train=False,
                transform=test_tf, download=self.download,
            )

        elif self.dataset_name == "cifar10":
            self._train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True,
                transform=train_tf, download=self.download,
            )
            self._test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False,
                transform=test_tf, download=self.download,
            )

        elif self.dataset_name == "emnist":
            self._train_dataset = torchvision.datasets.EMNIST(
                root=str(self.data_dir), split="byclass", train=True,
                transform=train_tf, download=self.download,
            )
            self._test_dataset = torchvision.datasets.EMNIST(
                root=str(self.data_dir), split="byclass", train=False,
                transform=test_tf, download=self.download,
            )

        elif self.dataset_name == "domainnet":
            self._train_dataset, self._test_dataset = self._load_domainnet(
                train_tf, test_tf,
            )

        elif self.dataset_name == "inaturalist":
            self._train_dataset, self._test_dataset = self._load_inaturalist(
                train_tf, test_tf,
            )

        elif self.dataset_name == "pathmnist":
            self._train_dataset, self._test_dataset = self._load_pathmnist(
                train_tf, test_tf,
            )

        elif self.dataset_name == "speech_commands":
            self._train_dataset, self._test_dataset = self._load_speech_commands()

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        logger.info(
            "Loaded %s: %d train, %d test samples",
            self.dataset_name,
            len(self._train_dataset),
            len(self._test_dataset),
        )
        return self._train_dataset, self._test_dataset

    # ------------------------------------------------------------------
    # Dataset-specific loaders
    # ------------------------------------------------------------------

    def _load_domainnet(
        self, train_tf: T.Compose, test_tf: T.Compose,
    ) -> Tuple[Dataset, Dataset]:
        """Load DomainNet (clipart domain for tractability).

        Uses torchvision ImageFolder on downloaded/extracted data.
        Raises FileNotFoundError if the data directory is missing —
        run ``scripts/prepare_domainnet.py`` first.
        """
        domain_dir = self.data_dir / "clipart"
        if domain_dir.exists():
            train_ds = torchvision.datasets.ImageFolder(
                root=str(domain_dir / "train"), transform=train_tf,
            )
            test_ds = torchvision.datasets.ImageFolder(
                root=str(domain_dir / "test"), transform=test_tf,
            )
            return train_ds, test_ds

        raise FileNotFoundError(
            f"DomainNet clipart directory not found at {domain_dir}. "
            f"Run 'python scripts/prepare_domainnet.py --data_dir {self.data_dir}' "
            f"to download and organise the dataset before training."
        )

    def _load_inaturalist(
        self, train_tf: T.Compose, test_tf: T.Compose,
    ) -> Tuple[Dataset, Dataset]:
        """Load iNaturalist subset (200 classes).

        Uses torchvision ImageFolder on pre-organized directory structure.
        Raises FileNotFoundError if the data directory is missing.
        """
        data_root = self.data_dir
        train_dir = data_root / "train"
        test_dir = data_root / "test"

        if train_dir.exists():
            train_ds = torchvision.datasets.ImageFolder(
                root=str(train_dir), transform=train_tf,
            )
            test_ds = torchvision.datasets.ImageFolder(
                root=str(test_dir), transform=test_tf,
            )
            return train_ds, test_ds

        raise FileNotFoundError(
            f"iNaturalist directory not found at {data_root}. "
            f"Expected train/ and test/ subdirectories with ImageFolder layout. "
            f"Download and organise the dataset before training."
        )

    def _load_pathmnist(
        self, train_tf: T.Compose, test_tf: T.Compose,
    ) -> Tuple[Dataset, Dataset]:
        """Load PathMNIST from the MedMNIST package.

        Raises ImportError if the medmnist package is not installed.
        """
        try:
            import medmnist
            from medmnist import PathMNIST as PathMNISTClass
        except ImportError:
            raise ImportError(
                "The 'medmnist' package is required for PathMNIST but is not "
                "installed. Install it with: pip install medmnist"
            ) from None

        train_ds = PathMNISTClass(
            split="train", transform=train_tf,
            download=self.download, root=str(self.data_dir),
        )
        test_ds = PathMNISTClass(
            split="test", transform=test_tf,
            download=self.download, root=str(self.data_dir),
        )
        # MedMNIST uses .labels instead of .targets; add targets attr
        if not hasattr(train_ds, "targets"):
            train_ds.targets = train_ds.labels.squeeze().tolist()
        if not hasattr(test_ds, "targets"):
            test_ds.targets = test_ds.labels.squeeze().tolist()
        return train_ds, test_ds

    def _load_speech_commands(self) -> Tuple[Dataset, Dataset]:
        """Load Google Speech Commands v2 as mel-spectrogram tensors.

        Converts 1-second audio clips into mel-spectrograms and packages
        them as TensorDatasets. Raises on failure instead of silently
        falling back to synthetic data.
        """
        spec = self.spec
        size = spec.image_size

        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "The 'torchaudio' package is required for Speech Commands "
                "but is not installed. Install it with: pip install torchaudio"
            ) from None

        train_ds_raw = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.data_dir), download=self.download,
            subset="training",
        )
        test_ds_raw = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(self.data_dir), download=self.download,
            subset="testing",
        )

        train_ds = self._speech_to_spectrograms(train_ds_raw, size)
        test_ds = self._speech_to_spectrograms(test_ds_raw, size)
        return train_ds, test_ds

    @staticmethod
    def _speech_to_spectrograms(
        raw_ds: Dataset, target_size: int,
    ) -> TensorDataset:
        """Convert a SpeechCommands dataset to mel-spectrogram TensorDataset."""
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160,
            n_mels=target_size,
        )

        # Build label mapping from unique labels
        all_labels_str = sorted(set(
            raw_ds[i][2] for i in range(len(raw_ds))
        ))
        label_to_idx = {lab: i for i, lab in enumerate(all_labels_str)}

        images_list: list[torch.Tensor] = []
        labels_list: list[int] = []

        for i in range(len(raw_ds)):
            waveform, sample_rate, label, *_ = raw_ds[i]
            # Pad/trim to 1 second
            if waveform.shape[1] < 16000:
                waveform = torch.nn.functional.pad(
                    waveform, (0, 16000 - waveform.shape[1])
                )
            else:
                waveform = waveform[:, :16000]

            mel = mel_transform(waveform)  # (1, n_mels, time)
            # Resize time dimension to target_size
            mel = torch.nn.functional.interpolate(
                mel.unsqueeze(0), size=(target_size, target_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            # Log-scale
            mel = torch.log(mel.clamp(min=1e-9))
            # Normalize to zero mean, unit variance
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)

            images_list.append(mel)
            labels_list.append(label_to_idx[label])

        images = torch.stack(images_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

        ds = TensorDataset(images, labels)
        ds.targets = labels_list  # type: ignore[attr-defined]
        return ds

    def _synthetic_placeholder(self) -> Tuple[TensorDataset, TensorDataset]:
        """Create a synthetic placeholder dataset for testing/fallback."""
        spec = self.spec
        n_train, n_test = 1000, 200
        ch = spec.in_channels
        sz = spec.image_size
        nc = spec.num_classes

        train_imgs = torch.randn(n_train, ch, sz, sz)
        train_labels = torch.randint(0, nc, (n_train,))
        test_imgs = torch.randn(n_test, ch, sz, sz)
        test_labels = torch.randint(0, nc, (n_test,))

        train_ds = TensorDataset(train_imgs, train_labels)
        test_ds = TensorDataset(test_imgs, test_labels)
        train_ds.targets = train_labels.tolist()  # type: ignore[attr-defined]
        test_ds.targets = test_labels.tolist()  # type: ignore[attr-defined]
        return train_ds, test_ds

    # ------------------------------------------------------------------
    # Probe set
    # ------------------------------------------------------------------

    def get_probe_set(self) -> Subset:
        """Return a stratified probe set drawn from the test split.

        The probe set uses a fixed seed (``probe_seed=999`` by default)
        that is **independent** of the run seed, guaranteeing identical
        probes across all experiments.

        Returns
        -------
        torch.utils.data.Subset
        """
        if self._probe_set is not None:
            return self._probe_set

        if self._test_dataset is None:
            raise RuntimeError("Call load() before get_probe_set().")

        targets = np.array(self._get_targets(self._test_dataset))
        num_classes = self.get_num_classes()
        per_class = self.probe_size // num_classes
        remainder = self.probe_size % num_classes

        rng = np.random.RandomState(self.probe_seed)
        selected_indices: List[int] = []

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            n_take = min(per_class, len(cls_indices))
            if cls < remainder:
                n_take = min(n_take + 1, len(cls_indices))
            if len(cls_indices) > 0 and n_take > 0:
                chosen = rng.choice(cls_indices, size=n_take, replace=False)
                selected_indices.extend(chosen.tolist())

        rng.shuffle(selected_indices)

        self._probe_set = Subset(self._test_dataset, selected_indices)
        logger.info(
            "Probe set: %d samples from %s test split (seed=%d)",
            len(self._probe_set), self.dataset_name, self.probe_seed,
        )
        return self._probe_set

    # ------------------------------------------------------------------
    # Test loader
    # ------------------------------------------------------------------

    def get_test_loader(self) -> DataLoader:
        """Return a DataLoader over the full test set."""
        if self._test_dataset is None:
            raise RuntimeError("Call load() before get_test_loader().")

        g = torch.Generator()
        g.manual_seed(self.probe_seed)
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def get_num_classes(self) -> int:
        """Return the number of classes for the loaded dataset."""
        return self.spec.num_classes

    def get_image_size(self) -> int:
        """Return the spatial resolution for this dataset."""
        return self.spec.image_size

    def get_in_channels(self) -> int:
        """Return the number of input channels for this dataset."""
        return self.spec.in_channels

    def get_semantic_clusters(self) -> Optional[Dict[int, List[Any]]]:
        """Return semantic cluster definitions for the current dataset.

        Returns
        -------
        dict or None
            Mapping ``{cluster_id: [class_identifiers]}``.
        """
        return self.spec.semantic_clusters

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_targets(dataset: Dataset) -> List[int]:
        """Extract integer targets from a torchvision dataset."""
        if hasattr(dataset, "targets"):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.tolist()
            return list(targets)
        if hasattr(dataset, "labels"):
            labels = dataset.labels
            if isinstance(labels, np.ndarray):
                return labels.squeeze().tolist()
            if isinstance(labels, torch.Tensor):
                return labels.tolist()
            return list(labels)
        raise AttributeError(
            f"Cannot find targets on {type(dataset).__name__}."
        )

    def get_fine_class_indices(self, superclass_names: List[str]) -> List[int]:
        """Return sorted CIFAR-100 fine-class indices for superclass names."""
        if self.dataset_name != "cifar100":
            raise RuntimeError(
                "get_fine_class_indices() is only available for CIFAR-100."
            )
        if self._train_dataset is None:
            raise RuntimeError("Call load() before get_fine_class_indices().")

        ds = self._train_dataset
        class_to_idx: Dict[str, int] = ds.class_to_idx  # type: ignore[attr-defined]
        indices: List[int] = []
        for sc_name in superclass_names:
            fine_names = SUPERCLASS_TO_FINE[sc_name]
            for fn in fine_names:
                if fn in class_to_idx:
                    indices.append(class_to_idx[fn])
                else:
                    logger.warning(
                        "Fine class %r not found for superclass %r",
                        fn, sc_name,
                    )
        return sorted(indices)

    def validate_and_visualize(
        self,
        num_samples: int = 5,
        output_dir: Optional[str] = None,
    ) -> None:
        """Print dataset statistics and optionally save sample images.

        Parameters
        ----------
        num_samples:
            Number of sample images to save (if images exist).
        output_dir:
            Directory to save sample images. If None, images are not saved.
        """
        if self._train_dataset is None or self._test_dataset is None:
            raise RuntimeError("Call load() before validate_and_visualize().")

        print("\n" + "="*70)
        print(f"DATASET VALIDATION: {self.dataset_name.upper()}")
        print("="*70)

        # ---- Basic Statistics ----
        n_train = len(self._train_dataset)
        n_test = len(self._test_dataset)
        print(f"\n📊 Dataset Size:")
        print(f"   Train samples: {n_train:,}")
        print(f"   Test samples:  {n_test:,}")
        print(f"   Total:         {n_train + n_test:,}")

        # ---- Get targets to analyze class distribution ----
        train_targets = np.array(self._get_targets(self._train_dataset))
        test_targets = np.array(self._get_targets(self._test_dataset))

        unique_train_classes = np.unique(train_targets)
        unique_test_classes = np.unique(test_targets)

        print(f"\n🏷️  Class Information:")
        print(f"   Train classes: {len(unique_train_classes)}")
        print(f"   Test classes:  {len(unique_test_classes)}")
        print(f"   Expected:      {self.spec.num_classes}")

        # Class distribution in train set
        train_class_counts = np.bincount(train_targets)
        print(f"\n   Train class distribution:")
        print(f"      Min samples/class:  {train_class_counts.min()}")
        print(f"      Max samples/class:  {train_class_counts.max()}")
        print(f"      Mean samples/class: {train_class_counts.mean():.1f}")
        print(f"      Std samples/class:  {train_class_counts.std():.1f}")

        # Class distribution in test set
        test_class_counts = np.bincount(test_targets)
        print(f"\n   Test class distribution:")
        print(f"      Min samples/class:  {test_class_counts.min()}")
        print(f"      Max samples/class:  {test_class_counts.max()}")
        print(f"      Mean samples/class: {test_class_counts.mean():.1f}")

        # ---- Image Metadata ----
        print(f"\n🖼️  Image Metadata:")
        print(f"   Size:        {self.spec.image_size}×{self.spec.image_size}")
        print(f"   Channels:    {self.spec.in_channels}")
        print(f"   Normalization mean: {self.spec.mean}")
        print(f"   Normalization std:  {self.spec.std}")

        # ---- Sample a batch to check shapes ----
        print(f"\n🔍 Sample Batch (first 3 samples):")
        try:
            for i in range(min(3, len(self._train_dataset))):
                sample = self._train_dataset[i]
                if isinstance(sample, tuple):
                    x, y = sample[0], sample[1]
                    if isinstance(x, torch.Tensor):
                        print(f"   Sample {i}: shape={tuple(x.shape)}, dtype={x.dtype}, "
                              f"class={y}")
                    else:
                        print(f"   Sample {i}: shape={np.array(x).shape}, class={y}")
        except Exception as e:
            logger.warning(f"Could not inspect sample batch: {e}")

        # ---- Save sample images ----
        if output_dir is not None:
            self._save_sample_images(num_samples, output_dir)

        print("\n" + "="*70)
        print("✅ Validation complete!\n")

    def _save_sample_images(self, num_samples: int, output_dir: str) -> None:
        """Save sample images from train and test sets."""
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning(
                "PIL or matplotlib not available; skipping image saving. "
                "Install with: pip install pillow matplotlib"
            )
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving sample images to {output_path}")

        # Save raw (untransformed) images if available
        if self.dataset_name in ("cifar100", "cifar10", "emnist"):
            raw_dataset = self._train_dataset  # Already raw for these
        else:
            # For other datasets, we may not have raw access
            raw_dataset = None

        # Try to visualize transformed images
        try:
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
            if num_samples == 1:
                axes = axes.reshape(2, 1)

            # Train set samples
            for i in range(min(num_samples, len(self._train_dataset))):
                x, y = self._train_dataset[i]
                if isinstance(x, torch.Tensor):
                    # Denormalize if possible
                    x_np = self._denormalize(x.numpy())
                    axes[0, i].imshow(x_np)
                elif isinstance(x, Image.Image):
                    axes[0, i].imshow(x)
                else:
                    axes[0, i].imshow(np.array(x))
                axes[0, i].set_title(f"Train - Class {y}")
                axes[0, i].axis("off")

            # Test set samples
            for i in range(min(num_samples, len(self._test_dataset))):
                x, y = self._test_dataset[i]
                if isinstance(x, torch.Tensor):
                    x_np = self._denormalize(x.numpy())
                    axes[1, i].imshow(x_np)
                elif isinstance(x, Image.Image):
                    axes[1, i].imshow(x)
                else:
                    axes[1, i].imshow(np.array(x))
                axes[1, i].set_title(f"Test - Class {y}")
                axes[1, i].axis("off")

            plt.suptitle(f"{self.dataset_name.upper()} Dataset Samples")
            plt.tight_layout()
            save_path = output_path / f"{self.dataset_name}_samples.png"
            plt.savefig(str(save_path), dpi=100, bbox_inches="tight")
            print(f"   ✓ Saved to {save_path}")
            plt.close()

        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize image from normalized values."""
        # x is CHW or HW
        mean = np.array(self.spec.mean).reshape(-1, 1, 1) if len(x.shape) == 3 else np.array(self.spec.mean)
        std = np.array(self.spec.std).reshape(-1, 1, 1) if len(x.shape) == 3 else np.array(self.spec.std)

        x_denorm = (x * std + mean).clip(0, 1)

        # Convert CHW to HWC if needed
        if len(x_denorm.shape) == 3 and x_denorm.shape[0] in (1, 3):
            x_denorm = np.transpose(x_denorm, (1, 2, 0))
            # Drop channel dim if grayscale
            if x_denorm.shape[-1] == 1:
                x_denorm = x_denorm.squeeze(-1)

        return x_denorm
