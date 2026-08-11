"""Unified multi-dataset loader for CIFAR-100, CIFAR-10, and EMNIST.

Handles downloading, normalization, probe-set extraction, and semantic
clustering for the three benchmark datasets used in this project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T

from superadditivity.utils.seed import PROBE_SEED, seed_worker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CIFAR-100 semantic structure
# ---------------------------------------------------------------------------

#: Mapping from CIFAR-100 superclass name to a list of fine-class names.
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

#: Four-way balanced semantic clustering for CIFAR-100.
#: Each cluster contains 5 superclasses (= 25 fine classes).
SEMANTIC_CLUSTERS: Dict[str, Dict[int, List[str]]] = {
    "cifar100": {
        0: [  # Animals
            "aquatic_mammals",
            "fish",
            "insects",
            "large_carnivores",
            "reptiles",
        ],
        1: [  # Artifacts
            "vehicles_1",
            "vehicles_2",
            "household_electrical_devices",
            "household_furniture",
            "food_containers",
        ],
        2: [  # Nature / Structures
            "flowers",
            "fruit_and_vegetables",
            "trees",
            "large_natural_outdoor_scenes",
            "large_man-made_outdoor_things",
        ],
        3: [  # Mammals / People
            "people",
            "medium_mammals",
            "small_mammals",
            "large_omnivores_and_herbivores",
            "non-insect_invertebrates",
        ],
    },
    "cifar10": {
        0: [2, 3, 4, 5, 6, 7],   # Animals: bird, cat, deer, dog, frog, horse
        1: [0, 1, 8, 9],          # Vehicles: airplane, automobile, ship, truck
    },
}

# ---------------------------------------------------------------------------
# Per-dataset normalization constants
# ---------------------------------------------------------------------------

_NORM_STATS: Dict[str, Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]]] = {
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "emnist": {
        "mean": (0.1751,),
        "std": (0.3332,),
    },
}

_SUPPORTED_DATASETS = {"cifar100", "cifar10", "emnist"}


class DatasetLoader:
    """Unified loader for CIFAR-100, CIFAR-10, and EMNIST.

    Parameters
    ----------
    dataset_name:
        One of ``"cifar100"``, ``"cifar10"``, or ``"emnist"``.
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
            "DatasetLoader initialised: dataset=%s, data_dir=%s",
            self.dataset_name,
            self.data_dir,
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
        stats = _NORM_STATS[self.dataset_name]
        mean, std = stats["mean"], stats["std"]

        if self.dataset_name in ("cifar100", "cifar10"):
            if train:
                return T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

        # EMNIST: 28x28 grayscale -> pad to 32x32 -> repeat to 3 channels
        if train:
            return T.Compose([
                T.Pad(2),  # 28x28 -> 32x32
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch -> 3ch
                T.Normalize(mean * 3, std * 3),
            ])
        return T.Compose([
            T.Pad(2),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean * 3, std * 3),
        ])

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
                root=str(self.data_dir),
                train=True,
                transform=train_tf,
                download=self.download,
            )
            self._test_dataset = torchvision.datasets.CIFAR100(
                root=str(self.data_dir),
                train=False,
                transform=test_tf,
                download=self.download,
            )
        elif self.dataset_name == "cifar10":
            self._train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir),
                train=True,
                transform=train_tf,
                download=self.download,
            )
            self._test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir),
                train=False,
                transform=test_tf,
                download=self.download,
            )
        elif self.dataset_name == "emnist":
            self._train_dataset = torchvision.datasets.EMNIST(
                root=str(self.data_dir),
                split="byclass",
                train=True,
                transform=train_tf,
                download=self.download,
            )
            self._test_dataset = torchvision.datasets.EMNIST(
                root=str(self.data_dir),
                split="byclass",
                train=False,
                transform=test_tf,
                download=self.download,
            )

        logger.info(
            "Loaded %s: %d train, %d test samples",
            self.dataset_name,
            len(self._train_dataset),
            len(self._test_dataset),
        )
        return self._train_dataset, self._test_dataset

    # ------------------------------------------------------------------
    # Probe set
    # ------------------------------------------------------------------

    def get_probe_set(self) -> Subset:
        """Return a stratified probe set drawn from the test split.

        The probe set uses a fixed seed (``probe_seed=999`` by default) that
        is **independent** of the run seed, guaranteeing identical probes
        across all experiments.

        Returns
        -------
        torch.utils.data.Subset
        """
        if self._probe_set is not None:
            return self._probe_set

        if self._test_dataset is None:
            raise RuntimeError(
                "Call load() before get_probe_set()."
            )

        targets = np.array(self._get_targets(self._test_dataset))
        num_classes = self.get_num_classes()
        per_class = self.probe_size // num_classes
        remainder = self.probe_size % num_classes

        rng = np.random.RandomState(self.probe_seed)
        selected_indices: List[int] = []

        for cls in range(num_classes):
            cls_indices = np.where(targets == cls)[0]
            # Some classes may have fewer samples than per_class
            n_take = min(per_class, len(cls_indices))
            if cls < remainder:
                n_take = min(n_take + 1, len(cls_indices))
            chosen = rng.choice(cls_indices, size=n_take, replace=False)
            selected_indices.extend(chosen.tolist())

        # Shuffle so classes are interleaved
        rng.shuffle(selected_indices)

        self._probe_set = Subset(self._test_dataset, selected_indices)
        logger.info(
            "Probe set: %d samples from %s test split (seed=%d)",
            len(self._probe_set),
            self.dataset_name,
            self.probe_seed,
        )
        return self._probe_set

    # ------------------------------------------------------------------
    # Test loader
    # ------------------------------------------------------------------

    def get_test_loader(self) -> DataLoader:
        """Return a DataLoader over the full test set.

        Returns
        -------
        torch.utils.data.DataLoader
        """
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
        """Return the number of classes for the loaded dataset.

        Returns
        -------
        int
            100 for CIFAR-100, 10 for CIFAR-10, 62 for EMNIST (byclass).
        """
        return {"cifar100": 100, "cifar10": 10, "emnist": 62}[self.dataset_name]

    def get_semantic_clusters(self) -> Optional[Dict[int, List[Any]]]:
        """Return semantic cluster definitions for the current dataset.

        Returns
        -------
        dict or None
            For CIFAR-100: ``{0: [superclass_names], 1: ..., 3: ...}``
            (4 clusters of 25 fine classes each).
            For CIFAR-10: ``{0: [class_indices], 1: [class_indices]}``
            (2 clusters).
            For EMNIST: ``None`` (no semantic structure).
        """
        if self.dataset_name in SEMANTIC_CLUSTERS:
            return SEMANTIC_CLUSTERS[self.dataset_name]
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_targets(dataset: Dataset) -> List[int]:
        """Extract integer targets from a torchvision dataset."""
        if hasattr(dataset, "targets"):
            return list(dataset.targets)
        if hasattr(dataset, "labels"):
            return list(dataset.labels)
        raise AttributeError(
            f"Cannot find targets on {type(dataset).__name__}."
        )

    def get_fine_class_indices(self, superclass_names: List[str]) -> List[int]:
        """Return sorted CIFAR-100 fine-class indices for a list of superclass names.

        Parameters
        ----------
        superclass_names:
            Superclass names from :data:`SUPERCLASS_TO_FINE`.

        Returns
        -------
        list[int]
            Sorted fine-class indices.

        Raises
        ------
        RuntimeError
            If the dataset is not CIFAR-100 or has not been loaded.
        """
        if self.dataset_name != "cifar100":
            raise RuntimeError(
                "get_fine_class_indices() is only available for CIFAR-100."
            )
        if self._train_dataset is None:
            raise RuntimeError("Call load() before get_fine_class_indices().")

        # Build superclass -> fine-class index mapping from the dataset's
        # class_to_idx and the CIFAR-100 meta information.
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
                        "Fine class %r not found in class_to_idx for superclass %r",
                        fn,
                        sc_name,
                    )
        return sorted(indices)
