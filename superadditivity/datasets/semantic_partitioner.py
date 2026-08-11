"""Dirichlet-semantic partitioner for non-IID data splits.

Allocates training-set indices to clients using a two-level scheme:

1. **Community assignment**: each client belongs to a community; each
   community is associated with a subset of semantic clusters (for
   CIFAR-100/10) or all classes (for EMNIST).
2. **Intra-community Dirichlet**: within its assigned class pool, each
   community draws per-class proportions from ``Dir(alpha)`` and
   distributes samples to its member clients.

Modes
-----
``"iid"``
    Uniform random allocation (ignores communities and clusters).
``"dirichlet_semantic"``
    Cluster-aware Dirichlet allocation as described above.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from superadditivity.utils.seed import derived_seed

logger = logging.getLogger(__name__)


class SemanticPartitioner:
    """Dirichlet-semantic non-IID partitioner.

    Parameters
    ----------
    train_dataset:
        Full training dataset (e.g. CIFAR-100 train split).
    semantic_clusters:
        Cluster definitions as returned by
        :meth:`DatasetLoader.get_semantic_clusters`.  May be ``None``
        (e.g. EMNIST), in which case ``"dirichlet_semantic"`` mode
        falls back to a pure Dirichlet over all classes within each
        community.
    community_assignments:
        Mapping ``{community_id: [client_id, ...]}`` describing how
        clients are grouped.  Every client id in ``range(num_clients)``
        must appear exactly once.
    alpha:
        Dirichlet concentration parameter.  Smaller values produce
        more heterogeneous (non-IID) partitions.
    mode:
        ``"iid"`` or ``"dirichlet_semantic"``.
    run_seed:
        Master run seed from which the Dirichlet seed is derived.
    num_clients:
        Total number of clients.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        semantic_clusters: Optional[Dict[int, List[Any]]],
        community_assignments: Dict[int, List[int]],
        alpha: float,
        mode: str,
        run_seed: int,
        num_clients: int = 128,
    ) -> None:
        if mode not in ("iid", "dirichlet_semantic"):
            raise ValueError(f"Unknown partitioner mode: {mode!r}")

        self.train_dataset = train_dataset
        self.semantic_clusters = semantic_clusters
        self.community_assignments = community_assignments
        self.alpha = alpha
        self.mode = mode
        self.run_seed = run_seed
        self.num_clients = num_clients

        # Extract targets
        self.targets = np.array(self._get_targets(train_dataset))
        self.num_classes = int(self.targets.max()) + 1

        # Validate community assignments cover all clients
        all_clients = sorted(
            c for clients in community_assignments.values() for c in clients
        )
        expected = list(range(num_clients))
        if all_clients != expected:
            raise ValueError(
                f"community_assignments must cover clients 0..{num_clients - 1} "
                f"exactly once.  Got {len(all_clients)} client entries."
            )

        logger.info(
            "SemanticPartitioner: mode=%s, alpha=%.3f, %d clients, "
            "%d communities, %d classes",
            mode,
            alpha,
            num_clients,
            len(community_assignments),
            self.num_classes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partition(self) -> Dict[int, List[int]]:
        """Partition training indices across clients.

        Returns
        -------
        dict[int, list[int]]
            ``{client_id: [sample_indices]}``.
        """
        if self.mode == "iid":
            return self._partition_iid()
        return self._partition_dirichlet_semantic()

    def get_stats(self) -> np.ndarray:
        """Return a ``(num_clients, num_classes)`` count matrix.

        Must be called after :meth:`partition`.

        Returns
        -------
        np.ndarray
            Shape ``(num_clients, num_classes)``, dtype ``int64``.
        """
        assignment = self.partition()
        counts = np.zeros((self.num_clients, self.num_classes), dtype=np.int64)
        for client_id, indices in assignment.items():
            for idx in indices:
                label = int(self.targets[idx])
                counts[client_id, label] += 1
        return counts

    # ------------------------------------------------------------------
    # IID partitioning
    # ------------------------------------------------------------------

    def _partition_iid(self) -> Dict[int, List[int]]:
        """Uniform random split, ignoring communities and clusters."""
        seed = derived_seed(self.run_seed, kind="dirichlet")
        rng = np.random.RandomState(seed)

        all_indices = np.arange(len(self.targets))
        rng.shuffle(all_indices)

        splits = np.array_split(all_indices, self.num_clients)
        result: Dict[int, List[int]] = {
            i: split.tolist() for i, split in enumerate(splits)
        }

        logger.info("IID partition: ~%d samples per client", len(splits[0]))
        return result

    # ------------------------------------------------------------------
    # Dirichlet-semantic partitioning
    # ------------------------------------------------------------------

    def _partition_dirichlet_semantic(self) -> Dict[int, List[int]]:
        """Cluster-aware Dirichlet allocation."""
        seed = derived_seed(self.run_seed, kind="dirichlet")
        rng = np.random.RandomState(seed)

        # Build per-class index pools
        class_indices: Dict[int, List[int]] = {
            c: np.where(self.targets == c)[0].tolist()
            for c in range(self.num_classes)
        }
        # Shuffle each class pool
        for c in class_indices:
            rng.shuffle(class_indices[c])

        result: Dict[int, List[int]] = {i: [] for i in range(self.num_clients)}

        # Determine which classes each community can draw from
        community_class_pools = self._build_community_class_pools()

        for comm_id, client_ids in self.community_assignments.items():
            class_pool = community_class_pools[comm_id]
            n_clients_in_comm = len(client_ids)

            for cls in class_pool:
                cls_idx = class_indices[cls]
                if len(cls_idx) == 0:
                    continue

                # Draw Dirichlet proportions for this class within this community
                proportions = rng.dirichlet(
                    np.full(n_clients_in_comm, self.alpha)
                )
                # Convert proportions to counts
                proportions = proportions / proportions.sum()
                counts = (proportions * len(cls_idx)).astype(int)
                # Distribute remainder to first clients to avoid losing samples
                remainder = len(cls_idx) - counts.sum()
                for r in range(remainder):
                    counts[r % n_clients_in_comm] += 1

                # Assign indices
                ptr = 0
                for local_idx, cid in enumerate(client_ids):
                    n = counts[local_idx]
                    result[cid].extend(cls_idx[ptr : ptr + n])
                    ptr += n

        # Log statistics
        sizes = [len(v) for v in result.values()]
        logger.info(
            "Dirichlet-semantic partition: alpha=%.3f, samples per client "
            "min=%d, max=%d, mean=%.1f",
            self.alpha,
            min(sizes),
            max(sizes),
            np.mean(sizes),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_community_class_pools(self) -> Dict[int, List[int]]:
        """Map each community to its allowed class indices.

        For datasets with semantic clusters, each community gets the
        classes from its assigned clusters.  When ``semantic_clusters``
        is ``None`` (EMNIST), every community gets all classes.
        """
        if self.semantic_clusters is None:
            # No semantic structure: all communities draw from all classes
            all_classes = list(range(self.num_classes))
            return {
                comm_id: all_classes
                for comm_id in self.community_assignments
            }

        num_communities = len(self.community_assignments)
        num_clusters = len(self.semantic_clusters)

        # Distribute clusters across communities as evenly as possible.
        # If more communities than clusters, communities may share clusters.
        cluster_ids = sorted(self.semantic_clusters.keys())
        community_class_pools: Dict[int, List[int]] = {}

        for comm_idx, comm_id in enumerate(sorted(self.community_assignments.keys())):
            # Assign cluster(s) to this community (round-robin)
            assigned_clusters = [
                cluster_ids[j]
                for j in range(num_clusters)
                if j % num_communities == comm_idx % num_communities
            ]
            # If no cluster assigned (more communities than clusters),
            # wrap around
            if not assigned_clusters:
                assigned_clusters = [cluster_ids[comm_idx % num_clusters]]

            classes: List[int] = []
            for cl_id in assigned_clusters:
                cluster_def = self.semantic_clusters[cl_id]
                if isinstance(cluster_def[0], str):
                    # CIFAR-100: cluster_def is a list of superclass names
                    # We need to resolve to fine-class indices. This
                    # requires the dataset to provide class_to_idx.
                    classes.extend(
                        self._superclass_names_to_indices(cluster_def)
                    )
                else:
                    # CIFAR-10: cluster_def is already a list of class indices
                    classes.extend(int(x) for x in cluster_def)

            community_class_pools[comm_id] = sorted(set(classes))

        return community_class_pools

    def _superclass_names_to_indices(
        self, superclass_names: List[str]
    ) -> List[int]:
        """Resolve CIFAR-100 superclass names to fine-class indices.

        Uses the dataset's internal metadata when available; otherwise
        falls back to the canonical CIFAR-100 superclass ordering
        (5 fine classes per superclass, ordered by coarse label index).
        """
        from superadditivity.datasets.dataset_loader import SUPERCLASS_TO_FINE

        ds = self.train_dataset
        if hasattr(ds, "class_to_idx"):
            class_to_idx = ds.class_to_idx  # type: ignore[attr-defined]
            indices: List[int] = []
            for sc in superclass_names:
                fine_names = SUPERCLASS_TO_FINE.get(sc, [])
                for fn in fine_names:
                    if fn in class_to_idx:
                        indices.append(class_to_idx[fn])
            return indices

        # Fallback: use coarse labels from the CIFAR-100 dataset
        # CIFAR-100 stores coarse_targets when available
        if hasattr(ds, "targets") and hasattr(ds, "classes"):
            # Build a mapping from superclass name -> set of fine indices
            # by inspecting all samples.
            # This is a last resort and only runs once.
            logger.warning(
                "class_to_idx not found; falling back to sequential "
                "superclass index resolution."
            )
            all_superclass_names = list(SUPERCLASS_TO_FINE.keys())
            indices = []
            for sc in superclass_names:
                sc_idx = all_superclass_names.index(sc)
                # Each superclass has 5 consecutive fine classes
                for k in range(5):
                    indices.append(sc_idx * 5 + k)
            return indices

        raise RuntimeError(
            "Cannot resolve superclass names to fine-class indices."
        )

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
