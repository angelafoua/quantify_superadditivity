"""Orchestrates all evaluation: extraction, drift metrics, and accuracy.

The :class:`ExperimentEvaluator` ties together every analyzer and the
representation extractor into a single ``evaluate`` call that can be
invoked once per training round.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from superadditivity.evaluation.cka_analyzer import CKAAnalyzer
from superadditivity.evaluation.rsa_analyzer import RSAAnalyzer
from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer
from superadditivity.evaluation.fisher_analyzer import FisherAnalyzer
from superadditivity.evaluation.centroid_analyzer import CentroidAnalyzer
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.evaluation.representation_extractor import RepresentationExtractor
from superadditivity.models.model_utils import average_state_dicts

logger = logging.getLogger(__name__)


class ExperimentEvaluator:
    """Orchestrates per-round evaluation of representation drift.

    Extracts representations from all clients, computes CKA (on all layers),
    RSA, MMD, Fisher, and centroid metrics (on the primary layer), and
    records everything to a :class:`DriftTracker`.

    Parameters
    ----------
    extractor:
        Representation extractor configured with the desired layers.
    cka:
        CKA analyzer instance.
    rsa:
        RSA analyzer instance.
    mmd:
        MMD analyzer instance.
    fisher:
        Fisher analyzer instance.
    centroid:
        Centroid analyzer instance.
    drift_tracker:
        Tracker that accumulates metrics over rounds.
    probe_loader:
        DataLoader for the fixed probe set.
    community_map:
        Mapping from client ID to community ID.
    test_loader:
        Optional DataLoader for accuracy evaluation.
    primary_layer:
        Layer name used for RSA/MMD/Fisher/Centroid metrics.
    embedding_snapshot_rounds:
        If given, raw embeddings are stored at these round numbers.
    device:
        Device for inference.

    Examples
    --------
    >>> evaluator = ExperimentEvaluator(
    ...     extractor=extractor, cka=cka, rsa=rsa, mmd=mmd,
    ...     fisher=fisher, centroid=centroid, drift_tracker=tracker,
    ...     probe_loader=probe_loader, community_map=comm_map,
    ... )
    >>> metrics = evaluator.evaluate(clients, round_num=10)
    """

    def __init__(
        self,
        extractor: RepresentationExtractor,
        cka: CKAAnalyzer,
        rsa: RSAAnalyzer,
        mmd: MMDAnalyzer,
        fisher: FisherAnalyzer,
        centroid: CentroidAnalyzer,
        drift_tracker: DriftTracker,
        probe_loader: torch.utils.data.DataLoader,
        community_map: Dict[int, int],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        primary_layer: str = "layer4",
        embedding_snapshot_rounds: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ) -> None:
        self.extractor = extractor
        self.cka = cka
        self.rsa = rsa
        self.mmd = mmd
        self.fisher = fisher
        self.centroid = centroid
        self.drift_tracker = drift_tracker
        self.probe_loader = probe_loader
        self.community_map = community_map
        self.test_loader = test_loader
        self.primary_layer = primary_layer
        self.embedding_snapshot_rounds = set(
            embedding_snapshot_rounds or []
        )
        self.device = torch.device(device)

    def evaluate(
        self,
        clients: Dict[int, Any],
        round_num: int,
    ) -> Dict[str, float]:
        """Run full evaluation for a single training round.

        Performs the following steps:

        1. Extract community-level embeddings for all layers.
        2. Compute CKA community matrix and mean off-diagonal for each layer.
        3. On the primary layer, compute RSA, MMD, Fisher ratio, and
           centroid distances.
        4. Record all metrics to the drift tracker.
        5. Optionally store raw embedding snapshots.

        Parameters
        ----------
        clients:
            Mapping from client ID to client object with a ``.model``
            attribute.
        round_num:
            Current training round number.

        Returns
        -------
        Dict[str, float]
            Flat dictionary of all computed scalar metrics.
        """
        logger.info("Evaluating round %d with %d clients", round_num, len(clients))

        # Step 1: Extract community embeddings
        community_emb = self.extractor.extract_community_embeddings(
            clients, self.probe_loader, self.community_map
        )

        metrics: Dict[str, Any] = {}

        # Step 2: CKA on all layers
        for layer_name in self.extractor.layer_names:
            if layer_name not in community_emb:
                continue
            emb = community_emb[layer_name]
            cka_matrix = self.cka.compute_community_matrix(emb)
            mean_cka = self.cka.mean_off_diagonal(cka_matrix)
            metrics[f"cka_{layer_name}_matrix"] = cka_matrix
            metrics[f"cka_{layer_name}_mean"] = mean_cka

        # Step 3: Primary-layer metrics
        primary_emb = community_emb.get(self.primary_layer, {})
        if primary_emb:
            # RSA
            rsa_matrix = self.rsa.compute_community_matrix(primary_emb)
            metrics["rsa_matrix"] = rsa_matrix
            metrics["rsa_mean"] = self.rsa.mean_off_diagonal(rsa_matrix)

            # MMD
            mmd_matrix, p_matrix = self.mmd.compute_community_matrix(primary_emb)
            metrics["mmd_matrix"] = mmd_matrix
            metrics["mmd_p_matrix"] = p_matrix
            metrics["mmd_mean"] = self.mmd.mean_off_diagonal(mmd_matrix)

            # Fisher
            fisher_val = self.fisher.compute(primary_emb)
            metrics["fisher_ratio"] = fisher_val

            # Centroid
            centroids, dist_matrix = self.centroid.compute(primary_emb)
            metrics["centroid_distance_matrix"] = dist_matrix
            metrics["centroid_mean_distance"] = self.centroid.mean_distance(
                dist_matrix
            )

        # Step 4: Record to drift tracker
        self.drift_tracker.record(round_num, metrics)

        # Step 5: Optionally store raw embeddings
        if round_num in self.embedding_snapshot_rounds:
            self.drift_tracker.record_embeddings(round_num, community_emb)

        # Return only scalar metrics
        scalar_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        }

        logger.info(
            "Round %d evaluation complete: %d scalar metrics",
            round_num,
            len(scalar_metrics),
        )
        return scalar_metrics

    def evaluate_accuracy(
        self,
        clients: Dict[int, Any],
        round_num: int,
    ) -> Dict[str, float]:
        """Evaluate per-community and global accuracy.

        For each community, averages the state dicts of its constituent
        clients to create a community model, then evaluates on
        ``test_loader``.

        Parameters
        ----------
        clients:
            Mapping from client ID to client object with a ``.model``
            attribute.
        round_num:
            Current training round number.

        Returns
        -------
        Dict[str, float]
            Contains ``"global_accuracy"`` and ``"accuracy_community_{id}"``
            for each community.

        Raises
        ------
        RuntimeError
            If ``test_loader`` was not provided at construction time.
        """
        if self.test_loader is None:
            raise RuntimeError(
                "Cannot evaluate accuracy: test_loader was not provided"
            )

        # Group clients by community
        communities: Dict[int, List[int]] = defaultdict(list)
        for client_id, comm_id in self.community_map.items():
            if client_id in clients:
                communities[comm_id].append(client_id)

        results: Dict[str, float] = {}
        all_correct = 0
        all_total = 0

        for comm_id in sorted(communities.keys()):
            client_ids = communities[comm_id]

            # Average state dicts within community
            state_dicts = [
                clients[cid].model.state_dict() for cid in client_ids
            ]
            avg_state = average_state_dicts(state_dicts)

            # Create a temporary model with averaged weights
            reference_model = clients[client_ids[0]].model
            eval_model = type(reference_model)(
                *_get_constructor_args(reference_model)
            )
            eval_model.load_state_dict(avg_state)
            eval_model.to(self.device)
            eval_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    images, labels = batch[0].to(self.device), batch[1].to(
                        self.device
                    )
                    outputs = eval_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = correct / total if total > 0 else 0.0
            results[f"accuracy_community_{comm_id}"] = acc
            all_correct += correct
            all_total += total

        results["global_accuracy"] = (
            all_correct / all_total if all_total > 0 else 0.0
        )

        # Record to drift tracker
        self.drift_tracker.record(round_num, results)

        logger.info(
            "Round %d accuracy: global=%.4f, %d communities",
            round_num,
            results["global_accuracy"],
            len(communities),
        )
        return results


def _get_constructor_args(model: nn.Module) -> tuple:
    """Best-effort extraction of constructor args for model cloning.

    Falls back to empty tuple if not available.

    Parameters
    ----------
    model:
        The model to inspect.

    Returns
    -------
    tuple
        Constructor arguments, if recoverable.
    """
    # CIFARResNet stores enough info to reconstruct
    from superadditivity.models.resnet import CIFARResNet, BasicBlock

    if isinstance(model, CIFARResNet):
        # Infer num_classes from fc layer
        num_classes = model.fc.out_features
        # Infer block counts from layer sizes
        num_blocks = [
            len(model.layer1),
            len(model.layer2),
            len(model.layer3),
            len(model.layer4),
        ]
        return (BasicBlock, num_blocks, num_classes)

    return ()
