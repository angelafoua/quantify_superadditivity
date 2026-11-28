"""Orchestrates all evaluation: extraction, drift metrics, and accuracy.

The :class:`ExperimentEvaluator` ties together every analyzer and the
representation extractor into a single ``evaluate`` call that can be
invoked once per training round.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from superadditivity.evaluation.cka_analyzer import CKAAnalyzer, ClientCKAAnalyzer
from superadditivity.evaluation.rsa_analyzer import RSAAnalyzer
from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer
from superadditivity.evaluation.paired_mmd_analyzer import (
    PairedMMDAnalyzer,
    multiplier_tag,
)
from superadditivity.evaluation.fisher_analyzer import FisherAnalyzer
from superadditivity.evaluation.centroid_analyzer import CentroidAnalyzer
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.evaluation.representation_extractor import RepresentationExtractor

logger = logging.getLogger(__name__)


class ExperimentEvaluator:
    """Orchestrates per-round evaluation of representation drift.

    Extracts representations from all clients, computes CKA (on all layers),
    RSA, Fisher, and centroid metrics (on the primary layer), client-level
    paired MMD (on the configured layers), and records everything to a
    :class:`DriftTracker`.

    Parameters
    ----------
    extractor:
        Representation extractor configured with the desired layers.
    cka:
        CKA analyzer instance.
    rsa:
        RSA analyzer instance.
    mmd:
        Legacy community-level MMD analyzer, kept only for comparison.
        ``None`` disables it. Its estimator is biased for the paired probe
        samples used here; report :class:`PairedMMDAnalyzer` results instead.
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
        Layer name used for RSA, Fisher, centroid and legacy MMD metrics
        (paired MMD uses ``paired_mmd_layers``).
    embedding_snapshot_rounds:
        If given, raw embeddings are stored at these round numbers.
    device:
        Device for inference.
    paired_mmd:
        Client-level paired MMD analyzer. ``None`` disables it.
    paired_mmd_layers:
        Layers on which paired MMD is computed. ``None`` means every layer
        of the extractor.
    accuracy_every:
        If set (and ``test_loader`` is given), each client's own model is
        evaluated on the test set at rounds divisible by this, and at
        ``final_round``. ``None`` disables accuracy evaluation.
    final_round:
        Last training round (always evaluated for accuracy when enabled).
    client_cka:
        Client-level CKA analyzer (the primary CKA drift). ``None``
        disables it.
    client_cka_layers:
        Layers for client-level CKA; ``None`` means every extracted layer.

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
        mmd: Optional[MMDAnalyzer],
        fisher: FisherAnalyzer,
        centroid: CentroidAnalyzer,
        drift_tracker: DriftTracker,
        probe_loader: torch.utils.data.DataLoader,
        community_map: Dict[int, int],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        primary_layer: str = "layer4",
        embedding_snapshot_rounds: Optional[Sequence[int]] = None,
        device: str = "cpu",
        paired_mmd: Optional[PairedMMDAnalyzer] = None,
        paired_mmd_layers: Optional[Sequence[str]] = None,
        accuracy_every: Optional[int] = None,
        final_round: Optional[int] = None,
        client_cka: Optional[ClientCKAAnalyzer] = None,
        client_cka_layers: Optional[Sequence[str]] = None,
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
        self.paired_mmd = paired_mmd
        if accuracy_every is not None and accuracy_every <= 0:
            raise ValueError(f"accuracy_every must be positive, got {accuracy_every}")
        self.accuracy_every = accuracy_every
        self.final_round = final_round
        self.client_cka = client_cka
        self.client_cka_layers = self._check_layers(client_cka_layers, extractor)
        self.paired_mmd_layers = self._check_layers(paired_mmd_layers, extractor)

    @staticmethod
    def _check_layers(
        layers: Optional[Sequence[str]], extractor: RepresentationExtractor
    ) -> List[str]:
        """Return ``layers`` (default: all extracted layers), validated."""
        if layers is None:
            return list(extractor.layer_names)
        unknown = [name for name in layers if name not in extractor.layer_names]
        if unknown:
            raise ValueError(
                f"Layers {unknown} are not extracted layers {extractor.layer_names}"
            )
        return list(layers)

    def evaluate(
        self,
        clients: Dict[int, Any],
        round_num: int,
    ) -> Dict[str, float]:
        """Run full evaluation for a single training round.

        Performs the following steps:

        1. Extract per-client embeddings for all layers, and average them
           into community-level embeddings.
        2. Compute CKA community matrix and mean off-diagonal for each layer.
        3. On the primary layer, compute RSA, Fisher ratio, centroid
           distances and (if enabled) legacy community MMD.
        4. Compute client-level paired MMD on the configured layers.
        5. Record all metrics to the drift tracker.
        6. Optionally store raw embedding snapshots.

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

        # Step 1: Extract client embeddings once; average into communities
        client_emb = self.extractor.extract(clients, self.probe_loader)
        community_emb = self.extractor.average_by_community(
            client_emb, self.community_map
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

            # Legacy community-level MMD (comparison only)
            if self.mmd is not None:
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

        # Step 4: Client-level paired MMD and client-level CKA
        if self.paired_mmd is not None:
            metrics.update(self._paired_mmd_metrics(client_emb, clients))
        if self.client_cka is not None:
            metrics.update(self._client_cka_metrics(client_emb, clients))

        # Step 5: Client test accuracy (each client's own model)
        if self._accuracy_due(round_num):
            metrics.update(self.evaluate_client_accuracy(clients))

        # Step 6: Record to drift tracker
        self.drift_tracker.record(round_num, metrics)

        # Step 7: Optionally store raw embeddings
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

    def _client_cka_metrics(
        self,
        client_emb: Dict[str, Dict[int, np.ndarray]],
        clients: Dict[int, Any],
    ) -> Dict[str, Any]:
        """Client-level CKA metrics for every configured layer.

        Keys ``cka_client_{L}_between`` / ``_within`` (mean CKA similarity
        over client pairs in different / the same community), ``_drift``
        (``within - between``) and ``_matrix`` (``(C, C)``, rows ordered by
        sorted client ID).
        """
        assert self.client_cka is not None
        comm_map = {
            cid: comm for cid, comm in self.community_map.items() if cid in clients
        }
        out: Dict[str, Any] = {}
        for layer in self.client_cka_layers:
            res = self.client_cka.compute(
                {cid: client_emb[layer][cid] for cid in comm_map}, comm_map
            )
            prefix = f"cka_client_{layer}"
            out[f"{prefix}_between"] = res["between"]
            out[f"{prefix}_within"] = res["within"]
            out[f"{prefix}_drift"] = res["drift"]
            out[f"{prefix}_matrix"] = res["matrix"]
        return out

    def _paired_mmd_metrics(
        self,
        client_emb: Dict[str, Dict[int, np.ndarray]],
        clients: Dict[int, Any],
    ) -> Dict[str, Any]:
        """Compute paired-MMD metrics for every configured layer.

        Metric keys, for each layer ``L`` (``mmd_paired_{L}_...``):

        - ``drift`` / ``between`` / ``within``: B - W, B and W at the
          primary (median-heuristic) bandwidth.
        - ``drift_bw{m}`` / ``between_bw{m}`` / ``within_bw{m}``: the same
          at every other bandwidth multiplier ``m`` (sensitivity analysis).
        - ``bandwidth``: the primary bandwidth sigma.
        - ``p_value``: descriptive community-label permutation p-value of
          B - W (``nan`` if disabled).
        - ``matrix``: the ``(C, C)`` client MMD^2 matrix at the primary
          bandwidth, rows and columns ordered by sorted client ID.

        Parameters
        ----------
        client_emb:
            ``layer_name -> client_id -> embeddings`` from the extractor.
        clients:
            The clients being evaluated.

        Returns
        -------
        Dict[str, Any]
            Flat metric dictionary.
        """
        assert self.paired_mmd is not None
        comm_map = {
            cid: comm for cid, comm in self.community_map.items() if cid in clients
        }
        primary = PairedMMDAnalyzer.PRIMARY_MULTIPLIER
        out: Dict[str, Any] = {}
        for layer in self.paired_mmd_layers:
            layer_emb = {cid: client_emb[layer][cid] for cid in comm_map}
            res = self.paired_mmd.compute(layer_emb, comm_map)
            prefix = f"mmd_paired_{layer}"
            out[f"{prefix}_drift"] = res["drift"]
            out[f"{prefix}_between"] = res["between"]
            out[f"{prefix}_within"] = res["within"]
            out[f"{prefix}_bandwidth"] = res["bandwidth"]
            out[f"{prefix}_p_value"] = res["p_value"]
            out[f"{prefix}_matrix"] = res["matrix"]
            for mult, agg in res["sensitivity"].items():
                if mult == primary:
                    continue
                tag = multiplier_tag(mult)
                out[f"{prefix}_drift_{tag}"] = agg["drift"]
                out[f"{prefix}_between_{tag}"] = agg["between"]
                out[f"{prefix}_within_{tag}"] = agg["within"]
        return out

    def _accuracy_due(self, round_num: int) -> bool:
        if self.accuracy_every is None or self.test_loader is None:
            return False
        return round_num % self.accuracy_every == 0 or round_num == self.final_round

    def evaluate_client_accuracy(self, clients: Dict[int, Any]) -> Dict[str, Any]:
        """Test accuracy of every client's own model (with its own BN stats).

        Parameters
        ----------
        clients:
            Mapping from client ID to client object with a ``.model``.

        Returns
        -------
        Dict[str, Any]
            ``accuracy_client_mean`` / ``_std`` / ``_min`` / ``_max``
            (scalars) and ``accuracy_clients`` (array ordered by client ID).

        Raises
        ------
        RuntimeError
            If ``test_loader`` was not provided.
        """
        if self.test_loader is None:
            raise RuntimeError("Cannot evaluate accuracy: test_loader was not provided")
        accs = []
        for cid in sorted(clients):
            model = clients[cid].model
            was_training = model.training
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    correct += int((model(images).argmax(dim=1) == labels).sum().item())
                    total += int(labels.numel())
            model.train(was_training)
            accs.append(correct / total if total else float("nan"))
        arr = np.asarray(accs, dtype=np.float64)
        logger.info(
            "Client test accuracy: mean=%.4f min=%.4f max=%.4f",
            arr.mean(), arr.min(), arr.max(),
        )
        return {
            "accuracy_client_mean": float(arr.mean()),
            "accuracy_client_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "accuracy_client_min": float(arr.min()),
            "accuracy_client_max": float(arr.max()),
            "accuracy_clients": arr,
        }
