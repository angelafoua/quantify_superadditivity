"""Client-level paired MMD for measuring community representation drift.

This is the MMD protocol used for reported results. It replaces the
community-level :class:`~superadditivity.evaluation.mmd_analyzer.MMDAnalyzer`
(kept only for comparison) and differs from it in four ways:

1. **Unit of comparison: clients.** MMD^2 is computed between every pair of
   clients on their own embeddings of the probe set, and is not computed
   between activation-averaged "community models" that no client runs.
2. **Paired estimator.** Every client embeds the *same* probe images, so
   the samples are paired: ``x_i`` and ``y_i`` are two models' views of
   image ``i``. The estimator is the equal-size U-statistic of Gretton et
   al. (2012, eq. 3), which excludes ``i == j`` in *all* three kernel
   terms, including the cross term. This keeps it unbiased under pairing.
   The independent-sample estimator includes ``k(x_i, y_i)`` in the cross
   term and is biased downward when ``x_i`` and ``y_i`` are correlated.
3. **One kernel per evaluation.** A Gaussian RBF kernel is used, with a
   single bandwidth for all client pairs at a given evaluation: the median
   pairwise Euclidean distance of a fixed-index subsample of the pooled
   embeddings of all clients. All pairs are then measured with the same
   kernel, and the metric is invariant to a global rescaling of the
   features, so feature-norm growth during training is not counted as
   drift. Additional bandwidths (multiples of the median) are computed
   for sensitivity analysis.
4. **Aggregation and inference.** The drift statistic is
   ``B - W``: the mean MMD^2 over client pairs in *different* communities
   minus the mean over pairs in the *same* community. ``W`` is the noise
   floor (client disagreement irrespective of topology). Inference about
   the interaction term ``I`` is done across runs (seeds) downstream, not
   here. An optional descriptive permutation test of ``B - W`` permutes
   client-to-community labels (preserving community sizes).

All kernel computations use float64. They run on the given device, for
example CUDA, where all client pairs take seconds.

References
----------
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
(2012). A kernel two-sample test. *JMLR*, 13, 723-773.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


def multiplier_tag(multiplier: float) -> str:
    """Format a bandwidth multiplier as a metric-key-safe tag.

    Parameters
    ----------
    multiplier:
        Bandwidth multiplier, e.g. ``0.25``.

    Returns
    -------
    str
        Tag such as ``"bw0p25"`` (``.`` replaced by ``p``).
    """
    return "bw" + f"{float(multiplier):g}".replace(".", "p").replace("-", "m")


def median_bandwidth(
    pooled: torch.Tensor,
    subsample: int,
    seed: int,
) -> float:
    """Median pairwise Euclidean distance on a fixed-index subsample.

    Parameters
    ----------
    pooled:
        Pooled embeddings of shape ``(N, d)``.
    subsample:
        Maximum number of points used. If ``N > subsample`` a subset of
        ``subsample`` rows is drawn with a generator seeded by ``seed``,
        so the same rows are used for every call with the same ``N``.
    seed:
        Seed for the subsample indices.

    Returns
    -------
    float
        Median of the pairwise distances between distinct points, clamped
        to at least ``1e-12``.
    """
    n_total = pooled.shape[0]
    if n_total > subsample:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(n_total, generator=gen)[:subsample]
        pts = pooled[idx.to(pooled.device)]
    else:
        pts = pooled
    dists = torch.cdist(pts, pts)
    iu = torch.triu_indices(pts.shape[0], pts.shape[0], offset=1, device=pts.device)
    median = float(dists[iu[0], iu[1]].median().item())
    return max(median, 1e-12)


def paired_mmd2_matrix(
    embeddings: torch.Tensor,
    sigmas: Sequence[float],
) -> torch.Tensor:
    """Paired unbiased MMD^2 between every pair of clients.

    Parameters
    ----------
    embeddings:
        Tensor of shape ``(C, n, d)``: ``C`` clients, ``n`` paired probe
        samples (row ``i`` is the same probe image for every client),
        ``d`` features. Should be float64.
    sigmas:
        RBF bandwidths. One ``(C, C)`` matrix is returned per bandwidth.

    Returns
    -------
    torch.Tensor
        Shape ``(len(sigmas), C, C)``. Entry ``[s, a, b]`` is

        ``1/(n(n-1)) * sum_{i != j} [k(x_i,x_j) + k(y_i,y_j)
        - k(x_i,y_j) - k(x_j,y_i)]``

        with ``x = embeddings[a]``, ``y = embeddings[b]``, and kernel
        bandwidth ``sigmas[s]``. The matrix is symmetric with zero diagonal.
        Values can be slightly negative (unbiased estimator).
    """
    if embeddings.dim() != 3:
        raise ValueError(
            f"embeddings must have shape (C, n, d), got {tuple(embeddings.shape)}"
        )
    C, n, _ = embeddings.shape
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples, got n={n}")

    sig = torch.as_tensor(list(sigmas), dtype=embeddings.dtype, device=embeddings.device)
    gammas = 1.0 / (2.0 * sig ** 2)  # (S,)
    S = sig.shape[0]

    # Off-diagonal block sums: T[s, a, b] = sum_{i != j} k(x^a_i, x^b_j)
    T = torch.zeros((S, C, C), dtype=embeddings.dtype, device=embeddings.device)
    for a in range(C):
        rest = embeddings[a:]                                   # (C-a, n, d)
        # Squared distances between client a and clients a..C-1: (n, C-a, n)
        sq = torch.cdist(embeddings[a], rest.reshape(-1, rest.shape[-1])) ** 2
        sq = sq.reshape(n, C - a, n)
        # Paired diagonal (i == j) squared distances: (C-a, n)
        sq_diag = torch.diagonal(sq, dim1=0, dim2=2)
        for s in range(S):
            k_full = torch.exp(-gammas[s] * sq).sum(dim=(0, 2))       # (C-a,)
            k_diag = torch.exp(-gammas[s] * sq_diag).sum(dim=1)       # (C-a,)
            T[s, a, a:] = k_full - k_diag
            T[s, a:, a] = k_full - k_diag

    within = torch.diagonal(T, dim1=1, dim2=2)                   # (S, C)
    mmd2 = (within[:, :, None] + within[:, None, :] - 2.0 * T) / (n * (n - 1))
    idx = torch.arange(C, device=embeddings.device)
    mmd2[:, idx, idx] = 0.0
    return mmd2


def between_within(
    mmd2: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Aggregate a client MMD^2 matrix into between/within community means.

    Parameters
    ----------
    mmd2:
        Symmetric ``(C, C)`` client MMD^2 matrix.
    labels:
        Community label per client, shape ``(C,)``.

    Returns
    -------
    Dict[str, float]
        ``"between"`` (mean over pairs in different communities),
        ``"within"`` (mean over pairs in the same community) and
        ``"drift"`` (``between - within``). A mean with no qualifying pairs
        is ``nan``.
    """
    C = mmd2.shape[0]
    iu = np.triu_indices(C, k=1)
    vals = mmd2[iu]
    same = labels[iu[0]] == labels[iu[1]]
    between = float(vals[~same].mean()) if (~same).any() else float("nan")
    within = float(vals[same].mean()) if same.any() else float("nan")
    return {"between": between, "within": within, "drift": between - within}


def community_permutation_test(
    mmd2: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    seed: int,
) -> float:
    """Permutation p-value for ``B - W`` under random community labels.

    Shuffles the client-to-community labels (which preserves community
    sizes) and recomputes ``B - W`` from the fixed MMD^2 matrix. No kernel
    is recomputed, so 10,000 permutations take milliseconds.

    Treat this as a descriptive test (are clients in the same community
    more alike than clients in different communities, at this round of
    this run?). It assumes clients are exchangeable under the null. The
    paper's inferential claims about ``I`` use seed-level resampling
    downstream.

    Parameters
    ----------
    mmd2:
        Symmetric ``(C, C)`` client MMD^2 matrix.
    labels:
        Community label per client.
    n_permutations:
        Number of random relabelings.
    seed:
        RNG seed.

    Returns
    -------
    float
        ``(#{perm stat >= observed} + 1) / (n_permutations + 1)``.
    """
    C = mmd2.shape[0]
    iu = np.triu_indices(C, k=1)
    vals = mmd2[iu]
    observed = between_within(mmd2, labels)["drift"]

    rng = np.random.RandomState(seed)
    perms = np.stack([rng.permutation(labels) for _ in range(n_permutations)])
    same = perms[:, iu[0]] == perms[:, iu[1]]                     # (P, pairs)
    n_same = same.sum(axis=1)
    n_diff = same.shape[1] - n_same
    sum_same = (same * vals).sum(axis=1)
    sum_diff = vals.sum() - sum_same
    stats = sum_diff / n_diff - sum_same / n_same
    # Relabelings that give the same partition reproduce `observed` up to
    # float summation order; count them as ties with a relative tolerance.
    tol = 1e-9 * float(np.abs(vals).max()) if vals.size else 0.0
    count = int(np.sum(stats >= observed - tol))
    return float((count + 1) / (n_permutations + 1))


class PairedMMDAnalyzer:
    """Client-level paired MMD with between-minus-within aggregation.

    Parameters
    ----------
    bandwidth_multipliers:
        Multiples of the median-heuristic bandwidth to evaluate. Must
        include ``1.0`` (the primary bandwidth).
    bandwidth_subsample:
        Maximum number of pooled points used for the median heuristic.
    bandwidth_seed:
        Seed for the median-heuristic subsample indices. Keep it fixed
        across runs.
    n_permutations:
        Permutations for the descriptive community-label test of ``B - W``
        (primary bandwidth only). ``0`` disables the test.
    permutation_seed:
        Seed for the permutation test.
    device:
        Device for kernel computations (e.g. ``"cuda:0"``).

    Examples
    --------
    >>> analyzer = PairedMMDAnalyzer(bandwidth_multipliers=[0.5, 1.0, 2.0])
    >>> out = analyzer.compute(client_emb, community_map)
    >>> out["drift"], out["between"], out["within"]
    """

    PRIMARY_MULTIPLIER = 1.0

    def __init__(
        self,
        bandwidth_multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        bandwidth_subsample: int = 2000,
        bandwidth_seed: int = 0,
        n_permutations: int = 10000,
        permutation_seed: int = 0,
        device: str = "cpu",
    ) -> None:
        mults = [float(m) for m in bandwidth_multipliers]
        if self.PRIMARY_MULTIPLIER not in mults:
            raise ValueError(
                "bandwidth_multipliers must include 1.0 (the primary bandwidth); "
                f"got {mults}"
            )
        if any(m <= 0 for m in mults):
            raise ValueError(f"bandwidth multipliers must be positive, got {mults}")
        self.bandwidth_multipliers = mults
        self.bandwidth_subsample = int(bandwidth_subsample)
        self.bandwidth_seed = int(bandwidth_seed)
        self.n_permutations = int(n_permutations)
        self.permutation_seed = int(permutation_seed)
        self.device = torch.device(device)

    def compute(
        self,
        client_embeddings: Dict[int, np.ndarray],
        community_map: Dict[int, int],
        run_permutation_test: bool = True,
    ) -> Dict[str, Any]:
        """Compute client MMD^2 matrices and community aggregates.

        Parameters
        ----------
        client_embeddings:
            Mapping from client ID to that client's probe embeddings
            ``(n, d)``. All clients must use the same probe images in the
            same order.
        community_map:
            Mapping from client ID to community ID.
        run_permutation_test:
            If ``False``, skip the permutation test even if
            ``n_permutations > 0``.

        Returns
        -------
        Dict[str, Any]
            ``"client_ids"`` (sorted ``(C,)`` array), ``"bandwidth"``
            (primary sigma), ``"matrix"`` (primary ``(C, C)`` MMD^2),
            ``"between"``, ``"within"``, ``"drift"`` (primary bandwidth),
            ``"p_value"`` (or ``nan`` if not run), and
            ``"sensitivity"``: ``{multiplier: {"between", "within",
            "drift"}}`` for every multiplier.
        """
        client_ids = sorted(client_embeddings.keys())
        missing = [c for c in client_ids if c not in community_map]
        if missing:
            raise KeyError(f"Clients missing from community_map: {missing}")
        shapes = {np.asarray(client_embeddings[c]).shape for c in client_ids}
        if len(shapes) != 1:
            raise ValueError(
                f"All clients must have embeddings of the same shape, got {shapes}"
            )

        X = torch.as_tensor(
            np.stack([np.asarray(client_embeddings[c], dtype=np.float64)
                      for c in client_ids]),
            dtype=torch.float64,
            device=self.device,
        )
        labels = np.array([community_map[c] for c in client_ids])

        sigma = median_bandwidth(
            X.reshape(-1, X.shape[-1]), self.bandwidth_subsample, self.bandwidth_seed
        )
        sigmas = [sigma * m for m in self.bandwidth_multipliers]
        mats = paired_mmd2_matrix(X, sigmas).cpu().numpy()

        sensitivity: Dict[float, Dict[str, float]] = {}
        for m, mat in zip(self.bandwidth_multipliers, mats):
            sensitivity[m] = between_within(mat, labels)

        primary_idx = self.bandwidth_multipliers.index(self.PRIMARY_MULTIPLIER)
        primary = mats[primary_idx]
        p_value = float("nan")
        if run_permutation_test and self.n_permutations > 0:
            p_value = community_permutation_test(
                primary, labels, self.n_permutations, self.permutation_seed
            )

        result: Dict[str, Any] = {
            "client_ids": np.asarray(client_ids),
            "bandwidth": sigma,
            "matrix": primary,
            "p_value": p_value,
            "sensitivity": sensitivity,
        }
        result.update(sensitivity[self.PRIMARY_MULTIPLIER])
        return result
