# CLAUDE.md — Project context for Claude Code

This file orients an AI assistant (or a new contributor) working in this
repository. See the [Research Paper Proposal](Research%20Paper%20Proposal%20-%20Beyond%20Additive%20Heterogeneity.md)
for the full scientific context.

## What this project is

A reproducible research codebase that measures **superadditive interaction
effects** between data heterogeneity and network topology in **decentralized
federated learning (DFL / D-SGD)**. The primary analysis tests whether the
combined effect of non-IID data and community-structured networks on
representation drift exceeds the sum of their individual effects.

The headline quantity is the **interaction term**:
```
I = Drift(D) - Drift(B) - Drift(C) + Drift(A)
```
where A/B/C/D are the four cells of a 2×2 factorial (IID/Non-IID × Dense/Community),
and Drift is measured via layer-wise CKA on a fixed probe set.

## Mental model of the pipeline

```
config (Hydra) → graph (SBM/ER/WS) → partition (Dirichlet/IID/Quantity) → 128 clients
   → D-SGD rounds {local SGD → gossip mix → drift eval} → HDF5 results
   → statistical analysis (ANOVA + bootstrap I>0 test) → figures
```

The single integration point is `scripts/run_experiment.py::run`. Everything
else is library code under `superadditivity/`.

## Where things live

| Concern | Location |
|---------|----------|
| Seeding / reproducibility | `superadditivity/utils/seed.py` |
| Multi-dataset loading | `superadditivity/datasets/dataset_loader.py` |
| Non-IID partitioning | `superadditivity/datasets/semantic_partitioner.py` |
| Quantity-skew partitioning | `superadditivity/datasets/quantity_skew_partitioner.py` |
| Per-client dataset | `superadditivity/datasets/client_dataset.py` |
| Graphs + mixing matrix | `superadditivity/graphs/` |
| Models (ResNet-18, ConvNet) | `superadditivity/models/` |
| Training loop (D-SGD) | `superadditivity/training/dsgd_coordinator.py` |
| Gossip mixing (critical!) | `superadditivity/communication/gossip_mixer.py` |
| Drift metrics (CKA, RSA, MMD, Fisher, Centroid) | `superadditivity/evaluation/` |
| Superadditivity analysis (I, bootstrap, surface fit) | `superadditivity/analysis/interaction_analyzer.py` |
| Statistics (ANOVA, effect sizes, mixed-effects) | `superadditivity/analysis/` |
| Figures | `superadditivity/visualization/` |
| Logging (CSV, W&B, checkpoints) | `superadditivity/logging/` |
| Configs | `configs/` |

## Supported datasets

- **CIFAR-100** (primary) — 100 fine classes, 20 superclasses, 4 semantic clusters
- **CIFAR-10** — 10 classes, simpler task for robustness check
- **Federated EMNIST** — 62 classes (digits + letters), grayscale, different domain

## Supported models

- **ResNet-18** (CIFAR variant) — 3×3 stride-1 stem, layers 1-4 (64/128/256/512), feature_dim=512
- **SimpleConvNet** (4-layer) — lighter architecture for robustness check

## Supported topologies

- **SBM** (Stochastic Block Model) — primary, parameterised by p_in/p_out
- **Erdős-Rényi** — dense baseline, matched expected density
- **Watts-Strogatz** — small-world topology (new)
- **Ring-of-cliques** — deterministic community structure
- **Exponential graph** — degree-heterogeneous topology

## Data heterogeneity types

- **IID** — uniform random assignment
- **Dirichlet semantic** — label skew via Dirichlet(α) within semantic clusters
- **Quantity skew** — IID labels but Dirichlet-skewed sample counts (negative control)

## Invariants that MUST hold (do not break these)

1. **Mixing matrix W is doubly stochastic and symmetric**, with weight only on
   edges or the diagonal. Tested in `tests/test_mixing_matrix.py`.
2. **Gossip mixing computes `new[i] = Σ_j W[i,j] · params[j]` exactly.** This is
   the single most failure-prone step. Tested in `tests/test_gossip_mixer.py`.
3. **The probe set is identical across every run** (fixed seed `999`). Never make
   the probe seed depend on `run_seed`/`graph_seed`.
4. **All 128 clients start from the same initialisation** (seeded by `run_seed`).
5. **All drift metrics are computed in float64** for numerical stability.
6. **This codebase is independent** — no imports from `dfl_drift` or any other project.

## Conventions

- Type hints and docstrings on all public functions/classes.
- Logging via the module-level `logger = logging.getLogger(__name__)`.
- New config keys must be added to the relevant `configs/*.yaml` AND consumed in
  code — Hydra will not error on unused keys, so keep them in sync.
- Tests must run on CPU without dataset downloads (use synthetic fixtures in
  `tests/conftest.py`).

## Running things

```bash
# Environment check
python scripts/validate_setup.py

# Run tests (fast, CPU only)
pytest tests/ -q

# Single experiment
python scripts/run_experiment.py data=moderate_noniid graph=sbm_medium

# Full factorial sweep
python scripts/run_sweep.py --experiment core_factorial

# Extended 4×4 grid
python scripts/run_sweep.py --experiment extended_grid

# p_out parametric sweep (for I(α,β) surface)
python scripts/run_sweep.py --experiment pout_sweep

# Robustness checks (different datasets/models/topologies)
python scripts/run_sweep.py --experiment robustness_check

# Post-hoc analysis
python scripts/run_analysis.py --results_dir outputs/core_factorial
python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface

# Generate figures
python scripts/run_visualization.py --results_dir outputs/core_factorial
```

## Common extension points

- **New drift metric**: add an analyzer in `superadditivity/evaluation/`, wire it
  into `ExperimentEvaluator.evaluate`, record it in `DriftTracker`.
- **New topology**: add a generator in `superadditivity/graphs/`, dispatch it in
  `GraphManager.build`, add a `configs/graph/*.yaml`.
- **New data regime**: add a `configs/data/*.yaml` and (if a new mechanism) a
  branch in `SemanticPartitioner` or a new partitioner class.
- **New model**: add to `superadditivity/models/`, dispatch in
  `scripts/run_experiment.py::build_model`, add `configs/model/*.yaml`.
