# Experiment Workflow

This document explains the end-to-end workflow for running superadditivity experiments in this repository — from environment setup through data collection, analysis, and figure generation.

## Overview

The codebase measures **superadditive interaction effects** between data heterogeneity and network topology in decentralized federated learning (D-SGD). The central quantity is the interaction term:

```
I = Drift(D) - Drift(B) - Drift(C) + Drift(A)
```

where A/B/C/D are the four cells of a 2×2 factorial (IID/Non-IID × Dense/Community), and Drift is measured via layer-wise CKA on a fixed probe set.

The pipeline has five stages:

```
Setup → Configure → Run experiments → Analyze → Visualize
```

---

## Stage 1: Environment Setup

Before running any experiment, validate that all dependencies are installed and the package is importable:

```bash
python scripts/validate_setup.py
```

**File:** [`scripts/validate_setup.py`](../scripts/validate_setup.py)

This script checks:
- Core Python packages (PyTorch, NumPy, SciPy, NetworkX, h5py, Hydra, statsmodels, matplotlib)
- GPU / CUDA availability
- All `superadditivity.*` subpackage imports
- Config file presence under `configs/`
- Seeding utilities

A passing run prints `ALL CHECKS PASSED`. Fix any `[FAIL]` lines before proceeding.

**Dependencies** are declared in:
- [`requirements.txt`](../requirements.txt) — pip-installable list
- [`pyproject.toml`](../pyproject.toml) — package metadata and optional extras

---

## Stage 2: Configuration

All experiments are configured through **Hydra** composition. The root config is [`configs/config.yaml`](../configs/config.yaml), which pulls in four override groups:

```yaml
# configs/config.yaml
defaults:
  - data: moderate_noniid
  - graph: sbm_medium
  - model: resnet18
  - training: default
  - _self_

experiment_name: superadditivity_experiment
output_dir: outputs/${experiment_name}

n_clients: 32
n_communities: 4

run_seed: 42
graph_seed: 100
```

> **Note:** `n_clients` defaults to **32** (with `n_communities: 4`, i.e. 8
> clients per community). `n_clients` must be a multiple of `n_communities`,
> validated at startup. Sweeps and older notes may reference other client
> counts, but 32 is the current default.

### Config groups

| Group | Directory | Purpose |
|-------|-----------|---------|
| `data` | [`configs/data/`](../configs/data/) | Dataset and partition method |
| `graph` | [`configs/graph/`](../configs/graph/) | Network topology and parameters |
| `model` | [`configs/model/`](../configs/model/) | Neural network architecture |
| `training` | [`configs/training/`](../configs/training/) | Optimizer, LR schedule, rounds |
| `experiment` | [`configs/experiment/`](../configs/experiment/) | Sweep definitions |

### Data configs

Seven datasets are supported: **CIFAR-100** (primary), **CIFAR-10**,
**Federated EMNIST**, **DomainNet** (clipart), **iNaturalist** (200-class
subset), **PathMNIST**, and **Google Speech Commands**. The full registry lives
in [`superadditivity/datasets/dataset_loader.py`](../superadditivity/datasets/dataset_loader.py).

**CIFAR-100 (primary):**

| File | Partition method | Description |
|------|-----------------|-------------|
| [`configs/data/iid.yaml`](../configs/data/iid.yaml) | `iid` | Uniform random assignment |
| [`configs/data/mild_noniid.yaml`](../configs/data/mild_noniid.yaml) | `dirichlet` | Dirichlet α=1.0 label skew |
| [`configs/data/moderate_noniid.yaml`](../configs/data/moderate_noniid.yaml) | `dirichlet` | Dirichlet α=0.5 label skew |
| [`configs/data/severe_noniid.yaml`](../configs/data/severe_noniid.yaml) | `dirichlet` | Dirichlet α=0.1 label skew |
| [`configs/data/quantity_skew.yaml`](../configs/data/quantity_skew.yaml) | `quantity_skew` | IID labels, skewed sample counts (negative control) |

**Other datasets (robustness checks):**

| File | Partition method | Description |
|------|-----------------|-------------|
| [`configs/data/cifar10_iid.yaml`](../configs/data/cifar10_iid.yaml) | `iid` | CIFAR-10, IID |
| [`configs/data/cifar10_mild_noniid.yaml`](../configs/data/cifar10_mild_noniid.yaml) | `dirichlet` | CIFAR-10, mild Non-IID |
| [`configs/data/cifar10_noniid.yaml`](../configs/data/cifar10_noniid.yaml) | `dirichlet` | CIFAR-10, moderate Non-IID |
| [`configs/data/cifar10_severe_noniid.yaml`](../configs/data/cifar10_severe_noniid.yaml) | `dirichlet` | CIFAR-10, severe Non-IID |
| [`configs/data/emnist_iid.yaml`](../configs/data/emnist_iid.yaml) | `iid` | Federated EMNIST, IID |
| [`configs/data/emnist_mild_noniid.yaml`](../configs/data/emnist_mild_noniid.yaml) | `dirichlet` | Federated EMNIST, mild Non-IID |
| [`configs/data/emnist_noniid.yaml`](../configs/data/emnist_noniid.yaml) | `dirichlet` | Federated EMNIST, moderate Non-IID |
| [`configs/data/emnist_severe_noniid.yaml`](../configs/data/emnist_severe_noniid.yaml) | `dirichlet` | Federated EMNIST, severe Non-IID |
| [`configs/data/domainnet_iid.yaml`](../configs/data/domainnet_iid.yaml) | `iid` | DomainNet (clipart), IID |
| [`configs/data/domainnet_mild_noniid.yaml`](../configs/data/domainnet_mild_noniid.yaml) | `dirichlet` | DomainNet, mild Non-IID |
| [`configs/data/domainnet_noniid.yaml`](../configs/data/domainnet_noniid.yaml) | `dirichlet` | DomainNet, moderate Non-IID |
| [`configs/data/domainnet_severe_noniid.yaml`](../configs/data/domainnet_severe_noniid.yaml) | `dirichlet` | DomainNet, severe Non-IID |
| [`configs/data/inaturalist_iid.yaml`](../configs/data/inaturalist_iid.yaml) | `iid` | iNaturalist (200-class subset), IID |
| [`configs/data/inaturalist_noniid.yaml`](../configs/data/inaturalist_noniid.yaml) | `dirichlet` | iNaturalist, Non-IID |
| [`configs/data/pathmnist_iid.yaml`](../configs/data/pathmnist_iid.yaml) | `iid` | PathMNIST, IID |
| [`configs/data/pathmnist_mild_noniid.yaml`](../configs/data/pathmnist_mild_noniid.yaml) | `dirichlet` | PathMNIST, mild Non-IID |
| [`configs/data/pathmnist_noniid.yaml`](../configs/data/pathmnist_noniid.yaml) | `dirichlet` | PathMNIST, moderate Non-IID |
| [`configs/data/pathmnist_severe_noniid.yaml`](../configs/data/pathmnist_severe_noniid.yaml) | `dirichlet` | PathMNIST, severe Non-IID |
| [`configs/data/speech_commands_iid.yaml`](../configs/data/speech_commands_iid.yaml) | `iid` | Google Speech Commands, IID |
| [`configs/data/speech_commands_noniid.yaml`](../configs/data/speech_commands_noniid.yaml) | `dirichlet` | Google Speech Commands, Non-IID |

### Graph configs

| File | Topology | Description |
|------|----------|-------------|
| [`configs/graph/erdos_renyi.yaml`](../configs/graph/erdos_renyi.yaml) | `erdos_renyi` | Dense baseline |
| [`configs/graph/sbm_weak.yaml`](../configs/graph/sbm_weak.yaml) | `sbm` | Weak community structure |
| [`configs/graph/sbm_medium.yaml`](../configs/graph/sbm_medium.yaml) | `sbm` | Medium community structure (primary) |
| [`configs/graph/sbm_strong.yaml`](../configs/graph/sbm_strong.yaml) | `sbm` | Strong community structure |
| [`configs/graph/watts_strogatz.yaml`](../configs/graph/watts_strogatz.yaml) | `watts_strogatz` | Small-world topology |
| [`configs/graph/ring_of_cliques.yaml`](../configs/graph/ring_of_cliques.yaml) | `ring_of_cliques` | Deterministic community structure |
| [`configs/graph/exponential.yaml`](../configs/graph/exponential.yaml) | `exponential` | Degree-heterogeneous topology |

### Model configs

| File | Architecture |
|------|-------------|
| [`configs/model/resnet18.yaml`](../configs/model/resnet18.yaml) | ResNet-18 (CIFAR variant, primary) |
| [`configs/model/convnet.yaml`](../configs/model/convnet.yaml) | SimpleConvNet (4-layer, robustness check) |
| [`configs/model/wide_resnet.yaml`](../configs/model/wide_resnet.yaml) | WideResNet-28-2 (robustness check) |
| [`configs/model/vit_tiny.yaml`](../configs/model/vit_tiny.yaml) | ViT-Tiny (Vision Transformer, robustness check) |

The model registry in [`scripts/run_experiment.py`](../scripts/run_experiment.py)
(`MODEL_REGISTRY`) also exposes a `resnet34_cifar` architecture (deeper ResNet,
same interface), which can be selected via `model.architecture=resnet34_cifar`.

### Training config

[`configs/training/default.yaml`](../configs/training/default.yaml) — D-SGD with 500 rounds, SGD optimizer (lr=0.1, momentum=0.9), cosine LR decay, and evaluation every 10 rounds.

---

## Stage 3: Running Experiments

### Single experiment

**File:** [`scripts/run_experiment.py`](../scripts/run_experiment.py)

```bash
# Using default config with overrides
python scripts/run_experiment.py data=moderate_noniid graph=sbm_medium

# Override any config key directly
python scripts/run_experiment.py data=iid graph=erdos_renyi run_seed=123
```

The `run()` function executes these steps in order:

1. **Seed everything** — calls `set_all_seeds(run_seed)` from [`superadditivity/utils/seed.py`](../superadditivity/utils/seed.py). The probe set is always seeded with the fixed constant `PROBE_SEED=999`, independent of `run_seed`.

2. **Load dataset** — [`superadditivity/datasets/dataset_loader.py`](../superadditivity/datasets/dataset_loader.py) downloads/caches one of the seven supported datasets (CIFAR-100, CIFAR-10, EMNIST, DomainNet, iNaturalist, PathMNIST, Speech Commands) and exposes `get_semantic_clusters()` and `get_probe_set()`.

3. **Build graph** — [`superadditivity/graphs/graph_manager.py`](../superadditivity/graphs/graph_manager.py) constructs the adjacency graph via the appropriate generator:
   - [`superadditivity/graphs/sbm_generator.py`](../superadditivity/graphs/sbm_generator.py) — Stochastic Block Model
   - [`superadditivity/graphs/erdos_renyi_generator.py`](../superadditivity/graphs/erdos_renyi_generator.py) — Erdős-Rényi
   - [`superadditivity/graphs/watts_strogatz_generator.py`](../superadditivity/graphs/watts_strogatz_generator.py) — Watts-Strogatz
   - [`superadditivity/graphs/special_topologies.py`](../superadditivity/graphs/special_topologies.py) — ring-of-cliques, exponential
   - [`superadditivity/graphs/mixing_matrix.py`](../superadditivity/graphs/mixing_matrix.py) — produces the doubly-stochastic symmetric mixing matrix W
   - [`superadditivity/graphs/graph_metrics.py`](../superadditivity/graphs/graph_metrics.py) — spectral gap, conductance, etc.

4. **Partition data** — assigns training samples to the `n_clients` clients (32 by default):
   - [`superadditivity/datasets/semantic_partitioner.py`](../superadditivity/datasets/semantic_partitioner.py) — Dirichlet label skew within semantic clusters (IID or non-IID modes)
   - [`superadditivity/datasets/quantity_skew_partitioner.py`](../superadditivity/datasets/quantity_skew_partitioner.py) — IID labels with skewed sample counts
   - [`superadditivity/datasets/client_dataset.py`](../superadditivity/datasets/client_dataset.py) — wraps indices into per-client `Dataset` objects

5. **Initialise model** — all clients start from the same weights. The model is built by `build_model()` in [`scripts/run_experiment.py`](../scripts/run_experiment.py) and weight-initialised via [`superadditivity/models/model_utils.py`](../superadditivity/models/model_utils.py):
   - [`superadditivity/models/resnet.py`](../superadditivity/models/resnet.py) — CIFAR-adapted ResNet-18/34 with 3×3 stride-1 stem
   - [`superadditivity/models/convnet.py`](../superadditivity/models/convnet.py) — lightweight 4-layer ConvNet
   - [`superadditivity/models/wide_resnet.py`](../superadditivity/models/wide_resnet.py) — WideResNet-28-2
   - [`superadditivity/models/vit.py`](../superadditivity/models/vit.py) — ViT-Tiny

6. **Create clients** — `n_clients` [`superadditivity/training/decentralized_client.py`](../superadditivity/training/decentralized_client.py) instances (32 by default), each holding a model copy, local dataset, and SGD optimizer.

7. **Run D-SGD** — the coordinator drives the training loop. The algorithm is selected by `cfg.training.algorithm`:
   - `dsgd` (default): [`superadditivity/training/dsgd_coordinator.py`](../superadditivity/training/dsgd_coordinator.py) — local SGD followed by gossip mixing via [`superadditivity/communication/gossip_mixer.py`](../superadditivity/communication/gossip_mixer.py)
   - `fedavg`: [`superadditivity/training/fedavg_coordinator.py`](../superadditivity/training/fedavg_coordinator.py)
   - `local_only`: [`superadditivity/training/local_only_coordinator.py`](../superadditivity/training/local_only_coordinator.py)

   Within each D-SGD round:
   - Every client runs `local_steps` SGD steps on its local data ([`superadditivity/training/local_trainer.py`](../superadditivity/training/local_trainer.py))
   - Gossip mixer computes `new_params[i] = Σ_j W[i,j] · params[j]` exactly
   - LR is updated via [`superadditivity/training/lr_schedule.py`](../superadditivity/training/lr_schedule.py) (cosine decay with warmup)

8. **Evaluate drift** — every `eval_every` rounds, [`superadditivity/evaluation/experiment_evaluator.py`](../superadditivity/evaluation/experiment_evaluator.py) runs:
   - [`superadditivity/evaluation/representation_extractor.py`](../superadditivity/evaluation/representation_extractor.py) — extract layer activations on the probe set (float64)
   - [`superadditivity/evaluation/cka_analyzer.py`](../superadditivity/evaluation/cka_analyzer.py) — layer-wise CKA (primary drift metric)
   - [`superadditivity/evaluation/rsa_analyzer.py`](../superadditivity/evaluation/rsa_analyzer.py) — Representational Similarity Analysis
   - [`superadditivity/evaluation/mmd_analyzer.py`](../superadditivity/evaluation/mmd_analyzer.py) — Maximum Mean Discrepancy
   - [`superadditivity/evaluation/fisher_analyzer.py`](../superadditivity/evaluation/fisher_analyzer.py) — Fisher divergence
   - [`superadditivity/evaluation/centroid_analyzer.py`](../superadditivity/evaluation/centroid_analyzer.py) — centroid distance
   - [`superadditivity/evaluation/drift_tracker.py`](../superadditivity/evaluation/drift_tracker.py) — accumulates metrics and writes to HDF5

9. **Log and checkpoint** — [`superadditivity/logging/csv_logger.py`](../superadditivity/logging/csv_logger.py), [`superadditivity/logging/checkpoint_manager.py`](../superadditivity/logging/checkpoint_manager.py), [`superadditivity/logging/metadata_store.py`](../superadditivity/logging/metadata_store.py), and optionally [`superadditivity/logging/wandb_logger.py`](../superadditivity/logging/wandb_logger.py).

**Outputs** (written to `outputs/<experiment_name>/seed_<run_seed>_graph_<graph_seed>/`):

| File | Contents |
|------|----------|
| `summary.json` | Final metrics and graph statistics |
| `metadata.json` | Full resolved config |
| `metrics.csv` | Per-round training metrics |
| `drift_metrics.h5` | Layer-wise drift values across all rounds |
| `checkpoints/` | Model checkpoints every 50 rounds |
| `run.log` | Full log output |

---

### Sweep experiments

**File:** [`scripts/run_sweep.py`](../scripts/run_sweep.py)

Runs all cells of a named experiment, iterating over seeds:

```bash
# Core 2×2 factorial (60 runs: 4 cells × 5 run_seeds × 3 graph_seeds)
python scripts/run_sweep.py --experiment core_factorial

# Extended 4×4 grid
python scripts/run_sweep.py --experiment extended_grid

# Parametric p_out sweep (for I(α,β) surface)
python scripts/run_sweep.py --experiment pout_sweep

# Per-dataset p_out sweeps (same runner, different dataset)
python scripts/run_sweep.py --experiment pout_sweep_cifar10
python scripts/run_sweep.py --experiment pout_sweep_emnist

# Robustness checks (different datasets, models, topologies)
python scripts/run_sweep.py --experiment robustness_check
```

**Experiment configs** (under [`configs/experiment/`](../configs/experiment/)):

| File | Description |
|------|-------------|
| [`core_factorial.yaml`](../configs/experiment/core_factorial.yaml) | 4 cells (A/B/C/D), 5 run_seeds × 3 graph_seeds each |
| [`extended_grid.yaml`](../configs/experiment/extended_grid.yaml) | 4 data levels × 4 network levels |
| [`pout_sweep.yaml`](../configs/experiment/pout_sweep.yaml) | Varying p_out for I(α,β) surface fit (CIFAR-100) |
| [`pout_sweep_cifar10.yaml`](../configs/experiment/pout_sweep_cifar10.yaml) | p_out sweep on CIFAR-10 |
| [`pout_sweep_emnist.yaml`](../configs/experiment/pout_sweep_emnist.yaml) | p_out sweep on Federated EMNIST |
| [`pout_sweep_pathmnist.yaml`](../configs/experiment/pout_sweep_pathmnist.yaml) | p_out sweep on PathMNIST |
| [`pout_sweep_domainnet.yaml`](../configs/experiment/pout_sweep_domainnet.yaml) | p_out sweep on DomainNet |
| [`robustness_check.yaml`](../configs/experiment/robustness_check.yaml) | Alternative datasets (CIFAR-10, EMNIST), models (ConvNet), heterogeneity (quantity skew), and topologies (Watts-Strogatz) |

Any experiment whose name starts with `pout_sweep` (e.g. `pout_sweep_cifar10`)
is dispatched through the parametric p_out sweep runner, so the per-dataset
variants run with the same driver as `pout_sweep`.

#### Core factorial design

The primary experiment tests superadditivity in a 2×2 design:

|  | Dense (ER) | Community (SBM) |
|--|-----------|----------------|
| **IID** | Cell A — baseline | Cell C — network effect only |
| **Non-IID** | Cell B — data effect only | Cell D — both effects |

`I = Drift(D) - Drift(B) - Drift(C) + Drift(A) > 0` is the superadditivity hypothesis.

---

## Stage 4: Statistical Analysis

**File:** [`scripts/run_analysis.py`](../scripts/run_analysis.py)

```bash
# Factorial analysis (2×2 ANOVA + bootstrap interaction test)
python scripts/run_analysis.py --results_dir outputs/core_factorial

# Interaction surface fit (for p_out sweep data)
python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface

# Both
python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis both
```

**Analysis modules** (under [`superadditivity/analysis/`](../superadditivity/analysis/)):

| File | Purpose |
|------|---------|
| [`aggregator.py`](../superadditivity/analysis/aggregator.py) | Load and aggregate `summary.json` files into a DataFrame |
| [`anova.py`](../superadditivity/analysis/anova.py) | Two-way ANOVA and permutation ANOVA (10,000 permutations) |
| [`interaction_analyzer.py`](../superadditivity/analysis/interaction_analyzer.py) | Bootstrap I>0 test (10,000 resamples), p_out sweep aggregation, surface fitting |
| [`effect_sizes.py`](../superadditivity/analysis/effect_sizes.py) | Bootstrap confidence intervals |
| [`multiple_comparisons.py`](../superadditivity/analysis/multiple_comparisons.py) | Pairwise t-tests with correction |
| [`mixed_effects.py`](../superadditivity/analysis/mixed_effects.py) | Linear mixed-effects model |
| [`statistical_report.py`](../superadditivity/analysis/statistical_report.py) | Generates a Markdown report |

**Outputs** (written to `outputs/<experiment_name>/analysis/`):

| File | Contents |
|------|----------|
| `statistical_report.md` | Full analysis report |
| `interaction_test.json` | Bootstrap I estimate, CI, and p-value |
| `interaction_surface.json` | α/β grid values and surface fit parameters |
| `surface_report.md` | Surface fit summary |

---

## Stage 5: Visualization

**File:** [`scripts/run_visualization.py`](../scripts/run_visualization.py)

```bash
python scripts/run_visualization.py --results_dir outputs/core_factorial
```

**Visualization modules** (under [`superadditivity/visualization/`](../superadditivity/visualization/)):

| File | Figure |
|------|--------|
| [`interaction_plots.py`](../superadditivity/visualization/interaction_plots.py) | 2×2 factorial grid showing drift per cell |
| [`drift_trajectories.py`](../superadditivity/visualization/drift_trajectories.py) | CKA drift over training rounds |
| [`interaction_surface.py`](../superadditivity/visualization/interaction_surface.py) | 3-D surface of I(α,β) |
| [`cka_heatmaps.py`](../superadditivity/visualization/cka_heatmaps.py) | Layer-wise CKA heatmaps |
| [`topology_plots.py`](../superadditivity/visualization/topology_plots.py) | Graph topology diagrams |
| [`embedding_projections.py`](../superadditivity/visualization/embedding_projections.py) | UMAP/t-SNE of learned representations |
| [`pout_sweep_plots.py`](../superadditivity/visualization/pout_sweep_plots.py) | I vs p_out line plots |
| [`figure_style.py`](../superadditivity/visualization/figure_style.py) | Publication-quality matplotlib style |

**Outputs** (written to `outputs/<experiment_name>/figures/`):

| File | Contents |
|------|----------|
| `factorial_grid.pdf` | Main 2×2 interaction figure |
| `drift_trajectories.pdf` | Per-condition drift curves |
| `interaction_surface.pdf` | I(α,β) surface (if surface data exists) |

---

## Tests

```bash
pytest tests/ -q
```

Tests run on CPU without downloading datasets (synthetic fixtures in [`tests/conftest.py`](../tests/conftest.py)):

| File | What it tests |
|------|--------------|
| [`tests/test_mixing_matrix.py`](../tests/test_mixing_matrix.py) | W is doubly stochastic and symmetric |
| [`tests/test_gossip_mixer.py`](../tests/test_gossip_mixer.py) | Gossip step computes Σ_j W[i,j]·params[j] exactly |
| [`tests/test_cka.py`](../tests/test_cka.py) | CKA values and symmetry properties |
| [`tests/test_mmd.py`](../tests/test_mmd.py) | MMD estimator |
| [`tests/test_partitioner.py`](../tests/test_partitioner.py) | IID and Dirichlet partitioning correctness |
| [`tests/test_sbm_generator.py`](../tests/test_sbm_generator.py) | SBM graph properties |
| [`tests/test_interaction_analyzer.py`](../tests/test_interaction_analyzer.py) | Interaction term I and bootstrap test |
| [`tests/test_dsgd_coordinator.py`](../tests/test_dsgd_coordinator.py) | D-SGD coordinator round logic |
| [`tests/test_representation_extractor.py`](../tests/test_representation_extractor.py) | Layer extraction and float64 precision |
| [`tests/test_integration.py`](../tests/test_integration.py) | End-to-end mini-experiment smoke test |

---

## Complete Workflow Example

```bash
# 1. Validate environment
python scripts/validate_setup.py

# 2. Run the core 2×2 factorial sweep (60 runs)
python scripts/run_sweep.py --experiment core_factorial

# 3. Analyze results
python scripts/run_analysis.py --results_dir outputs/core_factorial

# 4. Generate figures
python scripts/run_visualization.py --results_dir outputs/core_factorial

# 5. (Optional) Run p_out sweep for interaction surface
python scripts/run_sweep.py --experiment pout_sweep
python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface
python scripts/run_visualization.py --results_dir outputs/pout_sweep
```

---

## Key Invariants

The following must never be broken:

1. **Mixing matrix W** is doubly stochastic and symmetric. Tested in [`tests/test_mixing_matrix.py`](../tests/test_mixing_matrix.py).
2. **Gossip mixing** computes `new[i] = Σ_j W[i,j] · params[j]` exactly. Tested in [`tests/test_gossip_mixer.py`](../tests/test_gossip_mixer.py).
3. **Probe set** uses the fixed seed `PROBE_SEED=999` in [`superadditivity/utils/seed.py`](../superadditivity/utils/seed.py) and never depends on `run_seed`.
4. **All clients** start from identical model weights (seeded by `run_seed`); `n_clients` must be a multiple of `n_communities`.
5. **All drift metrics** are computed in float64 for numerical stability.
