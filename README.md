# Superadditivity: Beyond Additive Heterogeneity

**Quantifying Superadditive Interaction Effects in Decentralized Federated Learning**

This repository implements a reproducible experimental pipeline for studying
whether data heterogeneity and network topology interact *superadditively* in
decentralized federated learning (DFL). That is, whether their combined effect on
representation drift exceeds the sum of their individual effects.

## Key Research Question

> Does the interaction between non-IID data partitioning and community-structured
> communication networks produce **superadditive** representation drift, measured
> by the interaction term I = Drift(D) - Drift(B) - Drift(C) + Drift(A) in a
> 2×2 factorial design?

## Method

- **Drift metric**: Layer-wise Centered Kernel Alignment (CKA) on a fixed probe set
- **Training**: Decentralized SGD (D-SGD) with Metropolis-Hastings gossip mixing
- **Factorial design**: IID/Non-IID × Dense/Community, extended to 4×4
- **Statistical validation**: Two-way ANOVA, bootstrap test for I > 0, I(α,β) surface fitting

## Experimental Variations

| Dimension | Options |
|-----------|---------|
| **Datasets** | CIFAR-100 (primary), CIFAR-10, Federated EMNIST |
| **Models** | ResNet-18 (primary), 4-layer ConvNet |
| **Data heterogeneity** | IID, Dirichlet semantic (α ∈ {0.1, 0.5, 10}), Quantity skew |
| **Topologies** | SBM (primary), Erdős-Rényi, Watts-Strogatz, Ring-of-cliques, Exponential |

## Quick Start

```bash
# Create environment
python -m venv venv && source venv/bin/activate
pip install -e .

# Validate setup
python scripts/validate_setup.py

# Run tests
pytest tests/ -q

# Single experiment
python scripts/run_experiment.py data=moderate_noniid graph=sbm_medium

# Core 2×2 factorial sweep (60 runs)
python scripts/run_sweep.py --experiment core_factorial

# Post-hoc analysis
python scripts/run_analysis.py --results_dir outputs/core_factorial_*
```

## Project Structure

```
superadditivity/
├── datasets/          # CIFAR-100/10, EMNIST, partitioning
├── models/            # ResNet-18, ConvNet
├── graphs/            # SBM, ER, Watts-Strogatz, ring-of-cliques, exponential
├── communication/     # Gossip mixing with doubly-stochastic W
├── evaluation/        # CKA, RSA, MMD, Fisher, centroid drift metrics
├── training/          # D-SGD, FedAvg, local-only coordinators
├── analysis/          # ANOVA, bootstrap, interaction surface, mixed-effects
├── visualization/     # Publication figures, interaction surface plots
├── logging/           # CSV, W&B, checkpoints, metadata
└── utils/             # Seeding, device, I/O
configs/               # Hydra YAML configs
scripts/               # Entry points (run, sweep, analyze, visualize)
tests/                 # Unit + integration tests
```

## Experiments

| Experiment | Purpose | Runs |
|------------|---------|------|
| `core_factorial` | 2×2 interaction test | 60 |
| `extended_grid` | 4×4 dose-response | 240 |
| `pout_sweep` | I(α,β) surface mapping | 288 |
| `robustness_check` | Dataset/model/topology variants | ~120 |

## Critical Invariants

1. Mixing matrix W is **doubly stochastic and symmetric**
2. Gossip mixing computes `new[i] = Σ_j W[i,j] · params[j]` **exactly**
3. The probe set uses a **fixed seed (999)** across all runs
4. All 128 clients start from the **same initialisation**
5. All drift metrics computed in **float64**

## Citation

If you use this code, please cite the research paper (forthcoming).

## License

MIT
