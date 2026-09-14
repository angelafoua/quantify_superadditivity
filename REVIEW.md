# Codebase Correctness Review

## Finding 1 — CRITICAL: Sign error in interaction term when using CKA similarity

**SEVERITY**: Critical (wrong results)
**LOCATION**: `superadditivity/analysis/interaction_analyzer.py:196-266` (`compute_sweep_interactions`)

**WHAT**: The `compute_sweep_interactions` method defaults to
`metric_col="final_cka_cross_community"` (line 200), which is a CKA
**similarity** value (higher = more aligned representations, less drift).
But the interaction formula `I = D - B - C + A` and the bootstrap test
for `I > 0` assume the metric is a **dissimilarity** (higher = more drift).

**WHY**: CKA similarity is highest for cell A (IID + Dense) and lowest for
cell D (Non-IID + Community). Computing `I = CKA(D) - CKA(B) - CKA(C) + CKA(A)`
yields a **negative** I when superadditive drift exists. The one-sided
bootstrap test for `I > 0` would report a p-value near 1.0, **failing to
detect superadditivity that is present**. The headline claim is inverted.

Worked example: if CKA values are A=0.95, B=0.80, C=0.85, D=0.60:
- With CKA (similarity): I = 0.60 − 0.80 − 0.85 + 0.95 = −0.10 (WRONG: appears non-superadditive)
- With 1−CKA (dissimilarity): I = 0.40 − 0.20 − 0.15 + 0.05 = +0.10 (CORRECT: superadditivity detected)

**FIX**: Convert similarity to dissimilarity in `compute_sweep_interactions`:
```python
# Inside the function, after reading metric values:
drift_D = 1.0 - df.loc[..., metric_col].values
```
Or add a `higher_is_more_drift: bool = False` parameter. The generic
`compute_interaction` and `bootstrap_interaction_test` are correct — the
bug is in what values `compute_sweep_interactions` feeds them.

---

## Finding 2 — CRITICAL: Data duplication when class pools overlap across communities

**SEVERITY**: Critical (wrong results for affected configurations)
**LOCATION**: `superadditivity/datasets/semantic_partitioner.py:184-210`

**WHAT**: When `n_communities > n_semantic_clusters` (e.g., CIFAR-10 with
4 communities and 2 clusters), `_build_community_class_pools()` maps
multiple communities to the same cluster via modular wrapping (line 270).
In the Dirichlet allocation loop, each community reads `cls_idx = class_indices[cls]`
— the **full** sample list for that class — and allocates all `len(cls_idx)`
samples to its clients. The `ptr` is local to each community's iteration,
so communities that share the same class pool assign **identical samples**
to different clients.

**WHY**: For CIFAR-10 with 4 communities, communities 0 and 2 both get
"Animals" classes and communities 1 and 3 both get "Vehicles" classes. Every
sample appears twice. This inflates between-community CKA similarity and
biases the interaction term toward zero.

The primary CIFAR-100 experiment (4 clusters, 4 communities, no overlap) is
**not affected**.

**FIX**: Track a shared cursor per class across communities so each sample
is assigned at most once:
```python
class_cursor = {c: 0 for c in range(self.num_classes)}

for comm_id, client_ids in self.community_assignments.items():
    for cls in community_class_pools[comm_id]:
        cursor = class_cursor[cls]
        remaining = class_indices[cls][cursor:]
        # ... allocate from remaining, advance cursor
        class_cursor[cls] = cursor + allocated_count
```

---

## Finding 3 — MODERATE: ViT weight initialization uses atypically small std

**SEVERITY**: Moderate (biased convergence)
**LOCATION**: `superadditivity/models/model_utils.py:44`, called at `scripts/run_experiment.py:229`

**WHAT**: `init_weights()` applies `nn.init.normal_(weight, std=0.01)` to
**all** `nn.Linear` modules, including the QKV projection, output projection,
and MLP layers in all 12 ViT transformer blocks. Standard ViT initialization
uses truncated normal with std=0.02 (Dosovitskiy et al.).

**WHY**: Attention logits start very small, yielding near-uniform attention
and slow early learning. This doesn't invalidate the factorial design (all
ViT cells share initialization), but if the ViT robustness experiment shows
weaker superadditivity, it may be an initialization artifact.

**FIX**: Add model-aware initialization or let each model define its own
`reset_parameters()` method.

---

## Finding 4 — MODERATE: ER–SBM density matching not validated

**SEVERITY**: Moderate (potential density confound)
**LOCATION**: `superadditivity/graphs/graph_manager.py:117-133`

**WHAT**: The ER generator uses `ref_p_in` / `ref_p_out` (defaulting to
0.25 / 0.01) to match expected degree to an SBM. These are not validated
against the SBM parameters used in the same factorial.

**WHY**: A mismatch means the Dense and Community conditions have different
expected degrees, introducing a density confound. The interaction term I
would absorb this.

**FIX**: Assert `ref_p_in == p_in` and `ref_p_out == p_out` when both
topologies are used in the same experiment, or compute the ER probability
directly from the SBM config.

---

## Finding 5 — MODERATE: BatchNorm statistics create a confound in CKA measurement

**SEVERITY**: Moderate (potential bias, not a bug)
**LOCATION**: `superadditivity/communication/gossip_mixer.py` (design),
`superadditivity/evaluation/representation_extractor.py`

**WHAT**: The gossip mixer correctly excludes BN running stats from averaging.
But CKA hooks extract activations after BN, so the measured drift captures
both parameter divergence AND BN-statistics divergence.

**WHY**: In cell D (Non-IID + Community), clients within a community share
similar data distributions, so BN stats converge within-community but diverge
between-communities. This BN-driven drift component would exist even with
perfect parameter consensus and could inflate the measured interaction term.

**FIX**: Discuss in the paper. Optionally add a BN-sync variant to disentangle.

---

## Finding 6 — MINOR: Fragile `primary_layer` selection

**SEVERITY**: Minor
**LOCATION**: `scripts/run_experiment.py:302`

**WHAT**: `primary_layer=layer_names[-2]` relies on the second-to-last
entry being the penultimate convolutional/transformer stage. Works for all
current models but would silently break for new architectures with different
`get_layer_names()` ordering.

**FIX**: Have models expose an explicit `primary_layer` property.

---

## Finding 7 — MINOR: Probe set may be too small for high-class-count datasets

**SEVERITY**: Minor
**LOCATION**: `superadditivity/datasets/dataset_loader.py:637-678`

**WHAT**: The default `probe_size=1000` yields only 2-5 samples per class for
DomainNet (345 classes) and iNaturalist (200 classes), leading to noisy CKA
estimates and higher variance in the interaction term.

**FIX**: Scale `probe_size` with `num_classes`, e.g.
`probe_size = max(1000, 5 * num_classes)`.

---

## Summary

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **Critical** | `interaction_analyzer.py` | Sign error: CKA similarity vs dissimilarity inverts the result |
| 2 | **Critical** | `semantic_partitioner.py` | Data duplication when class pools overlap |
| 3 | Moderate | `model_utils.py` | ViT Linear layers init std=0.01 vs standard 0.02 |
| 4 | Moderate | `graph_manager.py` | ER density not validated against SBM parameters |
| 5 | Moderate | Gossip mixer / extractor | BN stats excluded from gossip but captured by CKA |
| 6 | Minor | `run_experiment.py` | Fragile primary_layer index convention |
| 7 | Minor | `dataset_loader.py` | Probe set too small for high-class-count datasets |

Findings 1 and 2 are the most urgent. Finding 1 affects ALL experiments.
Finding 2 affects robustness experiments where `n_communities > n_semantic_clusters`.
