# **Research Paper Proposal**

**Title:** Beyond Additive Heterogeneity: Quantifying Superadditive Interaction Effects in Decentralized Federated Learning

**Submission type:** Full research paper (8 pages \+ references)

Lou Angela Foua  
Ph.D. Student  
Department of Computer Science  
University of Arkansas at Little Rock

---

## **1\. Problem Statement**

Decentralized federated learning (DFL) enables clients to train collaboratively through peer-to-peer gossip communication, without a central server. Two factors are widely known to degrade DFL performance: *data heterogeneity* (Non-IID local distributions) and *network topology* (sparse or community-structured graphs). Existing theoretical analyses treat these factors as independent contributors to convergence error — their bounds decompose into separate additive terms for data skew and topology quality, with no cross term.

**The central claim of this paper is that this additive assumption is wrong.**

When Non-IID data and community-structured topology coexist — as they do in virtually every real-world DFL deployment — they do not produce independent effects. Instead, they interact nonlinearly, generating a superadditive interaction term *I* at the level of neural representation geometry that is not predicted by, and cannot be detected through, conventional convergence metrics.

Formally, if we define Drift(·) as layer-wise CKA divergence between clients:

Drift(data \+ topology) \= Drift(data) \+ Drift(topology) \+ I \+ Drift(baseline)

We argue that I \> 0 in joint-heterogeneity settings, and that I grows as a non-trivial function of data skew severity α and community isolation strength β. Characterizing I — its magnitude, its layer-specific distribution, its trajectory over training, and its dependence on (α, β) — is the primary contribution of this paper.

---

## **2\. Motivation and Significance**

### **2.1 Why accuracy metrics are insufficient**

Standard DFL evaluation relies on test accuracy, training loss, and convergence speed. These metrics answer "does the model work?" but not "what is happening inside the model during training?" A system can achieve acceptable accuracy while internally developing large, persistent representational divergence between clients — a failure mode that is invisible to endpoint metrics but that will degrade robustness, generalization under distribution shift, and the system's ability to continue learning after deployment.

### **2.2 Why the additive assumption matters**

Every algorithm designed to improve DFL robustness under heterogeneity implicitly inherits some assumption about how the two heterogeneity sources interact. If the additive assumption is correct, one can optimize independently for data skew (e.g., FedProx-style regularization) and topology (e.g., better graph design), and the combination will be approximately as good as the sum of the parts. If the interaction term I is large, this modular strategy fails: the optimal intervention under joint heterogeneity is fundamentally different from the sum of the two optimal single-factor interventions.

Demonstrating that I \> 0 is therefore not merely an empirical curiosity — it has direct normative implications for algorithm design.

### **2.3 Why DFL specifically**

In centralized FL, a parameter server provides periodic global aggregation that partially resets representational divergence regardless of data distribution. In DFL, no such corrective mechanism exists. Each client's representation evolves under a combination of its local data gradient and the gossip signal from its topological neighborhood. When that neighborhood is itself skewed (community structure) and the local data is skewed (Non-IID), the two biases compound in a way that centralized analysis cannot capture.

---

## **3\. Research Questions**

This paper addresses three nested research questions:

**RQ1 (Existence).** Does joint Non-IID data and community-structured topology produce superadditive representation drift — i.e., is I \> 0 across training, and is it detectable layer-wise via CKA?

**RQ2 (Structure).** Where does I localize? Which layers, which training phases, and which network positions (cross-community vs. within-community client pairs) carry the largest interaction effects?

**RQ3 (Parameterization).** How does I scale as a function of data skew α (Dirichlet concentration) and community isolation β (SBM inter-community edge probability)? Is the relationship monotone, convex, or does it exhibit a threshold effect?

---

## **4\. Proposed Methodology**

### **4.1 Experimental design: 2×2 factorial baseline**

The foundational design crosses two binary factors:

|  | IID data | Non-IID data |
| ----- | ----- | ----- |
| **Dense graph** | Cell A — Baseline | Cell B — Data effect |
| **Community graph** | Cell C — Topology effect | Cell D — Joint condition |

The interaction term is computed as:

I \= Drift(D) − Drift(B) − Drift(C) \+ Drift(A)

This subtraction isolates the portion of drift in Cell D that cannot be attributed to either single factor alone. All four cells are run with the same architecture, optimizer, and communication budget. The only variables are the data distribution and graph topology.

### **4.2 Parametric sweep: modeling I(α, β)**

Beyond the binary 2×2, we propose a systematic sweep over:

* **α ∈ {0.05, 0.1, 0.3, 0.5, 1.0, ∞}** — Dirichlet concentration controlling data skew severity (lower α \= more extreme Non-IID)  
* **β ∈ {0.01, 0.05, 0.1, 0.2, 0.5}** — SBM inter-community edge probability controlling topology isolation (lower β \= stronger community structure)

For each (α, β) pair, we compute I as defined above. The output is an empirical interaction surface I(α, β) that characterizes whether the interaction is additive, superadditive, or subadditive across the parameter space.

### **4.3 Measurement: layer-wise CKA throughout training**

Centered Kernel Alignment (CKA) is used to measure pairwise representational similarity between clients at each layer and at each checkpoint during training. Key measurement choices:

* **Linear CKA** for efficiency at scale; RBF CKA as robustness check  
* **Measured at every epoch** to track trajectory, not just final state  
* **Three pairwise types:** within-community, cross-community, and random pairs — to test whether I is topology-position-dependent  
* **Layer-wise decomposition** to identify which depth carries the largest interaction effects (prior work suggests Layer 3→4 transitions are a bottleneck)

### **4.4 Architecture and dataset**

* **Architecture:** ResNet-18 or a standard CNN (e.g., 4-layer ConvNet used in federated learning benchmarks)  
* **Dataset:** CIFAR-10 and CIFAR-100 (standard DFL benchmarks); Federated EMNIST as a secondary dataset  
* **Graph model:** Stochastic Block Model (SBM) with k communities, n clients per community, inter-community probability β, intra-community probability p \>\> β  
* **DFL protocol:** gossip averaging with fixed communication rounds per epoch, no server

### **4.5 Statistical validation**

Because I is computed by subtraction of noisy estimates, statistical rigor is essential:

* All cells run with multiple random seeds (minimum 5\)  
* Confidence intervals on I at each (α, β, layer, time) point  
* Hypothesis test: H₀: I \= 0, tested against H₁: I \> 0 using bootstrap  
* Bonferroni correction across layer and time comparisons

---

## **5\. Expected Contributions**

**C1 (Empirical).** The first systematic measurement of the interaction term I between data heterogeneity and network topology in DFL, quantified via layer-wise CKA across a parametric grid.

**C2 (Methodological).** A factorial subtraction framework for isolating interaction effects in multi-factor DFL experiments, reusable for future heterogeneity studies.

**C3 (Theoretical implication).** Evidence that existing convergence bounds, which decompose heterogeneity additively, underestimate the true impact of joint heterogeneity at the representation level — motivating revised theoretical treatment.

**C4 (Practical guidance).** Identification of which (α, β) regimes produce dangerous I values, which layers require targeted intervention, and what this implies for topology–data co-design in real deployments.

---

## **6\. Positioning Against Prior Work**

| Work | Heterogeneity studied | Metric used | What this paper adds |
| ----- | ----- | ----- | ----- |
| Koloskova et al. (ICML 2020\) | Topology \+ data (additive bounds) | Convergence rate | Measures interaction I at representation level |
| Lian et al. (2017) | Topology only | Convergence | Extends to joint heterogeneity |
| Bellet et al. — D-Cliques (2022) | Both, but compensatory design | Accuracy | Studies nonlinear interaction, not compensation |
| FedCKA, FLAYER, FedIN | Data heterogeneity (centralized) | CKA similarity | Applies CKA framework to DFL with topology |
| This paper | Both — joint and factorial | CKA \+ interaction term I | First to isolate and parameterize I |

The key positioning claim: prior work studies either one factor at a time, or uses accuracy/convergence metrics that cannot detect the interaction. This paper does neither.

---

## **7\. Theoretical Framing**

We do not claim to disprove existing convergence theory. Instead, we establish a gap: existing bounds decompose as:

Error ≤ A \+ B(data) \+ C(topology)

and do not contain a cross term B × C or any interaction. We use this additive structure as a null hypothesis at the representation level, then test whether observed CKA drift violates it. The interaction term I is therefore an empirical quantity that existing theory does not predict and cannot bound.

A companion theoretical analysis — either a lower bound argument showing that additive bounds must miss I, or a new bound that explicitly includes an interaction term — is a stretch goal for the full paper, but is not required for the empirical contribution to stand.

---

## **8\. Limitations and Scope**

* Results are empirical; no closed-form expression for I(α, β) is claimed  
* CKA measures representational similarity, not causal drift mechanisms  
* Experiments use classification tasks; regression and generative settings are out of scope  
* The SBM graph model is a controlled proxy for real-world community structure; findings may not transfer directly to arbitrary real-world graphs

---

## **9\. Proposed Paper Structure**

1. **Introduction** — The additive assumption and why it fails; elevator pitch for I; contributions  
2. **Background** — DFL, CKA, convergence theory, Non-IID data  
3. **Related Work** — Positioned as above (Section 6\)  
4. **The Interaction Term I** — Formal definition, factorial derivation, null hypothesis  
5. **Experimental Setup** — Architecture, datasets, graph model, CKA protocol  
6. **Results: Existence of I** — 2×2 factorial, I \> 0 across layers and time (RQ1)  
7. **Results: Structure of I** — Layer-wise and position-wise analysis (RQ2)  
8. **Results: I(α, β) surface** — Parametric sweep results (RQ3)  
9. **Discussion** — Implications for algorithm design, theory revision, and deployment  
10. **Conclusion**

---

## **10\. Why This Paper Belongs at a Top Venue**

Top ML venues reward papers that identify a structural gap in existing understanding, not just improve a metric. This paper argues that the entire field of DFL has been operating under an implicit additive assumption that is both theoretically unjustified and empirically false in realistic settings. The framing is crisp, the claim is falsifiable, the methodology is replicable, and the implications touch both theory (convergence bounds need cross terms) and practice (algorithm designers should co-optimize topology and data simultaneously, not independently).

The use of CKA as a diagnostic — rather than a performance metric — is an underexplored angle that opens a new measurement paradigm for DFL research.

