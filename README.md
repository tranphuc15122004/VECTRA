# VECTRA — Vehicle-Edge-Coordination Transformer with Routed Attention

> **V**ehicle-**E**dge-**C**oordination **T**ransformer with **R**outed **A**ttention  
> A Scale-Balanced Graph-MoE MARL framework for Dynamic Vehicle Routing with Time Windows (DVRPTW)

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Core Assumptions](#2-core-assumptions)
3. [Mathematical Analysis](#3-mathematical-analysis)
   - 3.1 [Computational Complexity](#31-computational-complexity)
   - 3.2 [Solution Quality Preservation](#32-solution-quality-preservation)
   - 3.3 [Cross-Scale Generalisation](#33-cross-scale-generalisation)
4. [Architecture Overview](#4-architecture-overview)
5. [Component Details](#5-component-details)
   - 5.1 [Graph Encoder](#51-graph-encoder--hybrid-knn-transformer)
   - 5.2 [Latent Bottleneck](#52-latent-bottleneck)
   - 5.3 [Fleet Encoder & Coordination Memory](#53-fleet-encoder--coordination-memory)
   - 5.4 [Edge Feature Encoder](#54-edge-feature-encoder)
   - 5.5 [Cross-Edge Fusion](#55-cross-edge-fusion)
   - 5.6 [Ownership Head](#56-ownership-head)
   - 5.7 [Lookahead Head](#57-lookahead-head)
   - 5.8 [Score Fusion & MoE](#58-score-fusion--mixture-of-experts)
   - 5.9 [Candidate Shortlist (SBG)](#59-candidate-shortlist-sbg)
   - 5.10 [Adaptive Depth](#510-adaptive-depth)
6. [Training Objective](#6-training-objective)
7. [Inference Flow](#7-inference-flow)
8. [Configuration Reference](#8-configuration-reference)
9. [Quickstart](#9-quickstart)

---

## 1. Problem Formulation

DVRPTW is modelled as a **Sequential Multi-agent Markov Decision Process (sMAMDP)**.  
At each decision step exactly **one** vehicle acts; the others wait or travel.

| Symbol | Meaning |
|---|---|
| $\mathcal V = \{v_1,\dots,v_M\}$ | Fleet of $M$ vehicles |
| $\mathcal C_t = \{c_1,\dots,c_{L_t}\}$ | Known customer set at time $t$ (grows dynamically) |
| $s_t$ | Global state: positions, loads, current times, TW, dynamic arrivals |
| $a_t \in \mathcal C_t \cup \{\text{depot}\}$ | Action of the acting vehicle |
| $r_t$ | Reward: negative travel distance, lateness penalty |
| $\pi_\theta$ | Stochastic policy parameterised by $\theta$ |

**Objective:**

$$\max_\theta \; \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} r_t\right]$$

---

## 2. Core Assumptions

The following assumptions underpin the theoretical guarantees developed in §3.

**A1 — Feature boundedness.**  
All input features (coordinates, demands, time windows, vehicle states) are normalised to a bounded domain $\mathcal X \subset \mathbb R^d$ with $\|\mathbf x\|_\infty \le 1$.

**A2 — Lipschitz scoring.**  
The compatibility scorer $q_\theta : \mathcal X \to \mathbb R$ is $L_q$-Lipschitz with respect to the $\ell_2$ norm of its inputs.

**A3 — High-recall shortlist.**  
The cheap pre-filter (SBG) retains the optimal action $a^*$ in its shortlist of size $K$ with probability at least $1 - \delta$, where $\delta$ is small.

**A4 — Bounded approximation error.**  
The scorer's approximation error relative to the full dense scorer satisfies $\|\hat q - q\|_\infty \le \varepsilon$ uniformly on the shortlist.

**A5 — Scale-invariant features.**  
Coordinate and distance features are normalised per instance, making the feature distribution weakly dependent on the number of customers $L_c$.

---

## 3. Mathematical Analysis

### 3.1 Computational Complexity

Let $D$ = number of encoder layers, $d$ = model dimension, $L_c$ = number of customers,
$M$ = latent bottleneck tokens, $K$ = shortlist size, $k$ = KNN neighbourhood size.

| Module | Complexity |
|---|---|
| KNN-masked Graph Encoder (per layer) | $O(L_c k d)$ |
| Latent Bottleneck cross-attention | $O(L_c M d)$ |
| Fleet Encoder (per vehicle, per layer) | $O(L_v L_c d)$ |
| SBG candidate selection | $O(L_c)$ |
| Deep scoring on shortlist | $O(K d)$ |

**Total per decision step with adaptive depth $\bar D \le D$:**

$$\boxed{T_{\text{VECTRA}} = O\!\Big(\bar D\bigl(L_c k d + L_c M d\bigr) + K d\Big)}$$

Compared to the dense baseline:

$$T_{\text{dense}} = O\!\Big(D \cdot L_c^2 d\Big)$$

When $k, M, K \ll L_c$ and $\bar D \le D$, **VECTRA scales near-linearly** in $L_c$, whereas the dense baseline is quadratic.

**Corollary.** For fixed $k=15$, $M=32$, $K=16$, $\bar D = D/2$:

$$\frac{T_{\text{VECTRA}}}{T_{\text{dense}}} \approx \frac{(k+M)}{L_c} \cdot \frac{\bar D}{D} \xrightarrow{L_c\to\infty} 0$$

### 3.2 Solution Quality Preservation

**Proposition.** Under assumptions A1–A4, let $\gamma = q_{(1)} - q_{(2)}$ be the logit margin between the best and second-best candidate. If $\varepsilon < \gamma/2$, the argmax action is preserved:

$$\hat a^* = \arg\max_{j \in \mathcal S_K} \hat q_j = a^* \quad \text{whenever } a^* \in \mathcal S_K.$$

**Proof.** For any $j \neq a^*$ in the shortlist:

$$\hat q_{a^*} \ge q_{a^*} - \varepsilon > q_j + \gamma - \varepsilon > \hat q_j + \gamma - 2\varepsilon > \hat q_j$$

since $\varepsilon < \gamma/2$. $\square$

**Rollout quality bound.** Over a trajectory of $T$ steps, by union bound over shortlist misses and scoring errors:

$$\mathbb{E}[\Delta J] \le T\bigl(\delta \cdot \Delta_{\max} + \kappa \varepsilon\bigr)$$

Reducing $\delta$ (better shortlist filter) and $\varepsilon$ (better distillation) drives $\Delta J \to 0$.

### 3.3 Cross-Scale Generalisation

**Proposition.** Under A2 and A5, for instances of sizes $n$ and $m$ with feature distributions $\mu_n, \mu_m$:

$$\|\pi_n - \pi_m\|_{\mathrm{TV}} \le L_\pi \, W_1(\mu_n, \mu_m) + \xi$$

where $W_1$ is the Wasserstein-1 distance and $\xi > 0$ is reduced by the **multi-scale consistency loss** (see §6).

**Implication.** Curriculum training $n=20 \to 50 \to 100$ is theoretically stable: the policy TV-distance between consecutive scales is bounded by the distributional shift $W_1(\mu_n, \mu_m)$ plus residual training error.

---

## 4. Architecture Overview

```
INPUT: customers C_t, vehicles V, dynamic arrivals
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              STEP 1: Customer Encoding                  │
│   Depot + Customer Embeddings (Linear)                  │
│   ──► KNN-masked Graph Encoder (RBF edge bias)          │
│   ──► [optional Latent Bottleneck, M tokens]            │
│   ──► Dropout + Linear project → cust_repr  N×Lc×D     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│    STEP 2: Vehicle Context + Coordination Memory        │
│   Fleet Encoder (cross-att on cust_repr)               │
│   CoordinationMemory (per-vehicle GRU-like state)       │
│   ──► veh_repr  N×1×D                                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 3: Edge Feature Construction               │
│   dist, travel_time, arrival, wait, late,               │
│   slack, feasibility, cap_gap                           │
│   ──► EdgeFeatureEncoder → edge_emb  N×1×Lc×D          │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 4: SBG Candidate Shortlist                 │
│   cheap_score = -dist - λ·late + μ·slack + ω·owner     │
│   Select top-K feasible candidates → cand_idx           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         STEP 5: Deep Scoring on Shortlist               │
│   CrossEdgeFusion  → att_score                          │
│   OwnershipHead    → owner_bias                         │
│   LookaheadHead    → look_score                         │
│            │  z-normalise each stream                   │
│            ▼  Score Fusion MLP  (3→64→1)                │
│   ┌─── compat_base ───┐                                 │
│   │ MoE: uncertainty- │                                 │
│   │ gated blending    │                                 │
│   └───────────────────┘                                 │
│   Tanh exploration cap  → compat  N×1×K                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   log_softmax → scatter to full Lc
         │
   sample / greedy → cust_idx
         │
   Memory Update → CoordinationMemory
         │
   dyna.step → next state
```

---

## 5. Component Details

### 5.1 Graph Encoder — Hybrid KNN-Transformer

**Purpose:** Produce context-rich customer representations that capture both local spatial structure (KNN masking) and long-range dependencies (Transformer self-attention).

**RBF Edge Bias.** Raw Euclidean distances are embedded into $B=16$ Radial Basis Functions:

$$\phi_b(d) = \exp\!\left(-\frac{(d - c_b)^2}{2 w^2}\right), \quad c_b = \frac{b}{B-1}, \; b=0,\dots,B-1$$

These are projected to per-head attention biases via a small MLP, geometrically grounding attention in physical space.

**KNN mask.** For each node, only its $k$-nearest neighbours (in normalised distance space) are attended to; all other entries are set to $-\infty$ before softmax. Complexity per layer drops from $O(L_c^2)$ to $O(L_c k)$.

**Adaptive Depth.** Active layer count $D_{\text{use}}$ is reduced for easy instances:

$$D_{\text{use}} = D_{\min} + \left\lfloor (D - D_{\min}) \cdot \text{clamp}\!\left(\frac{r_{\text{easy}} - r_{\text{visible}}}{r_{\text{easy}}},\, 0,\, 1\right) \right\rfloor$$

where $r_{\text{visible}}$ is the fraction of unmasked customers.

**Layer equations:**

$$e_{ij}^{(h)} = \frac{\mathbf q_i^{(h)} \cdot \mathbf k_j^{(h)}}{\sqrt{d_h}} + \text{EdgeMLP}(\phi(d_{ij}))^{(h)}, \quad j \in \mathcal N_k(i)$$

$$\mathbf H'' = \text{LayerNorm}\!\bigl(\mathbf H' + \text{FFN}(\mathbf H')\bigr)$$

### 5.2 Latent Bottleneck

**Purpose:** Compress $L_c$ customer embeddings into $M \ll L_c$ latent tokens to reduce downstream attention cost.

**Mechanism:**

1. Select $M$ indices evenly spaced along the customer list (depot fixed at index 0).
2. Run the Graph Encoder only on these $M$ nodes.
3. For each full customer assign it the token embedding of its nearest (1-NN) latent node:

$$\tilde{\mathbf h}_j = \mathbf h_{\text{enc}}^{(M)}\!\left[\arg\min_{m \in [M]} \|x_j - x_m\|_2\right]$$

**Activation condition:** Only applied when $L_c \ge L_{\min}$ and $M < L_c$.

### 5.3 Fleet Encoder & Coordination Memory

**Fleet Encoder.** Each vehicle's state is projected and enriched by cross-attention over customer representations:

$$\mathbf h_v^{\text{veh}} = \text{FleetEncoder}(\mathbf s_v,\; \mathbf H^{\text{cust}})$$

**Coordination Memory.** A lightweight recurrent module tracks each vehicle's decision history for implicit multi-agent coordination:

$$\mathbf m_v^{t+1} = \tanh\!\left(W_x \cdot [\mathbf h_v^{\text{veh}};\, \mathbf h_{c^*}^{\text{cust}};\, \mathbf e_{v,c^*}] + W_h \mathbf m_v^t\right)$$

where $c^*$ is the customer just assigned to vehicle $v$. The memory persists across steps within an episode.

**Key property:** Update is $O(1)$ per step via `scatter` — only the acting vehicle's slot changes.

### 5.4 Edge Feature Encoder

**Purpose:** Encode relational features between the acting vehicle $v$ and each candidate customer $c$ into $\mathbf e_{v,c} \in \mathbb R^D$.

**Raw features** (8-dimensional):

| Feature | Formula |
|---|---|
| Euclidean distance | $d_{vc} = \|x_v - x_c\|_2$ |
| Travel time | $\tau_{vc} = d_{vc} / \text{speed}$ |
| Arrival time | $a_{vc} = t_v + \tau_{vc}$ |
| Wait time | $w_{vc} = \max(0,\; e_c - a_{vc})$ |
| Lateness | $\ell_{vc} = \max(0,\; a_{vc} - l_c)$ |
| TW slack | $\sigma_{vc} = \max(0,\; l_c - a_{vc})$ |
| Feasibility indicator | $f_{vc} = \mathbf 1[\ell_{vc} = 0]$ |
| Capacity gap | $g_{vc} = q_v - \text{dem}_c$ |

$$\mathbf e_{v,c} = \text{LayerNorm}\!\left(W_2 \cdot \text{ReLU}(W_1 \mathbf x_{vc})\right)$$

### 5.5 Cross-Edge Fusion

**Purpose:** Compute a compatibility score between vehicle and each customer, with edge embeddings as additive attention bias across all heads.

$$s_{vc}^{(h)} = \frac{\mathbf q^{(h)} \cdot \mathbf k_c^{(h)}}{\sqrt{d_h}} + (W_{\text{edge}} \mathbf e_{v,c})^{(h)}$$

$$\text{att\_score}_{vc} = \frac{1}{H} \sum_h s_{vc}^{(h)}$$

### 5.6 Ownership Head

**Purpose:** Estimate each vehicle's soft "ownership" over each customer to enable implicit coordination without message passing.

$$O = \text{softmax}\!\left(\frac{W_v \mathbf M \cdot (W_c \mathbf H^{\text{cust}})^\top}{\sqrt{D}}\right) \in \mathbb R^{L_v \times L_c}$$

The acting vehicle's row is used as a log-prior:

$$\text{owner\_bias}_{v^*,c} = \log O_{v^*,c}$$

This softly discourages multiple vehicles from targeting the same customer.

### 5.7 Lookahead Head

**Purpose:** Estimate the future value of assigning customer $c$ — a one-step critic that sees downstream feasibility pressure.

$$\text{look}_{v,c} = W_2 \cdot \text{ReLU}\!\left(W_1 \cdot [\mathbf h_v^{\text{veh}};\, \mathbf h_c^{\text{cust}};\, \mathbf e_{v,c}]\right) \in \mathbb R$$

The lookahead score penalises assignments likely to cause future lateness or infeasibility.

### 5.8 Score Fusion & Mixture of Experts

**Z-normalisation** across the candidate dimension ensures all three streams contribute equally:

$$\tilde s = \frac{s - \bar s}{\sigma_s + 10^{-8}}$$

**Fusion MLP:**

$$\text{compat\_base}_{v,c} = \text{MLP}_{3\to 64 \to 1}\!\left([\tilde s_{\text{att}};\, \tilde s_{\text{own}};\, \tilde s_{\text{look}}]\right)$$

**Mixture of Experts.** Two specialist experts blend with the base score:

| Expert | Formula | Regime |
|---|---|---|
| Local (myopic) | $0.7\tilde s_{\text{att}} + 0.9\tilde s_{\text{own}} - 0.2\tilde s_{\text{look}}$ | Low feasibility ($r_f < 0.35$) |
| Future-aware | $0.4\tilde s_{\text{att}} + 0.3\tilde s_{\text{own}} - 0.9\tilde s_{\text{look}}$ | High lookahead variance ($\sigma_{\text{look}} > 0.9$) |

**Uncertainty-gated blending:**

$$H = -\sum_c p_c \log p_c, \qquad \Delta = q_{(1)} - q_{(2)}$$

$$\alpha = \Bigl(\alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \cdot \tfrac{H / H_{\max}}{1 + \Delta}\Bigr) \cdot \mathbf 1\!\left[H > H_{\text{floor}} \;\wedge\; \Delta < \Delta_{\text{ceil}}\right]$$

$$\text{compat} = \text{compat\_base} + \alpha_{\text{local}} \cdot \delta_{\text{local}} + \alpha_{\text{fut}} \cdot \delta_{\text{fut}}$$

**Tanh exploration cap:**

$$\tilde{\text{compat}} = C \cdot \tanh\!\left(\text{compat} / C\right), \quad C = 10$$

### 5.9 Candidate Shortlist (SBG)

**Purpose:** Reduce $L_c$ candidates to $K \ll L_c$ before expensive deep scoring using only $O(L_c)$ scalar operations.

**Cheap score:**

$$s_c^{\text{cheap}} = -d_{vc} - \lambda_{\text{late}} \cdot \ell_{vc} + \mu_{\text{slack}} \cdot \sigma_{vc} + \omega_{\text{own}} \cdot O_{v^*,c}$$

**Adaptive $K$:**

| Feasibility ratio $r_f$ | Adjustment |
|---|---|
| $r_f > 0.6$ | $K \leftarrow \lfloor 1.5K \rfloor$ |
| $r_f < 0.3$ | $K \leftarrow \lfloor 0.75K \rfloor$ |
| otherwise | $K$ unchanged |

$K$ is clamped to $[K_{\min}, K_{\max}]$.

**Quality guarantee.** Under A3, retaining $a^*$ with probability $\ge 1-\delta$ combined with §3.2 gives expected rollout loss $\le T \delta \Delta_{\max}$.

### 5.10 Adaptive Depth

Both Graph Encoder and Fleet Encoder conditionally skip later Transformer layers for instances where a shallow pass already produces low-entropy, high-margin decisions. This is governed by the same `_resolve_layer_count` heuristic:

$$D_{\text{use}} = D_{\min} + \left\lfloor (D - D_{\min}) \cdot \text{hardness}\right\rfloor$$

where $\text{hardness} \in [0,1]$ is derived from the visible/feasible customer ratio relative to the `easy_ratio` threshold.

---

## 6. Training Objective

$$\mathcal L = \underbrace{\mathcal L_{\text{RL}}}_{\text{REINFORCE}} + \lambda_{\text{cons}} \underbrace{\mathcal L_{\text{scale}}}_{\text{multi-scale consistency}} + \lambda_{\text{dist}} \underbrace{\mathcal L_{\text{distill}}}_{\text{scorer distillation}} + \lambda_{\text{lb}} \underbrace{\mathcal L_{\text{moe-balance}}}_{\text{load balance}}$$

**REINFORCE loss** with baseline $b(\cdot)$:

$$\mathcal L_{\text{RL}} = -\mathbb{E}_{\pi_\theta}\!\left[\sum_t \bigl(R_t - b(\mathbf s_t)\bigr) \log \pi_\theta(a_t \mid \mathbf s_t)\right]$$

Supported baselines: `none`, `nearnb` (nearest-neighbour), `rollout`, `critic`.

**Multi-scale consistency** (reduces TV distance across scales):

$$\mathcal L_{\text{scale}} = \mathbb{E}_{n \ne m}\!\left[\text{KL}\!\left(\pi_\theta(\cdot \mid \mathbf s_n) \;\|\; \pi_\theta(\cdot \mid \mathbf s_m)\right)\right]$$

**Distillation loss** (shortlist logits ≈ dense logits):

$$\mathcal L_{\text{distill}} = \mathbb{E}\!\left[\text{KL}\!\left(\text{softmax}(\mathbf q_{\text{dense}}) \;\|\; \text{softmax}(\mathbf q_{\text{SBG}})\right)\right]$$

**MoE load-balance loss:**

$$\mathcal L_{\text{moe-balance}} = L_v \cdot \sum_e f_e \cdot P_e$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is the mean routing probability.

**Training infrastructure:** Mixed-precision AMP, gradient clipping, `LambdaLR` decay, `GradScaler`.

---

## 7. Inference Flow

```mermaid
flowchart TD
    A([Episode Start]) --> B[Reset CoordinationMemory\n_veh_memory = zeros N×Lv×H]
    B --> C{dyna.done?}
    C -- Yes --> Z([Return actions / logps / rewards])
    C -- No --> D{new_customers?}
    D -- Yes --> E[_encode_customers\nDepot + Cust Embedding\nKNN Graph Encoder\nLatent Bottleneck optional\nDropout + Linear project]
    D -- No --> F
    E --> F[_repr_vehicle\nFleet Encoder cross-att\n→ veh_repr  N×1×D]
    F --> G[_build_edge_features\ndist travel_time arrival\nwait late slack feasible cap_gap]
    G --> H[EdgeFeatureEncoder\n8-dim → D MLP + LayerNorm]
    H --> I[OwnershipHead\nveh_memory × cust_repr → O\nGather acting vehicle log-prior]
    I --> J{sbg_enable?}
    J -- Yes --> K[_select_sbg_candidates\ncheap_score topK\nAdaptive-K adjustment]
    K --> L[Gather cust_repr / edge_emb\n/ owner_bias on shortlist of size K]
    J -- No --> M[Use full Lc candidates]
    L --> N
    M --> N[LookaheadHead\nveh+cust+edge → MLP → look_score  N×1×K]
    N --> O[CrossEdgeFusion\nmulti-head att + edge bias\n→ att_score  N×1×K]
    O --> P[Z-normalise 3 streams\nScore Fusion MLP 3→64→1\n→ compat_base]
    P --> Q{sbg_moe_enable\nAND high entropy?}
    Q -- Yes --> R[Compute H and margin Δ\nGate strength α\nBlend expert_local / expert_future]
    R --> S
    Q -- No --> S[Tanh cap  C·tanh compat/C]
    S --> T[_get_logp\nlog_softmax mask infeasible]
    T --> U{sbg scatter?}
    U -- Yes --> V[Scatter logp_local → full Lc dim]
    U -- No --> W
    V --> W[Greedy argmax or Multinomial sample\n→ cust_idx]
    W --> X[_update_memory\nCoordination GRU update for v_cur]
    X --> Y[dyna.step cust_idx\nAdvance environment state]
    Y --> C
```

---

## 8. Configuration Reference

Key hyperparameters passed via `argparse` in [script/train_mardam.py](script/train_mardam.py):

| Parameter | Default | Description |
|---|---|---|
| `--model_size` | 128 | Hidden dimension $D$ |
| `--layer_count` | 3 | Transformer layer count |
| `--head_count` | 8 | Attention heads |
| `--ff_size` | 512 | FFN width |
| `--cust_k` | 15 | KNN neighbourhood $k$ |
| `--edge_feat_size` | 8 | Raw edge feature dim |
| `--memory_size` | None (=D) | Coordination memory width |
| `--lookahead_hidden` | 128 | Lookahead MLP width |
| `--dropout` | 0.1 | Dropout rate |
| `--tanh_xplor` | 10 | Tanh cap $C$ |
| **SBG** | | |
| `--sbg_enable` | False | Enable shortlisting |
| `--sbg_cand_k` | 0 | Base shortlist size $K$ |
| `--sbg_adaptive_k` | False | Adaptive $K$ by feasibility |
| `--sbg_k_min` | 8 | Minimum $K$ |
| `--sbg_k_max` | None | Maximum $K$ |
| `--sbg_late_penalty` | 2.0 | $\lambda_{\text{late}}$ |
| `--sbg_slack_weight` | 0.5 | $\mu_{\text{slack}}$ |
| `--sbg_owner_weight` | 0.5 | $\omega_{\text{own}}$ |
| **MoE** | | |
| `--sbg_moe_enable` | False | Enable MoE blending |
| `--sbg_moe_strength` | 0.15 | Maximum $\alpha$ |
| `--sbg_moe_uncertainty` | True | Gate by entropy/margin |
| `--sbg_moe_min_strength` | 0.01 | Minimum $\alpha$ |
| `--sbg_moe_entropy_floor` | 0.35 | $H_{\text{floor}}$ |
| `--sbg_moe_margin_ceil` | 1.5 | $\Delta_{\text{ceil}}$ |
| **Adaptive Depth** | | |
| `--adaptive_depth` | False | Enable adaptive layers |
| `--adaptive_min_layers` | 1 | $D_{\min}$ |
| `--adaptive_easy_ratio` | 0.6 | $r_{\text{easy}}$ |
| **Latent Bottleneck** | | |
| `--latent_bottleneck` | False | Enable compression |
| `--latent_tokens` | 32 | $M$ tokens |
| `--latent_min_nodes` | 64 | Activation threshold $L_{\min}$ |

### SBG-Train-Ready Preset

Pass `--sbg_train_ready` to activate a tuned preset for large-scale training (SBG + adaptive depth + bottleneck + MoE, all with conservative defaults):

```bash
python script/train_mardam.py --sbg_train_ready \
  --customers_count 100 --vehicles_count 5 --problem_type dvrptw
```

---

## 9. Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python script/train_mardam.py \
  --problem_type dvrptw \
  --customers_count 50 \
  --vehicles_count 3 \
  --epoch_count 300 \
  --batch_size 512 \
  --sbg_enable \
  --sbg_cand_k 16 \
  --sbg_adaptive_k \
  --sbg_moe_enable \
  --adaptive_depth \
  --latent_bottleneck \
  --baseline_type rollout
```

### Resume from Checkpoint

```bash
python script/train_mardam.py ... --resume_state ./mardam_output/chkpt_ep300.pyth
```

### Evaluation

```bash
python script/eval_learned_dyn.py \
  --model_path ./mardam_output/chkpt_ep300.pyth \
  --problem_type dvrptw \
  --customers_count 100 \
  --vehicles_count 5
```

---

## References

- Kool et al., *Attention, Learn to Solve Routing Problems!*, ICLR 2019  
- Gutierrez-Bucheli et al., *MARDAM*, 2022  
- Fedus et al., *Switch Transformers*, JMLR 2022  
- Velickovic et al., *Graph Attention Networks*, ICLR 2018  

---

*Theoretical bounds are stated under the assumptions in §2 and provide design intuition rather than PAC-style worst-case guarantees.*