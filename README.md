# MARDAM (C-MAP) Model Overview

This repository implements a Coordinated Memory + Adaptive Planning (C‑MAP) model for dynamic vehicle routing under sMMDP (sequential multi‑agent decision). A single shared policy acts for one vehicle at a time, while each vehicle keeps its own memory state to coordinate fleet behavior.

## High‑level idea
The model combines:
- **Spatial graph encoding** of customers (and depot) with edge‑biased attention.
- **Fleet encoding** of vehicle states with k‑NN masked attention.
- **Edge‑aware vehicle–customer fusion** to score feasible next visits.
- **Coordination memory** to avoid conflicts between vehicles.
- **Lookahead head** to penalize short‑sighted choices.

At each decision step, the acting vehicle scores all customers using a weighted mixture of attention score, ownership bias, and lookahead penalty, then samples (or greedily picks) the next customer.

## Component‑level description

### 1) Customer GraphEncoder
**Concept**: Encode spatial relationships between customers using a graph transformer with RBF distance bias.

**Inputs**:
- `customers`: $N \times L_c \times D_c$ node features (includes depot at index 0).
- `mask` (optional): $N \times L_c$ customer mask (hidden/padded/dynamic).
- `coords` (optional): $N \times L_c \times 2$ raw positions.

**Outputs**:
- `cust_repr`: $N \times L_c \times D$ encoded customer representations.

**Processing**:
1. Embed depot and customers into `D`.
2. Compute pairwise distances (or use provided cost matrix).
3. Apply RBF distance bias inside multi‑head attention.
4. Layer‑norm + feed‑forward sublayers for each encoder layer.
5. Masked nodes are zeroed.

---

### 2) FleetEncoder (Vehicle‑Vehicle)
**Concept**: Encode fleet interactions using attention over nearby vehicles (k‑NN mask).

**Inputs**:
- `vehicles`: $N \times L_v \times D_v$ vehicle state.
- `fleet_edges`: $N \times L_v \times L_v \times E_f$ pairwise features (distance, time gap, capacity gap).

**Outputs**:
- `fleet_repr`: $N \times L_v \times D$ vehicle representations.

**Processing**:
1. Project `vehicles` into model size `D`.
2. Build k‑NN mask over vehicle distances.
3. Apply mixed‑score multi‑head attention per layer.
4. Layer‑norm + feed‑forward sublayers.

---

### 3) EdgeFeatureEncoder (Vehicle–Customer)
**Concept**: Encode rich edge features between acting vehicle and each customer.

**Inputs**:
- `edge_feat`: $N \times 1 \times L_c \times E$ where $E=8$ by default.

**Outputs**:
- `edge_emb`: $N \times 1 \times L_c \times D$.

**Processing**:
MLP (Linear → ReLU → Linear → LayerNorm).

**Edge features used**:
distance, travel time, arrival time, wait time, late time, slack, feasibility, capacity gap.

---

### 4) CrossEdgeFusion (Vehicle–Customer Attention)
**Concept**: Compute edge‑biased attention scores between acting vehicle and customers.

**Inputs**:
- `veh_repr`: $N \times 1 \times D$ (acting vehicle).
- `cust_repr`: $N \times L_c \times D$.
- `edge_emb`: $N \times 1 \times L_c \times D$.

**Outputs**:
- `att_score`: $N \times 1 \times L_c$.

**Processing**:
Multi‑head dot‑product between vehicle query and customer keys, plus a learned edge bias.

---

### 5) CoordinationMemory
**Concept**: Maintain per‑vehicle hidden state to coordinate fleet decisions.

**Inputs**:
- `memory`: $N \times L_v \times H$.
- `veh_idx`: $N \times 1$ current vehicle index.
- `veh_repr`, `cust_repr`, `edge_emb` of selected customer.

**Outputs**:
- Updated `memory` with only the acting vehicle state changed.

**Processing**:
Concatenate acting vehicle, selected customer, and edge embedding → MLP update → scatter to memory slot.

---

### 6) OwnershipHead
**Concept**: Predict which vehicle should serve each customer (anti‑conflict bias).

**Inputs**:
- `veh_memory`: $N \times L_v \times H$.
- `cust_repr`: $N \times L_c \times D$.

**Outputs**:
- `owner_logits`: $N \times L_v \times L_c$.

**Processing**:
Project memory and customer features → dot product → softmax over vehicles → log‑bias for acting vehicle.

---

### 7) LookaheadHead
**Concept**: Estimate future cost‑to‑go to penalize greedy choices.

**Inputs**:
- `veh_repr`: $N \times 1 \times D$.
- `cust_repr`: $N \times L_c \times D$.
- `edge_emb`: $N \times 1 \times L_c \times D$.

**Outputs**:
- `lookahead`: $N \times 1 \times L_c$.

**Processing**:
Concatenate (vehicle, customer, edge) → MLP → scalar per customer.

---

## Scoring and decision rule
Final compatibility score for customer $j$:
$$
	ext{compat}_j = w_1 \cdot \text{att\_score}_j + w_2 \cdot \log(\text{owner\_prob}_j) - w_3 \cdot \text{lookahead}_j
$$
Then apply feasibility mask and select action by sampling or greedy argmax.

## Execution flow (per step)
1. **Customer encode**: build `cust_repr` once and update when new customers appear.
2. **Fleet encode**: compute `fleet_repr` for all vehicles.
3. **Select acting vehicle**: use `cur_veh_idx` from environment.
4. **Edge features**: build vehicle–customer features and encode them.
5. **Ownership bias**: compute per‑vehicle ownership logits and extract for acting vehicle.
6. **Lookahead**: estimate cost‑to‑go for each customer.
7. **Score & select**: mix scores, mask infeasible, sample/greedy.
8. **Update memory**: update only the acting vehicle’s memory state.


## Flowchart (true execution per `step()`)

> Note: `cust_repr` is computed **outside** `step()` and only refreshed when `dyna.new_customers == True`:
> `self._encode_customers(dyna.nodes, dyna.cust_mask)` → `self.cust_repr`.

```text
Outside step() (only when new customers appear)
┌───────────────────────────────┐
│ dyna.nodes + dyna.cust_mask    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ _encode_customers()            │
│   └─ cust_encoder (GraphEnc)   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ self.cust_repr  (cached)       │
└───────────────────────────────┘


Inside step(dyna): one decision for the currently acting vehicle

   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Inputs                                                                   │
   │  • dyna.vehicles, dyna.nodes                                             │
   │  • dyna.cur_veh_idx, dyna.cur_veh_mask                                   │
   │  • self.cust_repr (cached), self._veh_memory (cached)                    │
   └───────────────┬───────────────────────────────┬─────────────────────────┘
                   │                               │
                   │                               │
                   ▼                               ▼
┌───────────────────────────────┐      ┌──────────────────────────────────────┐
│ Fleet context for acting veh  │      │ Raw edge features (constraints)       │
│ _build_fleet_edges()          │      │ _build_edge_features()                │
│ fleet_encoder (FleetEnc)      │      │ dist, travel_time, arrival,           │
│  → fleet_repr                 │      │ wait, late, slack, feasible, cap_gap  │
│ gather(cur_veh_idx)           │      └───────────────┬──────────────────────┘
│  → veh_repr                   │                      │
└───────────────┬───────────────┘                      ▼
                │                         ┌──────────────────────────────────────┐
                │                         │ edge_encoder (EdgeFeatureEncoder)    │
                │                         │  → edge_emb                          │
                │                         └───────────────┬──────────────────────┘
                │                                         │
                └───────────────┬─────────────────────────┘
                                │
                                ▼
                   ┌──────────────────────────────────────┐
                   │ cross_fusion (CrossEdgeFusion)        │
                   │ (veh_repr, cust_repr, edge_emb)       │
                   │  → att_score                          │
                   └───────────────┬──────────────────────┘
                                   │
                                   ▼
      ┌─────────────────────────────────────────┐
      │ Auxiliary heads                          │
      │  OwnershipHead(veh_memory, cust_repr)    │
      │    → owner_logits → softmax → owner_bias │
      │  LookaheadHead(veh_repr, cust_repr,      │
      │              edge_emb) → lookahead       │
      └───────────────┬─────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Score + mask + select                                                       │
│ compat = w1*att_score + w2*owner_bias - w3*lookahead                         │
│ logp = masked_log_softmax(compat, cur_veh_mask)                              │
│ cust_idx = argmax(logp) or sample(exp(logp))                                 │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Memory update (only acting vehicle slot)                                     │
│ _update_memory(cur_veh_idx, cust_idx, veh_repr, edge_emb [+ cust_sel])       │
│   → self._veh_memory (updated)                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Output: cust_idx, logp(cust_idx)
```

## Inputs/Outputs summary
- **Inputs**: customer node features, vehicle states, dynamic masks, time windows, and capacity constraints (from environment).
- **Outputs**: action (next customer per step), log probabilities, and rewards for RL training.

## Notes
- The model is **shared across vehicles** (sMMDP), but **memory is per‑vehicle**.
- Edge features incorporate **time‑window feasibility** and **capacity gap** for better dynamic decision‑making.
- `tanh_xplor` controls exploration amplitude in the final logits.

