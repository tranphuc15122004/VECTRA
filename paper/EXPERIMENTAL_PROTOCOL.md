# COAST Experimental Protocol: Hypothesis-Driven Validation

**Purpose:** This document provides a complete experimental checklist to validate the four core hypotheses of COAST's structured decomposition approach for DVRPTW.

**Core Hypotheses:**
- **H1:** Coordination memory + ownership reduces inter-vehicle conflicts
- **H2:** Candidate-conditioned lookahead reduces myopia and improves isolated customer handling
- **H3:** Edge-aware scoring improves feasibility under tight constraints
- **H4:** Structured decomposition generalizes better than monolithic scoring under distribution shift

---

## PHASE 1: BASELINE IMPLEMENTATION

### 1.1 Baseline Models (Required)

**Purpose:** Establish fair comparison points that isolate different mechanisms.

| Baseline ID | Description | Components |
|-------------|-------------|------------|
| **B0-None** | No memory, no ownership, no lookahead | Cross-attention only |
| **B1-Memory** | Memory only, no ownership head | Memory updated, but not used for scoring |
| **B2-Own** | Ownership only, no memory | Ownership from vehicle state (no history) |
| **B3-Look** | Lookahead only, no coordination | Lookahead + attention, no ownership |
| **B4-Mono** | Monolithic attention | Single large attention head, same param budget |
| **B5-Linear** | Linear fusion | Replace MLP fusion with fixed weighted sum |
| **COAST** | Full model | Memory + ownership + lookahead + MLP fusion |

**Implementation Checklist:**
- [ ] Implement B0-None (only CrossEdgeFusion, no ownership bias, no lookahead)
- [ ] Implement B1-Memory (update memory but don't use in ownership)
- [ ] Implement B2-Own (ownership from current state only)
- [ ] Implement B3-Look (lookahead only)
- [ ] Implement B4-Mono (single attention head with equivalent parameters)
- [ ] Implement B5-Linear (replace score_fusion MLP with `score = w1*att + w2*own + w3*look`)
- [ ] Verify all baselines have similar parameter counts (within ±10%)
- [ ] Verify all baselines use same training protocol (optimizer, batch size, episodes)

### 1.2 Classical Heuristic Baselines

**Purpose:** Validate neural approaches against strong OR methods.

| Heuristic ID | Description | Tool/Method |
|--------------|-------------|-------------|
| **H-Insert** | Sequential insertion heuristic | OR-Tools or custom |
| **H-Greedy** | Greedy nearest neighbor | Custom implementation |
| **H-Sweep** | Sweep algorithm with time windows | Custom implementation |
| **H-LKH** | LKH3 with recourse (if applicable) | LKH3 binary |

**Implementation Checklist:**
- [ ] Implement/integrate insertion heuristic baseline
- [ ] Implement greedy nearest-neighbor baseline
- [ ] Integrate OR-Tools for comparison (if feasible)
- [ ] Test heuristic baselines on same instance set
- [ ] Report runtime for fairness (neural vs. heuristic)

### 1.3 Training Configuration

**Standardization:** All learned baselines use identical training setup.

```python
# Training Config (all models)
OPTIMIZER = 'Adam'
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
TRAINING_EPISODES = 100_000
BASELINE_TYPE = 'exponential_moving_average'  # for REINFORCE
BASELINE_DECAY = 0.999
ADVANTAGE_NORMALIZATION = True
GRADIENT_CLIP = 1.0  # for stability
SEEDS = [42, 123, 456, 789, 1024]  # 5 random seeds minimum

# Training Data Distribution
N_CUSTOMERS_TRAIN = [50, 60, 70, 80, 90, 100]  # sample uniformly
M_VEHICLES_TRAIN = [3, 4, 5]                   # sample uniformly
ARRIVAL_RATE_TRAIN = [0.1, 0.15, 0.2]          # Poisson λ per time unit
TW_SLACK_TRAIN = [10, 15, 20]                  # uniform time window slack
SPATIAL_DIST_TRAIN = 'uniform'                 # customers in [0,1]×[0,1]
```

**Reproducibility Checklist:**
- [ ] Fix all random seeds (numpy, torch, env)
- [ ] Set `torch.backends.cudnn.deterministic = True` for training
- [ ] Log all hyperparameters to config file
- [ ] Save model checkpoints every 10k episodes
- [ ] Log training curve (avg return, policy loss, baseline loss) every 100 episodes

---

## PHASE 2: IN-DISTRIBUTION EVALUATION

### 2.1 Primary Cost Metrics

**Purpose:** Establish baseline performance on standard metrics.

**Test Set:**
- 1000 instances, generated with same distribution as training
- Fixed seed for reproducibility
- Evaluate in greedy mode (no sampling)

**Metrics:**
| Metric | Formula | Unit |
|--------|---------|------|
| Avg. Total Cost | $\frac{1}{N}\sum_{i=1}^N c_i$ | time/distance |
| Std. Dev. Cost | $\sigma(c_1, \ldots, c_N)$ | time/distance |
| Feasibility Rate | $\frac{\#\text{feasible}}{\#\text{total}}$ | % |
| Avg. TW Violations | per violated customer | time units |
| Avg. Capacity Violations | per violated vehicle | demand units |

**Checklist:**
- [ ] Generate 1000 test instances with fixed seed
- [ ] Evaluate all baselines (B0-B5, COAST, H-Insert, H-Greedy)
- [ ] Report mean ± std across 5 training seeds for each model
- [ ] Compute statistical significance (paired t-test, p<0.05)
- [ ] Generate comparison table with confidence intervals
- [ ] Highlight best result in each row (bold if statistically significant)

**Expected Result (if H1-H4 hold):**
- COAST should outperform B0-B5 by 5-15% on avg. cost
- COAST should match or exceed heuristics on feasibility
- Ablations should show that removing any component degrades performance

### 2.2 Runtime and Inference Speed

**Purpose:** Ensure neural methods are practical for online deployment.

**Metrics:**
| Metric | Description | Unit |
|--------|-------------|------|
| Forward Pass Time | Per-step decision latency | ms |
| Episode Time | Full instance solve time | seconds |
| GPU Memory | Peak memory usage | GB |

**Checklist:**
- [ ] Measure forward pass time averaged over 1000 steps
- [ ] Measure full episode time on instances of varying size (n=50,100,150)
- [ ] Report GPU memory usage (if applicable)
- [ ] Compare inference time: COAST vs. B4-Mono vs. H-Insert
- [ ] Establish if COAST is real-time feasible (<50ms per decision)

**Expected Result:**
- COAST should achieve <50ms per decision for n≤100
- Full episode should solve in <5 seconds for n=100
- Trade-off analysis: cost improvement vs. latency increase

---

## PHASE 3: HYPOTHESIS H1 - COORDINATION VALIDATION

### H1 Statement
**"Coordination memory and ownership prediction reduce inter-vehicle conflicts."**

### 3.1 Conflict Metrics

**Metrics to Measure:**

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| **Service Region Overlap** | % of customers where multiple vehicles come within threshold | For each customer, count vehicles that visit within distance δ |
| **Vehicle Switches per Cluster** | Avg. # of different vehicles serving same spatial cluster | Cluster customers by location, count unique vehicles per cluster |
| **Ownership Concentration Entropy** | Shannon entropy of ownership distribution per customer | $H_j = -\sum_{i=1}^m O_{ij} \log O_{ij}$ |
| **Inter-Vehicle Distance Variance** | Variance of distances between simultaneously active vehicles | Track vehicle positions, compute variance over time |

**Ablation Comparison:**
Compare COAST vs. B0-None, B1-Memory, B2-Own

**Checklist:**
- [ ] Implement service region overlap metric (δ = 0.1 × problem scale)
- [ ] Implement clustering algorithm (k-means or DBSCAN on customer positions)
- [ ] Track which vehicles serve which clusters, compute switch count
- [ ] Log ownership probabilities $O_{ij}$ at each decision step
- [ ] Compute entropy $H_j$ for all customers, report average
- [ ] Track vehicle trajectories, compute distance variance over time
- [ ] Generate comparison table: COAST vs. B0/B1/B2 on all conflict metrics
- [ ] Run paired statistical test for significance

**Expected Result (if H1 holds):**
- COAST should have 18-24% lower service region overlap than B0-None
- COAST should have 15-20% fewer vehicle switches per cluster than B0
- Ownership entropy should be lower (more concentrated) for COAST
- B1-Memory should show partial improvement (memory helps, but ownership matters)
- B2-Own should show minimal improvement (ownership without history is weak)

### 3.2 Qualitative Visualization

**Purpose:** Show that ownership emerges as interpretable service regions.

**Visualizations to Generate:**

1. **Ownership Heatmap Over Time**
   - X-axis: episode timestep
   - Y-axis: customers
   - Color: ownership probability for a chosen vehicle
   - Shows how ownership concentrates/stabilizes over episode

2. **Spatial Ownership Map**
   - Plot customer positions as points
   - Color each customer by vehicle with highest ownership
   - Overlay vehicle trajectories
   - Shows emergent service regions

3. **Memory Evolution t-SNE**
   - Collect vehicle memory states $h_i$ over episode
   - Project to 2D using t-SNE
   - Color by vehicle ID
   - Shows how memories diverge/cluster per vehicle

**Checklist:**
- [ ] Implement logging of $O_{ij}$ at each step
- [ ] Generate ownership heatmap for 10 representative instances
- [ ] Generate spatial ownership map for same instances
- [ ] Collect memory states $h_i$, apply t-SNE, plot
- [ ] Include visualizations in paper/appendix
- [ ] Provide narrative interpretation (e.g., "Vehicle 0 claims eastern customers")

**Expected Result:**
- Clear spatial separation in ownership maps
- Ownership heatmaps show stable regions (not random)
- Memory t-SNE shows vehicle-specific clusters

---

## PHASE 4: HYPOTHESIS H2 - ANTICIPATION VALIDATION

### H2 Statement
**"Candidate-conditioned lookahead reduces myopia and improves isolated customer handling."**

### 4.1 Isolated Customer Performance

**Definition:** An "isolated customer" is one whose nearest neighbor distance exceeds 90th percentile of all pairwise distances.

**Metrics:**

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| **Isolated Customer Regret** | Extra cost incurred on isolated customers vs. optimal timing | Run oracle that serves isolated customers at best time, compare |
| **Late Service Rate on Isolated** | % of isolated customers served late (violating time window) | Track isolated customers, count violations |
| **Detour Cost for Isolated** | Avg. extra distance traveled to/from isolated customers | Measure deviation from shortest path |

**Ablation Comparison:**
Compare COAST vs. B0-None, B3-Look

**Checklist:**
- [ ] Identify isolated customers in each test instance (90th percentile distance)
- [ ] Compute regret: oracle serves isolated at optimal time, compare to model
- [ ] Track late service rate specifically on isolated customers
- [ ] Measure detour cost (distance traveled minus direct distance)
- [ ] Generate comparison table: COAST vs. B0 vs. B3 on isolated metrics
- [ ] Run significance test

**Expected Result (if H2 holds):**
- COAST should reduce isolated customer regret by 10-15% vs. B0
- Late service rate on isolated should be 8-12% lower
- Detour cost should decrease by 12-18%
- B3-Look should show similar improvement (lookahead is key)

### 4.2 Greedy Override Analysis

**Purpose:** Measure how often lookahead corrects greedy-nearest choices.

**Metrics:**

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| **Lookahead Intervention Rate** | % of decisions where lookahead changes the greedy choice | Compare $\arg\max(\alpha)$ vs. $\arg\max(\text{score})$ |
| **Intervention Benefit** | Cost difference when lookahead overrides greedy | Track instances where lookahead intervened, compare outcomes |
| **Greedy Failure Cases** | # of instances where greedy-only fails but lookahead succeeds | B0-greedy violates constraints, COAST does not |

**Checklist:**
- [ ] At each decision step, compute greedy choice $\arg\max(\alpha^{\text{att}})$
- [ ] Compare to final choice $\arg\max(\text{score})$
- [ ] Count interventions, compute rate
- [ ] Track episodes where intervention occurred, measure cost delta
- [ ] Identify cases where B0 fails feasibility but COAST succeeds
- [ ] Generate intervention statistics table
- [ ] Provide case study examples (visualize 2-3 instances)

**Expected Result:**
- Lookahead intervenes on 15-25% of decisions
- Interventions yield 3-8% cost benefit on average
- Greedy failures reduced by 20-30% with lookahead

### 4.3 Lookahead Calibration Analysis

**Purpose:** Verify lookahead scores are predictive of actual downstream cost.

**Metrics:**

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| **Lookahead Correlation** | Pearson correlation between $\ell_j$ and actual future cost | Track predicted $\ell_j$ and realized cost after choosing $j$ |
| **Calibration Error** | Mean squared error between $\ell_j$ and MC return | Compute MSE over all decisions |

**Checklist:**
- [ ] Log predicted lookahead $\ell_j$ for chosen customer at each step
- [ ] Compute actual return (future cost from that step onward)
- [ ] Compute correlation between predicted and actual
- [ ] Compute calibration error (MSE)
- [ ] Plot lookahead vs. actual cost scatter plot
- [ ] Verify correlation is positive and significant (r > 0.5)

**Expected Result:**
- Correlation r ≈ 0.5-0.7 (strong enough to guide decisions)
- Calibration error decreases during training
- Lookahead is not perfect, but informative

---

## PHASE 5: HYPOTHESIS H3 - EDGE-AWARENESS VALIDATION

### H3 Statement
**"Edge-aware scoring improves feasibility under tight constraints."**

### 5.1 Constraint Violation Analysis

**Purpose:** Test model performance under increasingly tight constraints.

**Test Regimes:**
1. **Loose TW:** time window slack = 20 time units
2. **Medium TW:** slack = 10 time units
3. **Tight TW:** slack = 5 time units
4. **Capacity Stress:** vehicle capacity 0.8× normal

**Metrics:**

| Metric | Definition | Unit |
|--------|------------|------|
| **TW Violation Rate** | % of customers served outside time window | % |
| **Capacity Violation Rate** | % of vehicles exceeding capacity | % |
| **Feasibility Drop** | Change in feasible solution rate vs. loose regime | % |

**Ablation Comparison:**
Compare COAST vs. B0-None (no edge features) vs. attention-only baseline

**Checklist:**
- [ ] Generate test sets with loose/medium/tight TW
- [ ] Generate test set with reduced capacity
- [ ] Evaluate all baselines on each regime
- [ ] Track violation rates (TW, capacity)
- [ ] Compute feasibility drop: (feasible_loose - feasible_tight) / feasible_loose
- [ ] Generate comparison table across regimes
- [ ] Test significance

**Expected Result (if H3 holds):**
- COAST maintains <5% violation rate even in tight regime
- B0-None has 15-25% violation rate in tight regime
- Feasibility drop for COAST is 10-15%, for B0 is 30-40%

### 5.2 Edge Feature Ablation

**Purpose:** Validate that edge features (distance, tw_slack, capacity_gap, etc.) contribute.

**Ablations:**
- Remove all edge features (use only node features)
- Remove time window features (wait, late, slack)
- Remove capacity features (cap_gap)
- Remove spatial features (distance)

**Checklist:**
- [ ] Implement edge-free baseline (no EdgeFeatureEncoder)
- [ ] Implement ablations removing specific edge feature groups
- [ ] Evaluate on tight-constraint test set
- [ ] Measure violation rates and cost
- [ ] Generate ablation table

**Expected Result:**
- Removing TW features increases TW violations by 20-30%
- Removing capacity features increases capacity violations by 15-25%
- Removing spatial features degrades cost by 10-15%

---

## PHASE 6: HYPOTHESIS H4 - GENERALIZATION VALIDATION

### H4 Statement
**"Structured decomposition generalizes better than monolithic scoring under distribution shift."**

### 6.1 Out-of-Distribution Test Regimes

**Training Distribution:** $n \in [50,100]$, $m \in [3,5]$, λ = 0.15, slack = 15

**OOD Test Distributions:**

| OOD ID | Shift Type | Config | Purpose |
|--------|-----------|--------|---------|
| **OOD-Scale** | More customers | $n = 150$ | Test capacity scaling |
| **OOD-Fleet** | More vehicles | $m = 8$ | Test coordination scaling |
| **OOD-Burst** | Bursty arrivals | λ varies [0.05, 0.4] | Test dynamic adaptation |
| **OOD-Tight** | Tight time windows | slack = 3-5 | Test constraint handling |
| **OOD-Cluster** | Clustered spatial | Gaussian mixture (4 clusters) | Test spatial structure change |
| **OOD-Sparse** | Sparse spatial | Customers in [0,2]×[0,2] with gaps | Test long-distance routing |

**Metrics:**

| Metric | Definition | How to Compute |
|--------|------------|----------------|
| **Cost Degradation** | (Cost_OOD - Cost_train) / Cost_train | % |
| **Feasibility Drop** | (Feasible_train - Feasible_OOD) / Feasible_train | % |
| **Generalization Score** | 1 - (Cost_deg + Feas_drop)/2 | 0-1 scale |

**Checklist:**
- [ ] Generate 500 instances for each OOD regime with fixed seed
- [ ] Evaluate all baselines (B0-B5, COAST) on all OOD regimes
- [ ] Compute cost degradation for each model on each OOD
- [ ] Compute feasibility drop for each model on each OOD
- [ ] Generate OOD comparison table (models × OOD regimes)
- [ ] Highlight which model degrades least on each OOD
- [ ] Run statistical tests for significance

**Expected Result (if H4 holds):**
- COAST cost degradation: 10-15% on average across OOD
- B4-Mono cost degradation: 25-35% on average
- COAST feasibility drop: <10%, B4-Mono: 20-30%
- Structured decomposition (memory+ownership+lookahead) is key

### 6.2 Fine-Tuning vs. Zero-Shot Generalization

**Purpose:** Test if decomposition helps with transfer learning.

**Protocol:**
1. Evaluate zero-shot (no fine-tuning)
2. Fine-tune for 10k episodes on OOD distribution
3. Evaluate after fine-tuning

**Metrics:**
- Zero-shot performance
- Fine-tuned performance
- Sample efficiency (episodes to reach 95% of final performance)

**Checklist:**
- [ ] Evaluate COAST and B4-Mono zero-shot on OOD-Scale, OOD-Burst
- [ ] Fine-tune both models for 10k episodes on each OOD
- [ ] Track learning curves during fine-tuning
- [ ] Compute sample efficiency (episodes to 95%)
- [ ] Generate learning curve plots
- [ ] Compare zero-shot gap: COAST vs. B4-Mono

**Expected Result:**
- COAST zero-shot gap: 15-20%
- B4-Mono zero-shot gap: 30-40%
- COAST fine-tunes faster (fewer episodes to 95%)

---

## PHASE 7: ABLATION STUDIES (COMPREHENSIVE)

### 7.1 Ablation Matrix

**Purpose:** Systematically test every component combination.

| Ablation ID | Memory | Ownership | Lookahead | Edge Features | Fusion |
|-------------|--------|-----------|-----------|---------------|--------|
| **A0** | ✗ | ✗ | ✗ | ✗ | - |
| **A1** | ✓ | ✗ | ✗ | ✗ | - |
| **A2** | ✗ | ✓ | ✗ | ✗ | - |
| **A3** | ✗ | ✗ | ✓ | ✗ | - |
| **A4** | ✗ | ✗ | ✗ | ✓ | - |
| **A5** | ✓ | ✓ | ✗ | ✗ | Linear |
| **A6** | ✓ | ✗ | ✓ | ✗ | Linear |
| **A7** | ✗ | ✓ | ✓ | ✗ | Linear |
| **A8** | ✓ | ✓ | ✓ | ✗ | Linear |
| **A9** | ✓ | ✓ | ✓ | ✓ | Linear |
| **COAST** | ✓ | ✓ | ✓ | ✓ | MLP |

**Checklist:**
- [ ] Implement all 11 ablation variants
- [ ] Train each variant with 3 seeds (for efficiency)
- [ ] Evaluate on in-distribution test set
- [ ] Compute cost, feasibility, conflict rate, isolated regret
- [ ] Generate ablation table with all metrics
- [ ] Identify which components contribute most
- [ ] Test interactions (e.g., does ownership help only with memory?)

**Expected Result:**
- Memory alone (A1) gives ~5% improvement
- Ownership alone (A2) gives ~3% improvement
- Lookahead alone (A3) gives ~6% improvement
- Combinations (A5-A9) show additive or super-additive effects
- MLP fusion (COAST) outperforms linear fusion (A9) by 2-4%

### 7.2 Sensitivity Analysis

**Purpose:** Test robustness to hyperparameter changes.

**Hyperparameters to Vary:**
- Memory size: [64, 128, 256]
- Lookahead hidden size: [64, 128, 256]
- Number of attention heads: [4, 8, 16]
- Fusion MLP depth: [2, 3, 4 layers]
- Exploration amplitude (tanh_xplor): [5, 10, 20]

**Checklist:**
- [ ] For each hyperparameter, train 3 variants
- [ ] Evaluate each variant on test set
- [ ] Plot performance vs. hyperparameter value
- [ ] Identify sweet spots and sensitivity
- [ ] Report recommended hyperparameter ranges

**Expected Result:**
- Memory size: performance saturates at 128-256
- Lookahead hidden: 128 is sufficient
- Attention heads: 8 is optimal (4 too few, 16 no gain)
- Fusion depth: 2-3 layers sufficient
- Exploration: 10 is robust

---

## PHASE 8: BEHAVIORAL ANALYSIS

### 8.1 Decision Attribution

**Purpose:** Understand which signal (attention, ownership, lookahead) dominates in different scenarios.

**Method:**
- At each decision step, record $\bar{\alpha}$, $\bar{\beta}$, $\bar{\ell}$ (normalized scores)
- Compute contribution weights from MLP fusion gradient or sensitivity
- Categorize decisions by dominant signal

**Metrics:**

| Metric | Definition |
|--------|------------|
| Attention-Dominated | % decisions where $|\bar{\alpha}| > |\bar{\beta}| + |\bar{\ell}|$ |
| Ownership-Dominated | % decisions where $|\bar{\beta}| > |\bar{\alpha}| + |\bar{\ell}|$ |
| Lookahead-Dominated | % decisions where $|\bar{\ell}| > |\bar{\alpha}| + |\bar{\beta}|$ |

**Checklist:**
- [ ] Log normalized scores at each decision
- [ ] Classify decisions by dominant signal
- [ ] Compute percentages
- [ ] Analyze temporal patterns (early episode vs. late episode)
- [ ] Analyze spatial patterns (clustered vs. isolated customers)
- [ ] Generate pie chart or bar chart of signal dominance

**Expected Result:**
- Attention dominates ~50% of decisions (base compatibility)
- Ownership dominates ~25% (conflict situations)
- Lookahead dominates ~25% (isolated/difficult customers)
- Early episode: ownership matters more (coordination setup)
- Late episode: lookahead matters more (optimization phase)

### 8.2 Failure Mode Analysis

**Purpose:** Identify when COAST fails and why.

**Method:**
- Select 50 worst-performing instances (highest cost vs. oracle)
- Manually inspect ownership heatmaps, trajectories, lookahead scores
- Categorize failure modes

**Failure Categories:**
1. Ownership collapse (all vehicles claim same region)
2. Lookahead miscalibration (underestimates future cost)
3. Edge feature misinterpretation (ignores tight constraints)
4. Exploration failure (stuck in local optimum)

**Checklist:**
- [ ] Identify 50 worst instances
- [ ] Generate full diagnostic output (ownership, lookahead, edge scores)
- [ ] Manually review and categorize failures
- [ ] Compute distribution of failure modes
- [ ] Provide 2-3 detailed case studies
- [ ] Propose mitigation strategies for each failure mode

**Expected Result:**
- Most common failure: lookahead miscalibration (30-40% of failures)
- Second: ownership collapse under high competition (20-30%)
- Third: edge feature misinterpretation under extreme TW (15-25%)
- Strategies: better lookahead supervision, ownership regularization, edge attention weighting

---

## PHASE 9: COMPARISON WITH STATE-OF-THE-ART

### 9.1 Literature Baselines

**Purpose:** Position COAST relative to published methods.

**Methods to Compare (if code/data available):**
- Attention Model (Kool et al. 2019)
- POMO (Kwon et al. 2020)
- RL4CO baselines (if DVRPTW variants exist)
- MARDAM (if ancestor codebase)

**Checklist:**
- [ ] Obtain code/checkpoints for each baseline (or reimplement)
- [ ] Evaluate on same test set as COAST
- [ ] Ensure fair comparison (same input format, no oracle info)
- [ ] Report cost, feasibility, runtime
- [ ] Generate comparison table with literature baselines
- [ ] Highlight COAST improvements

**Expected Result:**
- COAST outperforms Attention Model by 10-20% (dynamic adaptation)
- COAST matches or exceeds POMO on feasibility
- COAST has competitive runtime (<2× slower than lightweight baselines)

### 9.2 OR Solver Comparison

**Purpose:** Validate that learning-based approach is practical.

**Solvers:**
- OR-Tools with insertion and local search
- Gurobi/CPLEX (if model as MILP)
- LKH3 (if time-window variant exists)

**Checklist:**
- [ ] Integrate OR-Tools insertion heuristic
- [ ] Run on same test set with time limits (1s, 10s, 60s)
- [ ] Compare cost at each time limit
- [ ] Report anytime performance curves
- [ ] Analyze when neural is better (small instances, tight time?) vs. OR (large instances, loose time?)

**Expected Result:**
- COAST beats OR-Tools at <1s time limit
- OR-Tools catches up or exceeds at 10s+ time limit
- COAST is preferable for time-critical online scenarios

---

## PHASE 10: REPRODUCIBILITY AND RELEASE

### 10.1 Code and Data Release

**Checklist:**
- [ ] Clean and document all code
- [ ] Provide requirements.txt or conda environment
- [ ] Include training scripts with all hyperparameters
- [ ] Include evaluation scripts for all metrics
- [ ] Provide pretrained checkpoints for COAST and baselines
- [ ] Provide test data splits (in-distribution and OOD)
- [ ] Provide visualization scripts for ownership, lookahead, trajectories
- [ ] Write README with quickstart instructions
- [ ] Add LICENSE file
- [ ] Publish to GitHub (preferred) or paper supplementary

### 10.2 Reproducibility Checklist

**Ensures results can be replicated:**
- [ ] All random seeds documented
- [ ] Exact PyTorch/CUDA versions specified
- [ ] Training logs included
- [ ] Evaluation logs included
- [ ] Intermediate checkpoints available
- [ ] Verification script that runs mini-experiment (10 episodes, <5 min)
- [ ] Expected output documented (cost within ±2% of reported)

### 10.3 Ethical and Practical Considerations

**Checklist:**
- [ ] Discuss computational cost (GPU hours for training)
- [ ] Discuss carbon footprint (if relevant)
- [ ] Discuss limitations (when COAST fails, when to use OR instead)
- [ ] Discuss societal impact (logistics optimization → emissions, labor)
- [ ] Discuss dual-use concerns (if applicable)

---

## TIMELINE AND MILESTONES

**Recommended Execution Order:**

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Phase 1 | All baselines implemented and trained (5 seeds) |
| 3 | Phase 2 | In-distribution evaluation complete, primary metrics table |
| 4 | Phase 3 | H1 validation complete, conflict metrics + visualizations |
| 5 | Phase 4 | H2 validation complete, isolated customer analysis + case studies |
| 6 | Phase 5 | H3 validation complete, edge-awareness ablation |
| 7-8 | Phase 6 | H4 validation complete, OOD generalization results |
| 9 | Phase 7 | Full ablation study and sensitivity analysis |
| 10 | Phase 8 | Behavioral analysis, failure mode study |
| 11 | Phase 9 | Literature and OR baseline comparisons |
| 12 | Phase 10 | Code release, reproducibility verification |

**Critical Milestones:**
- **End of Week 3:** Confirm COAST significantly outperforms B0-None
- **End of Week 6:** Confirm H1 and H2 (coordination and anticipation work)
- **End of Week 8:** Confirm H4 (generalization advantage)
- **End of Week 12:** Paper-ready figures, tables, and code release

---

## SUMMARY CHECKLIST

**Before Paper Submission, Verify:**

### Primary Results
- [ ] COAST outperforms all baselines on in-distribution cost (with significance)
- [ ] COAST maintains high feasibility (>95%)
- [ ] Runtime is practical (<50ms per decision)

### Hypothesis Validation
- [ ] H1: Ownership reduces conflicts by 18-24% (p<0.05)
- [ ] H2: Lookahead reduces isolated customer regret by 10-15% (p<0.05)
- [ ] H3: Edge features reduce TW violations by 15-25% under tight constraints (p<0.05)
- [ ] H4: Decomposition retains 85-90% performance under OOD, vs. 70-80% for monolithic (p<0.05)

### Ablations
- [ ] All 11 ablation variants trained and evaluated
- [ ] Ablation table shows additive/super-additive effects
- [ ] Sensitivity analysis confirms hyperparameter robustness

### Visualizations
- [ ] Ownership heatmaps (10 instances)
- [ ] Spatial ownership maps (10 instances)
- [ ] Memory t-SNE plots
- [ ] Greedy override case studies (2-3 instances)
- [ ] Learning curves (in-dist and fine-tuning)
- [ ] OOD generalization plots

### Reproducibility
- [ ] All code and data released
- [ ] Pretrained checkpoints available
- [ ] Verification script passes on fresh environment
- [ ] README and documentation complete

### Writing
- [ ] Abstract clearly states decomposition thesis
- [ ] Introduction frames coordination-anticipation entanglement problem
- [ ] Method section emphasizes architectural principle over module details
- [ ] Results section is hypothesis-driven (H1-H4)
- [ ] Discussion addresses failure modes and limitations
- [ ] Conclusion restates key finding: decomposition > monolithic

---

**END OF EXPERIMENTAL PROTOCOL**

**Next Steps:**
1. Review this protocol with collaborators
2. Prioritize experiments based on available compute
3. Begin Phase 1 (baseline implementation)
4. Report preliminary results after Phase 2
5. Iterate on hypotheses if evidence is weak
6. Prepare paper draft aligned with experimental findings
