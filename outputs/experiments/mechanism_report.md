# Two-Region Mechanism Classification Report

## Overview

Five task-model combinations were trained and evaluated for inter-region
(A -> B) interaction mechanisms. Each model uses 64 hidden units per region.
Training used early stopping (patience = 5 eval cycles at eval_every = 2 steps,
target accuracy >= 0.95). Mechanism classification was run 8 times per model
(8 batches of 32 trials each, totalling 256 trials per model).

Three proxy indicators characterise the A -> B pathway:

| Indicator | What it measures | Threshold |
|-----------|-----------------|-----------|
| **E_J** (Realized Effect) | Overall difference in B's readout between coupled and decoupled networks | 0.12 |
| **D_J** (Direct Forcing) | Instantaneous one-step change in B's output due to A, at fixed hidden state | 0.06 |
| **M_J** (Modulation) | Whether A shifts B's hidden state orthogonal to the readout, altering B's autonomous dynamics | 0.04 |

These yield four mechanism classes: (0,0) none, (1,0) pure direct forcing,
(0,1) pure modulation, (1,1) mixed.

---

## 1. Training Summary

| Experiment | Model | Task | Best Val Acc | Early-Stop Step |
|------------|-------|------|-------------|-----------------|
| binary_additive | TwoRegionAdditiveRNN | BinaryCategorizationTask | 1.0000 | 10 |
| context_gated | TwoRegionGatedRNN | ContextDependentDecisionTask | 0.9922 | 64 |
| delayed_gated | TwoRegionGatedRNN | DelayedCategorizationTask | 1.0000 | 10 |
| redundant_lowrank | TwoRegionLowRankCommRNN | RedundantInputControlTask | 1.0000 | 20 |
| state_setting_reciprocal | TwoRegionReciprocalContextRNN | StateSettingTask | 0.9941 | 72 |

All models reached >= 99.2% validation accuracy. Simple tasks (binary, delayed,
redundant) converged in 10-20 steps. Tasks requiring context-dependent routing
(context, state-setting) needed 64-72 steps.

---

## 2. Mechanism Classification Results

### 2.1 Class Assignments

| Experiment | Mechanism Class | Consensus (out of 8 batches) |
|------------|----------------|------------------------------|
| binary_additive | **(1,1)** mixed | 8/8 unanimous |
| context_gated | **(1,1)** mixed | 8/8 unanimous |
| delayed_gated | **(1,1)** mixed | 8/8 unanimous |
| redundant_lowrank | **(1,0)** pure direct forcing | 7/8 (1,0), 1/8 (1,1) |
| state_setting_reciprocal | **(1,1)** mixed | 8/8 unanimous |

### 2.2 Aggregate Class Distribution (across all 40 batches)

| Class | Count | Percentage |
|-------|-------|------------|
| (1,1) mixed | 33 | 82.5% |
| (1,0) pure direct forcing | 7 | 17.5% |
| (0,1) pure modulation | 0 | 0.0% |
| (0,0) none | 0 | 0.0% |

Every model shows a significant realized effect (E_J = 1 in all 40 batches).
Direct forcing (D_J = 1) is universal. The differentiating factor is modulation.

---

## 3. Raw Proxy Values — Distribution per Experiment

### 3.1 Realized Effect (E_J): max_abs_delta_q

| Experiment | Mean | Std | Min | Max | Threshold | Ratio |
|------------|------|-----|-----|-----|-----------|-------|
| binary_additive | 2.117 | 0.065 | 2.046 | 2.254 | 0.12 | 17.6x |
| context_gated | 1.960 | 0.111 | 1.790 | 2.161 | 0.12 | 16.3x |
| delayed_gated | 1.908 | 0.037 | 1.878 | 1.979 | 0.12 | 15.9x |
| redundant_lowrank | 1.228 | 0.024 | 1.199 | 1.275 | 0.12 | 10.2x |
| state_setting_reciprocal | 1.497 | 0.062 | 1.416 | 1.624 | 0.12 | 12.5x |

All models are 10-18x above the E_J threshold. The A -> B pathway has a large
overall effect on B's output in every case. Binary additive shows the largest
realized effect; redundant lowrank the smallest (consistent with B having its
own access to useful inputs).

### 3.2 Direct Forcing (D_J): max_abs_direct

| Experiment | Mean | Std | Min | Max | Threshold | Ratio |
|------------|------|-----|-----|-----|-----------|-------|
| binary_additive | 0.104 | 0.003 | 0.101 | 0.108 | 0.06 | 1.7x |
| context_gated | 0.093 | 0.006 | 0.083 | 0.102 | 0.06 | 1.5x |
| delayed_gated | 0.076 | 0.002 | 0.075 | 0.079 | 0.06 | 1.3x |
| redundant_lowrank | 0.079 | 0.002 | 0.077 | 0.083 | 0.06 | 1.3x |
| state_setting_reciprocal | 0.105 | 0.003 | 0.101 | 0.110 | 0.06 | 1.7x |

Direct forcing is consistently above threshold but modest (1.3-1.7x).
The standard deviations are small, indicating stable estimates across batches.
Binary additive and state-setting show the strongest direct forcing.

### 3.3 Modulation (M_J): max_abs_modulation

| Experiment | Mean | Std | Min | Max | Threshold | Ratio |
|------------|------|-----|-----|-----|-----------|-------|
| context_gated | **0.443** | 0.026 | 0.393 | 0.467 | 0.04 | **11.1x** |
| state_setting_reciprocal | **0.369** | 0.021 | 0.314 | 0.382 | 0.04 | **9.2x** |
| binary_additive | 0.093 | 0.005 | 0.085 | 0.103 | 0.04 | 2.3x |
| delayed_gated | 0.063 | 0.001 | 0.062 | 0.064 | 0.04 | 1.6x |
| redundant_lowrank | 0.035 | 0.005 | 0.028 | 0.046 | 0.04 | **0.88x** |

This is the most informative indicator. There is a clear separation into
three tiers:

- **High modulation (9-11x):** context_gated, state_setting_reciprocal.
  These tasks require region A to set a computational regime in region B
  (which context to attend to, which cue-regime to adopt). A reshapes B's
  dynamics heavily.

- **Moderate modulation (1.6-2.3x):** binary_additive, delayed_gated.
  Some regime-shaping occurs alongside direct forcing, but it is not the
  primary mechanism.

- **Below threshold (0.88x):** redundant_lowrank. Since region B has its
  own access to useful inputs, A does not need to reshape B's dynamics.
  The A -> B pathway contributes only through direct output pushing.

### 3.4 Time to Divergence (coupled vs decoupled)

| Experiment | Mean (timesteps) | Std |
|------------|-----------------|-----|
| delayed_gated | 35.1 | 0.7 |
| context_gated | 27.2 | 1.7 |
| binary_additive | 25.8 | 0.6 |
| state_setting_reciprocal | 24.7 | 2.0 |
| redundant_lowrank | 22.1 | 0.7 |

Delayed gated has the longest time-to-divergence, consistent with its
delay period where both coupled and decoupled networks are idle, postponing
the point at which A's influence on B becomes visible.

---

## 4. Mean (trial-averaged) Proxy Values

These complement the max values by showing how strong each effect is on
average across all timesteps and trials, not just at the peak.

| Experiment | mean_abs_delta_q | mean_abs_direct | mean_abs_modulation |
|------------|-----------------|-----------------|---------------------|
| binary_additive | 0.420 | 0.033 | 0.017 |
| context_gated | 0.400 | 0.026 | 0.041 |
| delayed_gated | 0.411 | 0.022 | 0.019 |
| redundant_lowrank | 0.298 | 0.025 | 0.006 |
| state_setting_reciprocal | 0.412 | 0.030 | 0.036 |

The mean modulation values reinforce the same pattern: context_gated (0.041)
and state_setting (0.036) show sustained modulation, while redundant_lowrank
(0.006) shows almost none.

---

## 5. Key Findings

1. **All five models learn significant A -> B interactions** (E_J = 1
   unanimously). The inter-region pathway is always used.

2. **Direct forcing is universal but modest.** All models push B's output
   directly through the A -> B connection, but the effect sizes are only
   1.3-1.7x above threshold.

3. **Modulation is the differentiating factor.** Tasks that require
   context-dependent routing or regime-setting (context_gated,
   state_setting_reciprocal) produce 9-11x modulation above threshold.
   Simpler tasks produce 1.6-2.3x. Redundant input tasks fall below
   threshold entirely.

4. **The redundant_lowrank model is the sole outlier**, classified as
   (1,0) in 7/8 batches. When B can solve the task on its own, A's
   influence reduces to direct output pushing without reshaping B's
   internal dynamics.

5. **Classification is highly stable.** Four of five models show 8/8
   unanimous agreement. The redundant_lowrank model shows 7/8 agreement,
   with the single (1,1) batch having max_abs_modulation = 0.046 —
   barely above the 0.04 threshold.

6. **Training converges rapidly.** With 64-unit networks, even the
   hardest task (state_setting) converges in 72 steps. Simple tasks
   converge in 10 steps.
