# Two Region Mechanisms

A PyTorch research codebase for studying emergent inter-region interaction mechanisms in trained **two-region continuous-time rate RNNs**.

The central question is:

> After training on a task, what kind of interaction strategy emerges from region A to region B — and how does that strategy depend on what the task demands?

## Approach

The codebase provides three primitive proxy signals measured at every timestep:

- **D(t)** — Direct forcing: the instantaneous one-step change in B's readout due to the A->B source, holding state fixed.
- **M(t)** — Modulation: whether A->B shifts B's hidden state orthogonal to the readout gradient, altering B's future autonomous dynamics.
- **E(t)** — Realized effect: the total divergence between coupled and decoupled B readout trajectories.

These are shared measurement primitives. But the *interpretation* is task-specific: each task gets its own analysis contract with task-aligned outcome metrics, epoch structure, trial splits, and strategy labels.

## Cross-task results

Five task-model combinations have been profiled:

| Experiment | Task | Model | Strategy | Key Evidence |
|---|---|---|---|---|
| binary_additive | BinaryCategorization | Additive | **direct_dominant** | D unique=0.81, D onset first (t=16), M adds nothing unique |
| delayed_gated | DelayedCategorization | Gated | **M_first_delay_supported** | M onset first (t=24), delay M unique=0.78, stim->late_delay M unique=0.86 |
| context_gated | ContextDecision | Gated | **routing_modulatory** | Progressive routing selectivity: stim 0.24 -> delay 0.52 -> response 0.95 |
| state_setting | StateSetting | ReciprocalContext | **off_axis_mode_control** | Beta(orthogonal) separation 6.6x alpha(readout), triplet spread 0.32 vs 0.02 decoupled |
| redundant_lowrank | RedundantInput | LowRankComm | **direct_backup** | D unique=0.77, M unique=0.03, ~65% relative decoupling cost at all difficulties |

## What is included

- Four model families: `TwoRegionAdditiveRNN`, `TwoRegionGatedRNN`, `TwoRegionLowRankCommRNN`, `TwoRegionReciprocalContextRNN`
- Six task families: `BinaryCategorizationTask`, `DelayedCategorizationTask`, `ContextDependentDecisionTask`, `CueGatingTask`, `StateSettingTask`, `RedundantInputControlTask`
- Config-driven training with early stopping
- Shared D/M/E measurement engine (`src/analysis/mechanism_classification.py`)
- Task-specific profiling engine (`src/analysis/task_profiles.py`)
- Threshold sweep and robustness analysis
- Per-trial distribution diagnostics (histograms, time traces)
- Task-specific analyses:
  - Routing regression for context task
  - Paired-triplet evaluation and hidden-state geometry for state-setting task
  - Difficulty-conditioned decoupling cost for redundant task

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

### Smoke tests

```bash
python scripts/train_one.py --config configs/smoke_tests/binary_additive.yaml
python scripts/evaluate_model.py --run-dir outputs/smoke/binary_additive
python scripts/classify_mechanisms.py --run-dir outputs/smoke/binary_additive
```

### Train all five experiments

```bash
python scripts/train_one.py --config configs/experiments/binary_additive.yaml
python scripts/train_one.py --config configs/experiments/delayed_gated.yaml
python scripts/train_one.py --config configs/experiments/context_gated.yaml
python scripts/train_one.py --config configs/experiments/state_setting_reciprocal.yaml
python scripts/train_one.py --config configs/experiments/redundant_lowrank.yaml
```

Training uses early stopping (patience-based on validation accuracy). Most experiments converge in 10-72 steps.

### Run the full analysis pipeline

```bash
# Per-trial distributions and threshold sweeps
python scripts/classify_mechanisms.py --run-dir outputs/experiments/binary_additive
python scripts/threshold_sweep.py --run-dirs outputs/experiments/*

# Task-specific profiling
python scripts/run_task_analysis.py --all

# Final profiles and cross-task summary
python scripts/generate_profiles.py
```

### Task-specific deep analyses

```bash
# Delayed-gated: temporal staging, within-coherence splits, lag analysis
python scripts/delayed_gated_analysis.py

# State-setting: rule-aligned transfer, hidden-state geometry, paired triplets
python scripts/state_setting_analysis.py
```

## Project layout

```text
two_region_mechanisms/
  README.md
  requirements.txt
  configs/
    tasks/              # task parameter defaults
    models/             # model parameter defaults
    experiments/        # full experiment configs (5 tasks)
    smoke_tests/        # lightweight test configs
    analysis/           # per-task analysis contracts (YAML)
  src/
    tasks/              # task definitions (6 families)
    models/             # two-region RNN models (4 families)
    train/              # trainer with early stopping
    analysis/           # D/M/E engine, task profiles, classification
    utils/              # config, IO, plotting
  scripts/
    train_one.py                # train a single model
    evaluate_model.py           # evaluate and save metrics
    classify_mechanisms.py      # D/M/E classification + distributions
    threshold_sweep.py          # threshold robustness analysis
    run_task_analysis.py        # task-specific profiling (all 5)
    generate_profiles.py        # final profile JSONs + summary table
    delayed_gated_analysis.py   # deep delayed-task analysis
    state_setting_analysis.py   # deep state-setting analysis
    run_ablations.py            # ablation experiments
    make_figures.py             # aggregate figure generation
    sweep.py                    # hyperparameter sweeps
  tests/
  outputs/
```

## Task families

### BinaryCategorizationTask
Noisy evidence during stimulus, report left/right during response. No delay. Tests direct sensory-to-decision transmission.

### DelayedCategorizationTask
Like binary, but with a delay between stimulus and response. Tests whether A->B supports maintenance through delay-period modulation or early direct write-in.

### ContextDependentDecisionTask
Two evidence streams with a context cue selecting which is relevant. Tests whether A->B implements progressive routing — amplifying relevant evidence while suppressing irrelevant leakage.

### CueGatingTask
A cue selects one of four mappings: report, ignore, invert, or hold a signal.

### StateSettingTask
A cue selects one of three computational regimes (integrate, memory, transient) for later inputs. Tests whether A->B creates cue-dependent hidden-state regimes off the readout axis.

### RedundantInputControlTask
Both regions receive the same evidence. Tests whether A->B acts as a proportional direct backup or becomes unused when B has redundant input.

## Mechanism proxies

These implement **practical proxies**, not theorem-level identification.

### D(t) — Direct forcing
At each timestep, compare B's output with and without A->B input while holding hidden state fixed. Measures the instantaneous output push.

### M(t) — Modulation
Project the A->B-induced displacement in B's hidden state onto the readout gradient; subtract that aligned component. Then test whether the remaining orthogonal displacement changes B's next-step output. Measures whether A reshapes B's dynamics rather than just pushing its output.

### E(t) — Realized effect
Compare the full B readout trajectory in coupled vs decoupled networks. Measures the total consequence of A->B over time.

## Analysis pipeline

The analysis proceeds in layers:

1. **Shared D/M/E measurement** — same engine for every task
2. **Per-trial distributions** — histograms, time traces, AUC (scripts/classify_mechanisms.py)
3. **Threshold robustness** — sweep D/M thresholds, compare max vs max+mean rules (scripts/threshold_sweep.py)
4. **Task-specific profiling** — each task uses its own outcome metric, epoch emphasis, and trial splits (scripts/run_task_analysis.py)
5. **Final profiles** — one strategy label per task, justified by task-aligned headline metrics (scripts/generate_profiles.py)

Key design principle: the same D/M/E primitives are used everywhere, but each task defines what counts as the relevant outcome (routing selectivity for context, off-axis mode separation for state-setting, final |q| for binary/delayed/redundant).

## Output files

Each run directory stores:

- `config.yaml`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/train_history.csv`, `metrics/val_history.csv`
- `metrics/eval_metrics.json`
- `metrics/mechanism_summary.json`
- `metrics/{name}_final_profile.json`
- `metrics/raw/per_trial_metrics.csv`, `metrics/raw/traces.npz`
- `metrics/sweep/sweep_max.npz`, `metrics/sweep/sweep_maxmean.npz`
- `figures/*.png`
- Task-specific exports (`.npz`, `.csv`)

Cross-task outputs in `outputs/experiments/`:
- `final_cross_task_summary.csv`
- `threshold_robustness_summary.csv`
- `mechanism_report.md`

## License

This code is provided as a research starter template without warranty.
