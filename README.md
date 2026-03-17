# Two Region Mechanisms

A PyTorch research codebase for studying emergent inter-region interaction mechanisms in trained **two-region continuous-time rate RNNs**.

The central question is:

> After training on a task, what kind of interaction strategy emerges from region A to region B — and how does that strategy depend on what the task demands?

## Cross-task results

Five task-model combinations have been profiled:

| Experiment | Task | Strategy | Key Evidence |
|---|---|---|---|
| binary_additive | Binary Categorization | **direct_dominant** | D unique=0.81, D onset first (t=16), M adds nothing unique |
| delayed_gated | Delayed Categorization | **M_first_delay_supported** | M onset first (t=24), delay M unique=0.78, stim->late_delay M unique=0.86 |
| context_gated | Context Decision | **routing_modulatory** | Progressive routing selectivity: stim 0.24 -> delay 0.52 -> response 0.95 |
| state_setting | State Setting | **off_axis_mode_control** | Beta(orthogonal) separation 6.6x alpha(readout), triplet spread 0.32 vs 0.02 decoupled |
| redundant_lowrank | Redundant Input | **direct_backup** | D unique=0.77, M unique=0.03, ~65% relative decoupling cost at all difficulties |

---

## Reproduce from scratch

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train all five models

```bash
python scripts/train_one.py --config configs/experiments/binary_additive.yaml
python scripts/train_one.py --config configs/experiments/delayed_gated.yaml
python scripts/train_one.py --config configs/experiments/context_gated.yaml
python scripts/train_one.py --config configs/experiments/state_setting_reciprocal.yaml
python scripts/train_one.py --config configs/experiments/redundant_lowrank.yaml
```

Training uses early stopping (patience=5, eval every 2 steps). Most experiments converge in 10–72 steps. Each produces `training_curves.png` — check that loss dropped and accuracy plateaued.

### 3. Evaluate and classify

```bash
for d in binary_additive delayed_gated context_gated state_setting_reciprocal redundant_lowrank; do
  python scripts/evaluate_model.py --run-dir outputs/experiments/$d
  python scripts/classify_mechanisms.py --run-dir outputs/experiments/$d
done
```

This generates per-run:
- `coupled_vs_decoupled.png` — does decoupling change B's readout? (sanity check)
- `pca_hidden_states.png` — basic hidden-state geometry
- `traces_E_D_M.png` — D, M, E proxy traces with individual trial lines
- `hist_D_M_trial.png` — trial-level distributions of max D, max M, mean M

### 4. Threshold robustness

```bash
python scripts/threshold_sweep.py --run-dirs outputs/experiments/*
```

Generates `sweep_comparison.png` per run (max vs max+mean classification across 400 threshold pairs) and `threshold_robustness_summary.csv` across all runs.

### 5. Task-specific profiling

```bash
python scripts/run_task_analysis.py --all
```

This is where each task gets its own analysis. Generates per-run:
- `{name}.png` — canonical 4-panel traces (q, D, M, E) with epoch shading
- `{name}_hilo.png` — high vs low final |q| split (binary, delayed, redundant)
- `{name}_ctx.png` — by-context traces (context task only)
- `{name}_mode.png` — by-mode traces (state-setting only)
- `{name}_within_*_split_d.png` and `_m.png` — within-group D and M traces

### 6. Final profiles and summary

```bash
python scripts/generate_profiles.py
```

Produces `{name}_final_profile.json` per run and `final_cross_task_summary.csv` across all runs. This is the main results table.

### 7. Slide-ready figures

```bash
python scripts/make_slide_figures.py
```

Produces the presentation figures (see figure guide below).

### 8. Deep task-specific analyses (optional)

```bash
python scripts/delayed_gated_analysis.py    # temporal staging, within-coherence, lag
python scripts/delayed_gated_figures.py      # canonical delayed figures

python scripts/state_setting_analysis.py     # rule-aligned transfer, hidden-state geometry, triplets
```

### 9. Systematic task x architecture study

This repo now also includes a parallel study layer for the factorial task x architecture comparison. It does **not** overwrite the legacy `outputs/experiments/` results.

```bash
python scripts/generate_task_arch_grid.py
python scripts/run_task_arch_grid.py --skip-existing
python scripts/aggregate_task_arch_results.py
python scripts/make_task_arch_summary_figures.py
python scripts/prune_study_run_figures.py
```

Study outputs live under:

- `outputs/studies/task_arch_grid/runs/` — one run per `task__architecture__seed`
- `outputs/studies/task_arch_grid/summaries/` — aggregate CSV tables
- `outputs/studies/task_arch_grid/figures/` — study-level comparison figures

By default, each study run now keeps a **minimal** figure set:

- `traces.png` — the main per-run `E/D/M` trajectory figure

The most useful study-level figures are:

- `emd_summary_grid.png` — raw `E/D/M` trajectory summaries by task x architecture
- `emd_summary_grid_peak_normalized.png` — peak-normalized `E/D/M` summaries for comparing temporal shape
- `mechanism_map.png` — `D_J` vs `M_J`
- `onset_plot.png` — onset comparisons
- `task_arch_profile_grid.png` — compact strategy-label grid

---

## Interpret from existing results

If you have the outputs already (from the zip or a previous run), here is what to look at and in what order.

### Start here: cross-task summary

Open `outputs/experiments/final_cross_task_summary.csv`. This has one row per experiment with the strategy label and headline metrics.

### Per-experiment: what to check and where

For each experiment, figures are in `outputs/experiments/{name}/figures/`. The figures below are listed in the order you should read them.

#### Binary additive — "Is this mainly direct transmission?"

**Task:** Region A receives noisy evidence during a stimulus epoch. Region B receives a go cue during the response epoch. The network reports left/right. No delay — this is the simplest task.

**What the figures show:**

| Order | Figure | What to check |
|---|---|---|
| 1 | `training_curves.png` | Did training converge? Loss should drop, accuracy should plateau at ~1.0 |
| 2 | `binary_additive.png` | 4-panel traces showing |q|, |D|, |M|, |delta_q| over time with epoch shading (gray=fixation, blue=stimulus, green=response). Look for: D rises during stimulus, q follows shortly after |
| 3 | `slide_unique_contribution.png` | Left: bar plot of "unique contribution" — how much D predicts response q after controlling for M and difficulty (0.81 = strong), and vice versa (-0.55 = M adds nothing once D is known). Right: onset ordering showing D turns on first |
| 4 | `binary_additive_hilo.png` | Trials split by median final |q|: red = high output trials, blue = low. Shows D is larger on high-output trials |
| 5 | `traces_E_D_M.png` | D, M, E traces with thin lines for individual trials and thick line for trial mean. Shows trial-to-trial variability |
| 6 | `hist_D_M_trial.png` | Three histograms: trial-max |D|, trial-max |M|, trial-mean |M| with threshold lines. Shows what fraction of trials exceed classification thresholds |
| 7 | `sweep_comparison.png` | Heatmap of mechanism class across 400 threshold pairs (D threshold on y-axis, M threshold on x-axis). Color = class label. Black X = current operating thresholds. Shows how sensitive the classification is to threshold choice |

**Conclusion:** D-first, direct-dominant. M is collinear with D and adds no independent information.

#### Delayed gated — "Does delay-period modulation matter?"

**Task:** Same as binary, but with a delay epoch between stimulus and response where no input arrives. The network must hold the decision across the delay. Epochs: fixation, stimulus, delay, response.

**Key analysis concepts:**
- *Onset ordering:* which signal (D, M, or q) turns on first? Measured with baseline subtraction and a 3-step persistence criterion
- *Unique contribution:* partial correlation of D (or M) with later q, after controlling for the other signal and difficulty. Positive = independently predictive. Near zero = redundant with the other signal
- *Strict temporal precedence:* predictors are measured only from epochs that end before the outcome epoch begins, so the correlation is not tautological

| Order | Figure | What to check |
|---|---|---|
| 1 | `training_curves.png` | Convergence |
| 2 | `delayed_gated.png` | 4-panel traces with epoch shading (gray=fix, blue=stim, orange=delay, green=resp). Look for: M ramps during stimulus, D follows later, q emerges in delay |
| 3 | `slide_unique_contribution.png` | Left: onset ordering — M first (t=24), then D (t=29), then q (t=34). Middle: stimulus->response unique contribution — M=0.80, D=-0.63 (M carries the independent early signal). Right: delay->response — same pattern |
| 4 | `canonical_delayed_gated.png` | High vs low final |q| split with epoch shading. Red=high, blue=low. The groups separate during the delay epoch, visible in all four panels. This is the main evidence that delay-period D and M track later response |
| 5 | `canonical_within_coherence.png` | Same hi/lo split done separately for each coherence level (0.2, 0.4, 0.6, 0.8). The split persists within each coherence — the effect is not just stimulus strength |
| 6 | `traces_E_D_M.png` | Individual trial spread |
| 7 | `sweep_comparison.png` | Threshold robustness — this task is borderline in the old threshold-based classification, which is why the deeper temporal analysis matters more |

**Conclusion:** M-first, delay-supported. Stimulus M uniquely predicts later q (0.86 for stim->late_delay), even after controlling for D and difficulty.

#### Context gated — "Is this routing or just magnitude?"

**Task:** Two noisy evidence streams arrive simultaneously during stimulus. A context cue (during cue epoch) tells the network which stream is behaviorally relevant. The network must report the sign of the relevant stream, ignoring the irrelevant one. Epochs: cue, stimulus, delay, response.

**Key analysis concepts:**
- *Routing selectivity:* at each timestep, regress B's readout on relevant coherence and irrelevant coherence. The "relevant slope" measures sensitivity to the correct stream; the "irrelevant slope" (leakage) measures contamination from the wrong stream
- *Normalized selectivity:* (relevant - |irrelevant|) / (|relevant| + |irrelevant|). Ranges from 0 (no routing) to 1 (perfect routing)
- *Context selectivity:* how much do D or M traces differ between context=0 and context=1? High context selectivity in M means M carries context-dependent information

**Important:** for this task, the right outcome metric is routing quality, not final |q| magnitude. The slide figures focus on relevant vs irrelevant slopes, not D/M unique contributions.

| Order | Figure | What to check |
|---|---|---|
| 1 | `training_curves.png` | Convergence |
| 2 | `slide_routing_timecourse.png` | **Main figure.** Top panel: relevant slope (red) grows over time, irrelevant slope (blue) rises during stimulus then falls. Bottom panel: normalized selectivity rises from ~0.07 in early stimulus to ~0.95 in late response. This shows routing sharpens progressively |
| 3 | `slide_routing_epoch.png` | Bar chart per epoch: relevant slope (red) vs |irrelevant| slope (blue). During stimulus they're comparable; by response, relevant is 40x larger |
| 4 | `slide_routing_coupled_decoupled.png` | Coupled routing selectivity = 2.0, decoupled = 0.0. Without A->B, B cannot route at all |
| 5 | `context_gated.png` | 4-panel traces — overall shape |
| 6 | `context_gated_ctx.png` | Traces split by context (ctx=0 vs ctx=1). M shows larger context differences than D |
| 7 | `sweep_comparison.png` | Threshold robustness |

**Conclusion:** Progressive routing. During stimulus both streams influence B comparably, but across stimulus->delay->response the relevant stream is amplified and irrelevant leakage is suppressed. A->B is necessary for any routing.

#### State setting — "Are the modes implemented on or off the readout axis?"

**Task:** A cue selects one of three computational modes — *integrate* (accumulate evidence over time), *memory* (remember an early pulse, ignore later noise), or *transient* (respond to recent/late evidence). The same signal channel arrives during the signal epoch, but the correct answer depends on which mode was cued. Epochs: cue, signal, delay, response.

**Key analysis concepts:**
- *Paired triplets:* generate trial triplets where all three modes receive the exact same input signal but different cue. Any difference in B's output must come from the cue, not the input
- *Rule-aligned transfer:* for each mode, regress q(t) on the mode's rule-relevant variable (cumulative sum for integrate, early pulse for memory, late signal for transient). The slope tells you how sensitive q is to the correct statistic at each timepoint
- *Alpha vs beta decomposition:* project B's hidden state onto the readout weight vector (alpha = readout-aligned component) and the orthogonal complement (beta = off-axis component). Mode separation in beta means the modes live in B's hidden state in a direction the readout can't directly see — this is the signature of regime setting rather than direct output manipulation
- *Mode spread:* standard deviation of per-mode response |q|. Large coupled spread + near-zero decoupled spread = A->B is necessary for mode differentiation

| Order | Figure | What to check |
|---|---|---|
| 1 | `training_curves.png` | Convergence |
| 2 | `triplet_traces.png` | **Key figure.** Three panels (integrate/memory/transient), each showing sign-aligned q over time. Red = coupled, blue = decoupled. Same input for all three — coupled produces distinct trajectories per mode, decoupled is flat near zero |
| 3 | `rule_aligned_transfer.png` | Three panels showing the slope of q on each mode's rule-relevant variable over time. Each mode develops its own sensitivity curve. Decoupled (blue) is zero everywhere |
| 4 | `readout_vs_orthogonal.png` | Top: alpha (readout-aligned hidden state) by mode — modest separation. Bottom: beta (orthogonal hidden state) — much larger separation, especially in delay/response. Solid = coupled, dashed = decoupled |
| 5 | `dme_geometry_bridge.png` | Top: mode separation over time in D, M, alpha, and beta — beta (orange) dominates and leads. Bottom: within-triplet separation — coupled beta grows steeply, decoupled is flat. This bridges the hidden-state result back to the D/M framework |
| 6 | `slide_mode_summary.png` | Three compact bar plots. Left: alpha vs beta mode separation by epoch — beta >> alpha in delay and response. Middle: response mode spread — 0.19 coupled vs 0.02 decoupled. Right: within-triplet spread — 0.32 coupled vs 0.02 decoupled |
| 7 | `state_setting_reciprocal.png` | 4-panel traces — overall shape |
| 8 | `state_setting_reciprocal_mode.png` | Traces split by mode (integrate/memory/transient) |
| 9 | `sweep_comparison.png` | Threshold robustness |

**Conclusion:** Off-axis mode control. A->B creates cue-dependent hidden states in B mostly orthogonal to the readout, which later drive different outputs. The mode separation emerges first in beta (off-axis), before it becomes visible in alpha (readout) or in the D/M proxies.

#### Redundant lowrank — "Does A->B matter when B has the same input?"

**Task:** Both region A and region B receive the exact same noisy evidence signal. B also receives the go cue. Since B already has the evidence, A->B is not strictly necessary — but does the network use it anyway? Epochs: fixation, stimulus, response.

**Key analysis concepts:**
- *Decoupling cost:* difference in response |q| between coupled and decoupled networks. Shown per difficulty level to test whether A->B matters more when the task is hard
- *Relative cost:* decoupling cost / coupled |q|. If constant across difficulty, A->B provides proportional support. If rising with difficulty, A->B is a difficulty-dependent backup
- *Unique contribution:* same as binary — does D or M independently predict response q after controlling for the other and difficulty?

| Order | Figure | What to check |
|---|---|---|
| 1 | `training_curves.png` | Convergence |
| 2 | `redundant_lowrank.png` | 4-panel traces: D rises first during stimulus, M follows weakly |
| 3 | `slide_cost_and_unique.png` | **Main figure.** Left: grouped bars showing coupled (red) vs decoupled (blue) response |q| at each coherence level, with relative cost percentage above each pair (~65% everywhere — proportional use, not difficulty-gated). Right: D unique=0.77 (strong), M unique=0.03 (nothing) |
| 4 | `redundant_lowrank_hilo.png` | High vs low final |q| split |
| 5 | `traces_E_D_M.png` | Individual trial spread — M barely visible above baseline |
| 6 | `sweep_comparison.png` | Threshold robustness |

**Conclusion:** Direct backup. A->B consistently provides ~65% of B's output through direct forcing at all difficulty levels. M contributes nothing unique. This is proportional parallel support, not a contingency mechanism.

---

## Approach

Three proxy signals measured at every timestep:

- **D(t)** — Direct forcing: instantaneous one-step change in B's readout due to A->B, holding state fixed.
- **M(t)** — Modulation: whether A->B shifts B's hidden state orthogonal to the readout gradient, altering B's future autonomous dynamics.
- **E(t)** — Realized effect: total divergence between coupled and decoupled B readout trajectories.

These are shared measurement primitives. The *interpretation* is task-specific: each task gets its own analysis contract with task-aligned outcome metrics, epoch structure, trial splits, and strategy labels.

Key design principle: the same D/M/E engine is used everywhere, but each task defines what counts as the relevant outcome:
- **Binary / Delayed / Redundant**: final |q| prediction with unique D vs M contributions
- **Context**: routing selectivity (relevant vs irrelevant coherence slopes)
- **State-setting**: off-axis hidden-state separation and rule-aligned transfer curves

## Project layout

```text
two_region_mechanisms/
  README.md
  requirements.txt
  configs/
    experiments/        # 5 experiment configs
    smoke_tests/        # lightweight test configs
    analysis/           # per-task analysis contracts
    tasks/, models/     # parameter defaults
  src/
    tasks/              # 6 task families
    models/             # 4 model families
    train/              # trainer with early stopping
    analysis/           # D/M/E engine, task profiles, classification
    utils/              # config, IO, plotting
  scripts/
    train_one.py                # train a single model
    evaluate_model.py           # basic evaluation
    classify_mechanisms.py      # D/M/E + trial distributions
    threshold_sweep.py          # robustness analysis
    run_task_analysis.py        # task-specific profiling (all 5)
    generate_profiles.py        # final profiles + summary table
    make_slide_figures.py       # presentation figures
    generate_task_arch_grid.py  # build study configs
    run_task_arch_grid.py       # run task x architecture study
    aggregate_task_arch_results.py
    make_task_arch_summary_figures.py
    prune_study_run_figures.py  # keep study per-run figures minimal
    delayed_gated_analysis.py   # deep delayed-task analysis
    delayed_gated_figures.py    # canonical delayed figures
    state_setting_analysis.py   # deep state-setting analysis
  tests/
  outputs/
```

## Task families

| Task | Epochs | What it tests | A sees | B sees |
|---|---|---|---|---|
| BinaryCategorization | fix, stim, resp | Direct sensory-to-decision | Evidence | Go cue |
| DelayedCategorization | fix, stim, delay, resp | Maintenance through delay | Evidence | Go cue |
| ContextDependentDecision | cue, stim, delay, resp | Routing by context | Evidence + context | Context + go cue |
| StateSettingTask | cue, signal, delay, resp | Regime setting | Signal + mode cue | Mode cue + go cue |
| RedundantInputControl | fix, stim, resp | Use when redundant | Evidence | Evidence + go cue |
| CueGatingTask | cue, stim, delay, resp | Rule-gated mapping | Evidence + rule cue | Rule cue + go cue |

## Output files

Each run directory (`outputs/experiments/{name}/`) stores:

```text
config.yaml
checkpoints/best.pt, last.pt
metrics/
  train_history.csv, val_history.csv
  eval_metrics.json
  mechanism_summary.json
  {name}_final_profile.json        # final strategy label + headline metrics
  raw/per_trial_metrics.csv        # per-trial D/M features
  raw/traces.npz                   # full per-trial D/M/E traces
  sweep/sweep_max.npz, sweep_maxmean.npz
figures/
  training_curves.png              # step 2: training convergence
  coupled_vs_decoupled.png         # step 3: decoupling sanity check
  pca_hidden_states.png            # step 3: hidden-state overview
  traces_E_D_M.png                 # step 3: D/M/E with individual trials
  hist_D_M_trial.png               # step 3: trial-level distributions
  sweep_comparison.png             # step 4: threshold robustness
  {name}.png                       # step 5: canonical 4-panel traces
  {name}_hilo.png                  # step 5: high/low split
  {name}_within_*_split_*.png      # step 5: within-group D and M
  slide_*.png                      # step 7: presentation figures
  (task-specific deep analysis figures)
```

Cross-task outputs in `outputs/experiments/`:
- `final_cross_task_summary.csv` — one row per experiment with strategy label
- `threshold_robustness_summary.csv` — threshold sensitivity per experiment

Study outputs in `outputs/studies/task_arch_grid/`:
- `runs/{task}__{architecture}__seed{k}/` — per-run configs, metrics, checkpoints, and a minimal `figures/traces.png`
- `summaries/by_run.csv` — one row per study run
- `summaries/by_task_arch_seed.csv` — same rows grouped for downstream analysis
- `summaries/by_task_arch_mean.csv` — task x architecture means
- `figures/emd_summary_grid.png` — raw `E/D/M` summary trajectories
- `figures/emd_summary_grid_peak_normalized.png` — peak-normalized `E/D/M` summary trajectories
- `figures/mechanism_map.png` — task x architecture mechanism map
- `figures/onset_plot.png` — onset comparison
- `figures/task_arch_profile_grid.png` — compact profile-label grid

## License

This code is provided as a research starter template without warranty.
