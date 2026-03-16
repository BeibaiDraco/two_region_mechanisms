# Two Region Mechanisms

A PyTorch research codebase for studying emergent inter-region interaction mechanisms in trained **two-region continuous-time rate RNNs**.

The central question is:

> After training on a task, what kind of interaction emerges from region A to region B?

The code provides a practical first-pass mechanism taxonomy using three proxy indicators:

- **Direct forcing** `D_J`
- **Autonomous-field modulation** `M_J`
- **Realized quotient effect** `E_J`

Combined, these give one of four mechanism labels:

- `(0,0)` no quotient-level mechanism
- `(1,0)` pure direct forcing
- `(0,1)` pure autonomous-field modulation
- `(1,1)` mixed mechanism

## What is included

- Four model families:
  - `TwoRegionAdditiveRNN`
  - `TwoRegionGatedRNN`
  - `TwoRegionLowRankCommRNN`
  - `TwoRegionReciprocalContextRNN`
- Six task families:
  - `BinaryCategorizationTask`
  - `DelayedCategorizationTask`
  - `ContextDependentDecisionTask`
  - `CueGatingTask`
  - `StateSettingTask`
  - `RedundantInputControlTask`
- End-to-end training, evaluation, ablation, and plotting scripts
- Config-driven experiments
- Practical intervention-based mechanism proxies
- Smoke tests and lightweight unit tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke tests

```bash
python scripts/train_one.py --config configs/smoke_tests/binary_additive.yaml
python scripts/evaluate_model.py --run-dir outputs/smoke/binary_additive
python scripts/classify_mechanisms.py --run-dir outputs/smoke/binary_additive
```

A second smoke test uses a context-dependent task and a gated model:

```bash
python scripts/train_one.py --config configs/smoke_tests/context_gated.yaml
python scripts/evaluate_model.py --run-dir outputs/smoke/context_gated
python scripts/classify_mechanisms.py --run-dir outputs/smoke/context_gated
```

## One larger example

```bash
python scripts/train_one.py --config configs/experiments/context_gated.yaml
python scripts/classify_mechanisms.py --run-dir outputs/experiments/context_gated
python scripts/run_ablations.py --run-dir outputs/experiments/context_gated
python scripts/make_figures.py --root outputs/experiments
```

## Project layout

```text
two_region_mechanisms/
  README.md
  requirements.txt
  configs/
    tasks/
    models/
    experiments/
    smoke_tests/
  src/
    tasks/
    models/
    train/
    analysis/
    utils/
  scripts/
  tests/
  outputs/
```

## Task families

### BinaryCategorizationTask
Noisy evidence appears during a stimulus epoch. The network reports left/right during response.

### DelayedCategorizationTask
Like binary categorization, but with a delay between stimulus and response.

### ContextDependentDecisionTask
Two evidence streams are present. A context cue determines which one is behaviorally relevant.

### CueGatingTask
A cue selects one of several mappings: report, ignore, invert, or hold a signal.

### StateSettingTask
An initial cue selects a computational regime for later inputs, encouraging regime-setting rather than simple content passing.

### RedundantInputControlTask
Useful content is available both upstream and locally, so the learned `A -> B` pathway can become partly redundant.

## Mechanism proxies

These scripts implement **practical proxies**, not theorem-level identification.

### Realized effect `E_J`
Compare the region-B readout trajectory in the coupled network versus a decoupled rollout where the `A -> B` source is removed.

### Direct forcing `D_J`
Estimate the immediate one-step change in region-B readout due to the `A -> B` source while holding the current state fixed.

### Modulation `M_J`
Approximate whether `A -> B` changes the future autonomous quotient flow of region B through a private/latent state displacement orthogonal to the local readout gradient.

## Output files

Each run directory stores:

- `config.yaml`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics/train_history.csv`
- `metrics/val_history.csv`
- `metrics/eval_metrics.json`
- `metrics/mechanism_summary.json`
- `figures/*.png`
- optional ablation summaries

## Notes on the Yang multitask repo

This repo was built as a fresh PyTorch codebase rather than a port of `gyyang/multitask`. The Yang repo is still useful task inspiration: its README describes an older TensorFlow 1.8 / Python 2.7–3.6 stack and uses Mante-style context tasks such as `contextdm1`, with `mante` as the default quick-start training example. citeturn656047view0

## Recommended workflow

1. Run a smoke test.
2. Inspect `outputs/.../metrics/mechanism_summary.json`.
3. Train multiple seeds for several task × architecture combinations.
4. Aggregate with `scripts/make_figures.py`.

## License

This code is provided as a research starter template without warranty.
