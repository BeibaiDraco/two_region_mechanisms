#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.eval_cache import default_eval_cache_path, load_or_create_eval_batch
from src.analysis.rollout import load_run_artifacts
from src.analysis.study_outputs import prune_run_figures
from src.utils.io import load_json, save_json


def run_step(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def maybe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def build_final_profile(run_dir: Path) -> dict:
    metrics_dir = run_dir / "metrics"
    training = pd.read_csv(metrics_dir / "training_summary.csv").iloc[0].to_dict()
    mechanism = load_json(metrics_dir / "mechanism_summary.json")
    eval_metrics = load_json(metrics_dir / "eval_metrics.json")

    task_profile_path = metrics_dir / f"{run_dir.name}_profile.json"
    task_profile = load_json(task_profile_path) if task_profile_path.exists() else {}

    threshold_path = metrics_dir / "threshold_robustness_summary.csv"
    threshold_row = {}
    if threshold_path.exists():
        threshold_row = pd.read_csv(threshold_path).iloc[0].to_dict()

    final_profile = {
        "run_name": run_dir.name,
        "task": mechanism.get("task"),
        "model": mechanism.get("model"),
        "seed": mechanism.get("seed"),
        "best_val_accuracy": training.get("best_val_accuracy"),
        "best_step": training.get("best_step"),
        "mechanism_class_mode": mechanism.get("mechanism_class_mode"),
        "strategy_label": task_profile.get("strategy_label", mechanism.get("mechanism_class_mode")),
        "threshold_robustness": threshold_row.get("robustness_max"),
        "threshold_robustness_maxmean": threshold_row.get("robustness_maxmean"),
        "task_profile": task_profile,
        "mechanism_summary": mechanism,
        "eval_metrics": eval_metrics,
    }
    return final_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--cache-trials", type=int, default=256)
    parser.add_argument("--eval-cache", type=str, default=None)
    parser.add_argument("--refresh-eval-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-all-figures", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    final_profile_path = metrics_dir / "final_profile.json"
    if args.skip_existing and final_profile_path.exists():
        print(f"Skipping {run_dir} because final_profile.json already exists")
        return

    cfg, task, _, _ = load_run_artifacts(run_dir, checkpoint_name=args.checkpoint, map_location="cpu")
    cache_path = Path(args.eval_cache) if args.eval_cache else default_eval_cache_path(run_dir)
    load_or_create_eval_batch(
        task,
        n_trials=args.cache_trials,
        device="cpu",
        split="val",
        cache_path=cache_path,
        refresh=args.refresh_eval_cache,
    )

    py = sys.executable
    run_step([py, "scripts/evaluate_model.py", "--run-dir", str(run_dir), "--checkpoint", args.checkpoint, "--batch-size", str(args.cache_trials), "--eval-cache", str(cache_path)])
    run_step([py, "scripts/classify_mechanisms.py", "--run-dir", str(run_dir), "--checkpoint", args.checkpoint, "--batch-size", str(args.cache_trials), "--num-batches", "1", "--eval-cache", str(cache_path)])
    run_step([py, "scripts/threshold_sweep.py", "--run-dirs", str(run_dir), "--checkpoint", args.checkpoint, "--batch-size", str(args.cache_trials), "--num-batches", "1", "--eval-cache", str(cache_path), "--output-dir", str(metrics_dir)])
    run_step([py, "scripts/run_task_analysis.py", "--run-dir", str(run_dir), "--n-trials", str(args.cache_trials), "--eval-cache", str(cache_path)])

    classification_row = {
        **load_json(metrics_dir / "mechanism_summary.json"),
        **load_json(metrics_dir / "eval_metrics.json"),
    }
    pd.DataFrame([classification_row]).to_csv(metrics_dir / "classification_summary.csv", index=False)

    final_profile = build_final_profile(run_dir)
    save_json(final_profile, final_profile_path)

    maybe_copy(figures_dir / "traces_E_D_M.png", figures_dir / "traces.png")
    if not args.keep_all_figures:
        prune_run_figures(figures_dir)

    print(f"Saved bundle outputs in {run_dir}")
    print(f"  cache: {cache_path}")
    print(f"  profile: {final_profile_path}")


if __name__ == "__main__":
    main()
