#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def run_step(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default=str(ROOT / "configs" / "studies" / "task_arch_grid" / "generated"))
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--arch", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--cache-trials", type=int, default=256)
    args = parser.parse_args()

    config_paths = sorted(Path(args.config_dir).glob("*.yaml"))
    py = sys.executable

    for config_path in config_paths:
        cfg = load_config(config_path)
        study = cfg.get("study", {})
        if args.task and study.get("task_key") not in args.task:
            continue
        if args.arch and study.get("architecture_key") not in args.arch:
            continue
        if args.seed and int(study.get("seed", -1)) not in args.seed:
            continue

        run_dir = Path(cfg["output_dir"])
        final_profile = run_dir / "metrics" / "final_profile.json"
        if args.skip_existing and final_profile.exists():
            print(f"Skipping {run_dir}")
            continue

        run_step([py, "scripts/train_one.py", "--config", str(config_path)])
        run_step([
            py,
            "scripts/run_profile_bundle.py",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            args.checkpoint,
            "--cache-trials",
            str(args.cache_trials),
            *([] if not args.skip_existing else ["--skip-existing"]),
        ])


if __name__ == "__main__":
    main()
