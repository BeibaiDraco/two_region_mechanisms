#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.study_outputs import prune_run_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "runs"),
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())

    total_removed = 0
    for run_dir in run_dirs:
        removed = prune_run_figures(run_dir / "figures")
        total_removed += len(removed)

    print(f"Pruned figures in {len(run_dirs)} run directories")
    print(f"Removed {total_removed} extra figure files")


if __name__ == "__main__":
    main()
