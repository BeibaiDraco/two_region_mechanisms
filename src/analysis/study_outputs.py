from __future__ import annotations

import shutil
from pathlib import Path


DEFAULT_RUN_FIGURES = {"traces.png"}


def ensure_primary_trace_figure(figures_dir: str | Path) -> Path | None:
    figures_dir = Path(figures_dir)
    traces_png = figures_dir / "traces.png"
    legacy_trace = figures_dir / "traces_E_D_M.png"
    if not traces_png.exists() and legacy_trace.exists():
        shutil.copyfile(legacy_trace, traces_png)
    return traces_png if traces_png.exists() else None


def prune_run_figures(figures_dir: str | Path, keep: set[str] | None = None) -> list[Path]:
    figures_dir = Path(figures_dir)
    keep = set(DEFAULT_RUN_FIGURES if keep is None else keep)
    ensure_primary_trace_figure(figures_dir)

    removed: list[Path] = []
    if not figures_dir.exists():
        return removed

    for path in figures_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in keep:
            continue
        path.unlink()
        removed.append(path)
    return removed
