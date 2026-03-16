from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.io import load_json


def collect_mechanism_summaries(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in root.rglob("mechanism_summary.json"):
        summary = load_json(path)
        summary["run_dir"] = str(path.parent.parent)
        rows.append(summary)
    return pd.DataFrame(rows)


def class_counts(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}
    if "mechanism_class" in df.columns:
        return dict(Counter(df["mechanism_class"].tolist()))
    if "mechanism_class_mode" in df.columns:
        return dict(Counter(df["mechanism_class_mode"].tolist()))
    if "mechanism_class_counts" in df.columns:
        total = Counter()
        for entry in df["mechanism_class_counts"]:
            if isinstance(entry, dict):
                total.update(entry)
        return dict(total)
    return {}
