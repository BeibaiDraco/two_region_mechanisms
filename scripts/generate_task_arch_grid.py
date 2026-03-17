#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, save_config


def build_run_name(task_key: str, arch_key: str, seed: int) -> str:
    return f"{task_key}__{arch_key}__seed{seed}"


def generate_configs(manifest_path: Path) -> list[Path]:
    manifest = load_config(manifest_path)
    output_root = Path(manifest["output_root"])
    generated_dir = Path(manifest.get("generated_config_dir", manifest_path.parent / "generated"))
    tasks_dir = ROOT / "configs" / "tasks"
    models_dir = ROOT / "configs" / "models"

    generated_paths = []
    for task_key in manifest["tasks"]:
        allowed_arches = manifest.get("task_arch_allowlist", {}).get(task_key, manifest["architectures"])
        task_cfg = load_config(tasks_dir / f"{task_key}.yaml")
        for arch_key in allowed_arches:
            model_cfg = load_config(models_dir / f"{arch_key}.yaml")
            for seed in manifest["seeds"]:
                run_name = build_run_name(task_key, arch_key, int(seed))
                cfg = {
                    "study": {
                        "name": manifest["study_name"],
                        "task_key": task_key,
                        "architecture_key": arch_key,
                        "seed": int(seed),
                    },
                    "seed": int(seed),
                    "device": "auto",
                    "task": deepcopy(task_cfg),
                    "model": deepcopy(model_cfg),
                    "train": deepcopy(manifest.get("train_template", {})),
                    "analysis": deepcopy(manifest.get("analysis_template", {})),
                    "output_dir": str(output_root / run_name),
                }
                cfg["task"].setdefault("params", {})
                cfg["task"]["params"]["seed"] = int(seed)

                out_path = generated_dir / f"{run_name}.yaml"
                save_config(cfg, out_path)
                generated_paths.append(out_path)
    return generated_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "configs" / "studies" / "task_arch_grid" / "manifest.yaml"),
    )
    args = parser.parse_args()

    paths = generate_configs(Path(args.manifest))
    print(f"Generated {len(paths)} configs")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
