from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from tqdm import trange

from src.train.losses import (
    activity_regularization,
    communication_regularization,
    masked_mse,
    weight_regularization,
)
from src.utils.config import save_config
from src.utils.io import ensure_dir, save_checkpoint
from src.utils.plotting import plot_training_curves


@dataclass
class TrainerConfig:
    steps: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    eval_every: int = 50
    val_batches: int = 10
    grad_clip: float = 1.0
    activity_reg: float = 1e-4
    weight_reg: float = 1e-6
    communication_reg: float = 0.0
    device: str = "cpu"
    early_stop_patience: int = 0
    early_stop_target_acc: float = 1.0


class Trainer:
    def __init__(self, model, task, cfg: Dict[str, Any], run_dir: str | Path):
        self.model = model
        self.task = task
        self.cfg = cfg
        self.run_dir = Path(run_dir)

        train_cfg = cfg.get("train", {})
        self.tcfg = TrainerConfig(
            steps=int(train_cfg.get("steps", 500)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            eval_every=int(train_cfg.get("eval_every", 50)),
            val_batches=int(train_cfg.get("val_batches", 10)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            activity_reg=float(train_cfg.get("activity_reg", 1e-4)),
            weight_reg=float(train_cfg.get("weight_reg", 1e-6)),
            communication_reg=float(train_cfg.get("communication_reg", 0.0)),
            device=str(train_cfg.get("device", cfg.get("device", "cpu"))),
            early_stop_patience=int(train_cfg.get("early_stop_patience", 0)),
            early_stop_target_acc=float(train_cfg.get("early_stop_target_acc", 1.0)),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tcfg.learning_rate)
        self.train_history = []
        self.val_history = []

    def _compute_loss(self, rollout, batch):
        task_loss = masked_mse(rollout["outputs"], batch.targets, batch.mask)
        act_reg = activity_regularization(rollout, self.tcfg.activity_reg)
        w_reg = weight_regularization(self.model, self.tcfg.weight_reg)
        comm_reg = communication_regularization(self.model, self.tcfg.communication_reg)
        total = task_loss + act_reg + w_reg + comm_reg
        return total, {
            "task_loss": float(task_loss.detach().cpu()),
            "activity_reg": float(act_reg.detach().cpu()),
            "weight_reg": float(w_reg.detach().cpu()),
            "communication_reg": float(comm_reg.detach().cpu()),
        }

    @torch.no_grad()
    def evaluate(self, split: str = "val", num_batches: int | None = None) -> Dict[str, float]:
        self.model.eval()
        num_batches = self.tcfg.val_batches if num_batches is None else num_batches

        losses, accs = [], []
        for _ in range(num_batches):
            batch = self.task.sample_batch(self.tcfg.batch_size, split=split, device=self.tcfg.device)
            rollout = self.model(batch.inputs, add_noise=False)
            loss, _ = self._compute_loss(rollout, batch)
            acc = self.task.compute_accuracy(rollout["outputs"], batch)
            losses.append(float(loss.detach().cpu()))
            accs.append(acc)

        return {"loss": float(sum(losses) / len(losses)), "accuracy": float(sum(accs) / len(accs))}

    def train(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ensure_dir(self.run_dir / "checkpoints")
        ensure_dir(self.run_dir / "metrics")
        ensure_dir(self.run_dir / "figures")
        save_config(self.cfg, self.run_dir / "config.yaml")

        best_val = float("-inf")
        best_step = -1
        patience_counter = 0

        progress = trange(1, self.tcfg.steps + 1, desc="training", leave=False)
        for step in progress:
            self.model.train()
            batch = self.task.sample_batch(self.tcfg.batch_size, split="train", device=self.tcfg.device)
            rollout = self.model(batch.inputs, add_noise=self.model.training)
            loss, loss_terms = self._compute_loss(rollout, batch)

            self.optimizer.zero_grad()
            loss.backward()
            if self.tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.optimizer.step()

            train_acc = self.task.compute_accuracy(rollout["outputs"], batch)
            train_row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "accuracy": train_acc,
                **loss_terms,
            }
            self.train_history.append(train_row)
            progress.set_postfix(loss=f"{train_row['loss']:.3f}", acc=f"{train_acc:.2f}")

            if step == 1 or step % self.tcfg.eval_every == 0 or step == self.tcfg.steps:
                val = self.evaluate(split="val")
                val_row = {"step": step, **val}
                self.val_history.append(val_row)

                payload = {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.cfg,
                    "step": step,
                    "val": val,
                }
                save_checkpoint(self.run_dir / "checkpoints" / "last.pt", payload)
                improved = val["accuracy"] > best_val
                if val["accuracy"] >= best_val:
                    best_val = val["accuracy"]
                    best_step = step
                    save_checkpoint(self.run_dir / "checkpoints" / "best.pt", payload)
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (self.tcfg.early_stop_patience > 0
                        and best_val >= self.tcfg.early_stop_target_acc
                        and patience_counter >= self.tcfg.early_stop_patience):
                    progress.close()
                    print(f"Early stop at step {step}: val acc {best_val:.4f} "
                          f"stable for {patience_counter} evals")
                    break

        train_df = pd.DataFrame(self.train_history)
        val_df = pd.DataFrame(self.val_history)
        train_df.to_csv(self.run_dir / "metrics" / "train_history.csv", index=False)
        val_df.to_csv(self.run_dir / "metrics" / "val_history.csv", index=False)
        plot_training_curves(train_df, val_df, self.run_dir / "figures" / "training_curves.png")

        summary = {
            "best_val_accuracy": best_val,
            "best_step": best_step,
            "final_step": self.tcfg.steps,
        }
        pd.DataFrame([summary]).to_csv(self.run_dir / "metrics" / "training_summary.csv", index=False)
        return self.run_dir
