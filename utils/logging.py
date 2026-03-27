# utils/logging.py
import json
import time
from typing import Dict, Optional

import torch


class DistributedLogger:
    """
    Rank-0 training logger with optional WandB integration.

    Tracks a rolling loss window and computes throughput automatically.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        log_interval: int = 10,
        seq_len: int = 4096,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.log_interval = log_interval
        self.seq_len = seq_len

        self._start = time.time()
        self._step_start = time.time()
        self._loss_window: list[float] = []

        # WandB (optional)
        self._wandb = None
        if rank == 0 and wandb_project:
            try:
                import wandb  # type: ignore
                wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)
                self._wandb = wandb
            except ImportError:
                print("[logging] wandb not installed — skipping WandB integration")

    # ──────────────────────────────────────────────────────────────────────

    def log(
        self,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
    ) -> None:
        if self.rank != 0:
            return

        self._loss_window.append(loss)

        if step % self.log_interval != 0 or not self._loss_window:
            return

        avg_loss = sum(self._loss_window) / len(self._loss_window)
        elapsed = max(time.time() - self._step_start, 1e-6)
        tokens_per_sec = (self.log_interval * self.seq_len * self.world_size) / elapsed
        ppl = torch.tensor(avg_loss).exp().item()

        parts = [
            f"step={step:>7}",
            f"loss={avg_loss:.4f}",
            f"ppl={ppl:.2f}",
            f"lr={lr:.2e}",
            f"tps={tokens_per_sec:,.0f}",
        ]
        if metrics:
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}")

        print(" | ".join(parts))

        if self._wandb is not None:
            log_dict = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
            }
            if metrics:
                log_dict.update({f"train/{k}": v for k, v in metrics.items()})
            self._wandb.log(log_dict, step=step)

        self._loss_window = []
        self._step_start = time.time()

    def save_log(self, filename: str, data: Dict) -> None:
        if self.rank != 0:
            return
        with open(filename, "a") as f:
            f.write(json.dumps(data) + "\n")

    def finish(self) -> None:
        if self.rank == 0 and self._wandb is not None:
            self._wandb.finish()


# ── Module-level singleton ─────────────────────────────────────────────────────

_logger: Optional[DistributedLogger] = None


def init_logging(
    rank: int,
    world_size: int,
    log_interval: int = 10,
    seq_len: int = 4096,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> None:
    global _logger
    _logger = DistributedLogger(
        rank=rank,
        world_size=world_size,
        log_interval=log_interval,
        seq_len=seq_len,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


def get_logger() -> DistributedLogger:
    global _logger
    if _logger is None:
        # Lazy fallback: non-distributed no-op logger
        _logger = DistributedLogger(rank=0, world_size=1)
    return _logger
