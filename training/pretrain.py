# training/pretrain.py
import argparse
import contextlib
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig)
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Transformer, count_parameters
from kernels.gemm import FP8Linear          # FP8Linear lives in kernels/gemm.py
from utils.checkpoint import CheckpointManager
from utils.distributed import setup_distributed, cleanup_distributed, all_reduce_mean
from utils.logging import init_logging, get_logger


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WarmupCosineDecayScheduler(_LRScheduler):
    """
    Linear warmup → cosine decay → flat minimum LR.

    Phase 1  [0, warmup_steps):           LR scales linearly from 0 to base_lr.
    Phase 2  [warmup_steps, total_steps): LR follows a cosine curve from base_lr down to base_lr * min_lr_ratio.
    Phase 3  [total_steps, ∞):            LR is fixed at base_lr * min_lr_ratio.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        elif step >= self.total_steps:
            factor = self.min_lr_ratio
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return [base_lr * factor for base_lr in self.base_lrs]


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All training hyperparameters in one place. `model_config` holds the full parsed YAML dict"""

    # Full parsed config dict (set by main())
    model_config: dict = field(default_factory=dict)

    # Paths
    data_path:       str = "data/pretrain_data.bin"
    checkpoint_dir:  str = "checkpoints/pretrain"

    # Data
    vocab_size:  int = 102400
    max_seq_len: int = 4096

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 50_000
    warmup_steps: int = 2_000

    # Optimisation
    lr: float = 2.2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Precision
    dtype: str  = "bf16"   # "fp16" | "bf16" | "fp32"
    fp8_enabled: bool = False

    # Gradient checkpointing
    use_checkpoint: bool = False

    # MoE load balancing
    balance_loss_alpha: float = 0.0001
    bias_update_speed: float = 0.001
    bias_update_every: int = 10     # optimiser steps between bias updates

    # Parallelism strategy
    use_fsdp: bool = True  # True → FSDP (shard params+grads+optim); False → DDP (replicate)

    # Logging / checkpointing
    save_every: int = 1_000
    log_every:  int = 100


# ── Dataset ────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Packed pre-training dataset backed by a flat token tensor.
    Each sample is (input, target) where input = tokens[i:i+L] and target = tokens[i+1:i+L+1] (next-token prediction).
    If the data file does not exist, a synthetic random dataset is created and saved so that training can proceed
    without real data.
    """

    def __init__(self, data_path: str, max_seq_len: int, vocab_size: int):
        self.max_seq_len = max_seq_len
        self.vocab_size  = vocab_size

        if os.path.exists(data_path):
            self.data = torch.load(data_path, weights_only=True)
        else:
            print(f"[warn] {data_path} not found — generating synthetic data")
            os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
            self.data = torch.randint(0, vocab_size, (1_000_000,))
            torch.save(self.data, data_path)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        chunk = self.data[start : start + self.max_seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


# ── Trainer ────────────────────────────────────────────────────────────────────

class Pretrainer:
    """
    Pre-training loop integrating all DeepSeek-V3 training features.

    Key Features
    -------------
    • Distributed training with FSDP *or* DDP and gradient accumulation for large effective batch sizes.
      Use ``use_fsdp=True`` (recommended for large models) to shard parameters, gradients & optimiser states across ranks;
      use ``use_fsdp=False`` for the simpler DDP replica strategy (requires the full model to fit in each GPU's memory).
    • Warmup + cosine decay learning rate schedule.
    • Mixed precision training with optional FP16/BF16 support.
    • MoE load-balancing loss and periodic bias updates using cached routing.
    • Checkpointing of model weights, optimiser state, and training metadata with safetensors for efficient I/O.
    • Resumable training from checkpoints, including scheduler and bias update state.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.world_size, self.rank, self.local_rank = setup_distributed()

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        init_logging(self.rank, self.world_size, config.log_every)
        self.logger = get_logger()

        # ── Model ──────────────────────────────────────────────────────────

        self._log(f"Rank {self.rank}: initialising model...")
        raw_model = Transformer(
            config.model_config,
            world_size=self.world_size,
            rank=self.rank,
            use_checkpoint=config.use_checkpoint,
        ).to(self.device)

        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")

        # FP8 weight swap must happen before DDP/FSDP so the wrapper sees the final parameter set
        if config.fp8_enabled:
            self._apply_fp8(raw_model)

        if self.world_size > 1:
            if config.use_fsdp:
                # FSDP shards parameters, gradients, AND optimiser states across ranks.
                # use_orig_params=True preserves the per-parameter weight-decay groups
                self.model: nn.Module = FSDP(
                    raw_model,
                    device_id=self.local_rank,
                    use_orig_params=True,
                )
                self._log("Using FSDP (parameters + gradients + optimiser states sharded)")
            else:
                self.model: nn.Module = DDP(
                    raw_model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
                self._log("Using DDP (full model replicated on each rank)")
        else:
            self.model = raw_model

        # Keep a reference to the unwrapped model for direct attribute access
        self.raw_model: Transformer = raw_model

        # ── Optimiser ──────────────────────────────────────────────────────
        # Weight decay is not applied to 1-D parameters (biases, norms).
        decay_params    = [p for p in raw_model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in raw_model.parameters() if p.dim() < 2]
        self.optimizer  = AdamW(
            [
                {"params": decay_params,    "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available(),
        )

        # ── Scheduler ──────────────────────────────────────────────────────
        self.scheduler = WarmupCosineDecayScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # ── AMP ────────────────────────────────────────────────────────────
        self.amp_dtype = (
            torch.float16   if config.dtype == "fp16"  else
            torch.bfloat16  if config.dtype == "bf16"  else
            None
        )
        self.scaler = GradScaler("cuda") if config.dtype == "fp16" else None

        # ── Checkpoint manager ─────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(config.checkpoint_dir)

        # ── Optimiser step counter ─────────────────────────────────────────
        # Tracks how many full optimiser steps have been taken, independent of icro-step counting.
        # Used for bias update scheduling.
        self._opt_steps: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            print(msg)

    def _apply_fp8(self, model: nn.Module) -> None:
        """
        Replace every ``nn.Linear`` in *model* with ``FP8Linear`` from ``kernels/fp8_kernel.py``.

        • Weights are quantised to FP8 E4M3FN once here.
        • Activations are quantised on every forward call inside ``FP8GEMM``.
        • No external dependencies — uses the project's own Triton kernels.
        • FP8 GEMM requires ``torch.float8_e4m3fn`` support
        """

        def _swap_linear(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if type(child) is nn.Linear:
                    setattr(module, name, FP8Linear.from_linear(child))
                else:
                    _swap_linear(child)

        _swap_linear(model)
        self._log(
            "FP8 enabled — nn.Linear layers replaced with FP8Linear "
            "(kernels/fp8_kernel.py + kernels/gemm.py, e4m3fn weights, FP32 accumulation)")

    def _amp_context(self):
        """Return the appropriate AMP context for the configured dtype.

        When fp8_enabled, FP8 arithmetic is handled inside FP8Linear.forward() so the outer autocast just needs to
        cover the remaining BF16 ops.
        """
        if self.amp_dtype is not None:
            return autocast("cuda", dtype=self.amp_dtype)
        return contextlib.nullcontext()

    def _update_moe_bias(self) -> None:
        """
        Update per-expert load-balancing biases using routing cached during the most recent forward pass.

        Uses `moe.update_gate_bias()` which reads `_last_indices` populated
        by `DeepSeekMoE.forward()` — no extra gate computation or embedding.
        """
        for moe in self.raw_model.moe_layers():
            moe.update_gate_bias(speed=self.config.bias_update_speed)

    def _moe_balance_loss(self) -> torch.Tensor:
        """
        Sum the load-balance auxiliary loss across all MoE layers.

        Each layer's loss is computed from cached routing — no forward re-run.
        Returns a scalar zero tensor if there are no MoE layers.
        """
        losses = [
            moe.get_load_balance_loss()
            for moe in self.raw_model.moe_layers()
        ]
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).sum()

    # ──────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        micro_step: int,
    ) -> Dict[str, float]:
        """
        Run one micro-step (forward + backward).  The optimiser is only
        stepped every `gradient_accumulation_steps` micro-steps.

        Args:
            tokens:     (bsz, seq_len) input token IDs
            targets:    (bsz, seq_len) target token IDs
            micro_step: global micro-step counter (not the optimiser step)

        Returns:
            dict with "loss" (CE) and "balance_loss" (MoE load balance)
            both unscaled and globally averaged across ranks.
        """
        is_opt_step = (micro_step + 1) % self.config.gradient_accumulation_steps == 0

        # Sync gradients only on the last micro-step of each accumulation window.
        sync_ctx = (
            contextlib.nullcontext()
            if (not isinstance(self.model, (DDP, FSDP)) or is_opt_step)
            else self.model.no_sync())

        with sync_ctx, self._amp_context():
            # use_cache=False: teacher-forced training does not use the KV cache. Passing True would read
            # stale/zero latents from MLA's persistent cache buffers, silently corrupting attention scores.
            logits = self.model(tokens, start_pos=0, use_cache=False)

            ce_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100)

            # MoE load-balance auxiliary loss (uses cached routing — no second gate call).
            # Scaled by balance_loss_alpha which is typically very small (0.0001) so it does not dominate the CE loss.
            balance_loss = self._moe_balance_loss()
            loss = (
                ce_loss + self.config.balance_loss_alpha * balance_loss
            ) / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Optimiser step ─────────────────────────────────────────────────
        if is_opt_step:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._opt_steps += 1

            # Update MoE load-balancing biases periodically using cached routing
            if self._opt_steps % self.config.bias_update_every == 0:
                self._update_moe_bias()

        # Average loss across ranks for accurate logging
        ce_scalar      = all_reduce_mean(ce_loss.detach())
        balance_scalar = all_reduce_mean(balance_loss.detach())

        return {
            "loss":         ce_scalar.item(),
            "balance_loss": balance_scalar.item(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        """Save model weights (safetensors), optimiser state, and metadata."""
        extra_meta = {
            "scheduler": self.scheduler.state_dict(),
            "opt_steps": self._opt_steps,
            "tag":       tag or f"step_{step}",
            "config":    asdict(self.config),
        }
        if isinstance(self.model, FSDP):
            # Gather the full (unsharded) state dict onto rank-0 CPU. All ranks must enter
            # this context together; only rank 0 gets a populated dict — other ranks receive empty dict.
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = self.model.state_dict()
            if self.rank == 0:
                self.ckpt_manager.save_state_dict(
                    model_state, self.optimizer, step, extra_meta=extra_meta
                )
        else:
            if self.rank != 0:
                return
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            self.ckpt_manager.save(model_to_save, self.optimizer, step, extra_meta=extra_meta)
        self._log(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, step: int) -> int:
        """
        Load model weights, optimiser state, and metadata from a checkpoint.

        Args:
            step: the step number to load (as saved by CheckpointManager)

        Returns:
            The micro-step to resume from.
        """
        if isinstance(self.model, FSDP):
            # Load the full state dict from disk, then apply it inside the
            # FSDP context so PyTorch can re-shard it across ranks correctly.
            weights, meta = self.ckpt_manager.load_weights(step, device=str(self.device))
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
                missing, unexpected = self.model.load_state_dict(weights, strict=False)
            if missing:
                self._log(f"[checkpoint] missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
            if unexpected:
                self._log(f"[checkpoint] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
            # Restore optimiser state if present
            optim_path = self.ckpt_manager.save_dir / f"optim_step_{step}.pt"
            if optim_path.exists():
                self.optimizer.load_state_dict(
                    torch.load(str(optim_path), map_location=str(self.device), weights_only=True)
                )
        else:
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            meta = self.ckpt_manager.load(
                model_to_load,
                step,
                device=str(self.device),
                optimizer=self.optimizer,
            )
        if "scheduler" in meta:
            self.scheduler.load_state_dict(meta["scheduler"])
        if "opt_steps" in meta:
            self._opt_steps = meta["opt_steps"]
        resumed_step = meta.get("step", step)
        self._log(f"Resumed from step {resumed_step}")
        return resumed_step

    def _find_latest_checkpoint(self) -> Optional[int]:
        """Return the latest checkpoint step number, or None."""
        return self.ckpt_manager.latest_step()

    # ──────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        dataset = PretrainDataset(
            self.config.data_path,
            self.config.max_seq_len,
            self.config.vocab_size,
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # Resume from latest checkpoint if available
        global_step = 0
        latest = self._find_latest_checkpoint()
        if latest is not None:
            try:
                global_step = self.load_checkpoint(latest)
            except Exception as exc:
                self._log(f"[warn] Could not load checkpoint: {exc}")

        self._log(
            f"Training from step {global_step} to {self.config.max_steps}  "
            f"(world_size={self.world_size})"
        )

        # Set train mode once — not inside the hot path
        self.model.train()

        epoch = 0
        while global_step < self.config.max_steps:
            sampler.set_epoch(epoch)
            epoch += 1

            for tokens, targets in tqdm(loader, disable=self.rank != 0):
                if global_step >= self.config.max_steps:
                    break

                tokens  = tokens.to(self.device,  non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                metrics = self.train_step(tokens, targets, global_step)

                if global_step % self.config.log_every == 0 and self.rank == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.log(
                        global_step,
                        metrics["loss"],
                        lr=lr,
                        metrics={"balance_loss": metrics["balance_loss"]},
                    )

                if (
                    global_step % self.config.save_every == 0
                    and global_step > 0
                ):
                    self.save_checkpoint(global_step)

                global_step += 1

        self.save_checkpoint(global_step, tag="final")
        self._log("Training complete.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V3 pre-training")
    parser.add_argument("--config",         type=str, default="configs/pretrain_config.yaml")
    parser.add_argument("--data-path",      type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume",         type=str, default=None,
                        help="Checkpoint step number to resume from")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--use-fsdp", action="store_true",
                        help="Use FSDP instead of DDP (recommended for large models)")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    t = yaml_cfg.get("training", {})
    d = yaml_cfg.get("data",     {})

    config = TrainingConfig(
        # Store the full config dict so the Transformer constructor gets both the "model" block and any nested keys.
        #
        model_config=yaml_cfg,

        data_path=args.data_path or d.get("train_data_path", "data/pretrain_data.bin"),
        checkpoint_dir=args.checkpoint_dir or t.get("save_dir", "checkpoints/pretrain"),

        batch_size=t.get("micro_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        max_steps=t.get("total_steps", 50_000),
        warmup_steps=t.get("warmup_steps", 2_000),

        lr=t.get("lr", 2.2e-4),
        min_lr_ratio=t.get("min_lr_ratio", 0.1),
        weight_decay=t.get("weight_decay", 0.1),
        max_grad_norm=t.get("grad_clip", 1.0),

        fp8_enabled=t.get("fp8_enabled", False),
        use_checkpoint=args.use_checkpoint or t.get("use_checkpoint", False),
        use_fsdp=args.use_fsdp or t.get("use_fsdp", False),

        balance_loss_alpha=t.get("balance_loss_alpha", 0.0001),
        bias_update_speed=t.get("bias_update_speed", 0.001),
        bias_update_every=t.get("bias_update_every", 10),

        save_every=t.get("save_interval", 1_000),
        log_every=t.get("log_interval", 100),
    )

    trainer = Pretrainer(config)

    if args.resume is not None:
        trainer.load_checkpoint(int(args.resume))

    trainer.train()
    cleanup_distributed()


if __name__ == "__main__":
    main()
