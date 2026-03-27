# utils/checkpoint.py
"""
Checkpoint manager for DeepSeek-V3-style models.

File layout per checkpoint step
--------------------------------
  model_step_{step}.safetensors   — model weights (FP8 stored as-is)
  scales_step_{step}.json         — per-block FP8 weight scales (omitted when no FP8)
  optim_step_{step}.pt            — optimiser state dict (torch.save)
  meta_step_{step}.json           — step number + any extra metadata

Atomic writes
-------------
Each file is first written to a sibling temp file (same directory, so on the same filesystem/mount point as the
destination) then renamed.  rename() is atomic on POSIX filesystems, so the directory either contains the complete
checkpoint or the previous one — never a partial write.
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)

_FP8_E4M3_MAX: float = 448.0


class CheckpointManager:
    """
    Save and load model checkpoints.

    Thread / process safety
    -----------------------
    In DDP training only rank 0 should call `save()`.  `load()` may be
    called on all ranks independently since it is read-only.

    Usage
    -----
    ckpt = CheckpointManager("checkpoints/pretrain")
    ckpt.save(model, optimizer, step=1000, extra_meta={"scheduler": sched.state_dict()})
    meta = ckpt.load(model, step=1000, device="cuda:0", optimizer=optimizer)
    latest = ckpt.latest_step()   # int or None
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """
        Atomically persist model weights, optimiser state, and metadata.

        Model weights are written via safetensors (no pickle, memory-safe).
        FP8 weight tensors and their associated `weight_scale` buffers are
        included automatically through model.state_dict()

        Args:
            model:      unwrapped nn.Module (not DDP-wrapped)
            optimizer:  the optimiser to checkpoint
            step:       current training step (used as filename key)
            extra_meta: optional dict of JSON-serialisable metadata
                        (e.g. scheduler state, config, opt_steps counter)
        """
        state = model.state_dict()

        # Separate FP8 tensors from the rest for reporting purposes.
        # All tensors (FP8 or otherwise) go into the safetensors file.
        # Per-block weight_scale buffers are included automatically.
        fp8_keys = [k for k, v in state.items() if v.dtype == torch.float8_e4m3fn]

        # ── Model weights (safetensors) ────────────────────────────────────
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        self._atomic_save_safetensors(state, weight_path)

        # ── Optimiser state ────────────────────────────────────────────────
        optim_path = self.save_dir / f"optim_step_{step}.pt"
        self._atomic_save_torch(optimizer.state_dict(), optim_path)

        # ── Metadata ───────────────────────────────────────────────────────
        meta: dict = {
            "step":     step,
            "fp8_keys": fp8_keys}
        if extra_meta:
            # Merge extra_meta; step is authoritative
            meta.update({k: v for k, v in extra_meta.items() if k != "step"})

        meta_path = self.save_dir / f"meta_step_{step}.json"
        self._atomic_save_json(meta, meta_path)

        logger.info("[checkpoint] saved step %d → %s", step, self.save_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────

    def load(
        self,
        model: torch.nn.Module,
        step: int,
        device: str = "cuda",
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
    ) -> dict:
        """
        Load model weights and optionally restore optimiser state.

        Args:
            model:     the nn.Module to load weights into (not DDP-wrapped)
            step:      step number to load
            device:    device string for weight placement (e.g. "cuda:0")
            optimizer: if provided, restores optimiser state from checkpoint
            strict:    if True (default), raise on missing or unexpected keys. Set False only for partial loading         

        Returns:
            metadata dict (includes "step", scheduler state if saved, etc.)

        Raises:
            FileNotFoundError: if the checkpoint file does not exist
            RuntimeError: if strict=True and there are missing/unexpected keys
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weight_path}\n"
                f"Available steps: {self._list_steps()}")

        weights = load_file(str(weight_path), device=device)

        missing, unexpected = model.load_state_dict(weights, strict=False)

        # Always report what happened
        if missing:
            msg = (
                f"[checkpoint] {len(missing)} missing key(s): "
                f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

        if unexpected:
            msg = (
                f"[checkpoint] {len(unexpected)} unexpected key(s): "
                f"{unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

        # ── Optimiser ──────────────────────────────────────────────────────
        if optimizer is not None:
            optim_path = self.save_dir / f"optim_step_{step}.pt"
            if optim_path.exists():
                opt_state = torch.load(
                    optim_path,
                    map_location=device,
                    weights_only=True)
                optimizer.load_state_dict(opt_state)
            else:
                logger.warning(
                    "[checkpoint] no optimiser state at %s — "
                    "optimizer will start from scratch",
                    optim_path,
                )

        # ── Metadata ───────────────────────────────────────────────────────
        meta_path = self.save_dir / f"meta_step_{step}.json"
        meta: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            logger.warning("[checkpoint] no metadata file at %s", meta_path)
            meta = {"step": step}

        logger.info("[checkpoint] loaded step %d from %s", step, self.save_dir)
        return meta

    def save_state_dict(
        self,
        state_dict: dict,
        optimizer: torch.optim.Optimizer,
        step: int,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """
        Atomically persist an already-gathered state dict, optimiser state, and metadata.

        Used by the FSDP save path in pretrain.py where the full state dict is gathered
        from all ranks before calling this method (only rank 0 calls it with a non-empty dict).

        Args:
            state_dict: already-gathered model state dict (e.g. from FSDP full_state_dict)
            optimizer:  the optimiser to checkpoint
            step:       current training step
            extra_meta: optional JSON-serialisable metadata
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        self._atomic_save_safetensors(state_dict, weight_path)

        optim_path = self.save_dir / f"optim_step_{step}.pt"
        self._atomic_save_torch(optimizer.state_dict(), optim_path)

        meta: dict = {"step": step}
        if extra_meta:
            meta.update({k: v for k, v in extra_meta.items() if k != "step"})
        meta_path = self.save_dir / f"meta_step_{step}.json"
        self._atomic_save_json(meta, meta_path)

        logger.info("[checkpoint] saved state_dict step %d → %s", step, self.save_dir)

    def load_weights(
        self,
        step: int,
        device: str = "cuda",
    ) -> tuple:
        """
        Load the raw model weights and metadata for a checkpoint step without
        applying them to a model.

        Used by the FSDP load path in pretrain.py, which needs to obtain the weights
        and then apply them inside an FSDP.state_dict_type() context.

        Args:
            step:   step number to load
            device: device string for weight placement (e.g. "cuda:0")

        Returns:
            (weights_dict, meta_dict) — the state dict and the metadata dict.

        Raises:
            FileNotFoundError: if the safetensors file does not exist.
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weight_path}\n"
                f"Available steps: {self._list_steps()}")

        weights = load_file(str(weight_path), device=device)

        meta_path = self.save_dir / f"meta_step_{step}.json"
        meta: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            logger.warning("[checkpoint] no metadata file at %s", meta_path)
            meta = {"step": step}

        logger.info("[checkpoint] loaded weights step %d from %s", step, self.save_dir)
        return weights, meta

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def latest_step(self) -> Optional[int]:
        """
        Return the highest complete step number in the save directory, or None.

        A step is considered complete only when all three files exist:
          model_step_{step}.safetensors
          optim_step_{step}.pt
          meta_step_{step}.json
        This prevents resuming from a partially-written checkpoint.
        """
        steps = self._list_steps()
        if not steps:
            return None
        # Return the latest step that has all three files
        for step in sorted(steps, reverse=True):
            if self._checkpoint_complete(step):
                return step
        return None

    def list_checkpoints(self) -> list:
        """Return all complete checkpoint step numbers, sorted ascending."""
        return sorted(s for s in self._list_steps() if self._checkpoint_complete(s))

    def delete_checkpoint(self, step: int) -> None:
        """Remove all files for a given checkpoint step."""
        for pattern in [
            f"model_step_{step}.safetensors",
            f"optim_step_{step}.pt",
            f"meta_step_{step}.json",
        ]:
            p = self.save_dir / pattern
            if p.exists():
                p.unlink()
        logger.info("[checkpoint] deleted step %d", step)

    def keep_last_n(self, n: int) -> None:
        """
        Delete all but the `n` most recent complete checkpoints.

        Safe to call after every save to prevent unbounded disk growth.
        """
        complete = self.list_checkpoints()
        for step in complete[:-n]:
            self.delete_checkpoint(step)

    # ──────────────────────────────────────────────────────────────────────
    # Internal atomic write helpers
    # ──────────────────────────────────────────────────────────────────────

    def _atomic_save_safetensors(
        self, state: dict, path: Path
    ) -> None:
        """Write a state dict as safetensors atomically via temp+rename."""
        # safetensors requires all tensors to be contiguous
        contiguous = {k: v.contiguous() for k, v in state.items()}
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=".safetensors.tmp")
        os.close(fd)
        try:
            save_file(contiguous, tmp)
            os.replace(tmp, path)   # atomic on POSIX
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _atomic_save_torch(self, obj, path: Path) -> None:
        """Pickle an object via torch.save atomically via temp+rename."""
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=".pt.tmp")
        os.close(fd)
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _atomic_save_json(self, obj: dict, path: Path) -> None:
        """Write a JSON file atomically via temp+rename."""
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=".json.tmp")
        os.close(fd)
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2, default=_json_default)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _list_steps(self) -> list:
        """Return all step numbers that have a safetensors file."""
        steps = []
        for p in self.save_dir.glob("model_step_*.safetensors"):
            try:
                steps.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        return steps

    def _checkpoint_complete(self, step: int) -> bool:
        """True iff all required files exist for this step."""
        return all(
            (self.save_dir / name).exists()
            for name in [
                f"model_step_{step}.safetensors",
                f"optim_step_{step}.pt",
                f"meta_step_{step}.json",
            ]
        )


def _json_default(obj):
    """JSON serialiser for types that json.dump cannot handle natively."""
    if isinstance(obj, torch.Tensor):
        # Scheduler state dicts may contain small tensors
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")