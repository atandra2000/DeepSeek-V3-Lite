# utils/distributed.py
"""
Distributed training setup and collective communication utilities.

All functions are safe to call in single-process (world_size=1) mode —
they either no-op or return sensible defaults when dist is not initialised.
"""
import datetime
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Default NCCL timeout — 10 minutes is more aggressive than the PyTorch
# default (30 min), surfacing hung ranks faster without being too short
# for legitimate large-model all-reduces.
_DEFAULT_TIMEOUT = datetime.timedelta(minutes=10)


def setup_distributed(
    timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
) -> Tuple[int, int, int]:
    """
    Initialise the NCCL process group from standard torchrun environment vars.

    Validates:
      • LOCAL_RANK >= 0
      • rank < world_size
      • CUDA is available before attempting NCCL init (not after)

    Args:
        timeout: NCCL collective timeout (default 10 minutes).
                 Shorter values surface hung ranks faster.

    Returns:
        (world_size, rank, local_rank)
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {local_rank}")

    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"RANK ({rank}) must be in [0, world_size={world_size})"
        )

    if world_size > 1:
        # Check CUDA availability BEFORE init_process_group — NCCL requires
        # CUDA and will produce a cryptic error if it is not available.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available but world_size > 1 requires NCCL.  "
                "Check your CUDA installation or use a single-GPU setup."
            )

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

        logger.info(
            "Distributed: rank=%d / world_size=%d, local_rank=%d, "
            "device=cuda:%d, timeout=%s",
            rank, world_size, local_rank, local_rank, timeout,
        )

    return world_size, rank, local_rank


def cleanup_distributed() -> None:
    """
    Tear down the process group cleanly.

    Safe to call even if dist was never initialised or has already been
    destroyed.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master() -> bool:
    """
    True iff this process is global rank 0.

    Uses dist.get_rank() when the process group is initialised (authoritative),
    falling back to the RANK environment variable for pre-init checks.
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    # Pre-init fallback (e.g. during argument parsing before setup_distributed)
    return int(os.environ.get("RANK", 0)) == 0


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return the world size (1 if not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce sum across ranks, then divide by world_size to get the mean.

    Returns a NEW tensor — does not modify the input in-place.
    This is safer than in-place mutation when the caller may inspect the
    original tensor after the call.

    Args:
        tensor: any floating-point tensor on a CUDA device

    Returns:
        A new tensor with the global mean value.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor.clone()
    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
    return result


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce sum across ranks.  Returns a new tensor.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor.clone()
    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def barrier() -> None:
    """
    Synchronise all ranks.  No-op if dist is not initialised.
    """
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    """
    Broadcast a Python object from rank `src` to all other ranks.

    Useful for sharing config dicts, step numbers, or other small objects
    without serialising them to a file.

    Args:
        obj:  the object to broadcast (only meaningful on rank `src`)
        src:  source rank

    Returns:
        The broadcast object on all ranks.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]


def reduce_dict(
    input_dict: dict,
    average: bool = True,
) -> dict:
    """
    Reduce a dict of scalar tensors across all ranks.

    Useful for logging metrics that are computed per-rank (e.g. loss,
    accuracy) so that rank 0 always logs the globally averaged value.

    Args:
        input_dict: dict mapping str → scalar torch.Tensor
        average:    if True, divide by world_size after the all-reduce

    Returns:
        dict with the same keys and reduced values (new tensors).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return {k: v.clone() for k, v in input_dict.items()}

    # Stack into one tensor for a single all-reduce call
    keys   = list(input_dict.keys())
    values = torch.stack([input_dict[k].float() for k in keys])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values.div_(dist.get_world_size())

    return {k: values[i] for i, k in enumerate(keys)}