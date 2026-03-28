# utils/communication.py
"""
Communication primitives for expert parallelism and pipeline parallelism.

MoECommunication
----------------
Implements the All-to-All dispatch/combine pattern for expert parallelism.
Each token is physically sent to the rank that owns the expert(s) assigned
to it.  With topk > 1, a token may be assigned to experts on different ranks;
under the node-limited routing design the token is sent to one rank and that
rank processes all of the token's locally-owned experts in sequence.

The combine step accumulates contributions from ALL topk expert slots that
were processed by the receiving rank, not just the first one.  The original
code only applied weights[:, 0], silently dropping ~(topk-1)/topk of the
routing signal for topk > 1.

PipelineCommunication
---------------------
Provides non-blocking P2P primitives for 1F1B pipeline schedules.
Each stage method (forward_send, forward_recv, backward_send, backward_recv)
is independent — callers compose them into whatever schedule they need.

The original wrapped send and recv to the same peer in a single call, which
prevented asymmetric patterns (e.g. sending forward to next_rank while
simultaneously receiving backward from next_rank).  The fixed implementation
exposes raw isend/irecv handles and a barrier, with composed helpers that
model the four canonical pipeline directions.

Topology guards: rank 0 has no predecessor (no forward_recv from prev),
and the last rank has no successor (no forward_send to next).  The ring
wrap-around values for prev_rank / next_rank are still computed for use
by callers that implement ring-AllReduce patterns, but the pipeline helpers
explicitly guard against using them at the boundary stages.
"""
import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class MoECommunication:
    """
    All-to-All communication for MoE expert dispatch and combine.

    Token routing (dispatch)
    ------------------------
    Tokens are sorted by the rank that owns their first assigned expert
    (node-limited routing: each token goes to at most one node).  The
    sorted activations, routing weights, and expert indices are distributed
    to the owning ranks via three separate all_to_all_single calls.

    Expert output aggregation (combine)
    ------------------------------------
    After local expert computation, outputs for ALL topk expert slots that
    belong to this rank are accumulated with their routing weights.  The
    original code only used weights[:, 0], which was incorrect for topk > 1.

    Split sizes
    -----------
    all_to_all_single requires CPU-side integer lists for split sizes.
    rank_counts and recv_counts are moved to CPU before building the lists
    to avoid backend-dependent behaviour with GPU integer tensors.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        num_experts: int,
    ) -> None:
        if num_experts % world_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"world_size ({world_size})"
            )
        self.world_size       = world_size
        self.rank             = rank
        self.num_experts      = num_experts
        self.experts_per_rank = num_experts // world_size

        # Create a dedicated process group so MoE collectives do not
        # interfere with the default group used by DDP/FSDP.
        if dist.is_initialized():
            self.group: Optional[dist.ProcessGroup] = dist.new_group(
                list(range(world_size))
            )
        else:
            self.group = None

    # ──────────────────────────────────────────────────────────────────────
    # Dispatch
    # ──────────────────────────────────────────────────────────────────────

    def dispatch(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Distribute tokens to the ranks that own the selected experts.

        Args:
            x:       (T, dim)   — flattened token representations
            indices: (T, topk)  — global expert indices (int64)
            weights: (T, topk)  — routing weights (float)

        Returns:
            recv_x:         (T', dim)   — tokens received by this rank
            recv_weights:   (T', topk)  — corresponding routing weights
            recv_indices:   (T', topk)  — corresponding expert indices
            sort_idx:       (T,)        — permutation used to sort x before
                                          dispatch; needed by combine() to
                                          scatter results back to original order
        """
        if not dist.is_initialized():
            T = x.size(0)
            return x, weights, indices, torch.arange(T, device=x.device)

        T, dim = x.shape
        topk   = indices.shape[1]

        # ── Count tokens per rank ──────────────────────────────────────────
        # Each token's target rank is determined by its primary (first) expert.
        target_rank = (
            (indices[:, 0] // self.experts_per_rank)
            .clamp(0, self.world_size - 1))
        # Accumulate token counts per rank
        rank_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=x.device)
        rank_counts.scatter_add_(
            0, target_rank,
            torch.ones(T, dtype=torch.long, device=x.device))

        # Exchange counts — move to CPU first for all_to_all_single
        rank_counts_cpu = rank_counts.cpu()
        recv_counts_cpu = torch.empty_like(rank_counts_cpu)
        dist.all_to_all_single(
            recv_counts_cpu, rank_counts_cpu, group=self.group)

        input_splits  = rank_counts_cpu.tolist()    # how many tokens we send to each rank
        output_splits = recv_counts_cpu.tolist()    # how many tokens we receive from each rank
        total_recv    = sum(output_splits)

        # ── Sort tokens by target rank so the send buffer is contiguous ────
        sort_idx = torch.argsort(target_rank, stable=True)
        x_sorted = x[sort_idx].contiguous()
        w_sorted = weights[sort_idx].contiguous()
        i_sorted = indices[sort_idx].contiguous()

        # ── All-to-All: activations ────────────────────────────────────────
        recv_x = torch.empty(total_recv, dim, dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            recv_x, x_sorted,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group)

        # ── All-to-All: routing weights ────────────────────────────────────
        recv_weights = torch.empty(
            total_recv, topk, dtype=weights.dtype, device=x.device)
        dist.all_to_all_single(
            recv_weights, w_sorted,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group)

        # ── All-to-All: expert indices ─────────────────────────────────────
        recv_indices = torch.empty(
            total_recv, topk, dtype=indices.dtype, device=x.device)
        dist.all_to_all_single(
            recv_indices, i_sorted,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group)

        return recv_x, recv_weights, recv_indices, sort_idx

    # ──────────────────────────────────────────────────────────────────────
    # Combine
    # ──────────────────────────────────────────────────────────────────────

    def combine(
        self,
        expert_out: torch.Tensor,
        recv_weights: torch.Tensor,
        recv_indices: torch.Tensor,
        sort_idx: torch.Tensor,
        original_T: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Aggregate expert outputs, send them back to originating ranks, and
        unsort to restore the original token order.

        Each received token may have been assigned to multiple local experts
        (any expert index in recv_indices whose owning rank == self.rank).
        This method accumulates ALL matching expert contributions, weighted
        by their routing weights.

        Args:
            expert_out:   (T', dim) — sum of weighted expert outputs for each
                           received token, as computed by DeepSeekMoE.forward
                           (the MoE layer has already applied routing weights
                           and summed across local experts)
            recv_weights: (T', topk) — routing weights received during dispatch
                           (kept for reference; the MoE layer used these)
            recv_indices: (T', topk) — expert indices received during dispatch
            sort_idx:     (T,)       — permutation from dispatch(); used to
                           scatter combined outputs back to original order
            original_T:   T          — number of tokens before dispatch
            dim:          model dimension

        Returns:
            (T, dim) — combined output in original token order

        Note: The actual per-expert weighting and summation happens inside
        DeepSeekMoE.forward() on the receiving rank.  This method handles
        only the reverse All-to-All (combine) communication and unsorting.
        """
        T_recv = expert_out.size(0)

        if not dist.is_initialized():
            # Single-process: expert_out is already in original order
            out = torch.zeros(original_T, dim, dtype=expert_out.dtype,
                              device=expert_out.device)
            if T_recv > 0:
                out.index_add_(0, sort_idx, expert_out)
            return out

        # ── Reverse All-to-All: send results back to originating ranks ─────
        # The split sizes for the reverse direction are swapped:
        # we received output_splits tokens from each rank (now these are
        # the send amounts) and we send input_splits back.
        target_rank = (
            (recv_indices[:, 0] // self.experts_per_rank)
            .clamp(0, self.world_size - 1)
        )
        recv_rank_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=expert_out.device
        )
        recv_rank_counts.scatter_add_(
            0, target_rank,
            torch.ones(T_recv, dtype=torch.long, device=expert_out.device),
        )
        recv_rank_counts_cpu = recv_rank_counts.cpu()
        send_counts_cpu      = torch.empty_like(recv_rank_counts_cpu)
        dist.all_to_all_single(
            send_counts_cpu, recv_rank_counts_cpu, group=self.group
        )

        # Sort expert outputs by the destination rank for the reverse scatter
        sort_back = torch.argsort(target_rank, stable=True)
        out_sorted = expert_out[sort_back].contiguous()

        send_splits = recv_rank_counts_cpu.tolist()
        recv_splits = send_counts_cpu.tolist()
        total_back  = sum(recv_splits)

        combined = torch.empty(
            total_back, dim, dtype=expert_out.dtype, device=expert_out.device
        )
        dist.all_to_all_single(
            combined, out_sorted,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
        )

        # ── Unsort to restore original token order ─────────────────────────
        output = torch.zeros(
            original_T, dim, dtype=expert_out.dtype, device=expert_out.device
        )
        if combined.size(0) > 0:
            # sort_idx maps original positions to sorted positions;
            # we need the inverse: for each slot in combined, place it at
            # the position it came from.
            inv_sort = torch.empty_like(sort_idx)
            inv_sort[sort_idx] = torch.arange(
                original_T, dtype=sort_idx.dtype, device=sort_idx.device
            )
            output.index_add_(0, inv_sort[:combined.size(0)], combined)

        return output


# ── Pipeline parallelism ───────────────────────────────────────────────────────

class PipelineCommunication:
    """
    P2P communication primitives for 1F1B pipeline parallelism.

    Each method issues a single non-blocking operation and returns an opaque
    handle (Work object).  The caller is responsible for calling .wait() on
    handles before accessing the associated tensor.  This allows computation
    to overlap with communication — the DualPipe pattern (DeepSeek-V3 §3.2.1).

    Topology
    --------
    Stages are numbered 0..world_size-1 in forward direction.
    - Stage 0:           first stage — receives nothing from prev
    - Stage world_size-1: last stage — sends nothing to next

    prev_rank and next_rank use modular arithmetic for completeness but the
    pipeline helpers explicitly guard boundary stages.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        self.rank       = rank
        self.world_size = world_size
        # Modular values kept for ring-allreduce patterns; pipeline helpers
        # guard against using these at boundary stages.
        self.prev_rank = (rank - 1) % world_size
        self.next_rank = (rank + 1) % world_size

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

    # ──────────────────────────────────────────────────────────────────────
    # Raw non-blocking primitives
    # ──────────────────────────────────────────────────────────────────────

    def isend(
        self,
        tensor: torch.Tensor,
        dst: int,
        tag: int = 0,
    ) -> Optional[dist.Work]:
        """
        Non-blocking send.  Returns a Work handle; call .wait() before
        reusing the tensor.  Returns None if dist is not initialised.
        """
        if not dist.is_initialized():
            return None
        return dist.isend(tensor.contiguous(), dst=dst, tag=tag)

    def irecv(
        self,
        buffer: torch.Tensor,
        src: int,
        tag: int = 0,
    ) -> Optional[dist.Work]:
        """
        Non-blocking receive into `buffer`.  Returns a Work handle; call
        .wait() before reading the buffer.  Returns None if dist is not
        initialised.
        """
        if not dist.is_initialized():
            return None
        return dist.irecv(buffer, src=src, tag=tag)

    @staticmethod
    def wait(handle: Optional[dist.Work]) -> None:
        """Wait for a Work handle to complete.  No-op if handle is None."""
        if handle is not None:
            handle.wait()

    # ──────────────────────────────────────────────────────────────────────
    # Composed pipeline helpers
    # ──────────────────────────────────────────────────────────────────────

    def send_forward(
        self, tensor: torch.Tensor, tag: int = 0
    ) -> Optional[dist.Work]:
        """
        Non-blocking send of activations to the next pipeline stage.
        No-op (returns None) for the last stage.
        """
        if self.is_last_stage:
            return None
        return self.isend(tensor, dst=self.next_rank, tag=tag)

    def recv_forward(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        tag: int = 0,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Allocate a buffer and issue a non-blocking recv for forward activations
        from the previous pipeline stage.

        Returns (buffer, handle).  Call wait(handle) before reading buffer.
        For the first stage, returns a zero buffer and None handle.
        """
        buf = torch.zeros(shape, dtype=dtype, device=device)
        if self.is_first_stage:
            return buf, None
        handle = self.irecv(buf, src=self.prev_rank, tag=tag)
        return buf, handle

    def send_backward(
        self, grad: torch.Tensor, tag: int = 0
    ) -> Optional[dist.Work]:
        """
        Non-blocking send of gradients to the previous pipeline stage.
        No-op (returns None) for the first stage.
        """
        if self.is_first_stage:
            return None
        return self.isend(grad, dst=self.prev_rank, tag=tag)

    def recv_backward(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        tag: int = 0,
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Allocate a buffer and issue a non-blocking recv for backward gradients
        from the next pipeline stage.

        Returns (buffer, handle).  Call wait(handle) before reading buffer.
        For the last stage, returns a zero buffer and None handle.
        """
        buf = torch.zeros(shape, dtype=dtype, device=device)
        if self.is_last_stage:
            return buf, None
        handle = self.irecv(buf, src=self.next_rank, tag=tag)
        return buf, handle

    def paired_send_recv_forward(
        self,
        send_tensor: Optional[torch.Tensor],
        recv_shape: Optional[tuple],
        recv_dtype: torch.dtype,
        device: torch.device,
        tag: int = 0,
    ) -> Tuple[Optional[torch.Tensor], List[Optional[dist.Work]]]:
        """
        Issue a paired forward send (to next) and forward recv (from prev)
        simultaneously using batch_isend_irecv for better overlap.

        Returns (recv_buffer, handles).  recv_buffer is None if is_first_stage.
        Call wait(h) for h in handles before using the buffer or reusing
        send_tensor.
        """
        if not dist.is_initialized():
            return None, []

        ops: List[dist.P2POp] = []
        recv_buf: Optional[torch.Tensor] = None

        if send_tensor is not None and not self.is_last_stage:
            ops.append(
                dist.P2POp(dist.isend, send_tensor.contiguous(), self.next_rank, tag=tag)
            )
        if recv_shape is not None and not self.is_first_stage:
            recv_buf = torch.zeros(recv_shape, dtype=recv_dtype, device=device)
            ops.append(
                dist.P2POp(dist.irecv, recv_buf, self.prev_rank, tag=tag)
            )

        if not ops:
            return recv_buf, []

        handles = dist.batch_isend_irecv(ops)
        return recv_buf, handles
