# kernels/fp8_kernel.py
"""
FP8 quantisation and GEMM Triton kernels.

Scale tensor conventions
------------------------
Activation scales  : shape (total_blocks,) where total_blocks = numel // BLOCK_SIZE.
                     Stored and passed as a flat 1-D tensor; callers reshape as needed.
Weight scales      : shape (M_blocks, N_blocks) = (M // BLOCK_SIZE, N // BLOCK_SIZE)
                     for a 2-D weight of shape (M, N).

FP8 format
----------
All kernels target FP8 E4M3FN (tl.float8e4m3fn).
Max representable value: 448.0  (FP8_E4M3_MAX).

UE8M0 scale format
------------------
DeepSeek-V3 uses power-of-two scales (UE8M0) for hardware alignment.
When scale_fmt="ue8m0", scales are rounded up to the next power of two
via ceil(log2(s)).  This is implemented in a separate kernel variant
(act_quant_ue8m0_kernel) rather than a runtime branch inside a single
kernel, so both variants compile cleanly without libdevice conditionals.
"""
import torch
import triton
import triton.language as tl
from typing import Tuple


FP8_E4M3_MAX: float = 448.0


# ── Activation quantisation kernels ───────────────────────────────────────────

@triton.jit
def act_quant_kernel(
    x_ptr, y_ptr, s_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Block-wise activation quantisation → FP8 E4M3FN with float32 scales.

    Each program instance handles one contiguous block of BLOCK_SIZE elements.
    Boundary blocks (last block when n_elements % BLOCK_SIZE != 0) are handled
    with a mask so the kernel is safe for any n_elements.

    Scale = amax(|x|) / FP8_E4M3_MAX, stored as float32 at s_ptr[pid].
    """
    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-12)   # prevent division by zero
    s    = amax / 448.0              # FP8_E4M3_MAX = 448.0

    y = tl.clamp(x / s, -448.0, 448.0).to(tl.float8e4m3fn)

    tl.store(y_ptr + offs, y,  mask=mask)
    tl.store(s_ptr  + pid, s)


@triton.jit
def act_quant_ue8m0_kernel(
    x_ptr, y_ptr, s_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Block-wise activation quantisation → FP8 E4M3FN with UE8M0 (power-of-two) scales.

    Identical to act_quant_kernel but rounds the scale up to the nearest
    power of two: s = 2^ceil(log2(amax / 448)).  Separate kernel avoids a
    runtime branch and libdevice conditional inside act_quant_kernel.
    """
    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-12)
    raw_s = amax / 448.0

    # Round up to nearest power-of-two (UE8M0 format)
    exp = tl.math.ceil(tl.math.log2(raw_s))
    s   = tl.math.exp2(exp)

    y = tl.clamp(x / s, -448.0, 448.0).to(tl.float8e4m3fn)

    tl.store(y_ptr + offs, y,  mask=mask)
    tl.store(s_ptr  + pid, s)


# ── Weight dequantisation kernel ───────────────────────────────────────────────

@triton.jit
def weight_dequant_kernel(
    x_ptr, s_ptr, y_ptr,
    M, N,
    M_BLOCKS, N_BLOCKS,          # precomputed: cdiv(M, BS), cdiv(N, BS)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantise a 2-D FP8 weight tensor → BF16.

    x: (M, N) FP8
    s: (M_BLOCKS, N_BLOCKS) float32 scale per tile
    y: (M, N) BF16

    Each program instance handles one (BLOCK_SIZE, BLOCK_SIZE) tile.
    Boundary tiles are masked so the kernel is safe for any (M, N).

    N_BLOCKS is passed explicitly to avoid recomputing cdiv(N, BLOCK_SIZE)
    inside every thread.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offs   = offs_m[:, None] * N + offs_n[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Scale index: row-major (pid_m, pid_n) in the (M_BLOCKS, N_BLOCKS) grid
    s = tl.load(s_ptr + pid_m * N_BLOCKS + pid_n)

    tl.store(y_ptr + offs, (x * s).to(tl.bfloat16), mask=mask)


# ── FP8 GEMM kernel ────────────────────────────────────────────────────────────

@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_s_ptr, b_s_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    K_BLOCKS,                    # precomputed: cdiv(K, BLOCK_K)
    N_BLOCKS,                    # precomputed: cdiv(N, BLOCK_N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP8 × FP8 → BF16 tiled GEMM with per-block scaling and FP32 accumulation.

    Computes C = A @ B where:
      A: (M, K) FP8,  a_s: (M_BLOCKS, K_BLOCKS) float32 scales
      B: (K, N) FP8,  b_s: (K_BLOCKS, N_BLOCKS) float32 scales
      C: (M, N) BF16

    Scale indexing (fixed from original):
      a_s[m_block, k_block]  →  a_s_ptr + pid_m * K_BLOCKS + i
      b_s[k_block, n_block]  →  b_s_ptr + i * N_BLOCKS + pid_n

    B is expected in (K, N) layout — each column is a different output
    neuron, contiguous along K.  The Python wrapper transposes weight
    matrices from the standard (out, in) = (N, K) layout before calling.

    Strides are passed explicitly to support non-contiguous inputs.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer initialisation using explicit strides
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Scale pointer initialisation
    # a_s: (M_BLOCKS, K_BLOCKS) → row = pid_m, col advances with i
    a_s_ptr_cur = a_s_ptr + pid_m * K_BLOCKS
    # b_s: (K_BLOCKS, N_BLOCKS) → row advances with i, col = pid_n
    b_s_ptr_cur = b_s_ptr + pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(K_BLOCKS):
        k_remaining = K - i * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a   = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b   = tl.load(b_ptrs, mask=b_mask, other=0.0)
        a_s = tl.load(a_s_ptr_cur + i)          # scale for this (m_block, k_block)
        b_s = tl.load(b_s_ptr_cur + i * N_BLOCKS)  # b_s[i, pid_n] in (K_BLOCKS, N_BLOCKS) layout

        acc += tl.dot(a.to(tl.float32), b.to(tl.float32)) * a_s * b_s

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs  = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask  = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ── Python wrapper ─────────────────────────────────────────────────────────────

class FP8Quantizer:
    """
    FP8 quantisation and GEMM utilities matching DeepSeek-V3's block-wise scaling.

    Scale layout
    ------------
    quantize_activation returns scales as a flat 1-D tensor of shape
    (total_blocks,) where total_blocks = x.numel() // block_size.
    This matches the flat grid used by the Triton kernel.  Callers that
    need to index by (batch, seq, block) must reshape after the call.

    fp8_gemm expects:
      a_s: (M_blocks, K_blocks) = (M // BS, K // BS)  float32
      b_s: (K_blocks, N_blocks) = (K // BS, N // BS)  float32
    where B has shape (K, N) — already transposed from weight convention.
    """

    FP8_MAX = FP8_E4M3_MAX

    def __init__(self, block_size: int = 128, scale_fmt: str = "e4m3"):
        """
        Args:
            block_size: number of elements per quantisation block (must be a
                        power of two and ≥ 16 for Triton vectorisation).
            scale_fmt:  "e4m3" for standard float32 scales, or
                        "ue8m0" for power-of-two (hardware-aligned) scales.
        """
        if scale_fmt not in ("e4m3", "ue8m0"):
            raise ValueError(f"scale_fmt must be 'e4m3' or 'ue8m0', got {scale_fmt!r}")
        self.block_size = block_size
        self.scale_fmt  = scale_fmt

    def quantize_activation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block-wise quantise x → FP8 E4M3FN.

        Args:
            x: any-shape BF16/FP32 contiguous tensor whose total element
               count is divisible by block_size.

        Returns:
            y: same shape as x, dtype float8_e4m3fn
            s: (total_blocks,) float32 scale tensor, flat.
               total_blocks = x.numel() // block_size.
               Reshape to (*x.shape[:-1], n_blocks) for per-row indexing.
        """
        if not x.is_contiguous():
            x = x.contiguous()

        n_elem = x.numel()
        if n_elem % self.block_size != 0:
            raise ValueError(
                f"Total elements {n_elem} must be divisible by "
                f"block_size {self.block_size}"
            )

        n_blocks = n_elem // self.block_size
        y        = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s        = torch.empty(n_blocks, dtype=torch.float32, device=x.device)

        grid = (n_blocks,)
        if self.scale_fmt == "ue8m0":
            act_quant_ue8m0_kernel[grid](x, y, s, n_elem, BLOCK_SIZE=self.block_size)
        else:
            act_quant_kernel[grid](x, y, s, n_elem, BLOCK_SIZE=self.block_size)

        return y, s

    def dequantize_weight(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantise a 2-D FP8 weight tensor → BF16.

        Args:
            x: (M, N) float8_e4m3fn
            s: (M // block_size, N // block_size) float32

        Returns:
            (M, N) bfloat16
        """
        if not x.is_contiguous():
            x = x.contiguous()
        if not s.is_contiguous():
            s = s.contiguous()
        if x.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Expected float8_e4m3fn weight, got {x.dtype}")
        if x.dim() != 2 or s.dim() != 2:
            raise ValueError(f"Expected 2-D tensors; x={x.shape}, s={s.shape}")

        M, N     = x.shape
        M_blocks = triton.cdiv(M, self.block_size)
        N_blocks = triton.cdiv(N, self.block_size)

        if s.shape != (M_blocks, N_blocks):
            raise ValueError(
                f"Scale shape {s.shape} does not match expected "
                f"({M_blocks}, {N_blocks}) for weight ({M}, {N}) "
                f"with block_size={self.block_size}"
            )

        y    = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        grid = (M_blocks, N_blocks)
        weight_dequant_kernel[grid](
            x, s, y, M, N, M_blocks, N_blocks,
            BLOCK_SIZE=self.block_size,
        )
        return y

    def fp8_gemm(
        self,
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        FP8 × FP8 → BF16 tiled GEMM.

        Args:
            a:   (M, K) float8_e4m3fn  — input activations
            a_s: (M // BS, K // BS) float32  — per-block scales for a
            b:   (K, N) float8_e4m3fn  — weight matrix, transposed to (K, N)
                 Caller must transpose from the standard (N, K) layout.
            b_s: (K // BS, N // BS) float32  — per-block scales for b

        Returns:
            (M, N) bfloat16
        """
        if a.dtype != torch.float8_e4m3fn:
            raise ValueError(f"a must be float8_e4m3fn, got {a.dtype}")
        if b.dtype != torch.float8_e4m3fn:
            raise ValueError(f"b must be float8_e4m3fn, got {b.dtype}")
        if not (a.is_contiguous() and b.is_contiguous()):
            a = a.contiguous()
            b = b.contiguous()

        K    = a.size(-1)
        M    = a.numel() // K
        N    = b.size(1)
        BS   = self.block_size

        K_blocks = triton.cdiv(K, BS)
        M_blocks = triton.cdiv(M, BS)
        N_blocks = triton.cdiv(N, BS)

        if a_s.shape != (M_blocks, K_blocks):
            raise ValueError(
                f"a_s shape {a_s.shape} != expected ({M_blocks}, {K_blocks})"
            )
        if b_s.shape != (K_blocks, N_blocks):
            raise ValueError(
                f"b_s shape {b_s.shape} != expected ({K_blocks}, {N_blocks})"
            )

        c    = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
        BM, BN, BK = 64, 64, BS

        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        fp8_gemm_kernel[grid](
            a, b, c,
            a_s, b_s,
            M, N, K,
            a.stride(0), a.stride(1),   # stride_am, stride_ak
            b.stride(0), b.stride(1),   # stride_bk, stride_bn
            K_blocks,
            N_blocks,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        return c
