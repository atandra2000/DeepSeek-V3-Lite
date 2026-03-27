# kernels/gemm.py
"""
FP8 linear layer: PyTorch-level dispatch integrating the Triton kernels
from fp8_kernel.py.

Design
------
FP8Linear is an nn.Module that stores weights in FP8 E4M3FN format and
maintains a precomputed per-block weight scale tensor.  The scale is computed
once at construction / weight assignment time — not on every forward call.

Forward path
  1. Quantise activations block-wise → FP8 + activation scales.
  2. Call the Triton FP8 GEMM kernel (A @ B with per-block scales, FP32 acc).
  3. Add bias (in BF16) if present.

Backward path
  Straight-through: activations are quantised on the forward pass; gradients
  flow through as if the quantisation did not happen (STE).  Weight gradients
  are computed in BF16/FP32 from the dequantised weight.

Fallback
  When Triton is not available (CPU, non-CUDA device, import error), the
  module falls back to dequantising the weight to BF16 and calling F.linear.
  This path is always numerically correct and is used for testing / debugging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from .fp8_kernel import FP8Quantizer, FP8_E4M3_MAX
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    FP8_E4M3_MAX = 448.0


class FP8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear with FP8-compressed weights.

    Weight storage
    --------------
    `self.weight` is stored as float8_e4m3fn, shape (out_features, in_features).
    `self.weight_scale` is a float32 tensor of shape
    (out_features // block_size, in_features // block_size) — precomputed
    at construction and updated whenever weights are replaced via `load_fp8`.

    Forward dispatch
    ----------------
    On CUDA with Triton available: uses FP8Quantizer.fp8_gemm (Triton kernel).
    Otherwise: dequantises weight → BF16, calls F.linear.

    Usage
    -----
    # From a pretrained BF16 linear layer:
    fp8_layer = FP8Linear.from_linear(linear_layer, block_size=128)

    # From scratch:
    fp8_layer = FP8Linear(in_features=2048, out_features=4096, block_size=128)
    """

    FP8_MAX = FP8_E4M3_MAX

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 128,
        scale_fmt: str = "e4m3",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.block_size   = block_size
        self.scale_fmt    = scale_fmt

        if in_features % block_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"block_size ({block_size})"
            )
        if out_features % block_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"block_size ({block_size})"
            )

        n_row_blocks = out_features // block_size
        n_col_blocks = in_features  // block_size

        # FP8 weight — not a Parameter so the optimiser never touches it.
        # Weights are stored quantised; gradients are accumulated in a
        # higher-precision master weight held by the optimiser.
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn,
                        device=device),
        )
        # Precomputed per-block scale — updated whenever weight is replaced.
        self.register_buffer(
            "weight_scale",
            torch.ones(n_row_blocks, n_col_blocks, dtype=torch.float32,
                       device=device),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self._quantizer = FP8Quantizer(block_size=block_size, scale_fmt=scale_fmt) \
            if _TRITON_AVAILABLE else None

    # ──────────────────────────────────────────────────────────────────────
    # Weight management
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def load_fp8(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        """
        Replace the stored weight with a pre-quantised FP8 tensor and its scale.

        Args:
            weight_fp8:   (out_features, in_features) float8_e4m3fn
            weight_scale: (out_features // block_size, in_features // block_size) float32
        """
        if weight_fp8.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected weight shape ({self.out_features}, {self.in_features}), "
                f"got {weight_fp8.shape}"
            )
        self.weight.copy_(weight_fp8)
        self.weight_scale.copy_(weight_scale)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 128,
        scale_fmt: str = "e4m3",
    ) -> "FP8Linear":
        """
        Construct an FP8Linear by quantising an existing nn.Linear weight.

        The quantised weight and precomputed scale are stored; the original
        BF16/FP32 weight tensor is not retained.
        """
        out_f, in_f = linear.weight.shape
        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=linear.bias is not None,
            block_size=block_size,
            scale_fmt=scale_fmt,
            device=linear.weight.device,
        )
        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.detach().to(torch.bfloat16))

        # Quantise the weight row-by-row so scale shape is (out_blocks, in_blocks).
        # Each row of out_features is treated as a flat (in_features,) vector;
        # we need (out_features // BS, in_features // BS) scales.
        # The cleanest way: reshape to (out_features * in_features,), quantise,
        # then fold the flat scale into (out_blocks, in_blocks).
        if _TRITON_AVAILABLE:
            q = FP8Quantizer(block_size=block_size, scale_fmt=scale_fmt)
            w_flat = linear.weight.detach().float().contiguous().view(-1)
            w_q, w_s_flat = q.quantize_activation(w_flat)
            n_col_blocks = in_f  // block_size
            n_row_blocks = out_f // block_size
            # w_s_flat: (out_features * in_features // BS,) →
            #           (out_features // BS, in_features // BS)
            # Each row of out_features contributes in_features // BS scale entries.
            # Rows in the flat scale correspond to blocks of in_features within each
            # output row, ordered as (row0_block0, row0_block1, ..., row1_block0, ...).
            # But quantize_activation flattens the whole weight and assigns one scale
            # per BLOCK_SIZE elements globally.  The first n_col_blocks scales cover
            # row 0, the next n_col_blocks cover row 1, and so on — only when
            # in_features is a multiple of block_size, which we've already checked.
            w_s = w_s_flat.view(n_row_blocks, n_col_blocks)
            layer.load_fp8(w_q.view(out_f, in_f), w_s)
        else:
            # Fallback: compute per-block scales in PyTorch
            w = linear.weight.detach().float()
            n_row_blocks = out_f // block_size
            n_col_blocks = in_f  // block_size
            w_blocks = w.view(n_row_blocks, block_size, n_col_blocks, block_size)
            amax = w_blocks.abs().amax(dim=(1, 3))         # (n_row_blocks, n_col_blocks)
            scale = (amax / FP8_E4M3_MAX).clamp(min=1e-12)
            w_scaled = (w_blocks / scale.unsqueeze(1).unsqueeze(3)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            w_q = w_scaled.view(out_f, in_f).to(torch.float8_e4m3fn)
            layer.load_fp8(w_q, scale)

        return layer

    def dequantize_weight(self) -> torch.Tensor:
        """
        Return the weight dequantised to BF16.  Used in the fallback path
        and for debugging.  Does not cache the result.
        """
        if _TRITON_AVAILABLE and self.weight.is_cuda:
            q = FP8Quantizer(self.block_size, self.scale_fmt)
            return q.dequantize_weight(self.weight, self.weight_scale)
        # Pure PyTorch fallback
        w = self.weight.float()
        n_row_blocks = self.out_features // self.block_size
        n_col_blocks = self.in_features  // self.block_size
        s = self.weight_scale.view(n_row_blocks, 1, n_col_blocks, 1)
        w = w.view(n_row_blocks, self.block_size, n_col_blocks, self.block_size)
        return (w * s).view(self.out_features, self.in_features).to(torch.bfloat16)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = x @ weight.T + bias

        FP8 path (CUDA + Triton): quantise x, call fp8_gemm kernel.
        Fallback path: dequantise weight → BF16, call F.linear.
        """
        use_triton = (
            _TRITON_AVAILABLE
            and x.is_cuda
            and x.numel() % self.block_size == 0
        )

        if use_triton:
            return self._forward_triton(x)
        return self._forward_fallback(x)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """Triton FP8 GEMM path."""
        q          = FP8Quantizer(self.block_size, self.scale_fmt)
        x_orig_shape = x.shape

        # Flatten leading dims: (*, in_features) → (M, in_features)
        M          = x.numel() // self.in_features
        x_2d       = x.reshape(M, self.in_features)
        x_q, x_s   = q.quantize_activation(x_2d.float().contiguous())

        K_blocks   = self.in_features  // self.block_size
        # x_s is (total_act_blocks,) = (M * K // BS,).
        # M_blocks = total_act_blocks // K_blocks avoids cdiv mismatch.
        total_act_blocks = x_s.numel()
        M_blocks   = total_act_blocks // K_blocks
        x_s_2d     = x_s.view(M_blocks, K_blocks)

        # Kernel expects B in (K, N) layout; weight is (N, K) = (out, in)
        w_t        = self.weight.t().contiguous()           # (in, out) = (K, N)
        w_s_t      = self.weight_scale.t().contiguous()     # (K_blocks, N_blocks)

        out        = q.fp8_gemm(x_q, x_s_2d, w_t, w_s_t)  # (M, out_features)

        # Restore leading dims
        out        = out.view(*x_orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantise → BF16 → F.linear fallback."""
        w_bf16 = self.dequantize_weight()
        out    = F.linear(x.to(torch.bfloat16), w_bf16)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, "
            f"scale_fmt={self.scale_fmt!r}"
        )


# ── Backward: straight-through estimator ──────────────────────────────────────

def _fp8_dequant_to_float(weight_fp8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantise a block-quantised FP8 weight to float32 in pure PyTorch.
    No nn.Module construction — safe to call from autograd Function.
    """
    out_features, in_features = weight_fp8.shape
    n_rb = weight_scale.shape[0]
    n_cb = weight_scale.shape[1]
    bs   = in_features // n_cb   # block_size along in_features dim
    w    = weight_fp8.float()
    s    = weight_scale.view(n_rb, 1, n_cb, 1)
    return (w.view(n_rb, bs, n_cb, bs) * s).view(out_features, in_features)


class _FP8LinearSTE(torch.autograd.Function):
    """
    Autograd function for FP8Linear with straight-through gradient estimation.

    Forward: quantise activations, call FP8GEMM (Triton) or fallback.
    Backward: pass gradients through as if no quantisation occurred (STE).
              Weight gradient computed from dequantised weight in float32.

    Avoids constructing nn.Module instances (which require __init__) inside
    autograd Functions — instead calls computation helpers directly.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        block_size: int,
        scale_fmt: str,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight_fp8, weight_scale)
        ctx.block_size = block_size
        ctx.scale_fmt  = scale_fmt
        ctx.has_bias   = bias is not None

        out_features = weight_fp8.shape[0]
        in_features  = weight_fp8.shape[1]

        use_triton = (
            _TRITON_AVAILABLE and x.is_cuda
            and x.numel() % block_size == 0
        )

        if use_triton:
            q        = FP8Quantizer(block_size, scale_fmt)
            M        = x.numel() // in_features
            x_q, x_s = q.quantize_activation(x.reshape(M, in_features).float().contiguous())
            K_blocks  = in_features // block_size
            M_blocks  = x_s.numel() // K_blocks
            x_s_2d    = x_s.view(M_blocks, K_blocks)
            w_t       = weight_fp8.t().contiguous()
            w_s_t     = weight_scale.t().contiguous()
            out       = q.fp8_gemm(x_q, x_s_2d, w_t, w_s_t).view(*x.shape[:-1], out_features)
        else:
            w_dq = _fp8_dequant_to_float(weight_fp8, weight_scale).to(torch.bfloat16)
            out  = F.linear(x.to(torch.bfloat16), w_dq).to(x.dtype)

        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight_fp8, weight_scale = ctx.saved_tensors
        block_size = ctx.block_size

        # Dequantise weight to float32 for gradient computation
        w_fp32 = _fp8_dequant_to_float(weight_fp8, weight_scale)

        # grad_x = grad_output @ W
        # grad_output: (*, out), w_fp32: (out, in)
        # (*, out) @ (out, in) → (*, in)  — no transpose needed
        grad_x = grad_output.float().matmul(w_fp32)

        # grad_W = grad_output^T @ x
        # Flatten to 2D: (M, out)^T @ (M, in) = (out, in)
        M           = x.numel() // x.shape[-1]
        grad_flat   = grad_output.reshape(M, -1)       # (M, out)
        x_flat      = x.reshape(M, -1).float()         # (M, in)
        grad_weight = grad_flat.mT.matmul(x_flat)      # (out, in)

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias else None
        )

        return (
            grad_x.to(x.dtype),
            grad_weight.to(weight_fp8.dtype),
            None,   # weight_scale — no gradient
            grad_bias,
            None,   # block_size
            None,   # scale_fmt
        )


# ── Convenience factory ────────────────────────────────────────────────────────

def replace_linear_with_fp8(
    model: torch.nn.Module,
    block_size: int = 128,
    scale_fmt: str = "e4m3",
    skip_modules: tuple = (),
) -> torch.nn.Module:
    """
    Recursively replace all nn.Linear layers in `model` with FP8Linear.

    Args:
        model:        the model to modify in-place
        block_size:   FP8 quantisation block size
        scale_fmt:    "e4m3" or "ue8m0"
        skip_modules: tuple of module name substrings to leave as nn.Linear
                      (e.g. ("head", "embed") to skip the LM head and embeddings)

    Returns:
        The modified model (same object, modified in-place).
    """
    for name, module in list(model.named_children()):
        if any(skip in name for skip in skip_modules):
            continue
        if isinstance(module, nn.Linear):
            # Only replace if dimensions are block_size-aligned
            if (
                module.in_features  % block_size == 0
                and module.out_features % block_size == 0
            ):
                setattr(model, name, FP8Linear.from_linear(module, block_size, scale_fmt))
        else:
            replace_linear_with_fp8(module, block_size, scale_fmt, skip_modules)
    return model