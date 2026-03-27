# models/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.distributed as dist

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network used in dense layers.
    FFN(x) = W2(silu(W1(x)) * W3(x))
    Architecturally identical to the Expert FFN in DeepSeekMoE so that dense and sparse layers are interchangeable.
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Single transformer block: pre-norm MLA attention + pre-norm SwiGLU/MoE FFN. The first `n_dense_layers` layers
    use a SwiGLUFFN; all subsequent layers use DeepSeekMoE. Layer 0 is always dense in the config.
    """

    def __init__(
        self,
        layer_id: int,
        config: dict,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.layer_id      = layer_id
        self.dim           = config["dim"]
        self.n_dense_layers = config["n_dense_layers"]

        self.attn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        # Pass world_size/rank so MLA can shard heads across tensor-parallel ranks
        self.attn = MultiHeadLatentAttention(config, layer_id, world_size, rank)

        self.ffn_norm = nn.RMSNorm(self.dim, eps=1e-6)
        if layer_id < self.n_dense_layers:
            self.ffn = SwiGLUFFN(self.dim, config["inter_dim"])
        else:
            self.ffn = DeepSeekMoE(config, world_size, rank)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, mask, use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ParallelEmbedding(nn.Module):
    """
    Vocabulary-parallel embedding.

    The vocabulary is partitioned across `world_size` ranks.  Each rank owns tokens in the range
    [vocab_start_idx, vocab_end_idx). Out-of-range token IDs are zeroed before the local lookup;
    the resulting partial embeddings are summed via all-reduce to produce the full embedding on every rank.
    The last rank absorbs any remainder tokens when vocab_size is not exactly divisible by world_size.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.dim         = dim
        self.world_size  = world_size
        self.rank        = rank

        base                  = vocab_size // world_size
        remainder             = vocab_size % world_size
        self.part_vocab_size  = base + remainder if rank == world_size - 1 else base
        self.vocab_start_idx  = rank * base
        self.vocab_end_idx    = self.vocab_start_idx + self.part_vocab_size

        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim))
        nn.init.normal_(self.weight, std=0.006)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            # Zero out token IDs that belong to other ranks before the lookup, then mask the resulting vectors back
            # to zero so they contribute nothing to the all-reduce sum.
            out_of_range = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x_local      = (x - self.vocab_start_idx).masked_fill(out_of_range, 0)
            y            = F.embedding(x_local, self.weight)
            y            = y.masked_fill(out_of_range.unsqueeze(-1), 0.0)
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        else:
            y = F.embedding(x, self.weight)

        return y


class Transformer(nn.Module):
    """
    DeepSeek-V3-style Transformer: MLA attention, MoE FFN, optional MTP.

    Construction
    ------------
    Takes a config dict directly (not a file path) so the model can be instantiated from memory, from tests, or
    after hydrating a checkpoint's stored config — without touching the filesystem.

    Causal mask caching
    -------------------
    `_build_causal_mask` caches the most recently built mask.  Calls with the same (seqlen, device) are free;
    a new mask is only allocated when seqlen or device changes.

    KV cache lifecycle
    ------------------
    During generation the KV cache is populated by each MLA layer. Call `reset_cache()` between independent
    generation requests to avoid stale context bleed and to return VRAM to the allocator.

    Gradient checkpointing
    ----------------------
    Set `use_checkpoint=True` to wrap each TransformerBlock in `torch.utils.checkpoint.checkpoint`.
    This reduces peak activation memory by recomputing activations on the backward pass at the cost of ~33% extra FLOPs.
    Essential for training at seq_len=4096 with 27 layers on memory-constrained hardware.
    """

    def __init__(
        self,
        config: dict,
        world_size: int = 1,
        rank: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        # Accept either a raw model-config dict or a nested {"model": {...}} dict
        model_cfg = config.get("model", config)

        self.world_size     = world_size
        self.rank           = rank
        self.use_checkpoint = use_checkpoint
        self.max_seq_len    = model_cfg["max_seq_len"]
        self.config         = model_cfg   # stored for downstream use (e.g. MTP)

        self.embed = ParallelEmbedding(
            model_cfg["vocab_size"], model_cfg["dim"], world_size, rank
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(i, model_cfg, world_size, rank)
                for i in range(model_cfg["n_layers"])
            ]
        )

        self.norm = nn.RMSNorm(model_cfg["dim"], eps=1e-6)

        # Output head: full-vocab, non-sharded.
        # ParallelEmbedding's all-reduce gives every rank the full embedding vector, so a non-sharded head is correct here.
        self.head = nn.Linear(model_cfg["dim"], model_cfg["vocab_size"], bias=False)

        # Causal mask cache: avoids re-allocating (S, S) on every forward call.
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_seqlen: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_causal_mask(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """
        Return an additive causal mask of shape (1, 1, seqlen, seqlen).

        Cached: free on repeated calls with the same seqlen and device.
        The mask contains 0 for attended positions and -inf for future tokens.
        """
        if (
            self._mask_cache is None
            or seqlen != self._mask_seqlen
            or self._mask_cache.device != device
        ):
            mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=device),
                diagonal=1,
            )
            self._mask_cache  = mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)
            self._mask_seqlen = seqlen
        return self._mask_cache

    def _run_layers(
        self,
        h: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
        use_cache: bool,
    ) -> torch.Tensor:
        """
        Run all TransformerBlocks, optionally with gradient checkpointing.

        Gradient checkpointing is applied per-block so each block's activations are recomputed independently
        during the backward pass, keeping peak memory proportional to block size rather than total depth.
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # checkpoint requires a function with all-tensor inputs/outputs.
                # start_pos, mask, use_cache are non-tensor — wrap in a closure.
                def _block(h, layer=layer, sp=start_pos, m=mask, uc=use_cache):
                    return layer(h, sp, m, uc)
                h = torch.utils.checkpoint.checkpoint(
                    _block, h, use_reentrant=False
                )
            else:
                h = layer(h, start_pos, mask, use_cache)
        return h

    def reset_cache(self) -> None:
        """
        Clear the KV cache in all MLA layers.
        Call this between independent generation sessions to avoid stale context bleed and to return VRAM to allocator.
        """
        for layer in self.layers:
            if hasattr(layer.attn, "reset_cache"):
                layer.attn.reset_cache()

    def moe_layers(self):
        """Iterate over all DeepSeekMoE layers in the model."""
        for layer in self.layers:
            if isinstance(layer.ffn, DeepSeekMoE):
                yield layer.ffn

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            tokens:    (bsz, seqlen) integer token IDs
            start_pos: KV-cache offset.  0 for training / full prefill incremented by seqlen each decode step.   
            use_cache: passed through to each MLA layer.

        Returns:
            logits: (bsz, seqlen, vocab_size) — consistent shape regardless of seqlen.  Callers that want the
                    last-position logits should index [:, -1, :] explicitly.
        """
        bsz, seqlen = tokens.shape
        h    = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h    = self._run_layers(h, start_pos, mask, use_cache)
        h    = self.norm(h)
        return self.head(h)   # (bsz, seqlen, vocab_size)

    def forward_with_hidden(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and final normalised hidden states. Used by MultiTokenPrediction in training.

        Returns:
            logits:  (bsz, seqlen, vocab_size)
            h_norm:  (bsz, seqlen, dim)
        """
        bsz, seqlen = tokens.shape
        h    = self.embed(tokens)
        mask = self._build_causal_mask(seqlen, tokens.device) if seqlen > 1 else None
        h    = self._run_layers(h, start_pos, mask, use_cache)
        h_norm = self.norm(h)
        logits = self.head(h_norm)
        return logits, h_norm

    # ──────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache, top-p and top-k sampling.

        Prefill phase: encode the full prompt in one forward pass, populating the KV cache.

        Decode phase: generate one token at a time, passing only the new token and advancing start_pos by 1 each step.
        Each decode step is O(seq_so_far) in attention (due to KV cache reads) but O(1) in compute per
        layer — O(n) total vs O(n²) without the cache.

        Args:
            input_ids:      (bsz, prompt_len) — prompt token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature:    softmax temperature; 1.0 = no change,
                            values < 1.0 sharpen the distribution
            top_p:          nucleus sampling threshold (0, 1]; 1.0 = disabled
            top_k:          top-k filtering; 0 = disabled

        Returns:
            (bsz, prompt_len + generated_len) token IDs
        """
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        was_training = self.training
        self.reset_cache()
        self.eval()

        bsz, prompt_len = input_ids.shape
        output          = input_ids.clone()

        # ── Prefill ────────────────────────────────────────────────────────
        
        # Encode the full prompt once; this populates the KV cache for all positions [0, prompt_len).
        prefill_logits = self.forward(
            output, start_pos=0, use_cache=True)   # (bsz, prompt_len, vocab)
        # The next token is sampled from the last prompt position's logits.
        next_logits = prefill_logits[:, -1, :]   # (bsz, vocab)

        # ── Decode ─────────────────────────────────────────────────────────
        
        for step in range(max_new_tokens):
            next_token = self._sample(next_logits, temperature, top_p, top_k)
            output     = torch.cat([output, next_token], dim=1)

            if output.size(1) >= self.max_seq_len:
                break

            # Decode step: one token in, one token out.
            # start_pos = prompt_len + step because the cache already holds positions [0, prompt_len + step).
            decode_pos    = prompt_len + step
            decode_logits = self.forward(
                next_token, start_pos=decode_pos, use_cache=True
            )   # (bsz, 1, vocab)
            next_logits   = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return output

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Sample the next token from logits.

        Applies temperature scaling, optional top-k truncation, and optional top-p (nucleus) truncation in sequence,
        then samples from the resulting distribution.  When temperature == 0 uses argmax (greedy).

        Args:
            logits: (bsz, vocab) unnormalised logits
            temperature: softmax temperature
            top_p: nucleus threshold; 1.0 disables
            top_k: keep only top-k logits; 0 disables

        Returns:
            (bsz, 1) sampled token IDs
        """
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k: zero out all but the k highest logits
        if top_k > 0:
            # Keep only the top-k values; fill the rest with -inf
            kth_vals = logits.topk(min(top_k, logits.size(-1)), dim=-1)[0][:, -1:]
            logits   = logits.masked_fill(logits < kth_vals, float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        # Top-p: truncate to the smallest set of tokens whose cumulative probability exceeds p, then renormalise.
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative               = sorted_probs.cumsum(dim=-1)
            # Remove tokens where cumulative prob BEFORE adding this token > top_p
            remove       = (cumulative - sorted_probs) > top_p
            # Use masked_fill rather than in-place mutation to avoid corrupting the autograd graph if called in a grad context
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            # Sample from the truncated distribution and map back to vocab indices
            sampled_idx  = torch.multinomial(sorted_probs, num_samples=1)
            next_token   = sorted_idx.gather(-1, sampled_idx)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token   # (bsz, 1)


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Return (total_params, trainable_params).
    Counts each unique parameter tensor exactly once even when weights are tied (e.g. shared output heads in MTP).
    The original code iterated model.parameters() which visits shared tensors multiple times.
    """
    seen       = set()
    total      = 0
    trainable  = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total     += n
        if p.requires_grad:
            trainable += n
    return total, trainable