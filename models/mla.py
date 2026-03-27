# models/mla.py
import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V3.

    Key ideas
    ---------
    • Low-rank KV compression: the KV cache stores the normalised latent
      c_KV ∈ R^{kv_lora_rank} instead of full per-head K/V tensors, giving
      a (n_heads * (qk_nope_head_dim + v_head_dim)) / kv_lora_rank ≈ 10–20×
      reduction in KV-cache memory.

    • Decoupled RoPE: positional encodings are applied only to the
      qk_rope_head_dim slice of Q/K.  The nope slice carries content-based
      similarity; the rope slice carries positional similarity.  The two
      score contributions are summed before softmax.

    • Absorption trick: instead of expanding c_KV → (K_nope, V) at every
      step, wkv_b is absorbed into q_nope at query time, so attention scores
      are computed directly against the cached latent.  V is also recovered
      from the latent post-softmax.  This avoids materialising full K/V
      tensors during decode.
      
    • YaRN-compatible softmax scaling: for long contexts, the softmax scale is
      multiplied by mscale > 1 to prevent underflow.
    """

    def __init__(
        self,
        config: dict,
        layer_idx: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.layer_idx  = layer_idx
        self.world_size = world_size
        self.rank       = rank

        # ── Dimensions ────────────────────────────────────────────────────

        self.dim              = config["dim"]                    # Model hidden size (2048)
        self.n_heads          = config["n_heads"]                # Total attention heads (16)
        self.q_lora_rank      = config["q_lora_rank"]            # 0 means no LoRA compression fot Q
        self.kv_lora_rank     = config["kv_lora_rank"]           # 512 - the key compression dimention
        self.qk_nope_head_dim = config["qk_nope_head_dim"]       # 128 - content based Q/K dimention
        self.qk_rope_head_dim = config["qk_rope_head_dim"]       # 64 - positional Q/K dimention
        self.v_head_dim       = config["v_head_dim"]             # 128 - value head dimention
        self.qk_head_dim      = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_seq_len      = config["max_seq_len"]

        if self.n_heads % world_size != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by world_size ({world_size})"
            )
        self.n_local_heads = self.n_heads // world_size

        # ── RoPE config ───────────────────────────────────────────────────

        self.rope_theta  = config["rope_theta"]
        self.rope_factor = config.get("rope_factor", 1.0)

        mscale_raw  = config.get("mscale", 1.0)
        self.mscale = (
            0.1 * mscale_raw * math.log(self.rope_factor) + 1.0
            if self.rope_factor > 1.0
            else mscale_raw
        )

        # ── Softmax scale ─────────────────────────────────────────────────

        # Base: 1/sqrt(qk_head_dim). YaRN corrects for extended contexts by multiplying by mscale^2

        self.softmax_scale = self.qk_head_dim ** -0.5
        if self.max_seq_len > 4096 and self.mscale != 1.0:
            self.softmax_scale *= self.mscale ** 2

        # ── Query projections ──────────────────────────────────────────────

        # When q_lora_rank > 0 the query is produced via a low-rank
        # bottleneck: x → (wq_a) → q_lora_rank → (RMSNorm) → (wq_b) → Q.
        # wq_b output is sized for local heads (tensor parallelism ready).

        if self.q_lora_rank > 0:
            self.wq_a   = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.wq_b   = nn.Linear(
                self.q_lora_rank,
                self.n_local_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.wq = nn.Linear(
                self.dim,
                self.n_local_heads * self.qk_head_dim,
                bias=False,
            )

        # ── KV projections with latent compression ─────────────────────────

        # wkv_a projects x → (latent ‖ k_rope).  Replicated across all ranks
        # because the latent must be written to the KV cache in full — sharding
        # it would require a gather before every cache write.

        self.wkv_a  = nn.Linear(
            self.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)

        # wkv_b expands latent → (K_nope ‖ V) for local heads only.
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_local_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection: (n_local_heads * v_head_dim) → dim.
        # Row-parallel in tensor-parallel setups; plain linear here.

        self.wo = nn.Linear(self.n_local_heads * self.v_head_dim, self.dim, bias=False)

        # ── KV cache ──────────────────────────────────────────────────────

        # Allocated lazily on first forward call; grown as needed.
        # reset_cache() releases memory between independent sessions.

        self._cache_batch: int              = 0
        self.kv_cache: Optional[torch.Tensor] = None
        self.pe_cache: Optional[torch.Tensor] = None

        # ── RoPE frequency table ───────────────────────────────────────────

        # Extended lazily up to the maximum position seen so far.
        self._rope_seq_len: int = 0
        self.register_buffer(
            "freqs_cis",
            torch.empty(0, self.qk_rope_head_dim // 2, dtype=torch.complex64),
            persistent=False,
        )

    # ──────────────────────────────────────────────────────────────────────
    # RoPE helpers
    # ──────────────────────────────────────────────────────────────────────

    def _extend_rope(self, seq_len: int, device: torch.device) -> None:
        """
        Extend the precomputed RoPE table to cover at least `seq_len` positions.
        No-op when the table is already large enough.  Called from forward()
        before any positional encoding is applied.
        """
        if seq_len <= self._rope_seq_len:
            return

        dim      = self.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        if self.rope_factor > 1.0:
            inv_freq = inv_freq / self.rope_factor

        # Grow to at least 2× the current length to amortise reallocation cost
        # during autoregressive generation where seq_len increments by 1 each step.

        grow_to = max(seq_len, self._rope_seq_len * 2, 64)
        grow_to = min(grow_to, self.max_seq_len)
        t = torch.arange(grow_to, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)                          # (grow_to, dim//2)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._rope_seq_len = grow_to

    def _apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int,
        seqlen: int,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings.

        Args:
            x:         (bsz, seqlen, n_heads, rope_dim)
            start_pos: absolute position of the first token in x
            seqlen:    number of tokens in x

        Returns:
            Tensor of same shape and dtype as x.
        """
        dtype = x.dtype
        x_c   = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # freqs: (seqlen, rope_dim//2) → broadcast (1, seqlen, 1, rope_dim//2)
        freqs = self.freqs_cis[start_pos : start_pos + seqlen].view(1, seqlen, 1, -1)
        return torch.view_as_real(x_c * freqs).flatten(-2).to(dtype)

    # ──────────────────────────────────────────────────────────────────────
    # Cache management
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Guarantee that kv_cache / pe_cache can hold at least `bsz` sequences.

        Growth policy: at least double the current batch capacity (floor 16)
        to amortise repeated calls and avoid constant reallocation.
        The cache is also reallocated when device or dtype changes (e.g. after
        a .to() call that moves the model to a different device).
        """
        need_alloc = (
            self.kv_cache is None
            or bsz > self._cache_batch
            or self.kv_cache.device != device
            or self.kv_cache.dtype  != dtype
        )
        if not need_alloc:
            return

        new_bsz = max(bsz, self._cache_batch * 2, 16)
        self.kv_cache     = torch.zeros(
            new_bsz, self.max_seq_len, self.kv_lora_rank,
            device=device, dtype=dtype,
        )
        self.pe_cache     = torch.zeros(
            new_bsz, self.max_seq_len, self.qk_rope_head_dim,
            device=device, dtype=dtype,
        )
        self._cache_batch = new_bsz

    def reset_cache(self) -> None:
        """
        Release KV-cache memory and reset tracking state.

        Call this between independent generation sessions (e.g. different
        prompts in a serving context) to avoid stale context bleed and to
        return VRAM to the allocator.
        """
        self.kv_cache     = None
        self.pe_cache     = None
        self._cache_batch = 0

    def prefill_cache(
        self,
        kv_latent: torch.Tensor,
        k_pe: torch.Tensor,
        start_pos: int,
    ) -> None:
        """
        Write pre-computed latents into the KV cache at an arbitrary offset.

        Useful for prefix / prompt caching: if a shared prompt prefix has been
        processed once, its latents can be injected here so subsequent decode
        steps benefit from the cached context without re-running the encoder.

        Args:
            kv_latent: (bsz, seqlen, kv_lora_rank) — already kv_norm'd
            k_pe:      (bsz, seqlen, rope_dim)      — already RoPE-rotated
            start_pos: token offset to begin writing at
        """
        bsz, seqlen, _ = kv_latent.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"prefill_cache: end_pos {end_pos} > max_seq_len {self.max_seq_len}"
            )
        self._extend_rope(end_pos, kv_latent.device)
        self._ensure_cache(bsz, kv_latent.device, kv_latent.dtype)
        self.kv_cache[:bsz, start_pos:end_pos] = kv_latent
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args
        ----
        x          (bsz, seqlen, dim)
        start_pos  First absolute token index being processed.
                   • 0 during training / full-sequence prefill.
                   • Current decode step position during autoregressive
                     generation (incremented by 1 each step).
        mask       Additive causal mask, shape (1, 1, seqlen_q, seqlen_k).
                   0 for attended positions, -inf for masked positions.
                   Pass None when seqlen_q == 1 (single-token decode) — no
                   masking is needed because the query attends to all cached
                   positions unconditionally.
        use_cache  When True, write the current step's KV latents into the
                   persistent cache and read the full context from it.
                   When False (typical for training), use only the current
                   sequence's latents — avoids touching the cache entirely,
                   which prevents stale-state bugs during teacher-forced runs.

        Returns
        -------
        (bsz, seqlen, dim)
        """
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen

        if end_pos > self.max_seq_len:
            raise RuntimeError(
                f"Layer {self.layer_idx}: end_pos {end_pos} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        # Extend the RoPE table on demand before any positional encoding
        self._extend_rope(end_pos, x.device)

        if use_cache:
            self._ensure_cache(bsz, x.device, x.dtype)

        # ── Queries ────────────────────────────────────────────────────────

        if self.q_lora_rank > 0:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        else:
            q = self.wq(x)
        # (bsz, seqlen, n_local_heads, qk_head_dim)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self._apply_rope(q_pe, start_pos, seqlen)

        # ── KV latent compression ──────────────────────────────────────────

        kv_a                 = self.wkv_a(x)
        kv_latent, k_pe_raw  = kv_a.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_normed            = self.kv_norm(kv_latent)

        # k_pe_raw: (bsz, seqlen, rope_dim) — temporarily add head dim for _apply_rope
        k_pe = self._apply_rope(k_pe_raw.unsqueeze(2), start_pos, seqlen).squeeze(2)
        # k_pe: (bsz, seqlen, rope_dim) — shared across all heads

        if use_cache:
            # Write this step's latents into the persistent cache …
            self.kv_cache[:bsz, start_pos:end_pos] = kv_normed
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe
            # … then read back the full context window (prefix + current tokens)
            ctx_kv = self.kv_cache[:bsz, :end_pos]   # (bsz, end_pos, kv_lora_rank)
            ctx_pe = self.pe_cache[:bsz, :end_pos]    # (bsz, end_pos, rope_dim)
        else:
            # Training / no-cache path: context is the current sequence only.
            ctx_kv = kv_normed   # (bsz, seqlen, kv_lora_rank)
            ctx_pe = k_pe        # (bsz, seqlen, rope_dim)

        # ── Absorption trick ───────────────────────────────────────────────

        # wkv_b weight: (n_local_heads * (qk_nope + v_head), kv_lora_rank)
        # Reshaped to: (n_local_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)

        wkv_b = self.wkv_b.weight.view(
            self.n_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )

        # Project q_nope into latent space so scores are computed directly
        # against ctx_kv — no need to materialise full K.
        # q_nope_proj: (bsz, seqlen_q, n_local_heads, kv_lora_rank)

        q_nope_proj = torch.einsum(
            "bshd,hdc->bshc",
            q_nope,
            wkv_b[:, : self.qk_nope_head_dim],
        )

        # ── Attention scores ───────────────────────────────────────────────

        # scores: (bsz, seqlen_q, n_local_heads, seqlen_k)
        # Two additive contributions:
        #   content term  : q_nope_proj @ ctx_kv^T  — shared-latent keys
        #   positional term: q_pe       @ ctx_pe^T  — decoupled RoPE keys
        # ctx_pe is (bsz, seqlen_k, rope_dim) and is head-shared (broadcast
        # over n_local_heads), consistent with the single-head k_pe design.

        scores = (
            torch.einsum("bshc,btc->bsht", q_nope_proj, ctx_kv)
            + torch.einsum("bshr,btr->bsht", q_pe, ctx_pe)
        ) * self.softmax_scale

        # ── Mask application ───────────────────────────────────────────────

        # Incoming mask: (1, 1, seqlen_q, seqlen_k)
        # Scores layout: (bsz, seqlen_q, n_local_heads, seqlen_k)
        # We need broadcasting over bsz (dim 0) AND n_local_heads (dim 2).
        # squeeze(1) removes the redundant heads dim from the mask,
        # unsqueeze(2) inserts a heads dim, giving (1, seqlen_q, 1, seqlen_k).

        if mask is not None:
            scores = scores + mask.squeeze(1).unsqueeze(2)

        # ── Softmax ────────────────────────────────────────────────────────

        # Float32 for numerical stability; cast back to input dtype after.
        attn = scores.softmax(dim=-1, dtype=torch.float32).to(x.dtype)

        # ── Output: weighted latent sum → project to v_head_dim → output ──
        # Weighted sum of latent KV: (bsz, seqlen_q, n_local_heads, kv_lora_rank)

        out = torch.einsum("bsht,btc->bshc", attn, ctx_kv)
        # Project to v_head_dim via wkv_b V rows:
        # (bsz, seqlen_q, n_local_heads, v_head_dim)

        out = torch.einsum("bshc,hdc->bshd", out, wkv_b[:, self.qk_nope_head_dim :])
        # Flatten heads and project to model dim:
        # (bsz, seqlen_q, n_local_heads * v_head_dim) → (bsz, seqlen_q, dim)

        return self.wo(out.flatten(2))