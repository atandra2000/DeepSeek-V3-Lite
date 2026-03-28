# models/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import torch.distributed as dist


class AuxLossFreeGate(nn.Module):
    """
    Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3, Section 2.3.3).

    Routing decision
    ----------------
    Each token is assigned to the top-k experts by a biased score:

        biased_score_e = sigmoid(x @ W_e^T) + bias_e

    The bias is NOT used when computing the final routing weights — only the raw sigmoid scores are load
    normalised and used as weights.  This separates load balancing (via bias) from the gradient
    signal (via raw scores) balancing (via bias) from the gradient signal (via raw scores).

    Group-limited routing
    ---------------------
    When n_groups > 1 the experts are divided into n_groups equal groups.
    Only topk_groups groups are selected per token (node-limited routing).
    Within each selected group the top-`group_topk` biased scores are summed
    to produce a group score; the top groups by that score are activated.

    Bias update
    -----------
    After each optimiser step the caller should invoke update_bias() with the per-expert token counts from the
    last forward pass.  Experts that are over-loaded (count > avg * (1 + upper_threshold)) have their bias
    decreased; under-loaded experts have their bias increased.  The bias is stored as a plain buffer
    (not a Parameter) so it does not appear in optimiser state dicts.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.dim = config["dim"]
        self.topk = config["n_activated_experts"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_groups = config.get("n_expert_groups", 1)
        self.topk_groups = config.get("n_limited_groups", 1)
        self.route_scale = config.get("route_scale", 1.0)
        # Number of top experts per group considered for group score. DeepSeek-V3 uses 2
        self.group_topk = config.get("group_topk", 2)
        # Bias update thresholds: experts loaded outside [avg*(1-lo), avg*(1+hi)] have their bias adjusted.
        # Separate thresholds allow asymmetric hysteresis to prevent bias thrashing on noisy load estimates.
        self.bias_upper = config.get("bias_upper_threshold", 0.10)
        self.bias_lower = config.get("bias_lower_threshold", 0.10)
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.dim))
        nn.init.normal_(self.weight, std=0.006)
        # Bias registered as a buffer (not a Parameter):
        #   • included in state_dict  → persists across checkpoints
        #   • excluded from model.parameters() → not touched by the optimiser
        #   • updated manually via update_bias()
        self.register_buffer("bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, speed: float = 0.001) -> None:
        """
        Adjust per-expert bias based on observed token counts.

        Args:
            counts: (n_routed_experts,) integer token counts from the last step.
                    Must be on CPU (avoids unnecessary device sync in the caller).
            speed:  step size for each bias adjustment.
        """
        counts = counts.float()
        avg = counts.mean()
        # Decrease bias for over-loaded experts → makes them less likely to be
        # selected in the next step, steering tokens to under-used experts.
        self.bias[counts > avg * (1.0 + self.bias_upper)] -= speed
        # Increase bias for under-loaded experts → attract more tokens.
        self.bias[counts < avg * (1.0 - self.bias_lower)] += speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, dim) — flattened token representations

        Returns:
            weights: (T, topk) — normalised routing weights (sum-to-1 per token,
                                  then scaled by route_scale)
            indices: (T, topk) — global expert indices
        """
        T = x.size(0)

        # Raw sigmoid scores — used for final weight computation
        scores = F.linear(x, self.weight).sigmoid()          # (T, E)

        # Biased scores — used for routing decision only
        biased = scores + self.bias.to(scores.dtype)         # (T, E)

        if self.n_groups > 1:
            experts_per_group = self.n_routed_experts // self.n_groups
            # (T, n_groups, experts_per_group)
            biased_grouped = biased.view(T, self.n_groups, experts_per_group)
            # Group score = sum of top-group_topk biased scores within each group
            group_scores = biased_grouped.topk(self.group_topk, dim=-1)[0].sum(dim=-1)
            # Select topk_groups groups per token
            top_groups = group_scores.topk(self.topk_groups, dim=-1)[1]  # (T, topk_groups)
            # Mask out non-selected groups
            group_mask = torch.ones(T, self.n_groups, dtype=torch.bool, device=x.device)
            group_mask.scatter_(1, top_groups, False)
            biased = biased_grouped.masked_fill(group_mask.unsqueeze(-1), float("-inf")).flatten(1)     # (T, E)

        # Select top-k experts by biased score
        indices = biased.topk(self.topk, dim=-1)[1]          # (T, topk)

        # Routing weights from raw (unbiased) scores at the selected positions
        weights = scores.gather(1, indices)                   # (T, topk)

        # Normalise so weights sum to 1 per token, then apply route_scale
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        weights = (weights * self.route_scale).to(x.dtype)

        return weights, indices


class Expert(nn.Module):
    """
    Single SwiGLU expert.

    FFN(x) = W2(silu(W1(x)) * W3(x))

    W1 and W3 are the gate/up projections (dim → inter_dim).
    W2 is the down projection (inter_dim → dim).
    """

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with shared experts and aux-loss-free load balancing.

    Expert parallelism
    ------------------
    Each rank owns a contiguous shard of n_local_experts routed experts
    (indices [experts_start, experts_end)).  Shared experts are replicated
    on all ranks and run unconditionally.

    All-reduce
    ------------
    Only the routed expert output (y_routed) is all-reduced across ranks.
    Shared expert outputs are computed locally on every rank and added AFTER
    the all-reduce, so they are never multiplied by world_size.

    Routing cache
    -------------
    The most recent (weights, indices) pair is stored in self._last_weights and self._last_indices
    after every forward pass. This allows get_load_balance_loss() and get_routing_stats() to reuse routing
    without a second gate call, and allows pretrain.py to call update_gate_bias() without re-embedding the input batch.
    """

    def __init__(self, config: dict, world_size: int = 1, rank: int = 0):
        super().__init__()
        self.dim              = config["dim"]
        self.n_routed_experts = config["n_routed_experts"]
        self.n_shared_experts = config["n_shared_experts"]
        self.moe_inter_dim    = config["moe_inter_dim"]
        self.world_size       = world_size
        self.rank             = rank

        if self.n_routed_experts % world_size != 0:
            raise ValueError(
                f"n_routed_experts ({self.n_routed_experts}) must be "
                f"divisible by world_size ({world_size})"
            )
        self.n_local_experts = self.n_routed_experts // world_size
        self.experts_start   = rank * self.n_local_experts
        self.experts_end     = self.experts_start + self.n_local_experts

        self.gate = AuxLossFreeGate(config)

        # Only instantiate experts local to this rank
        self.experts = nn.ModuleList(
            [Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_local_experts)])

        # Shared experts — replicated on all ranks, always executed
        self.shared_experts = nn.ModuleList(
            [Expert(self.dim, self.moe_inter_dim) for _ in range(self.n_shared_experts)])

        # Routing cache: populated during forward(), reused by auxiliary methods
        self._last_weights: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, dim) — flattened token representations
        Returns:
            Tensor of same shape as x.
        """
        shape = x.shape
        flat  = x.view(-1, self.dim)   # (B*T, dim)

        weights, indices = self.gate(flat)  # (B*T, topk), (B*T, topk)
        # Cache routing for auxiliary loss / stats / bias update
        self._last_weights = weights.detach()
        self._last_indices = indices.detach()

        # ── Routed expert dispatch ─────────────────────────────────────────

        y_routed = torch.zeros_like(flat)
        # Build boolean token-assignment matrix once to  eliminate per-expert CPU syncs.
        # token_expert_mask: (T, n_local_experts) — True where token t routes to the local expert at column e.
        global_ids   = torch.arange(self.experts_start, self.experts_end, device=flat.device)   # (n_local_experts,)
        # (T, topk, 1) == (1, 1, n_local_experts) → (T, topk, n_local_experts)
        token_expert_mask = (indices.unsqueeze(-1) == global_ids.view(1, 1, -1)).any(dim=1)   # (T, n_local_experts)

        for local_idx, expert in enumerate(self.experts):
            # Which tokens route to this expert?  All comparisons stay on GPU.
            token_mask = token_expert_mask[:, local_idx]   # (T,) bool
            token_ids  = token_mask.nonzero(as_tuple=True)[0]  # (k,)
            if token_ids.numel() == 0:
                continue

            # Find the topk position for this expert in each selected token's routing vector
            global_idx  = self.experts_start + local_idx
            # (k, topk) boolean; exactly one True per row by construction
            topk_match  = indices[token_ids] == global_idx  # (k, topk)
            pos_in_topk = topk_match.long().argmax(dim=-1)  # (k,)

            expert_out  = expert(flat[token_ids])           # (k, dim)
            scale       = weights[token_ids, pos_in_topk].unsqueeze(-1)  # (k, 1)
            y_routed.index_add_(0, token_ids, expert_out * scale)

        # ── All-reduce routed output only ──────────────────────────────────

        # Shared experts are replicated and must NOT be included in the all-reduce.
        if self.world_size > 1:
            dist.all_reduce(y_routed, op=dist.ReduceOp.SUM)

        # ── Shared experts (always executed, added after all-reduce) ───────
        if self.shared_experts:
            shared_out = torch.stack(
                [e(flat) for e in self.shared_experts], dim=0).sum(dim=0)
            y = y_routed + shared_out
        else:
            y = y_routed

        return y.view(shape)

    # ──────────────────────────────────────────────────────────────────────
    # Auxiliary methods (reuse cached routing — no second gate call)
    # ──────────────────────────────────────────────────────────────────────

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Sequence-level load-balance auxiliary loss from DeepSeek-V3.

        L_bal = n_experts * Σ_e (f_e * P_e)

        where:
          f_e = fraction of tokens routed to expert e  (load)
          P_e = mean routing probability for expert e  (affinity)

        Minimising this encourages tokens to spread evenly across experts while keeping routing probabilities
        aligned with actual assignments. Requires a preceding forward() call in the same step to have
        populated the routing cache.  Returns zero if cache is empty.
        """
        if self._last_weights is None or self._last_indices is None:
            return torch.tensor(0.0, device=self.gate.weight.device)

        weights = self._last_weights  # (T, topk)
        indices = self._last_indices  # (T, topk)
        T       = weights.size(0)

        # f_e: per-expert token fraction — count each assignment once
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts).float()
        f = counts / counts.sum().clamp(min=1e-10)   # (E,)

        # P_e: mean routing probability for expert e across all tokens.
        # Build a (T, E) sparse assignment matrix, multiply by weights, then average over tokens.
        # one_hot: (T*topk, E) → reshaped and summed → (T, E)
        one_hot = F.one_hot(indices.flatten(), num_classes=self.n_routed_experts).float()
        # weights.flatten(): (T*topk,); weight each assignment by its routing weight
        P_dense = (one_hot * weights.flatten().unsqueeze(-1)).view(
            T, -1, self.n_routed_experts
        ).sum(dim=1)                                  # (T, E)
        P = P_dense.mean(dim=0)                      # (E,)

        return (f * P).sum() * self.n_routed_experts

    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """
        Return per-expert routing statistics from the last forward pass.

        Useful for monitoring load imbalance during training without any
        additional gate computation.

        Returns a dict with:
          counts      (E,)  — integer number of token-expert assignments
          load        (E,)  — fraction of total assignments per expert
          mean_weight (E,)  — mean routing weight for each expert
          utilisation       — fraction of experts that received at least one token
        """
        if self._last_weights is None or self._last_indices is None:
            return {}

        weights = self._last_weights
        indices = self._last_indices
        E       = self.n_routed_experts

        counts      = torch.bincount(indices.flatten(), minlength=E).float()
        load        = counts / counts.sum().clamp(min=1e-10)

        one_hot     = F.one_hot(indices.flatten(), num_classes=E).float()
        weight_sum  = (one_hot * weights.flatten().unsqueeze(-1)).sum(dim=0)
        mean_weight = weight_sum / counts.clamp(min=1.0)   # avoid div-by-zero on empty experts

        utilisation = (counts > 0).float().mean()

        return {
            "counts":      counts,
            "load":        load,
            "mean_weight": mean_weight,
            "utilisation": utilisation,
        }

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """
        Update the gate's load-balancing bias using the cached token counts.

        This is the correct call site for the bias update — it uses routing state captured during the last
        forward() pass without any additional gate computation or input re-embedding.

        Replaces the pattern in pretrain.py that re-embedded tokens:
            h = model.embed(tokens); moe._update_moe_bias(h)
        """
        if self._last_indices is None:
            return
        counts = torch.bincount(
            self._last_indices.flatten().cpu(),
            minlength=self.n_routed_experts,
        )
        self.gate.update_bias(counts, speed=speed)
