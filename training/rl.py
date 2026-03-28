# training/rl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable
import copy


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    batch_size: int = 8
    group_size: int = 8          # samples per prompt
    epsilon: float = 0.2         # PPO clip ratio
    beta: float = 0.04           # KL penalty coefficient
    lr: float = 1e-6
    max_steps: int = 500
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    max_new_tokens: int = 512


# ── Reward Model ───────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """Scalar reward head on top of a base language model."""

    def __init__(self, base_model: nn.Module, config: dict):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(config["dim"], 1, bias=True)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.normal_(self.reward_head.weight, std=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.base_model.forward_with_hidden(input_ids)[1]  # (bsz, seq, dim)
        reward = self.reward_head(hidden[:, -1])        # last token repr
        return reward.squeeze(-1)                       # (bsz,)

    # ── Rule-based rewards ─────────────────────────────────────────────────

    @staticmethod
    def rule_based(response: str, question: str = "") -> float:
        """Heuristic reward for verifiable reasoning tasks."""
        score = 0.0
        if "\\boxed{" in response:
            score += 0.5
        if any(w in response.lower() for w in ("therefore", "step", "because", "∴")):
            score += 0.3
        if len(response.split()) > 20:
            score += 0.2
        return min(score, 1.0)


# ── GRPO Trainer ───────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    Group Relative Policy Optimisation (GRPO).

    For each prompt we sample G responses, score them, normalise rewards
    within the group, and update the policy with a clipped surrogate objective
    plus a KL penalty against a frozen reference model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: RewardModel,
        config: GRPOConfig,
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config

        # Frozen reference copy (detached; no grad)
        self.reference = copy.deepcopy(policy_model)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.lr, weight_decay=0.0
        )

    @torch.no_grad()
    def _compute_log_probs(
        self, model: nn.Module, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities for a sequence.

        Args:
            token_ids: (bsz, seq)
        Returns:
            log_probs: (bsz, seq-1) — log p(t_{i+1} | t_{≤i})
        """
        logits = model(token_ids[:, :-1])           # (bsz, seq-1, vocab)
        log_p = F.log_softmax(logits, dim=-1)
        # Gather log prob of the actual next token
        return log_p.gather(-1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalise rewards within the group."""
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        return (rewards - mean) / std

    def train_step(
        self,
        prompts: List[str],
        generate_fn: Callable,
        tokenizer,
    ) -> dict:
        """
        GRPO update step.

        Args:
            prompts:     list of prompt strings
            generate_fn: callable(prompt, model) → response string
            tokenizer:   HuggingFace tokenizer

        Returns:
            dict with loss metrics
        """
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n_updates = 0

        for prompt in prompts:
            # ── 1. Sample group of responses ──────────────────────────────
            responses = [generate_fn(prompt, self.policy) for _ in range(self.config.group_size)]

            # ── 2. Score responses ────────────────────────────────────────
            rewards = torch.tensor(
                [self.reward_model.rule_based(r, prompt) for r in responses],
                dtype=torch.float32,
                device=next(self.policy.parameters()).device,
            )

            advantages = self.compute_advantages(rewards)   # (G,)

            # ── 3. Policy gradient with PPO clip ─────────────────────────
            prompt_loss = torch.tensor(0.0, device=rewards.device)

            for i, response in enumerate(responses):
                if isinstance(response, str):
                    ids = tokenizer.encode(response, return_tensors="pt").cuda()
                else:
                    ids = response.unsqueeze(0).cuda() if response.dim() == 1 else response.cuda()

                if ids.numel() < 2:
                    continue

                # Log-probs under current policy
                with torch.enable_grad():
                    logits = self.policy(ids[:, :-1])
                    log_p = F.log_softmax(logits, dim=-1)
                    token_log_probs = log_p.gather(
                        -1, ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)                           # (1, seq-1)

                # Log-probs under reference (no grad)
                with torch.no_grad():
                    ref_logits = self.reference(ids[:, :-1])
                    ref_log_p = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_p.gather(
                        -1, ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)

                # Sequence-level log-prob ratio
                seq_log_ratio = (token_log_probs - ref_token_log_probs).sum(-1)  # (1,)
                ratio = seq_log_ratio.exp()

                adv = advantages[i]
                # Clipped surrogate
                clipped = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

                # Token-level KL: KL(policy ‖ ref) ≈ Σ p*(log p - log q)
                kl = (token_log_probs.exp() * (token_log_probs - ref_token_log_probs)).sum(-1).mean()
                kl_loss = self.config.beta * kl

                prompt_loss = prompt_loss + (policy_loss + kl_loss) / self.config.group_size
                total_kl_loss += kl_loss.item()
                n_updates += 1

            total_policy_loss += prompt_loss.item()
            prompt_loss.backward()

        # Single optimiser step per call
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / len(prompts),
            "kl_loss": total_kl_loss / n,
        }

    def update_reference(self):
        """Sync reference model with current policy (call periodically)."""
        self.reference.load_state_dict(self.policy.state_dict())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    config = GRPOConfig()
    print("GRPO training configured:", config)


if __name__ == "__main__":
    main()
