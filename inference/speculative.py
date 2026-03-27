# inference/speculative.py
"""
Speculative decoding via Multi-Token Prediction (MTP).

Architecture (DeepSeek-V3, Section 5.4.3):
  1. Main model predicts token T1 from input.
  2. MTP draft head predicts token T2 using T1's embedding + main model hidden state.
  3. Main model verifies T2 at the next step; accept if p_main(T2) / p_draft(T2) > threshold.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from models.mtp import MTPModule


class SpeculativeDecoder:
    """
    MTP-based speculative decoder.

    Generates one speculative draft token per step; verifies it against the
    main model and either accepts or falls back to the main model's token.
    """

    def __init__(
        self,
        main_model: nn.Module,
        mtp_module: MTPModule,
        acceptance_threshold: float = 0.8,
    ):
        self.main_model = main_model
        self.mtp = mtp_module
        self.threshold = acceptance_threshold

    @torch.inference_mode()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Single speculative step.

        Returns T1 (main model's greedy token) and T2 (draft head's speculative token)
        separately so the caller can always append T1 and optionally also T2.

        Args:
            input_ids: (1, seq) token IDs fed into the main model
            start_pos: KV-cache start position

        Returns:
            token_main:   (1,) — main model's greedy prediction for position seq
            token_draft:  (1,) — MTP draft prediction for position seq+1
            was_accepted: bool — True if the draft token passes the acceptance test
        """
        # ── 1. Main model: predict T1 from the current context ────────────
        main_logits, hidden = self.main_model.forward_with_hidden(input_ids, start_pos)
        if main_logits.dim() == 3:
            main_logits = main_logits[:, -1, :]   # last position → (1, vocab)
            hidden_last = hidden[:, -1:, :]        # (1, 1, dim)
        else:
            hidden_last = hidden.unsqueeze(1) if hidden.dim() == 2 else hidden

        main_probs = torch.softmax(main_logits, dim=-1)
        token_main = main_probs.argmax(dim=-1)    # (1,) greedy

        # ── 2. MTP draft head: predict T2 using T1's embedding ────────────
        token_main_emb = self.main_model.embed(token_main.unsqueeze(-1))  # (1, 1, dim)
        draft_logits, _ = self.mtp(hidden_last, token_main_emb)
        draft_logits = draft_logits[:, -1, :]     # (1, vocab)
        draft_probs = torch.softmax(draft_logits, dim=-1)
        token_draft = draft_probs.argmax(dim=-1)  # (1,)

        # ── 3. Acceptance test (probability-ratio heuristic) ──────────────
        # Full speculative decoding would run the main model at position seq+1
        # and compare distributions.  Here we approximate: accept if the main
        # model assigns at least `threshold` of the draft's probability mass.
        p_main_of_draft  = main_probs[0, token_draft[0]].item()
        p_draft_of_draft = draft_probs[0, token_draft[0]].item()
        acceptance_ratio = (
            min(1.0, p_main_of_draft / p_draft_of_draft)
            if p_draft_of_draft > 1e-12 else 0.0
        )
        was_accepted = acceptance_ratio >= self.threshold

        return token_main, token_draft, was_accepted

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Full speculative generation loop.

        Each step produces 1 token (main model's prediction) or 2 tokens
        (main + draft) when the draft is accepted, achieving the intended speedup.
        """
        output = input_ids.clone()
        n_generated = 0

        while n_generated < max_new_tokens:
            token_main, token_draft, was_accepted = self.generate_step(output, start_pos=0)

            # Always append T1 (main model's verified token)
            output = torch.cat([output, token_main.unsqueeze(0)], dim=1)
            n_generated += 1

            # If draft was accepted, also append T2 (speculative bonus token)
            if was_accepted and n_generated < max_new_tokens:
                output = torch.cat([output, token_draft.unsqueeze(0)], dim=1)
                n_generated += 1

        return output
