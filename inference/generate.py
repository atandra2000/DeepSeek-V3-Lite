# inference/generate.py
"""
Interactive generation with the DeepSeek-V3-Lite model.

Supports:
  • Standard autoregressive decoding (temperature / top-p sampling)
  • Speculative decoding via MTP draft heads
  • Multi-turn conversation with chat-template formatting
"""
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional

import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Transformer
from models.mtp import MTPModule
from utils.checkpoint import CheckpointManager
from inference.speculative import SpeculativeDecoder
from transformers import AutoTokenizer


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError("Config must be a dict with a 'model' section")
    return cfg


# ── Sampling helper ────────────────────────────────────────────────────────────

def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Apply temperature + top-p nucleus sampling and return next token ID."""
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = (cumulative - sorted_probs) > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        next_idx = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
    else:
        next_idx = probs.argmax(dim=-1, keepdim=True)

    return next_idx.squeeze(-1)   # (bsz,)


# ── Standard generation ────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive token generation loop."""
    output = input_ids.clone()
    max_seq = getattr(model, "max_seq_len", 4096)

    for _ in range(max_new_tokens):
        context = output[:, -max_seq:]
        logits = model(context, start_pos=0)
        if logits.dim() == 3:
            logits = logits[:, -1, :]      # (bsz, vocab)

        next_token = sample_token(logits, temperature, top_p)  # (bsz,)
        output = torch.cat([output, next_token.unsqueeze(-1)], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return output


# ── Interactive loop ───────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_interactive(
    model: torch.nn.Module,
    tokenizer,
    args,
    mtp_module: Optional[MTPModule] = None,
) -> None:
    print("DeepSeek-V3-Lite  |  /exit to quit  |  /clear to reset context")
    messages = []

    decoder: Optional[SpeculativeDecoder] = None
    if mtp_module is not None and args.use_speculative:
        decoder = SpeculativeDecoder(
            model, mtp_module, acceptance_threshold=args.acceptance_threshold
        )
        print("Speculative decoding enabled.")

    eos_id = tokenizer.eos_token_id

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input == "/exit":
            break
        if user_input == "/clear":
            messages.clear()
            print("[context cleared]")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        if decoder is not None:
            output_ids = decoder.generate(
                input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
        else:
            output_ids = generate_tokens(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=eos_id,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\nAssistant: {response}")
        messages.append({"role": "assistant", "content": response})


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="Run DeepSeek-V3-Lite inference")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir or file")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_speculative", action="store_true")
    parser.add_argument("--acceptance_threshold", type=float, default=0.8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    # Initialise model from the already-loaded config dict (not the file path string)
    print("Initialising model...")
    model = Transformer(cfg).to("cuda")
    model.eval()

    # Load weights
    ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else str(Path(args.checkpoint).parent)
    ckpt_mgr = CheckpointManager(ckpt_dir)

    # Resolve step number
    if os.path.isdir(args.checkpoint):
        step = ckpt_mgr.latest_step()
        if step is None:
            raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    else:
        # Try parsing step from filename, e.g. model_step_1000.safetensors
        stem = Path(args.checkpoint).stem
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            step = ckpt_mgr.latest_step()

    print(f"Loading checkpoint step {step}...")
    ckpt_mgr.load(model, step)

    # Optional MTP draft head
    mtp_module: Optional[MTPModule] = None
    if args.use_speculative:
        try:
            mtp_module = MTPModule(model_cfg, depth=1).to("cuda")
            mtp_module.eval()
            # Try to load MTP weights from the same checkpoint dir
            mtp_path = Path(ckpt_dir) / f"mtp_step_{step}.safetensors"
            if mtp_path.exists():
                from safetensors.torch import load_file
                mtp_module.load_state_dict(load_file(str(mtp_path), device="cuda"), strict=False)
                print("MTP weights loaded.")
            else:
                print(f"[warn] MTP weights not found at {mtp_path}; using uninitialised draft head.")
        except Exception as exc:
            print(f"[warn] Could not load MTP module: {exc}")
            mtp_module = None

    # Tokenizer
    tok_path = cfg.get("data", {}).get("tokenizer_path", "deepseek-ai/deepseek-coder-v2-lite")
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    generate_interactive(model, tokenizer, args, mtp_module)


if __name__ == "__main__":
    main()
