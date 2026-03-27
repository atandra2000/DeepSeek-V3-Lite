# data/prepare_data.py
"""
Data preparation pipeline for DeepSeek-V3-Lite.

Stages:
  pretrain  — download datasets, tokenise, pack into a flat binary tensor
  sft       — produce a minimal instruction-tuning JSON
  all       — both stages
"""
import argparse
import json
import os
from pathlib import Path

import torch


# ── Dataset download ───────────────────────────────────────────────────────────

# Ordered by priority; community mirrors / smaller splits used for fast demo
_DATASETS = [
    ("HuggingFaceFW/fineweb-edu", "data/fineweb"),   # high-quality web text. Size: 1.5GB
    ("bigcode/the-stack-v2-train-smol-ids", "data/code"),  # code-focused subset of The Stack. Size: 1.2GB
    ("lighteval/MATH", "data/math"),  # math problems for reasoning diversity. Size: 0.5GB
]


def download_and_prepare_dataset(output_dir: str = "data/datasets") -> str:
    """Download HuggingFace datasets and save as JSONL shards."""
    from datasets import load_dataset  # lazy import

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ok = 0

    for name, subdir in _DATASETS:
        try:
            print(f"Downloading {name} …")
            # Fetch a small sample for quick bootstrapping
            ds = load_dataset(name, split="train[:5000]", trust_remote_code=True)
            out_path = Path(output_dir) / subdir
            out_path.mkdir(parents=True, exist_ok=True)
            ds.to_json(str(out_path / "data.jsonl"))
            print(f"  → saved {len(ds):,} examples to {out_path}")
            ok += 1
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")

    if ok == 0:
        print("[warn] No datasets downloaded; tokenisation will produce synthetic data.")
    return output_dir


# ── Tokenisation & packing ─────────────────────────────────────────────────────

def _iter_texts(data_dir: str):
    """Yield text strings from all JSON / JSONL files under data_dir."""
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Support both list-of-dicts and single-dict JSONL
                        if isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, dict) and isinstance(item.get("text"), str):
                                    yield item["text"]
                        elif isinstance(obj, dict):
                            text = obj.get("text") or obj.get("content") or obj.get("problem") or ""
                            if text:
                                yield text
            except OSError as exc:
                print(f"  [warn] Could not read {fpath}: {exc}")


def tokenize_and_pack(
    data_dir: str,
    output_path: str,
    vocab_size: int = 102400,
    tokenizer=None,
    max_tokens: int = 50_000_000,
) -> str:
    """
    Tokenise all text files and concatenate into a flat token tensor.

    Args:
        data_dir:    directory containing .json / .jsonl files
        output_path: destination .bin path
        vocab_size:  fallback character tokenisation vocab size
        tokenizer:   HuggingFace tokenizer (optional; preferred)
        max_tokens:  cap on total tokens saved (50M default)

    Returns:
        output_path
    """
    all_tokens: list[int] = []
    n_texts = 0
    skipped = 0

    for text in _iter_texts(data_dir):
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                toks = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                toks = [ord(c) % vocab_size for c in text[:2000]]
        else:
            # Fallback: byte-level character encoding
            toks = [ord(c) % vocab_size for c in text[:2000]]

        all_tokens.extend(toks)
        n_texts += 1

        if len(all_tokens) >= max_tokens:
            print(f"  Reached {max_tokens:,} token cap after {n_texts:,} documents.")
            break

    if not all_tokens:
        print("[warn] No tokens collected — generating synthetic random data.")
        all_tokens = torch.randint(0, vocab_size, (1_000_000,)).tolist()
        skipped = -1

    tensor = torch.tensor(all_tokens[:max_tokens], dtype=torch.long)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)

    print(
        f"Saved {len(tensor):,} tokens to {output_path}  "
        f"(texts={n_texts}, skipped={skipped})"
    )
    return output_path


# ── SFT data ───────────────────────────────────────────────────────────────────

def prepare_sft_data(output_path: str = "data/sft_data.json") -> str:
    """Create a bootstrapping SFT dataset."""
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Python function to check if a number is prime."},
                {
                    "role": "assistant",
                    "content": (
                        "```python\n"
                        "def is_prime(n: int) -> bool:\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    for i in range(2, int(n**0.5) + 1):\n"
                        "        if n % i == 0:\n"
                        "            return False\n"
                        "    return True\n"
                        "```"
                    ),
                },
            ]
        },
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Prepared {len(examples)} SFT examples → {output_path}")
    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare pre-training and SFT data")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "sft", "all"],
        default="all",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name")
    parser.add_argument("--max-tokens", type=int, default=50_000_000)
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.stage in ("pretrain", "all"):
        ds_dir = download_and_prepare_dataset(os.path.join(args.output_dir, "datasets"))
        tokenize_and_pack(
            ds_dir,
            os.path.join(args.output_dir, "pretrain_data.bin"),
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
        )

    if args.stage in ("sft", "all"):
        prepare_sft_data(os.path.join(args.output_dir, "sft_data.json"))

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
