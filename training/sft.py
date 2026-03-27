# training/sft.py
import sys
from pathlib import Path
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import yaml

sys.path.append(str(Path(__file__).parent.parent))


# ── Dataset ────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset with chat-template formatting."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 8192) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        messages = item["messages"]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        tokens = torch.tensor(token_ids, dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]

        # Per-token loss mask (1 = train on this token)
        # Sample isolation: mask assistant tokens as 1, prompt as 0 (simplified: all 1)
        loss_mask = torch.ones_like(y, dtype=torch.float)

        return x, y, loss_mask


def sft_collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), -100, dtype=torch.long)   # -100 ignored by CE
    masks_pad = torch.zeros(len(masks), max_len, dtype=torch.float)

    for i, (x, y, m) in enumerate(zip(xs, ys, masks)):
        n = x.size(0)
        xs_pad[i, :n] = x
        ys_pad[i, :n] = y
        masks_pad[i, :n] = m

    return xs_pad, ys_pad, masks_pad


# ── Trainer ────────────────────────────────────────────────────────────────────

class SFTTrainer:
    """Supervised Fine-Tuning trainer with BF16 AMP."""

    def __init__(self, model: nn.Module, tokenizer, config: dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 5e-6),
            weight_decay=0.1,
        )

        max_steps = config.get("max_steps", 1000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=config.get("min_lr", 1e-6)
        )

        self.scaler = GradScaler("cuda")

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, loss_mask in dataloader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            loss_mask = loss_mask.cuda(non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x)                          # (bsz, seq, vocab)
                # Masked CE loss
                per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                loss = (per_token * loss_mask.reshape(-1)).sum() / loss_mask.sum().clamp(min=1e-10)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )


# ── Data preparation ───────────────────────────────────────────────────────────

def prepare_sft_data(output_path: str = "data/sft_data.json") -> str:
    """Create a minimal SFT dataset for bootstrapping."""
    reasoning = [
        {
            "messages": [
                {"role": "user", "content": "Solve: 2x + 5 = 15"},
                {
                    "role": "assistant",
                    "content": (
                        "Subtract 5 from both sides: 2x = 10\n"
                        "Divide by 2: x = 5\n\nAnswer: \\boxed{5}"
                    ),
                },
            ]
        },
    ]
    non_reasoning = [
        {
            "messages": [
                {"role": "user", "content": "Write a haiku about AI"},
                {
                    "role": "assistant",
                    "content": "Silicon dreams wake,\nNeural paths through data streams,\nWisdom learns to grow.",
                },
            ]
        },
    ]

    data = reasoning + non_reasoning
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Prepared {len(data)} SFT examples → {output_path}")
    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, required=True,
                        help="YAML model config path (e.g. configs/pretrain_config.yaml)")
    parser.add_argument("--model-path",  type=str, default=None,
                        help="Checkpoint directory to resume from (optional)")
    parser.add_argument("--data-path",   type=str, default="data/sft_data.json")
    parser.add_argument("--output-path", type=str, default="checkpoints/sft")
    parser.add_argument("--tokenizer",   type=str, default="deepseek-ai/deepseek-coder-v2-lite")
    parser.add_argument("--epochs",      type=int, default=2)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--lr",          type=float, default=5e-6)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        prepare_sft_data(args.data_path)

    from models.transformer import Transformer
    from utils.checkpoint import CheckpointManager
    from transformers import AutoTokenizer

    # Load model config from YAML then initialise the model
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    print("Initialising model from config...")
    model = Transformer(yaml_cfg).cuda()

    # Optionally load pretrained weights from a checkpoint directory
    if args.model_path is not None:
        ckpt_mgr = CheckpointManager(args.model_path)
        step = ckpt_mgr.latest_step()
        if step is not None:
            print(f"Loading checkpoint step {step} from {args.model_path}")
            ckpt_mgr.load(model, step, device="cuda", strict=False)
        else:
            print(f"[warn] No checkpoints found in {args.model_path}; starting from scratch.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = SFTDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    config = {
        "lr": args.lr,
        "min_lr": args.lr * 0.1,
        "max_steps": args.epochs * len(dataloader),
        "output_path": args.output_path,
    }
    trainer = SFTTrainer(model, tokenizer, config)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {loss:.4f}")
        ckpt = Path(args.output_path) / f"sft_epoch_{epoch+1}.pt"
        trainer.save_checkpoint(str(ckpt))

    print("SFT complete.")


if __name__ == "__main__":
    main()
