# training/distillation.py
import sys
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import yaml

sys.path.append(str(Path(__file__).parent.parent))


class ReasoningDistillation:
    """
    Knowledge Distillation from a DeepSeek-R1-style reasoning teacher.

    The combined objective is:
        L = α · KL(student ‖ teacher) + (1-α) · CE(student, labels)

    Temperature scaling on the teacher's distribution encourages the student
    to mimic the soft probability mass across plausible tokens.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: dict,
        tokenizer=None,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.tokenizer = tokenizer

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.get("lr", 1e-5),
            weight_decay=0.01,
        )

        max_steps = config.get("max_steps", 1000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=config.get("lr", 1e-5) * 0.1
        )

        self.scaler = GradScaler("cuda")
        self.temperature = config.get("temperature", 2.0)
        self.alpha = config.get("distill_alpha", 0.7)

    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_teacher_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a long-CoT response from the teacher model."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer required for teacher generation")

        ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        out = self.teacher.generate(
            ids, max_new_tokens=max_tokens, temperature=0.7, top_p=0.9
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ──────────────────────────────────────────────────────────────────────

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL(student ‖ teacher) with temperature scaling, scaled by T²."""
        T = self.temperature
        s = F.log_softmax(student_logits / T, dim=-1)
        t = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (T ** 2)

    # ──────────────────────────────────────────────────────────────────────

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single distillation step."""
        self.student.train()

        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        with autocast("cuda", dtype=torch.bfloat16):
            student_logits = self.student(input_ids)    # (bsz, seq, vocab)

            with torch.no_grad():
                teacher_logits = self.teacher(input_ids)

            # Flatten for loss computation
            s_flat = student_logits.reshape(-1, student_logits.size(-1))
            t_flat = teacher_logits.reshape(-1, teacher_logits.size(-1))

            distill_loss = self.compute_distillation_loss(s_flat, t_flat)
            task_loss = F.cross_entropy(s_flat, labels.reshape(-1), ignore_index=-100)

            total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item(),
        }


# ── Data helpers ───────────────────────────────────────────────────────────────

def prepare_distillation_data(output_path: str = "data/distill_data.json") -> str:
    """Seed distillation dataset with a reasoning example."""
    examples = [
        {
            "prompt": "A train travels 300 miles in 5 hours. What is its speed?",
            "teacher_response": (
                "Step 1: Identify knowns — distance = 300 mi, time = 5 h\n"
                "Step 2: Speed = distance / time = 300 / 5 = 60 mph\n"
                "Step 3: Verify: 60 × 5 = 300 ✓\n\nAnswer: \\boxed{60} mph"
            ),
            "student_response": "Speed = 300 / 5 = 60 mph → \\boxed{60} mph",
        }
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Distillation data → {output_path}")
    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True,
                        help="YAML model config path (shared architecture for student and teacher)")
    parser.add_argument("--student-path", type=str, required=True,
                        help="Checkpoint directory for the student model")
    parser.add_argument("--teacher-path", type=str, required=True,
                        help="Checkpoint directory for the teacher model")
    parser.add_argument("--tokenizer",    type=str, default="deepseek-ai/deepseek-r1")
    parser.add_argument("--data-path",    type=str, default="data/distill_data.json")
    parser.add_argument("--output-path",  type=str, default="checkpoints/distill")
    parser.add_argument("--epochs",       type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        prepare_distillation_data(args.data_path)

    from models.transformer import Transformer
    from utils.checkpoint import CheckpointManager
    from transformers import AutoTokenizer

    # Both student and teacher share the same architecture config
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    print("Initialising student model...")
    student = Transformer(yaml_cfg).cuda()
    student_ckpt = CheckpointManager(args.student_path)
    student_step = student_ckpt.latest_step()
    if student_step is not None:
        print(f"  Loading student checkpoint step {student_step}")
        student_ckpt.load(student, student_step, device="cuda", strict=False)

    print("Initialising teacher model...")
    teacher = Transformer(yaml_cfg).cuda()
    teacher_ckpt = CheckpointManager(args.teacher_path)
    teacher_step = teacher_ckpt.latest_step()
    if teacher_step is not None:
        print(f"  Loading teacher checkpoint step {teacher_step}")
        teacher_ckpt.load(teacher, teacher_step, device="cuda", strict=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    config = {
        "lr": args.lr,
        "distill_alpha": 0.7,
        "temperature": 2.0,
        "max_steps": args.epochs * 100,   # approximate
    }
    trainer = ReasoningDistillation(student, teacher, config, tokenizer)

    with open(args.data_path) as f:
        data = json.load(f)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for ex in data:
            # In production use a real DataLoader; this is a minimal loop
            input_ids = ex.get("input_ids", [])
            labels = ex.get("labels", [])
            if not input_ids:
                continue
            batch = {
                "input_ids": torch.tensor([input_ids]),
                "labels": torch.tensor([labels]),
            }
            losses = trainer.train_step(batch)
            epoch_loss += losses["total_loss"]
            n += 1

        avg = epoch_loss / max(n, 1)
        print(f"Epoch {epoch+1}/{args.epochs} — Loss: {avg:.4f}")

        ckpt = Path(args.output_path) / f"distill_epoch_{epoch+1}.pt"
        torch.save(student.state_dict(), ckpt)
        print(f"Checkpoint → {ckpt}")

    print("Distillation complete.")
