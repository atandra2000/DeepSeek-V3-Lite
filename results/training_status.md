# Training Status

## Current Stage: Implementation / Learning

All architecture code has been written. Pre-training has **not started** yet.

---

### Implementation Checklist

| Component | File | Status |
|---|---|---|
| Transformer backbone | `models/transformer.py` | ✅ Written |
| Multi-Head Latent Attention | `models/mla.py` | ✅ Written |
| DeepSeekMoE + AuxLossFree gate | `models/moe.py` | ✅ Written |
| Multi-Token Prediction | `models/mtp.py` | ✅ Written |
| FP8 Triton kernels | `kernels/fp8_kernel.py` | ✅ Written |
| FP8Linear drop-in | `kernels/gemm.py` | ✅ Written |
| Pre-training loop (FSDP) | `training/pretrain.py` | ✅ Written |
| SFT trainer | `training/sft.py` | ✅ Written |
| GRPO trainer | `training/rl.py` | ✅ Written |
| R1 distillation | `training/distillation.py` | ✅ Written |
| Autoregressive inference | `inference/generate.py` | ✅ Written |
| Speculative decoding | `inference/speculative.py` | ✅ Written |
| Checkpoint manager | `utils/checkpoint.py` | ✅ Written |
| MoE All-to-All comms | `utils/communication.py` | ✅ Written |
| Distributed utilities | `utils/distributed.py` | ✅ Written |
| Distributed logger | `utils/logging.py` | ✅ Written |
| Data preparation | `data/prepare_data.py` | ✅ Written |
| Pre-training config | `configs/pretrain_config.yaml` | ✅ Written |
| Post-training config | `configs/post-train_config.yaml` | ✅ Written |

---

### Expected Pre-training Configuration

| Hyperparameter | Value |
|---|---|
| Hardware | 8 × RTX 5090 (32 GB each) |
| Effective batch | 2 micro × 4 grad_accum × 8 GPUs = 64 |
| Learning rate | 2.2e-4 (WarmupCosine, 500-step warmup) |
| Total steps | 50 000 |
| Precision | FP8 forward + BF16 master weights |
| Parallelism | FSDP full-sharding |
| MTP loss weight | 0.3 |
| Checkpointing | safetensors (atomic temp-rename) |
| Tracking | Weights & Biases |

---

*This file will be updated with loss curves and benchmark results once training begins.*
