# Module 11 — Distributed Training, DeepSpeed, and Production Inference

**Level:** Master  
**Interview goal:** Reason from model size, memory, topology, and failure mode
to a distributed design instead of treating `deepspeed` as a launch command.

## 1. Why distributed training exists

Large-model training consumes memory for parameters, gradients, optimizer state,
activations, communication buffers, and input batches. GPU memory is frequently
the limiting resource before raw FLOPS are.

```text
model + optimizer + gradients + activations + communication buffers
                              ↓
      one accelerator does not fit / train fast enough / meet batch target
                              ↓
data parallelism, sharding, pipeline/tensor parallelism, accumulation, offload
```

Start with measurement: model parameter count/dtype, sequence length, batch
shape, activation checkpointing, optimizer, GPU VRAM, interconnect, target
throughput, and recovery SLO. Then choose a parallelism strategy.

## 2. DeepSpeed ZeRO fundamentals

ZeRO partitions model-training state across data-parallel ranks.

| ZeRO stage | Sharded state | Main benefit | Main trade-off |
| --- | --- | --- | --- |
| 1 | Optimizer states | Easy first memory reduction | Parameters and gradients still replicate |
| 2 | Optimizer states + gradients | Larger memory reduction for common training | More communication and configuration complexity |
| 3 | Optimizer states + gradients + parameters | Maximum state-memory sharding | More gather/communication complexity and tuning sensitivity |

Offload can move selected state to CPU/NVMe, but it trades GPU memory for PCIe/
storage bandwidth and operational complexity. It is not free capacity.

## 3. Concept → code: validate a ZeRO-2 configuration safely

```bash
python examples/framework_demos/deepspeed_zero_config.py
```

The default lab parses
[`configs/deepspeed_zero2.json`](../configs/deepspeed_zero2.json), validates
its key invariants, and estimates per-rank state memory for a 1B-parameter
model. It runs on CPU because configuration review and capacity reasoning are
valuable before a costly multi-GPU job.

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2, "overlap_comm": true}
}
```

> **Warning:** The estimate is a planning approximation. Validate actual peak
> memory with the exact model, sequence length, kernels, activation
> checkpointing, precision, GPU, and interconnect; allocator fragmentation and
> activation memory can dominate.

## 4. Real GPU launch checklist

Run an actual DeepSpeed job only on a compatible CUDA/multi-GPU environment
after validating drivers, package compatibility, storage, and a small smoke
batch. The command shape is typically:

```bash
deepspeed --num_gpus=4 train.py \
  --deepspeed configs/deepspeed_zero2.json \
  --model_revision APPROVED_REVISION \
  --output_dir /durable/checkpoints/run-001
```

This handbook intentionally does **not** claim that this command runs in a
CPU-only notebook. A production `train.py` must handle dataset lineage,
distributed sampler, tokenizer/model revision, checkpoint resume, metric
aggregation, signal handling, and artifact publication.

| Preflight | What to verify |
| --- | --- |
| Topology | GPU count/type/VRAM, CUDA and NCCL compatibility, interconnect bandwidth |
| Batch math | Global batch = micro-batch × gradient accumulation × data-parallel world size |
| Precision | BF16/FP16 support, loss scaling policy, numerical stability checks |
| Data | Sharding, deterministic split, no duplicate/leaked examples, throughput to each worker |
| Checkpoints | Atomic durable writes, rank-safe resume, retention and corruption checks |
| Observability | Throughput, tokens/sec, GPU memory/utilization, communication time, loss/grad norms, failures |

## 5. Common incident patterns

| Symptom | Likely investigation path |
| --- | --- |
| Out-of-memory at first step | Model/activation estimate, sequence length, micro-batch, precision, ZeRO stage, checkpointing |
| OOM after many steps | Memory leak, retained graph/tensors, evaluation accumulation, fragmentation, dataloader behavior |
| GPU underutilization | Input pipeline, host-to-device copies, synchronization, small batches, communication imbalance |
| Hang at initialization | Rank/world-size configuration, network interface, NCCL environment, inconsistent package/driver versions |
| Divergent loss | Data/label issue, learning rate, precision overflow, gradient clipping, resume state mismatch |
| Non-reproducible result | Seed propagation, data shuffle, kernel nondeterminism, checkpoint/config/model revision gaps |

## 6. Distributed training versus distributed inference

Do not use training terminology blindly for serving.

| Topic | Training | Inference / agent runtime |
| --- | --- | --- |
| Primary objective | Optimize model weights | Meet latency, throughput, cost, and availability targets |
| State | Optimizer/gradients/checkpoints | KV cache, requests, sessions, model artifacts |
| Scaling | Workers cooperate on one update | Replicas/batching/queues route independent requests |
| Failure recovery | Resume consistent checkpoint | Retry safely, route around unhealthy replica, preserve request semantics |
| Evaluation | Train/validation/test + offline quality | Offline gate + canary + online task/safety metrics |

For agent systems, combine model-serving capacity with tool/retrieval limits.
Increasing model replicas does not protect a downstream tool from a fan-out loop;
admission control, per-tenant budgets, concurrency limits, and cancellation are
still required.

## 7. Master-level interview drill

**Prompt:** “Fine-tune a 7B model for domain classification on four GPUs with
limited VRAM and weekly releases.”

Strong answer sequence:

1. Confirm whether full tuning is necessary; establish RAG/prompt/base/LoRA
   baselines on a versioned dataset.
2. Choose PEFT/LoRA first; use QLoRA only after validating GPU and quantization
   compatibility, quality, and serving path.
3. Derive global batch and memory estimates; start with a small micro-batch,
   gradient accumulation, BF16, activation checkpointing, and a ZeRO stage that
   matches measured pressure.
4. Define checkpoint/resume, failure, artifact, and dataset lineage behavior
   before the first long job.
5. Gate each weekly release on held-out quality/safety/calibration slices,
   canary serving, cost/latency, and rollback to the prior adapter/model.

## 8. Stack coverage checklist

| Requested topic | Runnable/local artifact | Prerequisite |
| --- | --- | --- |
| DeepSpeed / ZeRO | `deepspeed_zero_config.py` | None for config validation; CUDA + DeepSpeed for an actual distributed job |
| QLoRA | `qlora_prerequisites()` plus Module 10 Colab cell | Compatible GPU + bitsandbytes + reviewed base model |
| PEFT / LoRA | `pytorch_hf_peft.py` | CPU framework requirements |
| PyTorch / Hugging Face | `pytorch_hf_peft.py` | CPU framework requirements |
| JAX | `jax_training_patterns.py` | CPU framework requirements |

Continue with [Module 12: Data, Math, and Classical ML Foundations](12_data_math_and_classical_ml_foundations.md).
