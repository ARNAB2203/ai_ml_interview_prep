# Module 10 — PyTorch, JAX, Hugging Face, PEFT, LoRA, and QLoRA

**Level:** Intermediate → Master  
**Interview goal:** Explain how to train, adapt, evaluate, and operate models
without treating GPU hardware or a fine-tuning library as magic.

## 1. Stack map

| Technology | Primary strength | Senior interview focus |
| --- | --- | --- |
| PyTorch | Imperative/eager tensor programming and broad production ecosystem | Autograd, modules, data pipelines, checkpoints, distributed primitives |
| JAX | Functional transformations and compiler-oriented numerical programming | PRNG keys, `jit`, `grad`, `vmap`, shapes, compilation/cache trade-offs |
| Hugging Face Transformers | Model/config/tokenizer/trainer ecosystem | Revision pinning, local versus Hub artifacts, model cards, tokenization, licenses |
| PEFT | Parameter-efficient adaptation methods | Trainable adapter surface, base-model compatibility, merge/export/evaluation |
| LoRA | Low-rank update adapters | Rank, alpha/scaling, target modules, dropout, adapter composition |
| QLoRA | Quantized base weights plus LoRA training | 4-bit quantization, GPU/VRAM requirements, bitsandbytes compatibility, quality gate |

## 2. PyTorch fundamentals

The basic loop is simple; a production-grade loop records much more than loss.

```text
versioned data → deterministic split/seed → batch → forward → loss
                                               ↓       ↑
checkpoint + metrics ← evaluation ← optimizer ← backward/gradient controls
```

| Concept | What to explain |
| --- | --- |
| `nn.Module` | Parameter ownership and composable forward computation |
| Autograd | Builds gradients through tensor operations; use `no_grad` for evaluation/inference |
| Optimizer | Updates parameters; scheduler, clipping, and mixed precision are separate policies |
| Data split | Prevent leakage by splitting on entity/time/source as required, not random rows by habit |
| Checkpoint | Save model/adapters, optimizer/scheduler state, seed, data/code/config versions, and metric provenance |
| Evaluation | Aggregate and slice by meaningful cohorts; calibrate confidence where decisions depend on it |

### Concept → code: tiny CPU classifier

The framework lab trains a small synthetic PyTorch classifier and asserts that
loss falls. The dataset is deliberately tiny so you can focus on the mechanics
before adding datasets, GPUs, or distributed workers.

```bash
python examples/framework_demos/pytorch_hf_peft.py
```

Expected observations:

1. The post-training loss is lower than initial loss.
2. The Hugging Face model is created from a local `BertConfig`; nothing is
   downloaded from the Hub.
3. The LoRA run has fewer trainable parameters than total parameters.

## 3. JAX fundamentals

JAX transforms pure numerical functions. It rewards functional state handling
and exact shape discipline.

| Transform | Purpose | Common trap |
| --- | --- | --- |
| `jax.grad` | Differentiate a scalar loss | Mutating parameters or hiding non-array state |
| `jax.jit` | Compile repeated numerical work | Unexpected recompilation from changing shapes/static arguments |
| `jax.vmap` | Vectorize a function over a batch axis | Assuming it is a substitute for distributed data sharding |
| `jax.random.split` | Produce explicit independent PRNG keys | Reusing a key and believing results are independent |

```bash
python examples/framework_demos/jax_training_patterns.py
```

The demo uses a seeded linear problem and explicitly splits PRNG keys, computes
a gradient, JIT-compiles an update, and vectorizes per-example loss. It is a
small but real CPU exercise.

> **Interview probe:** “When would you prefer JAX over PyTorch?”
>
> Discuss team expertise, existing ecosystem, accelerator/distributed strategy,
> XLA compilation characteristics, debugging ergonomics, deployment tooling,
> and benchmarked workload shape. Do not claim one is universally faster.

## 4. Hugging Face engineering discipline

Hugging Face provides useful model, tokenizer, dataset, and training utilities.
For a production artifact, pin more than a model name.

| Record | Why |
| --- | --- |
| Model repository and immutable revision/digest | Reproduces weights and detects unexpected upstream change |
| Tokenizer files/revision | Token IDs and context budget must match the model/evaluation |
| Model card/license | Establishes intended use, limitations, attribution, and redistribution obligations |
| Configuration | Architecture, max length, generation settings, quantization/adapters |
| Security posture | `trust_remote_code` decision, scanned artifacts, isolated loading environment |
| Evaluation report | Task/safety slices, baseline comparison, known limitations |

The included lab uses `BertConfig` and `BertForSequenceClassification` created
locally. In a real Hub workflow, use a pinned revision and an approved artifact
registry rather than an unpinned runtime download.

## 5. PEFT and LoRA

Instead of updating every base-model weight, LoRA adds low-rank matrices to
selected linear layers. Conceptually:

```text
W' = W + scale × B × A

W       frozen base projection
A, B    small trainable low-rank adapters
scale   derived from alpha/rank policy
```

| Knob | Effect | Trade-off |
| --- | --- | --- |
| Rank `r` | Adapter capacity | More memory/compute and possible overfit |
| `lora_alpha` | Scaling of update | Changes effective update magnitude |
| Target modules | Which projections adapt | Too narrow underfits; too broad erodes PEFT benefit |
| Dropout | Regularization | May hurt tiny-data convergence if overused |
| Dataset quality | Controls learned behavior | Bad labels/data lineage overwhelm clever adapter settings |

### Concept → code: local LoRA step

`run_lora_step()` in
[`pytorch_hf_peft.py`](../examples/framework_demos/pytorch_hf_peft.py) wraps a
tiny random-initialized BERT classifier with PEFT `LoraConfig`, runs one backward
step, and reports trainable versus total parameters. It has no provider key,
Hub download, or GPU requirement.

## 6. QLoRA is an accelerator path, not a CPU checkbox

QLoRA normally loads a base model with 4-bit quantization (often NF4), keeps
the quantized base frozen, and trains LoRA adapters with carefully selected
compute dtype. It is valuable when VRAM is constrained, but its compatibility
and quality depend on hardware, CUDA, bitsandbytes, model architecture, and
sequence/batch length.

```python
# Colab GPU cell — intentionally opt-in and model-specific.
%pip install -q transformers peft accelerate bitsandbytes datasets

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
# Pin an approved model revision in a real project; do not use an unreviewed default.
# base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
#                                              quantization_config=quantization)
# adapter = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
#                                           target_modules=["q_proj", "v_proj"]))
```

Before release, evaluate the adapter against a held-out task/safety set; save
base-model revision, tokenizer, quantization configuration, adapter weights,
training data lineage, code/config digest, and metrics. Decide whether to serve
base + adapter or merge/export only after rollback and compatibility review.

## 7. Fine-tuning decision framework

| Option | Favor when | Main risk |
| --- | --- | --- |
| Prompt/RAG improvement | Knowledge or instruction is the gap | Treating a behavior/capability problem as a retrieval problem |
| Fine-tune / LoRA | Stable, repeated behavior with high-quality examples | Overfitting, data leakage, unmeasured safety regression |
| QLoRA | Large model adaptation must fit constrained GPU memory | Hardware/library coupling and performance trade-offs |
| Full fine-tune | Evidence supports it and resources/governance justify it | High cost, operational complexity, catastrophic regression |
| Distillation | Smaller/cheaper model needs a measurable target | Teacher leakage and loss of edge-case behavior |

## 8. Interview drill: adapt an agent evaluator model

**Prompt:** “Your team wants to fine-tune a model that judges agent answers.”

Strong answer:

1. Clarify decision impact; a judge should not become the sole safety gate.
2. Define labels, annotator guidelines, data lineage, privacy/tenant consent,
   leakage controls, and temporal split.
3. Establish prompt/RAG and base-model baselines before fine-tuning.
4. Start with a small LoRA experiment, record all artifact versions, and inspect
   calibration/slice metrics plus agreement with human labels.
5. Canary the judge as advisory; monitor disagreements and drift; retain the
   previous evaluator and rollback path.

Proceed to [Module 11: Distributed Training and DeepSpeed](11_distributed_training_and_deepspeed.md).
