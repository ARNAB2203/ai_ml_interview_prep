"""CPU-safe PyTorch, Hugging Face, and LoRA foundations.

Run after installing `examples/requirements-frameworks-cpu.txt`:
    python examples/framework_demos/pytorch_hf_peft.py

The Hugging Face model is initialized from a tiny local BertConfig: it does not
download a model, dataset, or execute remote code. The QLoRA helper explains the
GPU-only configuration boundary without falsely claiming it runs on CPU.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model


def seed_everything(seed: int = 7) -> None:
    """Keep toy training repeatable; production also records data/code/config versions."""
    torch.manual_seed(seed)


# %% Step 1 — PyTorch CPU training loop with a measurable learning signal
class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float
    final_loss: float
    accuracy: float


def train_tiny_classifier(steps: int = 80) -> TrainingResult:
    seed_everything()
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((64, 2), generator=generator)
    labels = ((features[:, 0] + 0.6 * features[:, 1]) > 0).long()
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_function = nn.CrossEntropyLoss()

    initial_loss = float(loss_function(model(features), labels).item())
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()

    logits = model(features)
    final_loss = float(loss_function(logits, labels).item())
    accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    return TrainingResult(initial_loss, final_loss, accuracy)


# %% Step 2 — Hugging Face local model construction, no Hub download required
def tiny_huggingface_batch() -> tuple[BertForSequenceClassification, dict[str, torch.Tensor]]:
    seed_everything()
    config = BertConfig(
        vocab_size=64,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (4, 10)),
        "attention_mask": torch.ones((4, 10), dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    return model, batch


def run_huggingface_forward() -> dict[str, Any]:
    model, batch = tiny_huggingface_batch()
    output = model(**batch)
    assert output.loss is not None
    return {"loss": float(output.loss.item()), "logit_shape": tuple(output.logits.shape)}


# %% Step 3 — PEFT LoRA changes a small trainable adapter set, not all base weights
def run_lora_step() -> dict[str, Any]:
    model, batch = tiny_huggingface_batch()
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["query", "value"],
    )
    lora_model = get_peft_model(model, lora_config)
    trainable_parameters = [name for name, parameter in lora_model.named_parameters() if parameter.requires_grad]
    total_parameters = sum(parameter.numel() for parameter in lora_model.parameters())
    trainable_count = sum(parameter.numel() for parameter in lora_model.parameters() if parameter.requires_grad)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in lora_model.parameters() if parameter.requires_grad), lr=1e-3
    )
    output = lora_model(**batch)
    assert output.loss is not None
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    return {
        "loss": float(output.loss.item()),
        "trainable_parameters": trainable_count,
        "total_parameters": total_parameters,
        "trainable_names": trainable_parameters,
    }


def qlora_prerequisites() -> dict[str, str | bool]:
    """Describe, rather than pretend to run, the accelerator-specific QLoRA path."""
    return {
        "requires_gpu": True,
        "requires_bitsandbytes": True,
        "quantization": "4-bit NF4 base-model loading with compute dtype selected for the GPU",
        "adapter": "LoRA adapters train while quantized base weights remain frozen",
        "release_gate": "Evaluate against a held-out task/safety set; save adapter, tokenizer, base revision, and config.",
    }


def demo() -> None:
    training = train_tiny_classifier()
    huggingface = run_huggingface_forward()
    lora = run_lora_step()
    print("PyTorch training:", training)
    print("Hugging Face local forward pass:", huggingface)
    print("PEFT LoRA step:", {key: value for key, value in lora.items() if key != "trainable_names"})
    print("QLoRA prerequisites:", qlora_prerequisites())
    assert training.final_loss < training.initial_loss
    assert training.accuracy >= 0.9
    assert huggingface["logit_shape"] == (4, 2)
    assert 0 < lora["trainable_parameters"] < lora["total_parameters"]
    print("PASS: PyTorch, Transformers, and PEFT LoRA CPU checks completed.")


if __name__ == "__main__":
    demo()
