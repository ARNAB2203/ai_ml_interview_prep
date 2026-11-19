"""Validate a DeepSpeed ZeRO configuration and estimate optimizer-state sharding.

Default CPU-safe run (does not require DeepSpeed, CUDA, or multiple GPUs):
    python examples/framework_demos/deepspeed_zero_config.py

Use the handbook's GPU launch command only on a compatible multi-GPU host.
"""

# %% Imports
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "deepspeed_zero2.json"


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    zero = config.get("zero_optimization")
    if not isinstance(zero, dict) or zero.get("stage") not in {1, 2, 3}:
        raise ValueError("zero_optimization.stage must be 1, 2, or 3")
    if config.get("train_batch_size", 0) <= 0:
        raise ValueError("train_batch_size must be positive")
    if config.get("gradient_accumulation_steps", 0) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return config


def estimate_memory_gib(parameter_count: int, world_size: int, zero_stage: int) -> dict[str, float]:
    """Approximate fp16 param + fp32 optimizer/gradient memory per rank.

    This intentionally illustrates a planning model, not a replacement for an
    actual DeepSpeed memory estimator or measured profiler trace.
    """
    if parameter_count <= 0 or world_size <= 0:
        raise ValueError("parameter_count and world_size must be positive")
    parameter_bytes = 2  # fp16/bf16 parameter
    gradient_bytes = 2
    optimizer_bytes = 4 + 4 + 4  # fp32 master parameter plus Adam m and v
    bytes_per_parameter_state = parameter_bytes + gradient_bytes + optimizer_bytes
    baseline = parameter_count * bytes_per_parameter_state
    if zero_stage == 1:
        # Parameters and gradients are replicated; optimizer states are partitioned.
        per_rank = parameter_count * (parameter_bytes + gradient_bytes) + parameter_count * optimizer_bytes / world_size
    elif zero_stage == 2:
        # Parameters stay replicated; gradients and optimizer states are partitioned.
        per_rank = parameter_count * parameter_bytes + parameter_count * (gradient_bytes + optimizer_bytes) / world_size
    elif zero_stage == 3:
        per_rank = baseline / world_size
    else:
        raise ValueError("zero_stage must be 1, 2, or 3")
    return {"baseline_gib": baseline / 2**30, "per_rank_gib": per_rank / 2**30}


def demo() -> None:
    config = load_and_validate_config()
    stage = config["zero_optimization"]["stage"]
    estimate = estimate_memory_gib(parameter_count=1_000_000_000, world_size=4, zero_stage=stage)
    print("Validated DeepSpeed config:", config)
    print("1B parameter memory estimate:", estimate)
    assert stage == 2
    assert estimate["per_rank_gib"] < estimate["baseline_gib"]
    print("PASS: DeepSpeed ZeRO config and sharding-estimate checks completed.")


if __name__ == "__main__":
    demo()
