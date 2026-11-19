"""Local experiment registry, promotion gate, drift monitoring, and safe traces.

Run after installing examples/requirements-ml-ds.txt:
    python examples/ml_ds_demos/mlops_lifecycle.py

This is a dependency-light model of MLflow/W&B/DVC/managed-registry concepts.
It demonstrates the contract those systems should preserve: provenance,
measurable promotion, rollback, monitored data/segment behavior, and redacted
operational evidence.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    code_version: str
    data_version: str
    model_version: str
    metrics: Mapping[str, float]
    segment_recall: Mapping[str, float]


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    reasons: tuple[str, ...]
    candidate_run_id: str
    rollback_run_id: str | None


class LocalModelRegistry:
    """In-memory stand-in for a governed experiment/registry control plane."""

    def __init__(self) -> None:
        self._runs: dict[str, ExperimentRun] = {}
        self.champion_run_id: str | None = None

    def register(self, run: ExperimentRun) -> None:
        if run.run_id in self._runs:
            raise ValueError("run_id must be immutable and unique")
        self._runs[run.run_id] = run

    def promote(self, candidate_run_id: str, minimum_pr_auc: float, maximum_segment_gap: float) -> PromotionDecision:
        candidate = self._runs[candidate_run_id]
        reasons: list[str] = []
        if candidate.metrics.get("pr_auc", 0.0) < minimum_pr_auc:
            reasons.append("PR-AUC below release floor")
        recall_values = list(candidate.segment_recall.values())
        if not recall_values:
            reasons.append("missing segment recall evidence")
        elif max(recall_values) - min(recall_values) > maximum_segment_gap:
            reasons.append("segment recall gap exceeds policy")
        if candidate.metrics.get("safety_pass_rate", 0.0) < 1.0:
            reasons.append("safety regression detected")
        decision = PromotionDecision(
            allowed=not reasons,
            reasons=tuple(reasons),
            candidate_run_id=candidate_run_id,
            rollback_run_id=self.champion_run_id,
        )
        if decision.allowed:
            self.champion_run_id = candidate_run_id
        return decision


# %% Step 1 — data/prediction drift and slice quality are separate signals
def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate a simple PSI; use domain validation before acting on it."""
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("both reference and current distributions are required")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("reference distribution has insufficient variation")
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (reference_counts.sum() + epsilon * len(reference_counts))
    current_share = (current_counts + epsilon) / (current_counts.sum() + epsilon * len(current_counts))
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def monitoring_action(psi: float, segment_gap: float) -> str:
    if segment_gap > 0.15:
        return "page_owner_and_hold_promotion_for_segment_investigation"
    if psi > 0.25:
        return "open_drift_investigation_and_replay_evaluation"
    if psi > 0.10:
        return "watch_and_compare_next_window"
    return "healthy"


# %% Step 2 — trace attributes are useful only if they remain privacy-safe
SENSITIVE_ATTRIBUTE_NAMES = {"prompt", "response", "email", "access_token", "authorization"}


def redact_trace_attributes(attributes: Mapping[str, str | float | int]) -> dict[str, str | float | int]:
    return {
        key: "[REDACTED]" if key.lower() in SENSITIVE_ATTRIBUTE_NAMES else value
        for key, value in attributes.items()
    }


def demo() -> None:
    registry = LocalModelRegistry()
    registry.register(
        ExperimentRun(
            "baseline-2026-01",
            "git:abc123",
            "events:2026-01-01",
            "risk-model:1.0.0",
            {"pr_auc": 0.78, "safety_pass_rate": 1.0},
            {"us": 0.84, "eu": 0.81, "apac": 0.80},
        )
    )
    registry.champion_run_id = "baseline-2026-01"
    registry.register(
        ExperimentRun(
            "candidate-2026-02",
            "git:def456",
            "events:2026-02-01",
            "risk-model:1.1.0",
            {"pr_auc": 0.82, "safety_pass_rate": 1.0},
            {"us": 0.86, "eu": 0.84, "apac": 0.82},
        )
    )
    promotion = registry.promote("candidate-2026-02", minimum_pr_auc=0.80, maximum_segment_gap=0.08)

    rng = np.random.default_rng(9)
    reference = rng.normal(0, 1, size=1_000)
    current = rng.normal(0.9, 1.2, size=1_000)
    psi = population_stability_index(reference, current)
    action = monitoring_action(psi, segment_gap=0.04)
    safe_trace = redact_trace_attributes(
        {"trace_id": "trace-42", "tenant": "acme", "prompt": "customer-provided content", "latency_ms": 118}
    )

    print("Promotion decision:", promotion)
    print("Drift observation:", {"psi": psi, "action": action})
    print("Safe trace attributes:", safe_trace)
    assert promotion.allowed and registry.champion_run_id == "candidate-2026-02"
    assert action in {"open_drift_investigation_and_replay_evaluation", "watch_and_compare_next_window"}
    assert safe_trace["prompt"] == "[REDACTED]"
    print("PASS: provenance, promotion, drift, and trace-redaction checks completed.")


if __name__ == "__main__":
    demo()
