"""Tests for the optional ML/DS foundations extension."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


DEMO_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIRECTORY))

from data_classical_ml import (  # noqa: E402
    gradient_descent_linear_regression,
    make_demo_events,
    split_by_customer,
    validate_event_contract,
)
from deep_learning_nlp_foundations import (  # noqa: E402
    gradient_check,
    scaled_dot_product_attention,
)
from hybrid_retrieval_operations import (  # noqa: E402
    EvaluationCase,
    build_demo_index,
    evaluate_retrieval,
    little_law_concurrency,
)
from mlops_lifecycle import LocalModelRegistry, ExperimentRun, population_stability_index  # noqa: E402
from model_serving_api import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import numpy as np  # noqa: E402


class DataAndMathTests(unittest.TestCase):
    def test_contract_split_and_gradient_descent(self) -> None:
        frame = make_demo_events(rows=80)
        self.assertEqual(validate_event_contract(frame), [])
        train, test = split_by_customer(frame)
        self.assertFalse(set(train["customer_id"]).intersection(test["customer_id"]))
        initial_loss, final_loss = gradient_descent_linear_regression()
        self.assertLess(final_loss, initial_loss)


class DeepLearningTests(unittest.TestCase):
    def test_gradients_and_causal_mask(self) -> None:
        gradients = gradient_check()
        attention = scaled_dot_product_attention()
        self.assertLess(abs(gradients["autodiff"] - gradients["finite_difference"]), 0.02)
        self.assertEqual(attention["masked_attention_mass"], 0.0)


class RetrievalTests(unittest.TestCase):
    def test_hybrid_metrics_and_capacity_math(self) -> None:
        index = build_demo_index()
        metrics = evaluate_retrieval(
            index,
            [EvaluationCase("acme", "canary rollback", frozenset({"acme-canary-v1"}))],
        )
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(little_law_concurrency(10, 2), 20)


class MlopsTests(unittest.TestCase):
    def test_registry_gate_and_drift(self) -> None:
        registry = LocalModelRegistry()
        registry.register(
            ExperimentRun("run-1", "code-a", "data-a", "model-a", {"pr_auc": 0.9, "safety_pass_rate": 1.0}, {"us": 0.9, "eu": 0.88})
        )
        decision = registry.promote("run-1", minimum_pr_auc=0.8, maximum_segment_gap=0.1)
        self.assertTrue(decision.allowed)
        psi = population_stability_index(np.linspace(0, 1, 100), np.linspace(0.5, 1.5, 100))
        self.assertGreater(psi, 0.1)


class ServingTests(unittest.TestCase):
    def test_health_and_prediction_contract(self) -> None:
        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            response = client.post("/predict", json={"duration_ms": 1_000, "error_count": 1, "region": "us"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("risk_probability", response.json())


if __name__ == "__main__":
    unittest.main(verbosity=2)
