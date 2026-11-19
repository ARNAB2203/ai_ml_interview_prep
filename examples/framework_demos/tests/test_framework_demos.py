"""Framework-track tests; run them in the CPU framework virtual environment."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


DEMO_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIRECTORY))

from deepspeed_zero_config import estimate_memory_gib, load_and_validate_config  # noqa: E402
from jax_training_patterns import train  # noqa: E402
from langchain_llamaindex_rag import run_langchain_chain, run_llamaindex_retriever  # noqa: E402
from langgraph_approval_graph import run_request  # noqa: E402
from mcp_knowledge_server import calculate_capacity, lookup_runbook, validate_contract  # noqa: E402
from multi_agent_patterns import BoundedTeam, Task, TenantContext  # noqa: E402
from provider_gateway import (  # noqa: E402
    FakeModelClient,
    as_anthropic_tools,
    as_openai_tools,
    complete_with_retry,
    demo_request,
)
from pytorch_hf_peft import run_huggingface_forward, run_lora_step, train_tiny_classifier  # noqa: E402


class ProviderTests(unittest.TestCase):
    def test_fake_provider_retries_and_normalizes_tools(self) -> None:
        client = FakeModelClient(fail_once=True)
        result = complete_with_retry(client, demo_request())
        self.assertEqual(client.calls, 2)
        self.assertEqual(result.tool_calls[0].name, "calculate_capacity")
        request = demo_request()
        self.assertEqual(as_openai_tools(request.tools)[0]["function"]["name"], "calculate_capacity")
        self.assertEqual(as_anthropic_tools(request.tools)[0]["name"], "calculate_capacity")


class FrameworkRagTests(unittest.TestCase):
    def test_langchain_and_llamaindex_keep_tenant_scope(self) -> None:
        chain = run_langchain_chain("acme", "What must Acme do before a canary rollout?")
        index = run_llamaindex_retriever("acme", "deployment policy")
        self.assertEqual(chain["citations"], ["acme-deploy-v3"])
        self.assertTrue(index)
        self.assertTrue(all(item["tenant_id"] == "acme" for item in index))


class LangGraphTests(unittest.TestCase):
    def test_high_risk_side_effect_requires_approval(self) -> None:
        blocked = run_request("deploy package", approval_granted=False)
        approved = run_request("deploy package", approval_granted=True)
        self.assertEqual(blocked["result"], "approval required before side effect")
        self.assertTrue(approved["result"].startswith("executed"))


class McpTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_contract_and_tenant_guard(self) -> None:
        contract = await validate_contract()
        self.assertIn("lookup_runbook", contract["tools"])
        self.assertEqual(calculate_capacity(4, 2)["estimated_concurrent_runs"], 8)
        with self.assertRaises(PermissionError):
            lookup_runbook("acme", "eve@globex", "deployment")


class MultiAgentTests(unittest.TestCase):
    def test_side_effect_is_paused_without_trusted_approval(self) -> None:
        team = BoundedTeam()
        result = team.run(TenantContext("acme", "alice"), Task("deploy package", True), approval_granted=False)
        self.assertTrue(result.startswith("paused"))


class ModelEngineeringTests(unittest.TestCase):
    def test_cpu_training_hf_lora_and_jax(self) -> None:
        torch_result = train_tiny_classifier(steps=40)
        huggingface = run_huggingface_forward()
        lora = run_lora_step()
        jax_result = train(steps=15)
        self.assertLess(torch_result.final_loss, torch_result.initial_loss)
        self.assertEqual(huggingface["logit_shape"], (4, 2))
        self.assertLess(lora["trainable_parameters"], lora["total_parameters"])
        self.assertLess(jax_result["final_loss"], jax_result["initial_loss"])


class DeepSpeedPlanningTests(unittest.TestCase):
    def test_zero_two_config_shards_memory(self) -> None:
        config = load_and_validate_config()
        estimate = estimate_memory_gib(1_000_000, world_size=4, zero_stage=config["zero_optimization"]["stage"])
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertLess(estimate["per_rank_gib"], estimate["baseline_gib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
