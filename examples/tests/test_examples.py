"""Focused contract tests for the runnable handbook examples."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


EXAMPLES_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIRECTORY))

from agent_platform_api import AgentStore, create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from manifest_validator import (  # noqa: E402
    PackageValidationError,
    build_demo_package,
    validate_package,
)
from rag_evaluation import (  # noqa: E402
    EvaluationCase,
    GroundedResponder,
    build_demo_knowledge_base,
    evaluate,
)
from tokenizer_bpe import BPETrainer, SimpleTokenizerV1, SimpleTokenizerV2  # noqa: E402
from tool_using_agent import TenantContext, ToolCall, build_demo_agent  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def test_unknown_tokens_and_bpe_are_explicit(self) -> None:
        v1 = SimpleTokenizerV1.from_corpus("Hello, platform!")
        vocabulary = dict(v1.str_to_int)
        vocabulary["<unk>"] = len(vocabulary)
        v2 = SimpleTokenizerV2(vocabulary)
        self.assertIn(v2.str_to_int["<unk>"], v2.encode("unseen platform"))

        bpe = BPETrainer()
        merges = bpe.train("agent agents agentic", num_merges=5)
        self.assertTrue(merges)
        self.assertTrue(bpe.encode_word("agentic"))


class ToolAgentTests(unittest.TestCase):
    def test_policy_blocks_cross_tenant_tool_call(self) -> None:
        agent = build_demo_agent()
        context = TenantContext("acme", "alice", frozenset({"tools:calculate", "tools:knowledge"}))
        allowed = agent.answer(context, "calculate 7 * 6")
        denied = agent.runner.run(
            context,
            ToolCall("search_knowledge", {"query": "policy", "tenant_id": "globex"}),
        )
        self.assertEqual(allowed.output, "result=42")
        self.assertFalse(denied.success)
        self.assertEqual(denied.error_code, "cross_tenant_request")


class RagTests(unittest.TestCase):
    def test_evaluation_gate_and_tenant_filter(self) -> None:
        knowledge_base = build_demo_knowledge_base()
        responder = GroundedResponder(knowledge_base)
        report = evaluate(
            responder,
            knowledge_base,
            [
                EvaluationCase(
                    "deployment",
                    "acme",
                    "Which Acme deployment policy requires staging evaluation?",
                    frozenset({"acme-deploy-v3"}),
                )
            ],
        )
        retrieved = knowledge_base.retrieve("acme", "Globex policy human approval")
        self.assertTrue(report.passed)
        self.assertTrue(all(item.document.tenant_id == "acme" for item in retrieved))


class ApiTests(unittest.TestCase):
    def test_api_hides_cross_tenant_resource(self) -> None:
        client = TestClient(create_app(AgentStore()))
        payload = {
            "name": "policy-coach",
            "config": {
                "system_instruction": "Use only authorized policy documents.",
                "allowed_tools": ["search_knowledge"],
                "max_steps": 3,
            },
        }
        created = client.post("/agents", headers={"X-Tenant-ID": "acme"}, json=payload)
        agent_id = created.json()["id"]
        other_tenant = client.get(f"/agents/{agent_id}", headers={"X-Tenant-ID": "globex"})
        self.assertEqual(created.status_code, 201)
        self.assertEqual(other_tenant.status_code, 404)


class PackageTests(unittest.TestCase):
    def test_validator_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as directory:
            safe = Path(directory) / "safe.zip"
            unsafe = Path(directory) / "unsafe.zip"
            build_demo_package(safe)
            build_demo_package(unsafe, include_traversal_path=True)
            self.assertEqual(validate_package(safe).manifest.protocol_version, "1.0")
            with self.assertRaises(PackageValidationError):
                validate_package(unsafe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
