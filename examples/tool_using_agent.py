"""A deterministic, policy-controlled tool-using agent.

Run locally:
    python examples/tool_using_agent.py

Google Colab setup cell:
    # This lesson uses only the Python standard library.

Replace RuleBasedPlanner with an LLM adapter in production, but keep the typed
tool contract, deterministic policy checks, retry rules, and audit records.
"""

# %% Imports
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Mapping


# %% Step 1 — typed context and tool contracts
@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    subject_id: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, str]


ToolHandler = Callable[[TenantContext, Mapping[str, str]], str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    required_scope: str
    idempotent: bool
    handler: ToolHandler


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str
    attempts: int
    error_code: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    tenant_id: str
    subject_id: str
    tool_name: str
    outcome: str
    attempts: int
    latency_ms: float
    error_code: str | None = None


class PolicyDenied(Exception):
    """Raised when a valid-looking tool call is not authorized."""


class TransientToolError(Exception):
    """Raised by an idempotent dependency that is safe to retry."""


# %% Step 2 — planner selects a typed call; it does not execute anything
class RuleBasedPlanner:
    """A deterministic stand-in for an LLM structured-output planner."""

    def plan(self, question: str) -> ToolCall:
        lowered = question.lower().strip()
        if lowered.startswith("calculate "):
            return ToolCall("calculate", {"expression": question[len("calculate ") :]})
        if "policy" in lowered or "knowledge" in lowered or "deployment" in lowered:
            return ToolCall("search_knowledge", {"query": question})
        raise ValueError("The demo planner supports calculation and policy/knowledge questions only.")


# %% Step 3 — policy is separate from model planning
class PolicyEngine:
    """Enforces scope, tenant, and small input-budget constraints."""

    def authorize(self, context: TenantContext, spec: ToolSpec, call: ToolCall) -> None:
        if spec.required_scope not in context.scopes:
            raise PolicyDenied(f"missing_scope:{spec.required_scope}")

        requested_tenant = call.arguments.get("tenant_id")
        if requested_tenant is not None and requested_tenant != context.tenant_id:
            raise PolicyDenied("cross_tenant_request")

        if call.name == "calculate":
            expression = call.arguments.get("expression", "")
            if len(expression) > 80:
                raise PolicyDenied("expression_too_long")
        if call.name == "search_knowledge" and not call.arguments.get("query", "").strip():
            raise PolicyDenied("empty_query")


class ToolRunner:
    """Executes a registered tool only after deterministic policy checks."""

    def __init__(self, specs: list[ToolSpec], policy: PolicyEngine, max_attempts: int = 2) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._policy = policy
        self._max_attempts = max_attempts

    def run(self, context: TenantContext, call: ToolCall) -> ToolResult:
        spec = self._specs.get(call.name)
        if spec is None:
            return ToolResult(False, "", 0, "unknown_tool")

        try:
            self._policy.authorize(context, spec, call)
        except PolicyDenied as exc:
            return ToolResult(False, "", 0, str(exc))

        for attempt in range(1, self._max_attempts + 1):
            try:
                return ToolResult(True, spec.handler(context, call.arguments), attempt)
            except TransientToolError:
                if not spec.idempotent or attempt == self._max_attempts:
                    return ToolResult(False, "", attempt, "transient_dependency_error")
        raise AssertionError("All tool attempts must return or raise.")


# %% Step 4 — narrow tool implementations
def safe_calculate(_: TenantContext, arguments: Mapping[str, str]) -> str:
    """Evaluate a deliberately tiny arithmetic grammar; never use eval()."""
    expression = arguments["expression"]
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("Only numeric +, -, *, / expressions are allowed.")

    return f"result={visit(tree):g}"


class KnowledgeTool:
    """Tenant-scoped deterministic knowledge service with one simulated retry."""

    def __init__(self) -> None:
        self._retried_queries: set[tuple[str, str]] = set()
        self._documents = {
            "acme": "Acme deployment policy: evaluate in staging before a canary rollout.",
            "globex": "Globex deployment policy: regulated changes require human approval.",
        }

    def search(self, context: TenantContext, arguments: Mapping[str, str]) -> str:
        query = arguments["query"].lower()
        retry_key = (context.tenant_id, query)
        if "retry" in query and retry_key not in self._retried_queries:
            self._retried_queries.add(retry_key)
            raise TransientToolError("simulated temporary index timeout")
        return self._documents[context.tenant_id]


# %% Step 5 — the agent records an audit event around every outcome
@dataclass
class Agent:
    planner: RuleBasedPlanner
    runner: ToolRunner
    audit_events: list[AuditEvent] = field(default_factory=list)

    def answer(self, context: TenantContext, question: str) -> ToolResult:
        started = perf_counter()
        try:
            call = self.planner.plan(question)
            result = self.runner.run(context, call)
            tool_name = call.name
        except ValueError as exc:
            result = ToolResult(False, "", 0, "unsupported_request")
            tool_name = "planner"
            _ = exc  # Keep error text out of user-facing output in this teaching example.

        self.audit_events.append(
            AuditEvent(
                tenant_id=context.tenant_id,
                subject_id=context.subject_id,
                tool_name=tool_name,
                outcome="success" if result.success else "denied_or_failed",
                attempts=result.attempts,
                latency_ms=(perf_counter() - started) * 1_000,
                error_code=result.error_code,
            )
        )
        return result


def build_demo_agent() -> Agent:
    knowledge = KnowledgeTool()
    specs = [
        ToolSpec("calculate", "tools:calculate", True, safe_calculate),
        ToolSpec("search_knowledge", "tools:knowledge", True, knowledge.search),
    ]
    return Agent(RuleBasedPlanner(), ToolRunner(specs, PolicyEngine()))


def demo() -> None:
    """Run successful, retried, and denied cases with observable output."""
    agent = build_demo_agent()
    acme_user = TenantContext("acme", "alice", frozenset({"tools:calculate", "tools:knowledge"}))

    calculation = agent.answer(acme_user, "calculate (12 + 3) * 2")
    policy = agent.answer(acme_user, "Find the deployment policy and retry if needed")
    denied = agent.runner.run(
        acme_user,
        ToolCall("search_knowledge", {"query": "policy", "tenant_id": "globex"}),
    )

    print("Calculation:", calculation)
    print("Knowledge lookup (one safe retry):", policy)
    print("Cross-tenant request:", denied)
    print("Audit events:")
    for event in agent.audit_events:
        print(" ", event)

    assert calculation.success and calculation.output == "result=30"
    assert policy.success and policy.attempts == 2
    assert not denied.success and denied.error_code == "cross_tenant_request"
    assert all(event.tenant_id == "acme" for event in agent.audit_events)
    print("PASS: tool policy, retry, and audit checks completed.")


if __name__ == "__main__":
    demo()
