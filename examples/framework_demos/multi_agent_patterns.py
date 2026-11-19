"""Framework-independent bounded team workflow for AutoGen, CrewAI, and ADK.

Run locally:
    python examples/framework_demos/multi_agent_patterns.py

This is deliberately not a fake implementation of those SDKs. It teaches the
portable control-plane contract that should remain outside any multi-agent
framework: tenant context, role boundaries, review, termination, and audit.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    subject_id: str


@dataclass(frozen=True)
class Task:
    objective: str
    requires_side_effect: bool


@dataclass(frozen=True)
class Handoff:
    sender: str
    recipient: str
    payload: str


@dataclass
class BoundedTeam:
    """A planner/reviewer/executor workflow with explicit stop conditions."""

    max_handoffs: int = 3
    audit: list[Handoff] = field(default_factory=list)

    def run(self, context: TenantContext, task: Task, approval_granted: bool) -> str:
        plan = f"tenant={context.tenant_id}; plan: retrieve approved evidence for {task.objective}"
        self.audit.append(Handoff("planner", "reviewer", plan))

        reviewer_decision: Literal["approve", "needs_approval"]
        reviewer_decision = "needs_approval" if task.requires_side_effect and not approval_granted else "approve"
        self.audit.append(Handoff("reviewer", "executor", reviewer_decision))

        if len(self.audit) > self.max_handoffs:
            return "terminated: handoff budget exceeded"
        if reviewer_decision != "approve":
            return "paused: trusted approval required before side effect"
        return f"completed: bounded evidence-only response for {context.tenant_id}"


FRAMEWORK_MAPPING = {
    "AutoGen": "Agents exchange typed messages; configure termination and route tool access through policy.",
    "CrewAI": "Roles, tasks, and a process describe delegation; persist tenant context outside shared prose.",
    "LangGraph": "State graph provides explicit routing, checkpoints, interrupts, and replayable transitions.",
    "Google ADK": "Agents, sessions, tools, and evaluations require the same policy and trace boundaries.",
}


def demo() -> None:
    context = TenantContext("acme", "alice")
    safe_team = BoundedTeam()
    safe_result = safe_team.run(context, Task("deployment guidance", False), approval_granted=False)
    change_team = BoundedTeam()
    blocked_result = change_team.run(context, Task("deploy a package", True), approval_granted=False)

    print("Framework mapping:")
    for framework, control in FRAMEWORK_MAPPING.items():
        print(f"- {framework}: {control}")
    print("Read-only workflow:", safe_result)
    print("Side-effect workflow:", blocked_result)
    assert safe_result.startswith("completed")
    assert blocked_result.startswith("paused")
    assert all("tenant=acme" in handoff.payload or handoff.payload in {"approve", "needs_approval"} for handoff in safe_team.audit)
    print("PASS: bounded multi-agent handoff and approval checks completed.")


if __name__ == "__main__":
    demo()
