"""A narrow, local Model Context Protocol (MCP) tool server.

Validate the server contract without opening a port:
    python examples/framework_demos/mcp_knowledge_server.py

Run it over stdio for an MCP-capable host:
    python examples/framework_demos/mcp_knowledge_server.py --stdio

The demo accepts tenant/subject fields to make the contract visible. In a real
MCP deployment, derive those fields from authenticated transport/session context
and enforce authorization at the tool server—not in an agent prompt.
"""

# %% Imports
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mcp.server.fastmcp import FastMCP


RUNBOOKS = {
    "acme": {
        "deployment": "Run offline evaluation, then staging, then a canary before production rollout.",
        "incident": "Capture trace ID, model/prompt/tool versions, then rollback the unsafe change.",
    },
    "globex": {
        "deployment": "Obtain a named approval before release to the regulated production environment.",
    },
}


def lookup_runbook(tenant_id: str, subject_id: str, topic: str) -> dict[str, str]:
    """Return a narrow, tenant-scoped read result; no arbitrary file or HTTP access."""
    if not subject_id.endswith(f"@{tenant_id}"):
        raise PermissionError("subject is not authorized for the requested tenant")
    text = RUNBOOKS.get(tenant_id, {}).get(topic.lower())
    if text is None:
        return {"status": "not_found", "tenant_id": tenant_id, "topic": topic}
    return {"status": "ok", "tenant_id": tenant_id, "topic": topic.lower(), "text": text}


def calculate_capacity(requests_per_second: float, p95_seconds: float) -> dict[str, float]:
    """Safe, typed capacity calculation instead of exposing a general shell tool."""
    if requests_per_second < 0 or p95_seconds < 0:
        raise ValueError("capacity inputs must be non-negative")
    return {"estimated_concurrent_runs": requests_per_second * p95_seconds}


# %% Step 1 — actual FastMCP tool declarations generate discoverable schemas
server = FastMCP(
    "tenant-runbook-tools",
    instructions="Expose only authorized, read-oriented operational tools.",
)


@server.tool(name="lookup_runbook", description="Read a tenant-authorized deployment or incident runbook.")
def mcp_lookup_runbook(tenant_id: str, subject_id: str, topic: str) -> dict[str, str]:
    return lookup_runbook(tenant_id, subject_id, topic)


@server.tool(name="calculate_capacity", description="Calculate an estimated concurrent run count.")
def mcp_calculate_capacity(requests_per_second: float, p95_seconds: float) -> dict[str, float]:
    return calculate_capacity(requests_per_second, p95_seconds)


# %% Step 2 — contract validation exercises discovery and invocation locally
async def validate_contract() -> dict[str, Any]:
    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}
    if {"lookup_runbook", "calculate_capacity"} - tool_names:
        raise AssertionError("expected tools were not registered")

    result = await server.call_tool(
        "lookup_runbook",
        {"tenant_id": "acme", "subject_id": "alice@acme", "topic": "deployment"},
    )
    # SDK versions may return content blocks alone or ``(blocks, structured_output)``.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        payload = result[1]
    else:
        blocks = result[0] if isinstance(result, tuple) else result
        text = "\n".join(getattr(block, "text", str(block)) for block in blocks)
        payload = json.loads(text)
    if payload["tenant_id"] != "acme" or payload["status"] != "ok":
        raise AssertionError("unexpected MCP tool response")
    return {"tools": sorted(tool_names), "lookup_result": payload}


def demo() -> None:
    contract = asyncio.run(validate_contract())
    print("MCP tool contract:", contract)
    assert calculate_capacity(12, 3)["estimated_concurrent_runs"] == 36
    try:
        lookup_runbook("acme", "mallory@globex", "deployment")
    except PermissionError:
        denied = True
    else:
        denied = False
    assert denied
    print("PASS: MCP discovery, typed invocation, and tenant-policy checks completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true", help="Run as a blocking stdio MCP server.")
    arguments = parser.parse_args()
    if arguments.stdio:
        server.run(transport="stdio")
    else:
        demo()


if __name__ == "__main__":
    main()
