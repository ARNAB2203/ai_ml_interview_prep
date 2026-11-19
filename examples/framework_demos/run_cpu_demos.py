"""Run the optional CPU-safe framework demos after their requirements are installed."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


DEMOS = (
    "provider_gateway.py",
    "langchain_llamaindex_rag.py",
    "langgraph_approval_graph.py",
    "mcp_knowledge_server.py",
    "multi_agent_patterns.py",
    "pytorch_hf_peft.py",
    "jax_training_patterns.py",
    "deepspeed_zero_config.py",
)


def main() -> int:
    root = Path(__file__).resolve().parent
    for demo in DEMOS:
        print(f"\n{'=' * 12} {demo} {'=' * 12}", flush=True)
        completed = subprocess.run([sys.executable, str(root / demo)], check=False)
        if completed.returncode:
            return completed.returncode
    print("\nPASS: all optional CPU-safe framework demos completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
