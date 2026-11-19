"""Run every runnable handbook example and preserve each script's exit status."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


EXAMPLES = (
    "tokenizer_bpe.py",
    "tool_using_agent.py",
    "rag_evaluation.py",
    "agent_platform_api.py",
    "manifest_validator.py",
)


def main() -> int:
    examples_directory = Path(__file__).resolve().parent
    for example in EXAMPLES:
        print(f"\n{'=' * 12} {example} {'=' * 12}", flush=True)
        completed = subprocess.run([sys.executable, str(examples_directory / example)], check=False)
        if completed.returncode:
            print(f"FAILED: {example} exited with {completed.returncode}", file=sys.stderr)
            return completed.returncode
    print("\nPASS: all handbook examples completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
