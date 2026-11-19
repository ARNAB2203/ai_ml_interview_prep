"""Run ML/DS extension labs after installing requirements-ml-ds.txt."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


DEMOS = (
    "data_classical_ml.py",
    "deep_learning_nlp_foundations.py",
    "hybrid_retrieval_operations.py",
    "mlops_lifecycle.py",
    "model_serving_api.py",
)


def main() -> int:
    directory = Path(__file__).resolve().parent
    for demo in DEMOS:
        print(f"\n{'=' * 12} {demo} {'=' * 12}", flush=True)
        completed = subprocess.run([sys.executable, str(directory / demo)], check=False)
        if completed.returncode:
            return completed.returncode
    print("\nPASS: all ML/DS extension demos completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
