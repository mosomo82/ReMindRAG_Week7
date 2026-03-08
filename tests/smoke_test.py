"""
tests/smoke_test.py — Quick smoke test for ReMindRAG.

Delegates to the repo's proven repro_test.py and writes a
structured JSON result to artifacts/smoke_test_result.json.

Usage:
    python tests/smoke_test.py
    # or via Makefile:
    make smoke
"""
import sys
import os
import json
import subprocess
from datetime import datetime

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(REPO_ROOT, "Rag_Cache"), exist_ok=True)
os.makedirs(os.path.join(REPO_ROOT, "model_cache"), exist_ok=True)


def run_smoke_test():
    print("=" * 60)
    print("  ReMindRAG Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    repro_script = os.path.join(REPO_ROOT, "repro_test.py")
    result = subprocess.run(
        [sys.executable, repro_script],
        cwd=REPO_ROOT,
        capture_output=False,  # stream output live
    )

    passed = result.returncode == 0

    verdict = {
        "status": "PASS" if passed else "FAIL",
        "timestamp": datetime.now().isoformat() + "Z",
        "exit_code": result.returncode,
        "script": "repro_test.py",
    }

    result_path = os.path.join(ARTIFACTS_DIR, "smoke_test_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print()
    print(f"{'SUCCESS' if passed else 'FAIL'}: Smoke test {'passed' if passed else 'failed'}.")
    print(f"Result written to: {result_path}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
