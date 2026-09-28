"""Run one deterministic mutation against the quality gate.

The mutation must be killed by the gate tests. The source file is restored in
``finally`` so a local run cannot leave the checkout mutated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
TARGET = APP_ROOT / "presentation/gate.py"
ORIGINAL = b'if any(check.action == "fail" and not check.passed for check in checks):'
MUTANT = b'if any(check.action == "fail" and check.passed for check in checks):'


def main() -> int:
    source = TARGET.read_bytes()
    if source.count(ORIGINAL) != 1:
        print("Mutation target was not found exactly once.", file=sys.stderr)
        return 2

    TARGET.write_bytes(source.replace(ORIGINAL, MUTANT))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_gate.py"],
            cwd=APP_ROOT,
            check=False,
        )
    finally:
        TARGET.write_bytes(source)

    if result.returncode == 0:
        print("Mutation survived: the gate tests did not detect the broken condition.")
        return 1

    print("Mutation killed: the gate tests detected the broken condition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
