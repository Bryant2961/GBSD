"""Compatibility wrapper for official summary generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, str(ROOT / "experiments" / "summarize.py"), *sys.argv[1:]],
        cwd=ROOT,
    ))

