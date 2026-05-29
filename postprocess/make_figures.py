"""Generate the figure source manifest from official summaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(subprocess.call(
        [sys.executable, str(ROOT / "paper_tools" / "generate_figures.py"), *sys.argv[1:]],
        cwd=ROOT,
    ))
