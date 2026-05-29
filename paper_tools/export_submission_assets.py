"""Export official paper assets into a single folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/submission_assets")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    for rel in ["results/paper_tables/generated", "results/paper_figures"]:
        source = ROOT / rel
        if source.exists():
            dest = output / Path(rel).name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
    print(f"Exported assets to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

