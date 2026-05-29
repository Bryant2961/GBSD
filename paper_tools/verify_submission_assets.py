"""Check that generated paper tables and figure manifests exist."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        ROOT / "results/paper_tables/generated",
        ROOT / "results/paper_figures/figure_source_manifest.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing submission assets:")
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}")
        return 2
    print("Submission asset check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

