"""Build official GBSD summary CSV files from unified per-seed outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbsd.reporting.collect import (  # noqa: E402
    collect_seed_records,
    validate_required_columns,
    write_summaries,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/unified_blind_protocol")
    parser.add_argument("--output", default="results/paper_tables")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    raw = collect_seed_records(ROOT / args.input)
    missing = validate_required_columns(raw)
    if missing and not args.allow_incomplete:
        print("Official summary is missing required fields:")
        for field in missing:
            print(f"  - {field}")
        print("Use --allow-incomplete only for migration diagnostics.")
        return 2

    write_summaries(raw, ROOT / args.output)
    print(f"Wrote summaries to {ROOT / args.output}")
    print(f"Rows collected: {len(raw)}")
    if missing:
        print("Incomplete fields (diagnostic mode): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

