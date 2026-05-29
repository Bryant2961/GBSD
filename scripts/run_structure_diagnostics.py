from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gbsd.io.runtime_engine_adapter as adapter  # noqa: E402


def _parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run structure-discovery diagnostic baselines only."
    )
    parser.add_argument("--preset", default="full",
                        choices=["smoke", "quick_check", "preview", "medium", "full"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    adapter.ACTIVE_RESULTS_ROOT = (
        adapter.SMOKE_ROOT if args.preset == "smoke" else adapter.OFFICIAL_ROOT
    )
    outputs = adapter.run_structure_discovery_baselines(
        _parse_seeds(args.seeds),
        args.preset,
        execute=not args.plan_only,
        python=args.python,
    )
    print(f"Structure-diagnostic outputs: {len(outputs)}")
    if args.summarize:
        import subprocess

        input_dir = "results/smoke_test" if args.preset == "smoke" else "results/unified_blind_protocol"
        subprocess.check_call(
            [sys.executable, str(ROOT / "experiments" / "summarize.py"), "--input", input_dir],
            cwd=ROOT,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
