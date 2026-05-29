from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _run(cmd: list[str]) -> None:
    print("Running:")
    print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def stage_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--preset", default="full",
                        choices=["smoke", "quick_check", "preview", "medium", "full"])
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated seed list, for example 0,1,2,3,4.")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used by the migrated runtime engine.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Record the protocol plan without launching training.")
    parser.add_argument("--summarize", action="store_true",
                        help="Run experiments/summarize.py after the stage finishes.")
    return parser


def run_stage(stage: str, argv: list[str] | None = None, *, description: str) -> int:
    parser = stage_parser(description)
    args = parser.parse_args(argv)
    seeds = ",".join(str(seed) for seed in _parse_seeds(args.seeds))
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "generate_all.py"),
        "--stage", stage,
        "--preset", args.preset,
        "--seeds", seeds,
        "--n-mc", str(args.n_mc),
        "--python", args.python,
    ]
    if not args.plan_only:
        cmd.append("--execute")
    _run(cmd)
    if args.summarize:
        input_dir = "results/smoke_test" if args.preset == "smoke" else "results/unified_blind_protocol"
        _run([sys.executable, str(ROOT / "experiments" / "summarize.py"), "--input", input_dir])
    return 0


def run_single_problem(case: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Convenience runner for one runtime-engine GBSD problem: {case}. "
            "Official manuscript tables should still use the unified all-problem stage."
        )
    )
    parser.add_argument("--preset", default="full",
                        choices=["smoke", "quick_check", "preview", "medium", "full"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-clean", action="store_true",
                        help="Do not clean the case work directory before this targeted run.")
    args = parser.parse_args(argv)
    cmd = [
        args.python,
        "run_all_experiments.py",
        "--case", case,
        "--method", "bayesian",
        "--preset", args.preset,
        "--seed", str(args.seed),
        "--n_mc", str(args.n_mc),
    ]
    if not args.no_clean:
        cmd.append("--clean")
    _run_in_engine(cmd)
    return 0


def _run_in_engine(cmd: list[str]) -> None:
    engine = ROOT / "runtime_engine"
    print("Running runtime-engine command:")
    print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.check_call(cmd, cwd=engine)

