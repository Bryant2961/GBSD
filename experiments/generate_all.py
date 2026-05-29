"""Official experiment driver for GBSD paper reproduction.

By default this script records an auditable protocol plan only. With
``--execute`` it calls the integrated runtime engine and immediately normalizes
its artifacts into ``results/unified_blind_protocol``. Runtime work products are
never used as paper results until written through this official contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbsd.io.json_utils import write_json  # noqa: E402


STAGES = {
    "main": ["main_blind"],
    "runtime": ["runtime_params"],
    "baselines": ["strong_baselines"],
    "ablations": [
        "guard_ablation",
        "uq_calibration_ablation",
        "reconstruction_ablation",
        "zero_distillation_ablation",
    ],
    "sensitivity": ["threshold_sensitivity", "structure_stability"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), help="Protocol stage to run")
    parser.add_argument("--all", action="store_true", help="Plan all protocol stages")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the integrated runtime engine and write official unified outputs.",
    )
    parser.add_argument(
        "--preset",
        default="full",
        choices=["smoke", "quick_check", "preview", "medium", "full"],
        help="Runtime budget preset. smoke is path testing only and is excluded from official paper outputs.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seed_values",
        help="Seed to run. Repeat for multiple seeds. Defaults to 0-4.",
    )
    parser.add_argument(
        "--seeds",
        dest="seed_list",
        help="Comma-separated seed list, e.g. 0,1,2,3,4. Overrides repeated --seed.",
    )
    parser.add_argument("--n-mc", type=int, default=200, help="MC Dropout samples for main GBSD.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for the integrated runtime engine.",
    )
    args = parser.parse_args()

    if not args.all and not args.stage:
        parser.error("Specify --stage {main,runtime,baselines,ablations,sensitivity} or --all")

    selected_stages = list(STAGES) if args.all else [args.stage]
    selected = [item for stage in selected_stages for item in STAGES[stage]]
    if args.seed_list:
        seeds = [int(item.strip()) for item in args.seed_list.split(",") if item.strip()]
    else:
        seeds = args.seed_values or [0, 1, 2, 3, 4]

    executable_stages = {"main", "runtime", "baselines", "ablations", "sensitivity"}
    unsupported = [stage for stage in selected_stages if stage not in executable_stages]
    if args.execute and unsupported:
        print(
            "Execution adapters are currently migrated only for stages: "
            f"{', '.join(sorted(executable_stages))}."
        )
        print(f"Unsupported requested stage(s): {', '.join(unsupported)}")
        return 2

    run_id = dt.datetime.now().strftime("plan_%Y%m%d_%H%M%S_%f")
    plan = {
        "run_id": run_id,
        "protocol_id": "unified_blind_protocol_v1",
        "selected_experiments": selected,
        "selected_stages": selected_stages,
        "preset": args.preset,
        "seeds": seeds,
        "n_mc": args.n_mc,
        "python": args.python,
        "official_results_dir": (
            "results/smoke_test" if args.preset == "smoke"
            else "results/unified_blind_protocol"
        ),
        "status": "execute_requested" if args.execute else "plan_recorded",
        "execute_requested": bool(args.execute),
        "note": (
            "The integrated runtime engine writes the official per-seed result contract. "
            "Unnormalized runtime outputs remain work products only."
        ),
    }
    write_json(ROOT / "results" / "audit" / f"{run_id}.json", plan)
    print(f"Recorded official protocol plan: results/audit/{run_id}.json")
    for experiment in selected:
        print(f"  - {experiment}")
    if args.execute:
        try:
            from gbsd.io.runtime_engine_adapter import run_stage as run_runtime_stage
        except ModuleNotFoundError as exc:
            print(
                "Cannot import the runtime execution adapter. Run this command with the "
                "project environment that contains NumPy/PyTorch, for example "
                "python experiments\\generate_all.py ..."
            )
            print(f"Import error: {exc}")
            return 2
        total_outputs = 0
        for stage in selected_stages:
            print(f"\n=== Executing migrated stage: {stage} ===")
            outputs = run_runtime_stage(
                stage,
                execute=True,
                preset=args.preset,
                seeds=seeds,
                python=args.python,
                n_mc=args.n_mc,
            )
            total_outputs += len(outputs)
            for output in outputs:
                print(f"  wrote {output.official_dir.relative_to(ROOT)}")
        output_scope = "smoke-test" if args.preset == "smoke" else "official"
        result_input = "results/smoke_test" if args.preset == "smoke" else "results/unified_blind_protocol"
        print(f"\nExecution complete. {output_scope} per-seed outputs written: {total_outputs}")
        print(f"Next: python experiments/summarize.py --input {result_input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
