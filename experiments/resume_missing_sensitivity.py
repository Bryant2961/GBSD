"""Resume only missing official sensitivity outputs.

This script intentionally reuses the same runtime-engine adapter functions as
``run_sensitivity.py``. It differs only in the scheduler: existing official
per-seed outputs are skipped, and complete runtime archives are normalized
before retraining anything.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbsd.io import runtime_engine_adapter as runtime  # noqa: E402


CLUSTER_VALUES = ["0.03", "0.05", "0.08", "0.10", "0.15", "0.20"]
CLUSTER_CASES = ("Poisson", "Burgers_inv")
DERIVED_GUARD_VARIANTS = [
    *(f"accuracy_ratio_{str(v).replace('.', 'p')}" for v in [1.00, 1.03, 1.10, 1.25, 1.50]),
    *(f"minimum_compression_{str(v).replace('.', 'p')}" for v in [1.00, 1.10, 1.50, 2.00]),
]


def _official_dir(experiment: str, problem: str, variant: str, seed: int) -> Path:
    return runtime.OFFICIAL_ROOT / experiment / problem / variant / f"seed_{seed}"


def _official_complete(experiment: str, problem: str, variant: str, seed: int) -> bool:
    path = _official_dir(experiment, problem, variant, seed)
    return (path / "metrics.json").is_file() and (path / "run_manifest.json").is_file()


def _complete_archive(archive: Path, case: str) -> bool:
    return (
        (archive / "raw" / f"{case}_predictions.npz").is_file()
        and (archive / f"{case}_EXP").is_dir()
    )


def _latest_complete_archive(tag: str, seed: int, case: str) -> Path | None:
    root = runtime.ENGINE_ROOT / "Results" / "supplementary" / "ablations"
    candidates = sorted(
        root.glob(f"{tag}_s{seed}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for archive in candidates:
        if _complete_archive(archive, case):
            return archive
    return None


def _derive_structure_stability_one(problem: str, seed: int) -> runtime.RuntimeRun:
    source_dir = runtime._main_official_dir(problem, seed)
    with np.load(source_dir / "predictions_guard.npz", allow_pickle=True) as guard, np.load(
        source_dir / "predictions_blind.npz", allow_pickle=True
    ) as blind:
        guard_mean = runtime._array(guard, "bayesian_mean")
        blind_mean = runtime._array(blind, "bayesian_mean")
        guard_std = runtime._array(guard, "bayesian_std")
        blind_std = runtime._array(blind, "bayesian_std")
        source = str(runtime._scalar(blind, "bayesian_source", "guarded_final"))
        if guard_mean is None or blind_mean is None:
            raise KeyError(f"Reference predictions missing in {source_dir}")
        metrics = runtime._evaluate_selection(blind, blind_mean, blind_std, source)
        metrics["guard_decision"] = source
        return runtime._write_derived_run(
            "structure_stability",
            problem,
            "main_hac_across_seeds",
            seed,
            source_dir,
            guard,
            blind,
            guard_mean,
            blind_mean,
            guard_std,
            blind_std,
            source,
            metrics,
            {
                "decision_split": "guard_validation",
                "final_source": source,
                "guard_may_read_blind_labels": False,
            },
            {
                "reference_variant": True,
                "source_mode": "final",
                "stability_scope": "main_hac_outcome_across_seeds",
            },
        )


def _check_guard_threshold_derived(seeds: list[int]) -> list[str]:
    missing: list[str] = []
    for problem in runtime.PROBLEMS:
        for variant in DERIVED_GUARD_VARIANTS:
            for seed in seeds:
                if not _official_complete("threshold_sensitivity", problem, variant, seed):
                    missing.append(f"{problem}/{variant}/seed_{seed}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preset", default="full", choices=["full"])
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    runtime._require_runtime_engine()
    runtime.ACTIVE_RESULTS_ROOT = runtime.OFFICIAL_ROOT

    guard_missing = _check_guard_threshold_derived(seeds)
    if guard_missing:
        print("Missing derived guard-threshold outputs; run full sensitivity derivation first:")
        for item in guard_missing:
            print(f"  - {item}")
        return 2

    started = time.time()
    outputs: list[runtime.RuntimeRun] = []
    missing_clusters: list[tuple[str, str, int]] = []
    for case in CLUSTER_CASES:
        problem = runtime._case_to_problem(case)
        for value in CLUSTER_VALUES:
            variant = f"clustering_threshold_{value.replace('.', 'p')}"
            for seed in seeds:
                if not _official_complete("threshold_sensitivity", problem, variant, seed):
                    missing_clusters.append((case, value, seed))

    missing_stability: list[tuple[str, int]] = []
    for problem in runtime.PROBLEMS:
        for seed in seeds:
            if not _official_complete("structure_stability", problem, "main_hac_across_seeds", seed):
                missing_stability.append((problem, seed))

    print(f"Missing clustering-threshold runs: {len(missing_clusters)}")
    for case, value, seed in missing_clusters:
        print(f"  - {case} cluster_distance={value} seed={seed}")
    print(f"Missing structure-stability derived rows: {len(missing_stability)}")
    for problem, seed in missing_stability:
        print(f"  - {problem} main_hac_across_seeds seed={seed}")

    if not args.execute:
        print("\nPlan only. Add --execute to run the missing items.")
        return 0

    for case, value, seed in missing_clusters:
        variant = f"clustering_threshold_{value.replace('.', 'p')}"
        tag = f"{variant}_{case}"
        archive = _latest_complete_archive(tag, seed, case)
        if archive is not None:
            print(f"[normalize existing archive] {archive.name}")
        else:
            print(f"[train missing] {case} cluster_distance={value} seed={seed}")
            archive = runtime._run_config_ablation(
                case,
                tag,
                seed,
                {"cluster_distance": value},
                args.preset,
                True,
                args.python,
                args.n_mc,
            )
        outputs.append(
            runtime.normalize_config_ablation_archive(
                archive,
                "threshold_sensitivity",
                case,
                variant,
                seed,
                args.preset,
                {"cluster_distance": value},
            )
        )

    for problem, seed in missing_stability:
        print(f"[derive] structure_stability {problem} seed={seed}")
        outputs.append(_derive_structure_stability_one(problem, seed))

    runtime._write_json(
        ROOT / "results" / "audit" / f"resume_missing_sensitivity_{int(started)}.json",
        {
            "stage": "resume_missing_sensitivity",
            "preset": args.preset,
            "seeds": seeds,
            "missing_clusters_initial": [
                {"case": case, "cluster_distance": value, "seed": seed}
                for case, value, seed in missing_clusters
            ],
            "missing_stability_initial": [
                {"problem": problem, "seed": seed}
                for problem, seed in missing_stability
            ],
            "outputs": [runtime._relative(item.official_dir) for item in outputs],
            "elapsed_s": time.time() - started,
        },
    )
    print(f"\nResume complete. Outputs written: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
