"""Run one ablation with temporary config overrides and archive outputs."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd

from common import ABLATION_DIR, PROJECT_ROOT, ensure_output_dirs


def parse_key_value(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in override: {item}")
        overrides[key] = value
    return overrides


def update_config(path: Path, overrides: dict[str, str]) -> None:
    df = pd.read_csv(path, header=None)
    for key, value in overrides.items():
        mask = df.iloc[:, 0].astype(str).str.strip() == key
        if mask.any():
            df.loc[mask, 1] = value
        else:
            note = "temporary supplementary ablation override"
            df.loc[len(df)] = [key, value, note]
    df.to_csv(path, header=False, index=False)


def sanitize_tag(tag: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return clean or "ablation"


def archive_outputs(case: str, tag: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = ABLATION_DIR / f"{sanitize_tag(tag)}_s{seed}_{timestamp}"
    dst.mkdir(parents=True, exist_ok=True)

    case_src = PROJECT_ROOT / "Results" / f"{case}_EXP"
    if case_src.is_dir():
        shutil.copytree(case_src, dst / f"{case}_EXP",
                        ignore=shutil.ignore_patterns("__pycache__"))

    raw_src = PROJECT_ROOT / "results" / "raw" / f"{case}_predictions.npz"
    if raw_src.is_file():
        (dst / "raw").mkdir(exist_ok=True)
        shutil.copy2(raw_src, dst / "raw" / raw_src.name)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    if metrics_dir.is_dir():
        (dst / "metrics").mkdir(exist_ok=True)
        for item in metrics_dir.glob("*.csv"):
            if item.name.startswith(case) or item.name in {
                "calibration_metrics.csv",
                "parameter_inversion_metrics.csv",
            }:
                shutil.copy2(item, dst / "metrics" / item.name)

    figures_dir = PROJECT_ROOT / "results" / "figures"
    if figures_dir.is_dir():
        (dst / "figures").mkdir(exist_ok=True)
        for item in figures_dir.glob("*"):
            if case.lower() in item.name.lower():
                shutil.copy2(item, dst / "figures" / item.name)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True,
                        choices=["Laplace", "Poisson", "Burgers_inv"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="full")
    parser.add_argument("--method", default="bayesian",
                        choices=["bayesian", "baseline", "deterministic", "all"])
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        help="Temporary config override: key=value")
    parser.add_argument("--include-comparisons", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--student-pde-weight", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Cap hidden refinement/pretraining loops for path testing only.")
    args = parser.parse_args()

    ensure_output_dirs()
    overrides = parse_key_value(args.sets)
    cfg_path = PROJECT_ROOT / "Config" / f"{args.case}_EXP.csv"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    original = None
    backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    try:
        if overrides:
            original = cfg_path.read_bytes()
            backup_path.write_bytes(original)
            update_config(cfg_path, overrides)

        cmd = [
            sys.executable, "run_all_experiments.py",
            "--case", args.case,
            "--method", args.method,
            "--preset", args.preset,
            "--seed", str(args.seed),
            "--n_mc", str(args.n_mc),
        ]
        if args.include_comparisons or args.method in {"baseline", "deterministic"}:
            cmd.append("--include_comparisons")
        if args.clean:
            cmd.append("--clean")
        if args.predict_only:
            cmd.append("--predict_only")
        if args.figures_only:
            cmd.append("--figures_only")
        if args.student_pde_weight is not None:
            cmd.extend(["--student_pde_weight", str(args.student_pde_weight)])
        if args.smoke:
            cmd.append("--preset")
            cmd.append("smoke")

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        archive_dir = archive_outputs(args.case, args.tag, args.seed)
        print(f"Archived ablation: {archive_dir}")
    finally:
        if original is not None:
            cfg_path.write_bytes(original)
            if backup_path.is_file():
                backup_path.unlink()
            print(f"Restored config: {cfg_path}")


if __name__ == "__main__":
    main()
