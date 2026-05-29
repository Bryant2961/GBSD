"""Summarize stage-wise runtime and environment metadata for the GBSD paper.

This script is intentionally read-only with respect to experiment outputs. It
does not train models. It collects timing information produced by the
integrated main GBSD and baseline workflows, then writes compact paper-ready
tables plus raw traceability tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = PROJECT_ROOT / "Results" / "supplementary" / "tables"
OUT_PREFIX = "stagewise_runtime"

RUNTIME_RAW = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "runtime_and_params_raw.csv"
STRUCTURE_RAW = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "structure_discovery_baselines_raw.csv"
UQ_RAW = PROJECT_ROOT / "Results" / "supplementary" / "tables" / "uq_baselines_raw.csv"
RUNS_DIR = PROJECT_ROOT / "Results" / "supplementary" / "runs"


def mean_std(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan, math.nan, 0
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0, int(len(values))


def format_mean_std(mean: float, std: float, unit: str = "s") -> str:
    if math.isnan(mean):
        return "not recorded"
    if abs(mean) < 1.0:
        return f"{mean:.4f} +/- {std:.4f} {unit}"
    if abs(mean) < 10.0:
        return f"{mean:.3f} +/- {std:.3f} {unit}"
    if math.isnan(std):
        return f"{mean:.2f} {unit}"
    return f"{mean:.2f} +/- {std:.2f} {unit}"


def summarize_group(
    rows: list[dict],
    case: str,
    stage: str,
    values: pd.Series,
    source: str,
    note: str,
) -> None:
    mean, std, n = mean_std(values)
    rows.append(
        {
            "case": case,
            "stage": stage,
            "time_s_mean": mean,
            "time_s_std": std,
            "n": n,
            "time_s_mean_std": format_mean_std(mean, std),
            "source": source,
            "note": note,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def environment_rows(run_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for run_id in sorted(set(run_ids)):
        env_path = RUNS_DIR / run_id / "environment.json"
        if not env_path.is_file():
            continue
        env = read_json(env_path)
        rows.append(
            {
                "run_id": run_id,
                "seed": env.get("seed"),
                "preset": env.get("preset"),
                "cases": ";".join(env.get("cases", [])),
                "elapsed_seconds": env.get("elapsed_seconds"),
                "python_version": env.get("python_version"),
                "torch_version": env.get("torch_version"),
                "cuda_available": env.get("cuda_available"),
                "cuda_version": env.get("cuda_version"),
                "cudnn_version": env.get("cudnn_version"),
                "gpu_name": env.get("gpu_name"),
                "gpu_memory_total_gb": (
                    env.get("gpu_memory_total_bytes", math.nan) / 1024**3
                    if env.get("gpu_memory_total_bytes") is not None
                    else math.nan
                ),
                "platform": env.get("platform"),
                "cpu_count": env.get("cpu_count"),
                "n_mc": env.get("n_mc"),
                "argv": " ".join(env.get("argv", [])),
            }
        )
    return pd.DataFrame(rows)


def summarize_environment(env_raw: pd.DataFrame) -> pd.DataFrame:
    if env_raw.empty:
        return pd.DataFrame()
    fields = [
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "gpu_name",
        "gpu_memory_total_gb",
        "platform",
        "cpu_count",
        "n_mc",
    ]
    rows = []
    for field in fields:
        values = env_raw[field].dropna().astype(str).unique().tolist()
        rows.append({"item": field, "value": "; ".join(values)})
    elapsed_mean, elapsed_std, n = mean_std(env_raw["elapsed_seconds"])
    rows.append(
        {
            "item": "main_run_elapsed_seconds",
            "value": format_mean_std(elapsed_mean, elapsed_std),
            "n": n,
        }
    )
    return pd.DataFrame(rows)


def build_tables() -> dict[str, pd.DataFrame]:
    if not RUNTIME_RAW.is_file():
        raise FileNotFoundError(f"Missing main runtime table: {RUNTIME_RAW}")

    runtime = pd.read_csv(RUNTIME_RAW)
    runtime = runtime[runtime["run_id"].astype(str).str.match(r"full_s[0-4]_")]

    stage_rows: list[dict] = []
    for case, group in runtime.groupby("case", sort=True):
        summarize_group(
            stage_rows,
            case,
            "Teacher PINN training",
            group["teacher_time_s"],
            "v3.37 main GBSD runs: Clock time.csv",
            "Exact logged wall-clock time for deterministic teacher training.",
        )
        summarize_group(
            stage_rows,
            case,
            "Dense Bayesian student training",
            group["student_time_s"],
            "v3.37 main GBSD runs: Clock time.csv",
            "Exact logged wall-clock time for MC-Dropout student training/refinement.",
        )

    if STRUCTURE_RAW.is_file():
        structure = pd.read_csv(STRUCTURE_RAW)
        hac = structure[structure["structure_method"].astype(str).str.lower().eq("hac")]
        for case, group in hac.groupby("case", sort=True):
            summarize_group(
                stage_rows,
                case,
                "Structure discovery (HAC)",
                group["runtime_discovery_s"],
                "v3.38 structure-discovery baseline: HAC rows",
                "Measured from the frozen dense student checkpoint; no retraining of teacher/student.",
            )
            summarize_group(
                stage_rows,
                case,
                "Physics-informed structured reconstruction",
                group["runtime_reconstruction_s"],
                "v3.38 structure-discovery baseline: HAC rows",
                "Measured reconstruction of the structured candidate under PDE/data/anchor losses.",
            )
            summarize_group(
                stage_rows,
                case,
                "Deterministic blind-grid inference",
                group["runtime_inference_s"],
                "v3.38 structure-discovery baseline: HAC rows",
                "Single deterministic evaluation on the blind subset; not MC-Dropout sampling.",
            )
    else:
        for case in sorted(runtime["case"].unique()):
            for stage in [
                "Structure discovery (HAC)",
                "Physics-informed structured reconstruction",
                "Deterministic blind-grid inference",
            ]:
                summarize_group(
                    stage_rows,
                    case,
                    stage,
                    pd.Series(dtype=float),
                    "not available",
                    "No stage-specific timing was found.",
                )

    if UQ_RAW.is_file():
        uq = pd.read_csv(UQ_RAW)
        uq = uq[
            (uq["preset"].astype(str).str.lower().eq("full"))
            & (uq["method"].astype(str).str.lower().eq("direct_mc_dropout_pinn"))
        ]
        for case, group in uq.groupby("case", sort=True):
            summarize_group(
                stage_rows,
                case,
                "MC inference, 200 samples (dense-dropout proxy)",
                group["runtime_mc_or_ensemble_inference_s"],
                "v3.38 direct MC-Dropout PINN baseline",
                "Proxy timing for 200 stochastic forward passes on the blind subset. The main GBSD MC timing was not separately logged in v3.37.",
            )

    stagewise = pd.DataFrame(stage_rows)

    param_cols = [
        "case",
        "teacher_param_count",
        "dense_param_count",
        "structured_param_count",
        "param_count_compression",
        "teacher_model_size_bytes",
        "dense_model_size_bytes",
        "structured_model_size_bytes",
        "frozen_R_storage_bytes",
        "compression",
    ]
    params_raw = runtime[[c for c in param_cols if c in runtime.columns]].copy()
    params_summary_rows = []
    for case, group in params_raw.groupby("case", sort=True):
        row = {"case": case}
        for col in params_raw.columns:
            if col == "case":
                continue
            m, s, n = mean_std(group[col])
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
            row[f"{col}_n"] = n
        row["structured_vs_dense_trainable_params"] = (
            f"{row.get('structured_param_count_mean', math.nan):.1f} / "
            f"{row.get('dense_param_count_mean', math.nan):.1f}"
        )
        row["compression_mean_std"] = format_mean_std(
            row.get("param_count_compression_mean", math.nan),
            row.get("param_count_compression_std", math.nan),
            unit="x",
        )
        params_summary_rows.append(row)
    params_summary = pd.DataFrame(params_summary_rows)

    env_raw = environment_rows(runtime["run_id"])
    env_summary = summarize_environment(env_raw)

    source_notes = pd.DataFrame(
        [
            {
                "field": "teacher/student training",
                "source": str(RUNTIME_RAW),
                "status": "directly logged by main v3.37 full runs",
            },
            {
                "field": "structure discovery/reconstruction/deterministic inference",
                "source": str(STRUCTURE_RAW),
                "status": "directly logged by v3.38 HAC structure baseline using frozen v3.37 dense checkpoints",
            },
            {
                "field": "MC inference",
                "source": str(UQ_RAW),
                "status": "proxy from direct MC-Dropout PINN full baseline; main v3.37 GBSD did not separately log MC inference time",
            },
            {
                "field": "hardware/software",
                "source": str(RUNS_DIR / "<run_id>" / "environment.json"),
                "status": "direct environment manifest from archived runs",
            },
        ]
    )

    return {
        "stagewise_runtime_summary": stagewise,
        "params_summary": params_summary,
        "environment_raw": env_raw,
        "environment_summary": env_summary,
        "source_notes": source_notes,
    }


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = OUT_DIR / f"{OUT_PREFIX}_{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved: {out}")

    xlsx_path = OUT_DIR / f"{OUT_PREFIX}_workbook.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, df in tables.items():
                sheet = name[:31]
                df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"Saved: {xlsx_path}")
    except Exception as exc:  # pragma: no cover - optional convenience output.
        print(f"[warn] Could not write xlsx workbook: {exc}")


def main() -> None:
    tables = build_tables()
    write_outputs(tables)
    print("\nPaper-ready stagewise runtime summary:")
    preview = tables["stagewise_runtime_summary"][
        ["case", "stage", "time_s_mean_std", "n", "note"]
    ]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
