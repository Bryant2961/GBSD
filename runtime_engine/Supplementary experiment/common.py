"""Shared utilities for supplementary experiment post-processing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


CASES = ("Laplace", "Poisson", "Burgers_inv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPP_ROOT = PROJECT_ROOT / "Results" / "supplementary"
RUNS_DIR = SUPP_ROOT / "runs"
TABLE_DIR = SUPP_ROOT / "tables"
FIGURE_DIR = SUPP_ROOT / "figures"
COMMAND_DIR = SUPP_ROOT / "commands"
ABLATION_DIR = SUPP_ROOT / "ablations"


@dataclass(frozen=True)
class RunRecord:
    name: str
    seed: int | None
    root: Path
    raw_dir: Path
    metrics_dir: Path
    case_root_pattern: str

    def case_dir(self, case: str) -> Path:
        return self.root / self.case_root_pattern.format(case=case)


def ensure_output_dirs() -> None:
    for path in (SUPP_ROOT, TABLE_DIR, FIGURE_DIR, COMMAND_DIR, ABLATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def parse_seed(text: str) -> int | None:
    match = re.search(r"_s(\d+)(?:_|$)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"seed[_-]?(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def archived_runs() -> list[RunRecord]:
    if not RUNS_DIR.is_dir():
        return []
    records: list[RunRecord] = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        raw_dir = run_dir / "raw"
        metrics_dir = run_dir / "metrics"
        if raw_dir.is_dir():
            records.append(RunRecord(
                name=run_dir.name,
                seed=parse_seed(run_dir.name),
                root=run_dir,
                raw_dir=raw_dir,
                metrics_dir=metrics_dir,
                case_root_pattern="{case}_EXP",
            ))
    return records


def latest_record_per_seed(records: list[RunRecord]) -> list[RunRecord]:
    latest: dict[int, RunRecord] = {}
    unseeded: list[RunRecord] = []
    for record in sorted(records, key=lambda r: (r.root.stat().st_mtime, r.name)):
        if record.seed is None:
            unseeded.append(record)
        else:
            latest[record.seed] = record
    return sorted([*latest.values(), *unseeded], key=lambda r: r.name)


def _run_environment(record: RunRecord) -> dict:
    path = record.root / "environment.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_main_bayesian_full_run(record: RunRecord) -> bool:
    """Identify the full Bayesian guarded seed-sweep runs.

    Supplementary commands also archive quick checks, ablations, and classical
    comparison runs under Results/supplementary/runs.  Main multi-seed tables
    should use only the full Bayesian guarded runs with all three cases.
    """
    env = _run_environment(record)
    return (
        env.get("preset") == "full"
        and list(env.get("methods", [])) == ["bayesian"]
        and set(env.get("cases", [])) == set(CASES)
    )


def seed0_record(seed0_root: str | None) -> RunRecord | None:
    if not seed0_root:
        return None
    root = Path(seed0_root).expanduser().resolve()
    raw_dir = root / "results" / "raw"
    metrics_dir = root / "results" / "metrics"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"seed0 raw directory not found: {raw_dir}")
    return RunRecord(
        name="seed0_v3.34",
        seed=0,
        root=root,
        raw_dir=raw_dir,
        metrics_dir=metrics_dir,
        case_root_pattern="Results/{case}_EXP",
    )


def collect_runs(seed0_root: str | None = None) -> list[RunRecord]:
    archived = archived_runs()
    main_records = [r for r in archived if _is_main_bayesian_full_run(r)]
    records = latest_record_per_seed(main_records if main_records else archived)
    s0 = seed0_record(seed0_root)
    if s0 is not None:
        records = [r for r in records if r.seed != 0]
        records = [s0] + records
    return records


def load_npz(record: RunRecord, case: str):
    path = record.raw_dir / f"{case}_predictions.npz"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True)


def read_config(case: str) -> dict[str, str]:
    path = PROJECT_ROOT / "Config" / f"{case}_EXP.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path, header=None)
    cfg: dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip()
        value = str(row.iloc[1]).strip() if len(row) > 1 else ""
        if key and key.lower() not in {"key", "names", "nan"}:
            cfg[key] = value
    return cfg


def cfg_float(cfg: dict[str, str], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def as_flat(array) -> np.ndarray:
    return np.asarray(array, dtype=float).reshape(-1)


def scalar_string(data, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    try:
        return str(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def scalar_float(data, key: str, default: float = math.nan) -> float:
    if key not in data:
        return default
    try:
        return float(np.asarray(data[key]).reshape(-1)[0])
    except Exception:
        return default


def rel_l2(pred, exact) -> float:
    p = as_flat(pred)
    e = as_flat(exact)
    denom = float(np.sum(e ** 2))
    return float(np.sqrt(np.sum((p - e) ** 2) / max(denom, 1e-15)))


def coverage95(mean, std, exact) -> float:
    m = as_flat(mean)
    s = np.maximum(as_flat(std), 1e-12)
    e = as_flat(exact)
    return float(np.mean((e >= m - 1.96 * s) & (e <= m + 1.96 * s)))


def avg_interval_width(std) -> float:
    s = np.maximum(as_flat(std), 1e-12)
    return float(np.mean(2.0 * 1.96 * s))


def nll_gaussian(mean, std, exact) -> float:
    m = as_flat(mean)
    s = np.maximum(as_flat(std), 1e-12)
    e = as_flat(exact)
    var = s ** 2
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var)
                         + (e - m) ** 2 / (2.0 * var)))


def error_std_corr(mean, std, exact) -> float:
    err = np.abs(as_flat(mean) - as_flat(exact))
    s = as_flat(std)
    if float(np.std(err)) <= 1e-12 or float(np.std(s)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(err, s)[0, 1])


def evaluate_prediction(mean, std, exact) -> dict[str, float]:
    row = {"rel_l2": rel_l2(mean, exact)}
    if std is not None:
        row.update({
            "coverage95": coverage95(mean, std, exact),
            "corr": error_std_corr(mean, std, exact),
            "nll": nll_gaussian(mean, std, exact),
            "avg_interval_width": avg_interval_width(std),
        })
    return row


def calibration_eval_mask(data) -> np.ndarray | None:
    if "std_calibration_mask" not in data:
        return None
    mask = np.asarray(data["std_calibration_mask"]).astype(bool).reshape(-1)
    eval_mask = ~mask
    return eval_mask if int(eval_mask.sum()) >= 10 else None


def apply_mask(array, mask: np.ndarray | None):
    flat = as_flat(array)
    return flat[mask] if mask is not None else flat


def preferred_raw_std(data, case: str):
    keys = [
        "bayesian_raw_std",
        "bayesian_dense_raw_std",
        "bayesian_uncalibrated_std",
        "bayesian_dense_std",
    ]
    for key in keys:
        if key in data:
            return data[key], key
    if "bayesian_std" in data:
        return data["bayesian_std"], "bayesian_std"
    return None, ""


def read_parameter_error(record: RunRecord, case: str) -> float:
    if case != "Burgers_inv":
        return math.nan
    path = record.metrics_dir / "parameter_inversion_metrics.csv"
    if not path.is_file():
        return math.nan
    df = pd.read_csv(path)
    if df.empty or "relative_error" not in df.columns:
        return math.nan
    preferred = df[df.get("method", "").astype(str).str.contains(
        "Bayesian", case=False, na=False)]
    row = preferred.iloc[0] if not preferred.empty else df.iloc[0]
    try:
        return float(row["relative_error"])
    except Exception:
        return math.nan


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def summarize_mean_std(df: pd.DataFrame, group_cols: Iterable[str],
                       value_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(sub))
        for col in value_cols:
            if col in sub:
                row[f"{col}_mean"] = float(pd.to_numeric(
                    sub[col], errors="coerce").mean())
                row[f"{col}_std"] = float(pd.to_numeric(
                    sub[col], errors="coerce").std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
