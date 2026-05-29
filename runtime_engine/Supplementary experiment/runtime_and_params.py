"""Collect runtime, parameter count, compression, and model-size evidence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

from common import (
    CASES,
    PROJECT_ROOT,
    TABLE_DIR,
    collect_runs,
    ensure_output_dirs,
    load_npz,
    scalar_float,
    write_csv,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _state_dict_param_count(state_dict, kind: str = "dense") -> int | None:
    try:
        import torch
    except Exception:
        return None

    if not isinstance(state_dict, dict):
        return None
    skip_common = (
        "running_mean", "running_var", "num_batches_tracked",
        "positional_encoding", "fourier_modes_freq",
    )
    skip_structured = (".R", "relation_matrix")
    total = 0
    found = False
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(name)
        key_lower = key.lower()
        if any(tok in key_lower for tok in skip_common):
            continue
        if kind == "structured":
            # StructuredLinear.R is a frozen buffer that can be larger than the
            # dense weight matrix. It is storage overhead, not a trainable
            # structured parameter.
            if any(tok in key for tok in skip_structured):
                continue
        total += int(value.numel())
        found = True
    return total if found else None


def _module_param_count(obj) -> int | None:
    try:
        import torch
    except Exception:
        return None

    if hasattr(torch, "nn") and isinstance(obj, torch.nn.Module):
        return int(sum(p.numel() for p in obj.parameters()))
    return None


def count_checkpoint_params(path: Path, kind: str) -> tuple[int | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "structured":
            model = obj.get("model_object") if isinstance(obj, dict) else obj
            if hasattr(model, "count_parameters"):
                stats = model.count_parameters()
                count = stats.get("trainable") if isinstance(stats, dict) else None
                if count:
                    return int(count), "model_object.count_parameters.trainable"
            if isinstance(obj, dict):
                count = _state_dict_param_count(obj.get("model_state_dict"),
                                                kind="structured")
                if count:
                    return count, "model_state_dict_filtered_buffers"
            return None, "unavailable"

        module_count = _module_param_count(obj)
        if module_count:
            return module_count, "module.parameters"

        if isinstance(obj, dict):
            count = _state_dict_param_count(obj.get("model_state_dict"), kind=kind)
            if count:
                return count, "model_state_dict"
            count = _state_dict_param_count(obj, kind=kind)
            if count:
                return count, "plain_state_dict"
        return None, "unavailable"
    except Exception:
        return None, "load_failed"


def find_model_files(case_dir: Path, case: str) -> dict[str, Path | None]:
    model_dir = case_dir / "Models"
    if not model_dir.is_dir():
        return {"dense": None, "structured": None, "teacher": None}

    def first(patterns):
        for pattern in patterns:
            hits = sorted(model_dir.glob(pattern))
            if hits:
                return hits[0]
        return None

    return {
        "teacher": first([f"{case}_EXP_PINN_best.pth", f"{case}_EXP_PINN.pth"]),
        "dense": first([
            f"{case}_EXP_Student_MCDropout_student_mean_refined_best.pth",
            f"{case}_EXP_Student_MCDropout_student_best.pth",
            f"{case}_EXP_Student_MCDropout_student.pth",
        ]),
        "structured": first([
            f"{case}_EXP_bayesian_structured.pth",
            f"{case}_EXP_structured.pth",
        ]),
    }


def read_summary(case_dir: Path) -> dict:
    path = case_dir / "summary.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def read_clock(case_dir: Path) -> dict:
    path = case_dir / "Clock time.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def build_rows(seed0_root: str | None) -> list[dict]:
    rows: list[dict] = []
    for record in collect_runs(seed0_root):
        for case in CASES:
            case_dir = record.case_dir(case)
            files = find_model_files(case_dir, case)
            summary = read_summary(case_dir)
            clock = read_clock(case_dir)
            data = load_npz(record, case)
            compression = math.nan
            if data is not None:
                compression = scalar_float(data, "bayesian_structured_compression",
                                           math.nan)
            if math.isnan(compression) and "compression" in summary:
                try:
                    compression = float(summary["compression"])
                except Exception:
                    pass

            row = {
                "run_id": record.name,
                "seed": record.seed,
                "case": case,
                "teacher_time_s": summary.get(
                    "teacher_time_s", clock.get("Training Time", math.nan)),
                "student_time_s": summary.get(
                    "student_time_s", clock.get("Student Training Time", math.nan)),
                "train_steps": summary.get("train_steps", math.nan),
                "student_train_steps": summary.get("student_train_steps", math.nan),
                "mean_refine_steps": summary.get("mean_refine_steps", math.nan),
                "poisson_mean_refine_steps": summary.get(
                    "poisson_mean_refine_steps", math.nan),
                "anchor_pretrain_steps": summary.get(
                    "anchor_pretrain_steps", math.nan),
                "residual_pretrain_steps": summary.get(
                    "residual_pretrain_steps", math.nan),
                "compression": compression,
            }

            for label, path in files.items():
                row[f"{label}_model_path"] = str(path) if path else ""
                row[f"{label}_model_size_bytes"] = (
                    int(path.stat().st_size) if path and path.is_file() else math.nan)
                count, source = (
                    count_checkpoint_params(path, label)
                    if path else (None, "missing")
                )
                row[f"{label}_param_count"] = (
                    count if count is not None else math.nan)
                row[f"{label}_param_count_source"] = source
                if label == "structured":
                    r_bytes = math.nan
                    if path and path.is_file():
                        try:
                            import torch
                            obj = torch.load(path, map_location="cpu",
                                             weights_only=False)
                            state = (obj.get("model_state_dict", {})
                                     if isinstance(obj, dict) else {})
                            total_bytes = 0
                            for name, value in state.items():
                                if ".R" in str(name) and hasattr(value, "numel"):
                                    total_bytes += int(value.numel()) * int(value.element_size())
                            r_bytes = total_bytes
                        except Exception:
                            r_bytes = math.nan
                    row["frozen_R_storage_bytes"] = r_bytes

            dense_params = row.get("dense_param_count")
            struct_params = row.get("structured_param_count")
            try:
                dense_ok = dense_params and not math.isnan(float(dense_params))
                struct_ok = struct_params and not math.isnan(float(struct_params))
                comp_ok = not math.isnan(float(compression)) and float(compression) > 0
            except Exception:
                dense_ok = struct_ok = comp_ok = False
            if (not struct_ok) and dense_ok and comp_ok:
                struct_params = int(round(float(dense_params) / float(compression)))
                row["structured_param_count"] = struct_params
                row["structured_param_count_source"] = "dense_param_count / npz_compression"
                struct_ok = True
            if dense_ok and struct_ok and float(struct_params) > 0:
                row["param_count_compression"] = float(dense_params) / float(struct_params)
            else:
                row["param_count_compression"] = math.nan
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-root", default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    rows = build_rows(args.seed0_root)
    if not rows:
        raise SystemExit("No runtime/model outputs found.")
    df = pd.DataFrame(rows).sort_values(["case", "seed", "run_id"])
    write_csv(df, TABLE_DIR / "runtime_and_params_raw.csv")

    numeric_cols = [
        "teacher_time_s", "student_time_s", "compression",
        "teacher_model_size_bytes", "dense_model_size_bytes",
        "structured_model_size_bytes", "teacher_param_count",
        "dense_param_count", "structured_param_count",
        "param_count_compression",
    ]
    summary = df.groupby("case", dropna=False)[numeric_cols].agg(
        ["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    write_csv(summary.sort_values("case"),
              TABLE_DIR / "runtime_and_params_mean_std.csv")

    print("\nRuntime and parameter summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
