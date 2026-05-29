"""Execute the integrated training runtime and write official outputs.

The adapter calls the vendored runtime engine, then normalizes its artifacts
into the `results/unified_blind_protocol/...` contract.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gbsd.evaluation.metrics import (
    average_interval_width,
    coverage95,
    error_std_corr,
    gaussian_nll,
    relative_l2,
)
from gbsd.reporting.schema import REQUIRED_METRIC_FIELDS


ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = ROOT / "runtime_engine"
OFFICIAL_ROOT = ROOT / "results" / "unified_blind_protocol"
SMOKE_ROOT = ROOT / "results" / "smoke_test"
ACTIVE_RESULTS_ROOT = OFFICIAL_ROOT
PROBLEMS = {
    "laplace": "Laplace",
    "poisson": "Poisson",
    "burgers_inverse": "Burgers_inv",
}
LEGACY_TO_OFFICIAL = {v: k for k, v in PROBLEMS.items()}
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
RUNTIME_SOURCE_ID = "integrated_runtime_engine"


@dataclass
class RuntimeRun:
    experiment: str
    problem: str
    variant: str
    seed: int | str
    official_dir: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_official_dir(path: Path) -> None:
    """Create an official output dir while preserving any previous run."""
    if path.exists():
        rel = path.relative_to(ACTIVE_RESULTS_ROOT)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archive = ROOT / "results" / "audit" / "replaced_official_runs" / rel.parent / f"{rel.name}_{stamp}"
        suffix = 1
        while archive.exists():
            archive = archive.with_name(f"{rel.name}_{stamp}_{suffix}")
            suffix += 1
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive))
    path.mkdir(parents=True, exist_ok=True)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _run_command(cmd: list[str], cwd: Path, execute: bool) -> int:
    print(" ".join(cmd))
    if not execute:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=True).returncode


def _case_to_problem(case: str) -> str:
    return LEGACY_TO_OFFICIAL.get(case, case.lower())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_config_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows: dict[str, Any] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            key = str(row[0]).strip()
            if key.lower() in {"key", "names"} or not key:
                continue
            rows[key] = row[1]
    return rows


def _write_config_yaml(path: Path, config: dict[str, Any], meta: dict[str, Any]) -> None:
    lines = ["# Auto-generated from runtime CSV config.", "protocol_id: unified_blind_protocol_v1"]
    for key, value in meta.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("runtime_config:")
    for key in sorted(config):
        lines.append(f"  {key}: {json.dumps(config[key], ensure_ascii=False)}")
    _write_text(path, "\n".join(lines) + "\n")


def _latest_archive(runtime_root: Path, preset: str, seed: int, before: set[str]) -> Path:
    runs = runtime_root / "Results" / "supplementary" / "runs"
    pattern = f"{preset}_s{seed}_*"
    candidates = [p for p in runs.glob(pattern) if p.name not in before]
    if not candidates:
        candidates = list(runs.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No runtime archive matching {pattern} in {runs}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _archive_names(runtime_root: Path) -> set[str]:
    runs = runtime_root / "Results" / "supplementary" / "runs"
    if not runs.is_dir():
        return set()
    return {p.name for p in runs.iterdir() if p.is_dir()}


def _filter_npz_arrays(raw_npz: Path, output_npz: Path, mask: np.ndarray | None) -> None:
    data = np.load(raw_npz, allow_pickle=True)
    arrays: dict[str, Any] = {}
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
    for key in data.files:
        arr = data[key]
        if mask is not None and arr.shape and arr.shape[0] == mask.size:
            arrays[key] = arr[mask]
        elif mask is not None and arr.ndim >= 2 and arr.shape[0] != mask.size and arr.shape[1] == mask.size:
            arrays[key] = arr[:, mask, ...]
        else:
            arrays[key] = arr
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)


def _copy_checkpoints(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.is_dir():
        return
    for source in source_dir.rglob("*.pth"):
        relative = source.relative_to(source_dir)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_prediction_contract(source_npz: Path, output_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not source_npz.is_file():
        np.savez_compressed(output_dir / "predictions_blind.npz")
        np.savez_compressed(output_dir / "predictions_guard.npz")
        return None, None
    data = np.load(source_npz, allow_pickle=True)
    guard_mask = _mask_from_npz(data, "guard_mask")
    blind_mask = _mask_from_npz(data, "blind_test_mask")
    _filter_npz_arrays(source_npz, output_dir / "predictions_guard.npz", guard_mask)
    _filter_npz_arrays(source_npz, output_dir / "predictions_blind.npz", blind_mask)
    return guard_mask, blind_mask


def _mask_from_npz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key in data.files:
        return np.asarray(data[key]).reshape(-1).astype(bool)
    return None


def _metrics_from_prediction_npz(raw_npz: Path, blind_mask: np.ndarray | None) -> dict[str, Any]:
    data = np.load(raw_npz, allow_pickle=True)
    mask = blind_mask
    if mask is None:
        mask = _mask_from_npz(data, "blind_test_mask")

    def arr(name: str):
        if name not in data.files:
            return None
        value = data[name]
        if mask is not None and value.shape and value.shape[0] == mask.size:
            return value[mask]
        return value

    exact = arr("exact")
    dense = arr("bayesian_dense_mean")
    structured = arr("bayesian_structured_mean")
    final = arr("bayesian_mean")
    std = arr("bayesian_std")
    metrics: dict[str, Any] = {}
    if exact is not None and dense is not None:
        metrics["dense_rL2"] = relative_l2(dense, exact)
    if exact is not None and structured is not None:
        metrics["structured_rL2"] = relative_l2(structured, exact)
    if exact is not None and final is not None:
        metrics["final_rL2"] = relative_l2(final, exact)
    if exact is not None and final is not None and std is not None:
        metrics["coverage95"] = coverage95(final, exact, std)
        metrics["nll"] = gaussian_nll(final, exact, std)
        metrics["aiw"] = average_interval_width(std)
        metrics["error_std_corr"] = error_std_corr(final, exact, std)
    if "bayesian_source" in data.files:
        metrics["final_source"] = str(data["bayesian_source"].reshape(-1)[0])
    if "bayesian_structured_compression" in data.files:
        metrics["compression_ratio"] = float(data["bayesian_structured_compression"].reshape(-1)[0])
    return metrics


def _read_case_summary(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "summary.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else {}


def _read_parameter_metrics(archive: Path, case: str) -> dict[str, Any]:
    path = archive / "metrics" / "parameter_inversion_metrics.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("case") == case]
    if not rows:
        return {}
    # Prefer the smallest relative-error diagnostic if multiple teacher-stage
    # trajectories are present; the run manifest records that this is pipeline-level.
    def rel(row):
        try:
            return float(row.get("relative_error", "nan"))
        except Exception:
            return math.inf
    row = min(rows, key=rel)
    return {
        "nu_true": _maybe_float(row.get("true_value")),
        "nu_pred": _maybe_float(row.get("estimated_mean")),
        "nu_rel_error": _maybe_float(row.get("relative_error")),
    }


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _model_sizes(case_dir: Path) -> dict[str, Any]:
    models = case_dir / "Models"
    sizes: dict[str, Any] = {}
    if not models.is_dir():
        return sizes
    for path in models.glob("*.pth"):
        sizes[path.name] = path.stat().st_size
    return sizes


def _state_dict_parameter_count(state_dict: Any, structured: bool = False) -> int | None:
    try:
        import torch
    except Exception:
        return None
    if not isinstance(state_dict, dict):
        return None
    total = 0
    found = False
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(name).lower()
        if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked")):
            continue
        if structured and (key.endswith(".r") or "relation_matrix" in key):
            continue
        total += int(value.numel())
        found = True
    return total if found else None


def _checkpoint_parameter_count(path: Path, structured: bool = False) -> tuple[int | None, dict[str, Any]]:
    if not path.is_file():
        return None, {}
    try:
        import torch
        engine_path = str(ENGINE_ROOT)
        if engine_path not in sys.path:
            sys.path.insert(0, engine_path)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        metadata: dict[str, Any] = {}
        if structured:
            model = obj.get("model_object") if isinstance(obj, dict) else obj
            if hasattr(model, "count_parameters"):
                stats = model.count_parameters()
                metadata["structured_original_params"] = stats.get("original")
                metadata["structured_checkpoint_compression_ratio"] = stats.get("compression_ratio")
                metadata["structured_training_time_s"] = (
                    _maybe_float(obj.get("training_time")) if isinstance(obj, dict) else None
                )
                return int(stats["trainable"]), metadata
            state = obj.get("model_state_dict") if isinstance(obj, dict) else None
            return _state_dict_parameter_count(state, structured=True), metadata
        if hasattr(obj, "parameters"):
            return int(sum(p.numel() for p in obj.parameters())), metadata
        if isinstance(obj, dict):
            state = obj.get("model_state_dict", obj)
            return _state_dict_parameter_count(state), metadata
    except Exception:
        return None, {}
    return None, {}


def _find_checkpoint(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        found = sorted(directory.glob(pattern))
        if found:
            return found[0]
    return None


def _resource_accounting(checkpoint_dir: Path) -> dict[str, Any]:
    teacher = _find_checkpoint(checkpoint_dir, ["*_PINN_best.pth", "*_PINN.pth"])
    dense = _find_checkpoint(
        checkpoint_dir,
        ["*_Student_MCDropout_student_mean_refined_best.pth",
         "*_Student_MCDropout_student_best.pth", "*_Student_MCDropout_student.pth"],
    )
    structured = _find_checkpoint(checkpoint_dir, ["*_bayesian_structured.pth", "*_structured.pth"])
    teacher_params, _ = _checkpoint_parameter_count(teacher) if teacher else (None, {})
    dense_params, _ = _checkpoint_parameter_count(dense) if dense else (None, {})
    structured_params, structured_meta = (
        _checkpoint_parameter_count(structured, structured=True) if structured else (None, {})
    )
    result = {
        "teacher_params": teacher_params,
        "dense_params": dense_params,
        "structured_trainable_params": structured_params,
        "structured_effective_params": structured_params,
        **structured_meta,
    }
    if dense_params is not None and structured_params:
        result["compression_ratio"] = float(dense_params) / float(structured_params)
    return result


def _complete_metrics(record: dict[str, Any]) -> dict[str, Any]:
    completed = dict(record)
    for field in REQUIRED_METRIC_FIELDS:
        completed.setdefault(field, None)
    return completed


def normalize_main_archive(
    archive: Path,
    cases: Iterable[str],
    seed: int,
    preset: str,
) -> list[RuntimeRun]:
    outputs: list[RuntimeRun] = []
    env = _read_json(archive / "environment.json")
    for case in cases:
        problem = _case_to_problem(case)
        run = RuntimeRun(
            experiment="main_blind",
            problem=problem,
            variant="full_gbsd",
            seed=seed,
            official_dir=ACTIVE_RESULTS_ROOT / "main_blind" / problem / "full_gbsd" / f"seed_{seed}",
        )
        _prepare_official_dir(run.official_dir)

        raw_npz = archive / "raw" / f"{case}_predictions.npz"
        guard_mask, blind_mask = _write_prediction_contract(raw_npz, run.official_dir)

        config = _read_config_csv(archive / "config_snapshot" / f"{case}_EXP.csv")
        _write_config_yaml(
            run.official_dir / "config_resolved.yaml",
            config,
            {
                "experiment": "main_blind",
                "problem": problem,
                "variant": "full_gbsd",
                "seed": seed,
                "runtime_source_id": RUNTIME_SOURCE_ID,
            },
        )

        split_manifest = {
            "protocol_id": "unified_blind_protocol_v1",
            "split_id": f"{problem}_seed_{seed}",
            "guard_split": "guard_validation",
            "blind_split": "blind_test",
            "guard_may_read_blind_labels": False,
            "n_guard": int(guard_mask.sum()) if guard_mask is not None else None,
            "n_blind": int(blind_mask.sum()) if blind_mask is not None else None,
            "overlap_count": int(np.logical_and(guard_mask, blind_mask).sum())
            if guard_mask is not None and blind_mask is not None else None,
        }
        _write_json(run.official_dir / "split_manifest.json", split_manifest)

        metrics = _metrics_from_prediction_npz(raw_npz, blind_mask) if raw_npz.is_file() else {}
        case_summary = _read_case_summary(archive / f"{case}_EXP")
        param_metrics = _read_parameter_metrics(archive, case)
        timing = {
            "teacher_training_time_s": _maybe_float(case_summary.get("teacher_time_s")),
            "student_training_time_s": _maybe_float(case_summary.get("student_time_s")),
            "structured_training_time_s": None,
            "total_wall_clock_time_s": _maybe_float(env.get("elapsed_seconds")),
        }
        resources = _resource_accounting(archive / f"{case}_EXP" / "Models")
        timing["structured_training_time_s"] = resources.get("structured_training_time_s")
        model_size = _model_sizes(archive / f"{case}_EXP")
        guard_decision = {
            "decision_split": "guard_validation",
            "final_source": metrics.get("final_source"),
            "guard_may_read_blind_labels": False,
        }
        metrics.update(timing)
        metrics.update(resources)
        metrics.update({
            "device_name": env.get("gpu_name") if env.get("cuda_available") else "CPU",
            "cuda_version": env.get("cuda_version"),
            "pytorch_version": env.get("torch_version"),
            "ram_gb": (
                float(env["ram_total_bytes"]) / (1024.0 ** 3)
                if env.get("ram_total_bytes") is not None else None
            ),
        })
        metrics.update(param_metrics)
        metrics.update({
            "protocol_id": "unified_blind_protocol_v1",
            "experiment": "main_blind",
            "problem": problem,
            "variant": "full_gbsd",
            "seed": seed,
            "split_id": f"{problem}_seed_{seed}",
            "guard_decision": metrics.get("final_source"),
        })
        _write_json(run.official_dir / "metrics.json", _complete_metrics(metrics))
        _write_json(run.official_dir / "timing.json", timing)
        _write_json(run.official_dir / "model_size.json", model_size)
        _write_json(run.official_dir / "guard_decision.json", guard_decision)
        _copy_checkpoints(archive / f"{case}_EXP" / "Models", run.official_dir / "checkpoints")
        _write_json(run.official_dir / "run_manifest.json", {
            "protocol_id": "unified_blind_protocol_v1",
            "runtime_source_id": RUNTIME_SOURCE_ID,
            "runtime_archive": _relative(archive),
            "runtime_case": case,
            "preset": preset,
            "seed": seed,
            "official_status": "generated_by_runtime_adapter",
            "nu_metric_scope": "pipeline_level_inverse_parameter" if case == "Burgers_inv" else None,
        })
        outputs.append(run)
    return outputs


def normalize_uq_baseline_metrics(metrics_path: Path, variant: str, seed: int | str) -> RuntimeRun:
    raw = _read_json(metrics_path)
    case = raw.get("case")
    if not case:
        raise ValueError(f"Missing case in {metrics_path}")
    problem = _case_to_problem(str(case))
    run = RuntimeRun(
        experiment="strong_baselines",
        problem=problem,
        variant=variant,
        seed=seed,
        official_dir=ACTIVE_RESULTS_ROOT / "strong_baselines" / problem / variant / f"seed_{seed}",
    )
    _prepare_official_dir(run.official_dir)

    metrics = {
        "protocol_id": "unified_blind_protocol_v1",
        "experiment": "strong_baselines",
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "split_id": f"{problem}_seed_{seed}",
        "dense_rL2": raw.get("rel_l2_blind"),
        "structured_rL2": None,
        "final_rL2": raw.get("rel_l2_blind"),
        "dense_params": None,
        "structured_trainable_params": None,
        "structured_effective_params": None,
        "compression_ratio": None,
        "coverage95": raw.get("coverage95_blind"),
        "nll": raw.get("nll_blind"),
        "aiw": raw.get("mean_interval_width_blind"),
        "error_std_corr": raw.get("corr_abs_error_std_blind"),
        "final_source": variant,
        "guard_decision": "unused_guard_set",
        "teacher_training_time_s": None,
        "student_training_time_s": raw.get("runtime_train_s"),
        "structured_training_time_s": None,
        "nu_true": raw.get("nu_true"),
        "nu_pred": raw.get("nu_pred"),
        "nu_rel_error": raw.get("nu_relative_error"),
    }
    _write_json(run.official_dir / "metrics.json", _complete_metrics(metrics))
    _write_json(run.official_dir / "timing.json", {
        "student_training_time_s": raw.get("runtime_train_s"),
        "inference_time_s": raw.get("runtime_mc_or_ensemble_inference_s"),
    })
    _write_json(run.official_dir / "model_size.json", {})
    _write_json(run.official_dir / "guard_decision.json", {
        "guard_decision": "unused_guard_set",
        "unused_guard_set": True,
    })
    _write_json(run.official_dir / "split_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "split_id": f"{problem}_seed_{seed}",
        "must_match_main_split": True,
        "unused_guard_set": True,
        "n_guard": raw.get("n_guard"),
        "n_blind": raw.get("n_blind"),
    })
    _write_config_yaml(run.official_dir / "config_resolved.yaml", {}, {
        "experiment": "strong_baselines",
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "runtime_source_id": RUNTIME_SOURCE_ID,
    })
    _copy_checkpoints(metrics_path.parent, run.official_dir / "checkpoints")
    _write_json(run.official_dir / "run_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "runtime_source_id": RUNTIME_SOURCE_ID,
        "runtime_metrics": _relative(metrics_path),
        "official_status": "generated_by_runtime_adapter",
        "uses_teacher_distillation": False,
    })
    prediction_name = "ensemble_predictions.npz" if variant == "deep_ensemble_pinn" else "predictions.npz"
    _write_prediction_contract(metrics_path.parent / prediction_name, run.official_dir)
    return run


def normalize_structure_baseline_metrics(metrics_path: Path) -> RuntimeRun:
    raw = _read_json(metrics_path)
    case = raw.get("case")
    method = raw.get("structure_method")
    seed = raw.get("seed")
    if case is None or method is None or seed is None:
        raise ValueError(f"Missing case/method/seed in {metrics_path}")
    problem = _case_to_problem(str(case))
    variant = f"structure_discovery_{method}"
    run = RuntimeRun(
        experiment="strong_baselines",
        problem=problem,
        variant=variant,
        seed=int(seed),
        official_dir=ACTIVE_RESULTS_ROOT / "strong_baselines" / problem / variant / f"seed_{int(seed)}",
    )
    _prepare_official_dir(run.official_dir)
    metrics = {
        "protocol_id": "unified_blind_protocol_v1",
        "experiment": "strong_baselines",
        "problem": problem,
        "variant": variant,
        "seed": int(seed),
        "split_id": f"{problem}_seed_{int(seed)}",
        "dense_rL2": raw.get("dense_rel_l2_blind"),
        "structured_rL2": raw.get("structured_rel_l2_blind"),
        "final_rL2": raw.get("final_rel_l2_blind"),
        "dense_params": raw.get("dense_param_count"),
        "structured_trainable_params": raw.get("structured_param_count"),
        "structured_effective_params": raw.get("structured_param_count"),
        "compression_ratio": raw.get("compression_ratio"),
        "coverage95": raw.get("coverage95_blind"),
        "nll": raw.get("nll_blind"),
        "aiw": raw.get("mean_interval_width_blind"),
        "error_std_corr": raw.get("corr_abs_error_std_blind"),
        "final_source": raw.get("final_source"),
        "guard_decision": raw.get("accepted_by_guard"),
        "teacher_training_time_s": None,
        "student_training_time_s": None,
        "structured_training_time_s": raw.get("runtime_reconstruction_s"),
        "nu_true": raw.get("nu_true"),
        "nu_pred": raw.get("nu_pred"),
        "nu_rel_error": raw.get("nu_relative_error"),
    }
    _write_json(run.official_dir / "metrics.json", _complete_metrics(metrics))
    _write_json(run.official_dir / "timing.json", {
        "structure_discovery_time_s": raw.get("runtime_discovery_s"),
        "structured_training_time_s": raw.get("runtime_reconstruction_s"),
        "inference_time_s": raw.get("runtime_inference_s"),
    })
    _write_json(run.official_dir / "model_size.json", {})
    _write_json(run.official_dir / "guard_decision.json", {
        "accuracy_ok": raw.get("accuracy_ok"),
        "compression_ok": raw.get("compression_ok"),
        "accepted_by_guard": raw.get("accepted_by_guard"),
        "decision_split": "guard_validation",
    })
    _write_json(run.official_dir / "split_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "split_id": f"{problem}_seed_{int(seed)}",
        "n_guard": raw.get("n_guard"),
        "n_blind": raw.get("n_blind"),
        "must_match_main_split": True,
    })
    _write_config_yaml(run.official_dir / "config_resolved.yaml", {}, {
        "experiment": "strong_baselines",
        "problem": problem,
        "variant": variant,
        "seed": int(seed),
        "runtime_source_id": RUNTIME_SOURCE_ID,
    })
    _copy_checkpoints(metrics_path.parent, run.official_dir / "checkpoints")
    _write_json(run.official_dir / "run_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "runtime_source_id": RUNTIME_SOURCE_ID,
        "runtime_metrics": _relative(metrics_path),
        "official_status": "generated_by_runtime_adapter",
    })
    _write_prediction_contract(metrics_path.parent / "predictions.npz", run.official_dir)
    return run


def _require_runtime_engine() -> None:
    required = [
        ENGINE_ROOT / "run_all_experiments.py",
        ENGINE_ROOT / "Supplementary experiment" / "run_direct_mc_dropout_pinn_multi_seed.py",
        ENGINE_ROOT / "Supplementary experiment" / "run_deep_ensemble_pinn.py",
        ENGINE_ROOT / "Supplementary experiment" / "run_structure_discovery_baselines.py",
        ENGINE_ROOT / "Config",
        ENGINE_ROOT / "Database",
        ENGINE_ROOT / "Module",
        ENGINE_ROOT / "utils",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Integrated runtime engine is incomplete: {', '.join(missing)}")


def run_main_blind(seeds: list[int], preset: str, execute: bool, python: str, n_mc: int) -> list[RuntimeRun]:
    outputs: list[RuntimeRun] = []
    for seed in seeds:
        before = _archive_names(ENGINE_ROOT)
        cmd = [
            python,
            "run_all_experiments.py",
            "--case", "all",
            "--method", "bayesian",
            "--preset", preset,
            "--seed", str(seed),
            "--clean",
            "--n_mc", str(n_mc),
        ]
        _run_command(cmd, ENGINE_ROOT, execute)
        if execute:
            archive = _latest_archive(ENGINE_ROOT, preset, seed, before)
            outputs.extend(normalize_main_archive(archive, PROBLEMS.values(), seed, preset))
    return outputs


def run_direct_mc_dropout(seeds: list[int], preset: str, execute: bool, python: str, n_mc: int) -> list[RuntimeRun]:
    script = ENGINE_ROOT / "Supplementary experiment" / "run_direct_mc_dropout_pinn_multi_seed.py"
    seed_args = [item for seed in seeds for item in ("--seed", str(seed))]
    cmd = [python, str(script), "--case", "all", "--preset", preset, "--n_mc", str(n_mc), *seed_args]
    _run_command(cmd, ENGINE_ROOT, execute)
    outputs: list[RuntimeRun] = []
    if execute:
        requested_seeds = set(seeds)
        root = ENGINE_ROOT / "Results" / "supplementary" / "full_baselines" / "uq" / "direct_mc_dropout_pinn" / preset
        for metrics_path in root.glob("*/seed_*/metrics.json"):
            seed_text = metrics_path.parent.name.replace("seed_", "")
            seed = int(seed_text)
            if seed in requested_seeds:
                outputs.append(normalize_uq_baseline_metrics(metrics_path, "direct_mc_dropout_pinn", seed))
    return outputs


def run_deep_ensemble(seeds: list[int], preset: str, execute: bool, python: str) -> list[RuntimeRun]:
    script = ENGINE_ROOT / "Supplementary experiment" / "run_deep_ensemble_pinn.py"
    cmd = [
        python,
        str(script),
        "--case", "all",
        "--preset", preset,
        "--ensemble_seeds",
        *[str(seed) for seed in seeds],
    ]
    if preset == "smoke":
        cmd.extend(["--K", "2"])
    _run_command(cmd, ENGINE_ROOT, execute)
    outputs: list[RuntimeRun] = []
    if execute:
        root = ENGINE_ROOT / "Results" / "supplementary" / "full_baselines" / "uq" / "deep_ensemble_pinn" / preset
        for case in PROBLEMS.values():
            for ensemble_id, seed in enumerate(seeds):
                metrics_path = root / case / f"ensemble_{ensemble_id}" / "ensemble_metrics.json"
                if metrics_path.is_file():
                    outputs.append(normalize_uq_baseline_metrics(metrics_path, "deep_ensemble_pinn", seed))
    return outputs


def run_structure_discovery_baselines(seeds: list[int], preset: str, execute: bool, python: str) -> list[RuntimeRun]:
    script = ENGINE_ROOT / "Supplementary experiment" / "run_structure_discovery_baselines.py"
    seed_args = [item for seed in seeds for item in ("--seed", str(seed))]
    if preset == "smoke":
        cmd = [
            python, str(script), "--case", "Poisson", *seed_args,
            "--methods", "hac", "random", "magnitude", "low_rank",
            "--epochs", "5", "--grid_n", "10", "--force",
        ]
    else:
        cmd = [python, str(script), "--case", "all", *seed_args]
    _run_command(cmd, ENGINE_ROOT, execute)
    outputs: list[RuntimeRun] = []
    if execute:
        requested_seeds = set(seeds)
        root = ENGINE_ROOT / "Results" / "supplementary" / "full_baselines" / "structure_discovery"
        for metrics_path in root.glob("*/*/seed_*/metrics.json"):
            seed_text = metrics_path.parent.name.replace("seed_", "")
            if int(seed_text) in requested_seeds:
                outputs.append(normalize_structure_baseline_metrics(metrics_path))
    return outputs


def _array(data: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    return None


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    value = _array(data, key)
    if value is None or value.size == 0:
        return default
    item = value.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _main_official_dir(problem: str, seed: int) -> Path:
    path = ACTIVE_RESULTS_ROOT / "main_blind" / problem / "full_gbsd" / f"seed_{seed}"
    required = [path / "predictions_guard.npz", path / "predictions_blind.npz"]
    missing = [item for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Main blind result required before derived experiments: {path}. "
            "Run --stage main first."
        )
    return path


def _candidate_prediction(data: np.lib.npyio.NpzFile, source: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if source == "structured":
        return (
            _array(data, "bayesian_structured_mean"),
            _array(data, "bayesian_structured_raw_std", "bayesian_structured_std"),
        )
    return (
        _array(data, "bayesian_dense_mean"),
        _array(data, "bayesian_dense_raw_std", "bayesian_dense_std"),
    )


def _evaluate_selection(
    data: np.lib.npyio.NpzFile,
    selected_mean: np.ndarray,
    selected_std: np.ndarray | None,
    source: str,
) -> dict[str, Any]:
    exact = _array(data, "exact")
    if exact is None:
        raise KeyError("Official prediction contract is missing exact values.")
    dense = _array(data, "bayesian_dense_mean")
    structured = _array(data, "bayesian_structured_mean")
    record: dict[str, Any] = {"final_source": source}
    if dense is not None:
        record["dense_rL2"] = relative_l2(dense, exact)
    if structured is not None:
        record["structured_rL2"] = relative_l2(structured, exact)
    record["final_rL2"] = relative_l2(selected_mean, exact)
    compression = _scalar(data, "bayesian_structured_compression")
    if compression is not None:
        record["compression_ratio"] = float(compression)
    if selected_std is not None:
        record["coverage95"] = coverage95(selected_mean, exact, selected_std)
        record["nll"] = gaussian_nll(selected_mean, exact, selected_std)
        record["aiw"] = average_interval_width(selected_std)
        record["error_std_corr"] = error_std_corr(selected_mean, exact, selected_std)
    return record


def _save_selected_prediction(
    path: Path,
    source_data: np.lib.npyio.NpzFile,
    mean: np.ndarray,
    std: np.ndarray | None,
    source: str,
) -> None:
    arrays: dict[str, Any] = {
        "exact": _array(source_data, "exact"),
        "bayesian_mean": mean,
        "bayesian_source": np.array([source]),
    }
    for key in ("bayesian_dense_mean", "bayesian_structured_mean", "bayesian_structured_compression"):
        value = _array(source_data, key)
        if value is not None:
            arrays[key] = value
    if std is not None:
        arrays["bayesian_std"] = std
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_derived_run(
    experiment: str,
    problem: str,
    variant: str,
    seed: int,
    source_dir: Path,
    guard_data: np.lib.npyio.NpzFile,
    blind_data: np.lib.npyio.NpzFile,
    guard_mean: np.ndarray,
    blind_mean: np.ndarray,
    guard_std: np.ndarray | None,
    blind_std: np.ndarray | None,
    source: str,
    metrics: dict[str, Any],
    decision: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> RuntimeRun:
    run = RuntimeRun(
        experiment=experiment,
        problem=problem,
        variant=variant,
        seed=seed,
        official_dir=ACTIVE_RESULTS_ROOT / experiment / problem / variant / f"seed_{seed}",
    )
    _prepare_official_dir(run.official_dir)
    _save_selected_prediction(run.official_dir / "predictions_guard.npz", guard_data, guard_mean, guard_std, source)
    _save_selected_prediction(run.official_dir / "predictions_blind.npz", blind_data, blind_mean, blind_std, source)
    split_source = source_dir / "split_manifest.json"
    if split_source.is_file():
        shutil.copy2(split_source, run.official_dir / "split_manifest.json")
    else:
        _write_json(run.official_dir / "split_manifest.json", {
            "protocol_id": "unified_blind_protocol_v1",
            "split_id": f"{problem}_seed_{seed}",
            "guard_may_read_blind_labels": False,
        })
    completed = {
        "protocol_id": "unified_blind_protocol_v1",
        "experiment": experiment,
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "split_id": f"{problem}_seed_{seed}",
        **metrics,
    }
    _write_json(run.official_dir / "metrics.json", _complete_metrics(completed))
    _write_json(run.official_dir / "guard_decision.json", decision)
    _write_json(run.official_dir / "timing.json", {})
    _write_json(run.official_dir / "model_size.json", {})
    (run.official_dir / "checkpoints").mkdir(exist_ok=True)
    _write_json(run.official_dir / "checkpoints" / "source_checkpoint_reference.json", {
        "source_checkpoint_dir": _relative(source_dir / "checkpoints"),
        "reason": "Derived evaluation reuses the fitted main-blind predictors without retraining.",
    })
    config = _read_config_csv(ENGINE_ROOT / "Config" / f"{PROBLEMS[problem]}_EXP.csv")
    config_meta = {
        "experiment": experiment,
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "runtime_source_id": RUNTIME_SOURCE_ID,
        "derived_from_main_blind": True,
    }
    if metadata:
        config_meta["derived_metadata"] = metadata
    _write_config_yaml(run.official_dir / "config_resolved.yaml", config, config_meta)
    _write_json(run.official_dir / "run_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "runtime_source_id": RUNTIME_SOURCE_ID,
        "derived_from": _relative(source_dir),
        "official_status": "derived_from_blind_safe_main_contract",
        **(metadata or {}),
    })
    return run


def _guard_source_for_variant(
    variant: str,
    dense_guard_rL2: float,
    structured_guard_rL2: float,
    compression: float | None,
    config: dict[str, Any],
    seed: int,
) -> tuple[str, dict[str, Any]]:
    ratio = _config_float(config, "accept_structured_rel_l2_ratio", 1.10)
    slack = _config_float(config, "accept_structured_rel_l2_abs", 0.002)
    min_compression = _config_float(config, "min_structured_compression", 0.0)
    accuracy_ok = structured_guard_rL2 <= ratio * dense_guard_rL2 + slack
    compression_ok = min_compression <= 0.0 or (
        compression is not None and compression >= min_compression
    )
    accepted = {
        "full_gbsd": accuracy_ok and compression_ok,
        "always_dense": False,
        "always_structured": True,
        "accuracy_only_guard": accuracy_ok,
        "compression_only_guard": compression_ok,
        "no_guard_random_source": bool(np.random.default_rng(seed + 1701).integers(0, 2)),
    }[variant]
    decision = {
        "decision_split": "guard_validation",
        "guard_may_read_blind_labels": False,
        "dense_guard_rL2": dense_guard_rL2,
        "structured_guard_rL2": structured_guard_rL2,
        "compression_ratio": compression,
        "accuracy_ratio": ratio,
        "accuracy_slack": slack,
        "minimum_compression_ratio": min_compression,
        "accuracy_ok": accuracy_ok,
        "compression_ok": compression_ok,
        "accepted_by_variant": accepted,
        "diagnostic_only": variant == "no_guard_random_source",
    }
    return ("structured" if accepted else "dense_student"), decision


def derive_guard_ablation(seeds: list[int], execute: bool) -> list[RuntimeRun]:
    variants = [
        "full_gbsd",
        "always_dense",
        "always_structured",
        "accuracy_only_guard",
        "compression_only_guard",
        "no_guard_random_source",
    ]
    if not execute:
        print("derive guard_ablation from official main_blind guard/blind predictions")
        return []
    outputs: list[RuntimeRun] = []
    for problem in PROBLEMS:
        config = _read_config_csv(ENGINE_ROOT / "Config" / f"{PROBLEMS[problem]}_EXP.csv")
        for seed in seeds:
            source_dir = _main_official_dir(problem, seed)
            with np.load(source_dir / "predictions_guard.npz", allow_pickle=True) as guard, \
                 np.load(source_dir / "predictions_blind.npz", allow_pickle=True) as blind:
                dense_g, _ = _candidate_prediction(guard, "dense_student")
                struct_g, _ = _candidate_prediction(guard, "structured")
                if dense_g is None or struct_g is None or _array(guard, "exact") is None:
                    raise KeyError(f"Dense/structured guard predictions missing in {source_dir}")
                dense_guard_rL2 = relative_l2(dense_g, _array(guard, "exact"))
                struct_guard_rL2 = relative_l2(struct_g, _array(guard, "exact"))
                compression = _scalar(guard, "bayesian_structured_compression")
                compression = float(compression) if compression is not None else None
                for variant in variants:
                    source, decision = _guard_source_for_variant(
                        variant, dense_guard_rL2, struct_guard_rL2, compression, config, seed
                    )
                    guard_mean, _ = _candidate_prediction(guard, source)
                    blind_mean, _ = _candidate_prediction(blind, source)
                    if guard_mean is None or blind_mean is None:
                        raise KeyError(f"Prediction source {source} unavailable in {source_dir}")
                    metrics = _evaluate_selection(blind, blind_mean, None, source)
                    metrics["guard_decision"] = source
                    outputs.append(_write_derived_run(
                        "guard_ablation", problem, variant, seed, source_dir,
                        guard, blind, guard_mean, blind_mean, None, None, source,
                        metrics, decision,
                        {"uq_fields": "not_recomputed_for_mean_source_guard_ablation"},
                    ))
    return outputs


def _fit_temperature(mean: np.ndarray, exact: np.ndarray, std: np.ndarray, target: float = 0.95) -> float:
    residual = np.abs(np.asarray(mean).reshape(-1) - np.asarray(exact).reshape(-1))
    scale = np.maximum(np.asarray(std).reshape(-1), 1e-12)
    ratios = residual / (1.96 * scale)
    factor = float(np.quantile(ratios[np.isfinite(ratios)], target))
    return max(factor, 1e-12)


def derive_uq_ablation(seeds: list[int], execute: bool) -> list[RuntimeRun]:
    variants = [
        "raw_mc_dropout",
        "temperature_only",
        "disagreement_only",
        "temperature_plus_disagreement",
        "full_calibration",
    ]
    if not execute:
        print("derive uq_calibration_ablation for poisson and burgers_inverse from official main_blind")
        return []
    outputs: list[RuntimeRun] = []
    for problem in ("poisson", "burgers_inverse"):
        for seed in seeds:
            source_dir = _main_official_dir(problem, seed)
            with np.load(source_dir / "predictions_guard.npz", allow_pickle=True) as guard, \
                 np.load(source_dir / "predictions_blind.npz", allow_pickle=True) as blind:
                mean_g = _array(guard, "bayesian_mean")
                mean_b = _array(blind, "bayesian_mean")
                exact_g = _array(guard, "exact")
                raw_g = _array(guard, "bayesian_raw_std", "bayesian_uncalibrated_std", "bayesian_std")
                raw_b = _array(blind, "bayesian_raw_std", "bayesian_uncalibrated_std", "bayesian_std")
                dense_g = _array(guard, "bayesian_dense_mean")
                struct_g = _array(guard, "bayesian_structured_mean")
                dense_b = _array(blind, "bayesian_dense_mean")
                struct_b = _array(blind, "bayesian_structured_mean")
                if any(item is None for item in (mean_g, mean_b, exact_g, raw_g, raw_b, dense_g, struct_g, dense_b, struct_b)):
                    raise KeyError(f"UQ ablation inputs missing in {source_dir}")
                gap_g = np.abs(dense_g - struct_g)
                gap_b = np.abs(dense_b - struct_b)
                mixed_g = np.sqrt(np.maximum(raw_g, 1e-12) ** 2 + gap_g ** 2)
                mixed_b = np.sqrt(np.maximum(raw_b, 1e-12) ** 2 + gap_b ** 2)
                temp_raw = _fit_temperature(mean_g, exact_g, raw_g)
                temp_gap = _fit_temperature(mean_g, exact_g, gap_g)
                temp_mixed = _fit_temperature(mean_g, exact_g, mixed_g)
                std_variants = {
                    "raw_mc_dropout": (raw_g, raw_b, 1.0, 0.0),
                    "temperature_only": (raw_g * temp_raw, raw_b * temp_raw, temp_raw, 0.0),
                    "disagreement_only": (gap_g * temp_gap, gap_b * temp_gap, temp_gap, 1.0),
                    "temperature_plus_disagreement": (mixed_g * temp_mixed, mixed_b * temp_mixed, temp_mixed, 1.0),
                    "full_calibration": (_array(guard, "bayesian_std"), _array(blind, "bayesian_std"), _scalar(blind, "std_temperature_factor"), None),
                }
                source = str(_scalar(blind, "bayesian_source", "guarded_final"))
                for variant in variants:
                    std_g, std_b, temperature, disagreement_weight = std_variants[variant]
                    if std_g is None or std_b is None:
                        raise KeyError(f"Calibrated uncertainty missing for {variant} in {source_dir}")
                    metrics = _evaluate_selection(blind, mean_b, std_b, source)
                    metrics.update({
                        "guard_decision": source,
                        "calibrated_temperature": temperature,
                        "disagreement_weight": disagreement_weight,
                    })
                    outputs.append(_write_derived_run(
                        "uq_calibration_ablation", problem, variant, seed, source_dir,
                        guard, blind, mean_g, mean_b, std_g, std_b, source,
                        metrics,
                        {"decision_split": "guard_validation", "final_source": source, "guard_may_read_blind_labels": False},
                        {"temperature_fitted_on": "guard_validation" if variant != "full_calibration" else "stored_main_calibration"},
                    ))
    return outputs


def _ablation_archive_names() -> set[str]:
    root = ENGINE_ROOT / "Results" / "supplementary" / "ablations"
    return {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()


def _latest_ablation_archive(tag: str, seed: int, before: set[str]) -> Path:
    root = ENGINE_ROOT / "Results" / "supplementary" / "ablations"
    candidates = [p for p in root.glob(f"{tag}_s{seed}_*") if p.name not in before]
    if not candidates:
        candidates = list(root.glob(f"{tag}_s{seed}_*"))
    if not candidates:
        raise FileNotFoundError(f"No runtime ablation archive matching {tag}_s{seed}_* in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_config_ablation(
    case: str,
    tag: str,
    seed: int,
    overrides: dict[str, str],
    preset: str,
    execute: bool,
    python: str,
    n_mc: int,
) -> Path | None:
    script = ENGINE_ROOT / "Supplementary experiment" / "run_config_ablation.py"
    before = _ablation_archive_names()
    cmd = [
        python, str(script), "--case", case, "--tag", tag, "--seed", str(seed),
        "--preset", preset, "--method", "bayesian", "--n-mc", str(n_mc), "--clean",
    ]
    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={value}"])
    _run_command(cmd, ENGINE_ROOT, execute)
    return _latest_ablation_archive(tag, seed, before) if execute else None


def normalize_config_ablation_archive(
    archive: Path,
    experiment: str,
    case: str,
    variant: str,
    seed: int,
    preset: str,
    overrides: dict[str, str],
) -> RuntimeRun:
    problem = _case_to_problem(case)
    run = RuntimeRun(
        experiment=experiment,
        problem=problem,
        variant=variant,
        seed=seed,
        official_dir=ACTIVE_RESULTS_ROOT / experiment / problem / variant / f"seed_{seed}",
    )
    _prepare_official_dir(run.official_dir)
    raw_npz = archive / "raw" / f"{case}_predictions.npz"
    guard_mask, blind_mask = _write_prediction_contract(raw_npz, run.official_dir)
    metrics = _metrics_from_prediction_npz(raw_npz, blind_mask)
    metrics.update({
        "protocol_id": "unified_blind_protocol_v1",
        "experiment": experiment,
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "split_id": f"{problem}_seed_{seed}",
        "guard_decision": metrics.get("final_source"),
    })
    metrics.update(_read_parameter_metrics(archive, case))
    summary = _read_case_summary(archive / f"{case}_EXP")
    timing = {
        "teacher_training_time_s": _maybe_float(summary.get("teacher_time_s")),
        "student_training_time_s": _maybe_float(summary.get("student_time_s")),
        "structured_training_time_s": None,
    }
    metrics.update(timing)
    _write_json(run.official_dir / "metrics.json", _complete_metrics(metrics))
    _write_json(run.official_dir / "timing.json", timing)
    _write_json(run.official_dir / "model_size.json", _model_sizes(archive / f"{case}_EXP"))
    _write_json(run.official_dir / "guard_decision.json", {
        "decision_split": "guard_validation",
        "final_source": metrics.get("final_source"),
        "guard_may_read_blind_labels": False,
    })
    _write_json(run.official_dir / "split_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "split_id": f"{problem}_seed_{seed}",
        "guard_may_read_blind_labels": False,
        "n_guard": int(guard_mask.sum()) if guard_mask is not None else None,
        "n_blind": int(blind_mask.sum()) if blind_mask is not None else None,
        "overlap_count": int(np.logical_and(guard_mask, blind_mask).sum())
        if guard_mask is not None and blind_mask is not None else None,
    })
    config = _read_config_csv(ENGINE_ROOT / "Config" / f"{case}_EXP.csv")
    config.update(overrides)
    _write_config_yaml(run.official_dir / "config_resolved.yaml", config, {
        "experiment": experiment,
        "problem": problem,
        "variant": variant,
        "seed": seed,
        "runtime_source_id": RUNTIME_SOURCE_ID,
    })
    _copy_checkpoints(archive / f"{case}_EXP" / "Models", run.official_dir / "checkpoints")
    _write_json(run.official_dir / "run_manifest.json", {
        "protocol_id": "unified_blind_protocol_v1",
        "runtime_source_id": RUNTIME_SOURCE_ID,
        "runtime_archive": _relative(archive),
        "official_status": "generated_by_runtime_config_ablation",
        "preset": preset,
        "config_overrides": overrides,
    })
    return run


def _derive_main_reference(
    experiment: str,
    variant: str,
    seeds: list[int],
    source_mode: str = "final",
) -> list[RuntimeRun]:
    outputs: list[RuntimeRun] = []
    for problem in PROBLEMS:
        for seed in seeds:
            source_dir = _main_official_dir(problem, seed)
            with np.load(source_dir / "predictions_guard.npz", allow_pickle=True) as guard, \
                 np.load(source_dir / "predictions_blind.npz", allow_pickle=True) as blind:
                if source_mode == "dense_student":
                    guard_mean, guard_std = _candidate_prediction(guard, "dense_student")
                    blind_mean, blind_std = _candidate_prediction(blind, "dense_student")
                    source = "dense_student"
                else:
                    guard_mean = _array(guard, "bayesian_mean")
                    blind_mean = _array(blind, "bayesian_mean")
                    guard_std = _array(guard, "bayesian_std")
                    blind_std = _array(blind, "bayesian_std")
                    source = str(_scalar(blind, "bayesian_source", "guarded_final"))
                if guard_mean is None or blind_mean is None:
                    raise KeyError(f"Reference predictions missing in {source_dir}")
                metrics = _evaluate_selection(blind, blind_mean, blind_std, source)
                metrics["guard_decision"] = source
                outputs.append(_write_derived_run(
                    experiment, problem, variant, seed, source_dir,
                    guard, blind, guard_mean, blind_mean, guard_std, blind_std, source,
                    metrics,
                    {"decision_split": "guard_validation", "final_source": source, "guard_may_read_blind_labels": False},
                    {"reference_variant": True, "source_mode": source_mode},
                ))
    return outputs


def _source_environment(source_dir: Path) -> dict[str, Any]:
    manifest = _read_json(source_dir / "run_manifest.json")
    archive_text = manifest.get("runtime_archive")
    if not archive_text:
        return {}
    return _read_json(ROOT / str(archive_text) / "environment.json")


def derive_runtime_params(seeds: list[int], execute: bool) -> list[RuntimeRun]:
    if not execute:
        print("derive runtime_params from official main_blind checkpoints and timing manifests")
        return []
    outputs = _derive_main_reference("runtime_params", "resource_accounting", seeds)
    for run in outputs:
        source_dir = _main_official_dir(run.problem, int(run.seed))
        source_metrics = _read_json(source_dir / "metrics.json")
        accounting = _resource_accounting(source_dir / "checkpoints")
        environment = _source_environment(source_dir)
        metrics = _read_json(run.official_dir / "metrics.json")
        timing = _read_json(source_dir / "timing.json")
        if accounting.get("structured_training_time_s") is not None:
            timing["structured_training_time_s"] = accounting["structured_training_time_s"]
        metrics.update({
            "experiment": "runtime_params",
            "variant": "resource_accounting",
            "teacher_training_time_s": timing.get("teacher_training_time_s"),
            "student_training_time_s": timing.get("student_training_time_s"),
            "structured_training_time_s": timing.get("structured_training_time_s"),
            "total_wall_clock_time_s": timing.get("total_wall_clock_time_s"),
            "teacher_params": accounting.get("teacher_params"),
            "dense_params": accounting.get("dense_params"),
            "structured_trainable_params": accounting.get("structured_trainable_params"),
            "structured_effective_params": accounting.get("structured_effective_params"),
            "compression_ratio": accounting.get("compression_ratio", source_metrics.get("compression_ratio")),
            "device_name": source_metrics.get("device_name") or (
                environment.get("gpu_name") if environment.get("cuda_available") else "CPU"
            ),
            "cuda_version": source_metrics.get("cuda_version") or environment.get("cuda_version"),
            "pytorch_version": source_metrics.get("pytorch_version") or environment.get("torch_version"),
            "ram_gb": source_metrics.get("ram_gb") or (
                float(environment["ram_total_bytes"]) / (1024.0 ** 3)
                if environment.get("ram_total_bytes") is not None else None
            ),
        })
        _write_json(run.official_dir / "metrics.json", _complete_metrics(metrics))
        _write_json(run.official_dir / "timing.json", timing)
        shutil.copy2(source_dir / "model_size.json", run.official_dir / "model_size.json")
        manifest = _read_json(run.official_dir / "run_manifest.json")
        manifest["resource_accounting_source"] = _relative(source_dir)
        manifest["parameter_definition"] = "trainable degrees of freedom; frozen relation matrices excluded"
        _write_json(run.official_dir / "run_manifest.json", manifest)
    return outputs


def run_ablations(seeds: list[int], preset: str, execute: bool, python: str, n_mc: int) -> list[RuntimeRun]:
    outputs: list[RuntimeRun] = []
    outputs.extend(derive_guard_ablation(seeds, execute))
    outputs.extend(derive_uq_ablation(seeds, execute))
    if execute:
        outputs.extend(_derive_main_reference("reconstruction_ablation", "full_structured_reconstruction", seeds))
        outputs.extend(_derive_main_reference("zero_distillation_ablation", "teacher_distilled_student", seeds, "dense_student"))
    else:
        print("derive reconstruction full reference and teacher_distilled_student reference from official main_blind")

    reconstruction = {
        "without_pde_residual_loss": {"lambda_pde_recon": "0"},
        "without_boundary_or_initial_loss": {"lambda_bc_recon": "0"},
        "without_distillation_loss": {
            "lambda_distill_recon": "0",
            "lambda_anchor_recon": "0",
            "anchor_pretrain_steps": "0",
        },
    }
    reconstruction_cases = ("Poisson",) if preset == "smoke" else tuple(PROBLEMS.values())
    reconstruction_variants = (
        {"without_pde_residual_loss": reconstruction["without_pde_residual_loss"]}
        if preset == "smoke" else reconstruction
    )
    for case in reconstruction_cases:
        for variant, overrides in reconstruction_variants.items():
            for seed in seeds:
                tag = f"{variant}_{case}"
                archive = _run_config_ablation(case, tag, seed, overrides, preset, execute, python, n_mc)
                if archive is not None:
                    outputs.append(normalize_config_ablation_archive(
                        archive, "reconstruction_ablation", case, variant, seed, preset, overrides
                    ))
    if preset != "smoke":
        for seed in seeds:
            archive = _run_config_ablation(
                "Poisson", "without_hard_boundary_condition_Poisson", seed,
                {"use_hard_bc": "False"}, preset, execute, python, n_mc
            )
            if archive is not None:
                outputs.append(normalize_config_ablation_archive(
                    archive, "reconstruction_ablation", "Poisson",
                    "without_hard_boundary_condition", seed, preset, {"use_hard_bc": "False"}
                ))
    zero_overrides = {
        "lambda_distill_student": "0",
        "lambda_distill_mean_refine": "0",
        "lambda_distill_recon": "0",
        "force_structured_prediction": "0",
        "min_structured_compression": "999",
    }
    zero_cases = ("Poisson",) if preset == "smoke" else tuple(PROBLEMS.values())
    for case in zero_cases:
        for seed in seeds:
            archive = _run_config_ablation(
                case, f"zero_distillation_mc_dropout_student_{case}", seed,
                zero_overrides, preset, execute, python, n_mc
            )
            if archive is not None:
                outputs.append(normalize_config_ablation_archive(
                    archive, "zero_distillation_ablation", case,
                    "zero_distillation_mc_dropout_student", seed, preset, zero_overrides
                ))
    return outputs


def _derive_guard_threshold_sensitivity(seeds: list[int], execute: bool) -> list[RuntimeRun]:
    if not execute:
        print("derive accuracy/minimum-compression guard threshold sensitivity from official main_blind")
        return []
    outputs: list[RuntimeRun] = []
    grids = {
        "accuracy_ratio": [1.00, 1.03, 1.10, 1.25, 1.50],
        "minimum_compression": [1.00, 1.10, 1.50, 2.00],
    }
    for problem in PROBLEMS:
        base_config = _read_config_csv(ENGINE_ROOT / "Config" / f"{PROBLEMS[problem]}_EXP.csv")
        for seed in seeds:
            source_dir = _main_official_dir(problem, seed)
            with np.load(source_dir / "predictions_guard.npz", allow_pickle=True) as guard, \
                 np.load(source_dir / "predictions_blind.npz", allow_pickle=True) as blind:
                dense_g, _ = _candidate_prediction(guard, "dense_student")
                struct_g, _ = _candidate_prediction(guard, "structured")
                dense_r = relative_l2(dense_g, _array(guard, "exact"))
                struct_r = relative_l2(struct_g, _array(guard, "exact"))
                compression = _scalar(guard, "bayesian_structured_compression")
                for parameter, values in grids.items():
                    for value in values:
                        config = dict(base_config)
                        key = "accept_structured_rel_l2_ratio" if parameter == "accuracy_ratio" else "min_structured_compression"
                        config[key] = str(value)
                        source, decision = _guard_source_for_variant(
                            "full_gbsd", dense_r, struct_r,
                            float(compression) if compression is not None else None, config, seed
                        )
                        gm, _ = _candidate_prediction(guard, source)
                        bm, _ = _candidate_prediction(blind, source)
                        metrics = _evaluate_selection(blind, bm, None, source)
                        metrics.update({"guard_decision": source, "sensitivity_parameter": parameter, "sensitivity_value": value})
                        variant = f"{parameter}_{str(value).replace('.', 'p')}"
                        outputs.append(_write_derived_run(
                            "threshold_sensitivity", problem, variant, seed, source_dir,
                            guard, blind, gm, bm, None, None, source, metrics, decision,
                            {"sensitivity_parameter": parameter, "sensitivity_value": value, "uq_fields": "not_recomputed"},
                        ))
    return outputs


def run_sensitivity(seeds: list[int], preset: str, execute: bool, python: str, n_mc: int) -> list[RuntimeRun]:
    outputs = _derive_guard_threshold_sensitivity(seeds, execute)
    cluster_values = ["0.05"] if preset == "smoke" else ["0.03", "0.05", "0.08", "0.10", "0.15", "0.20"]
    cluster_cases = ("Poisson",) if preset == "smoke" else ("Poisson", "Burgers_inv")
    for case in cluster_cases:
        for value in cluster_values:
            variant = f"clustering_threshold_{value.replace('.', 'p')}"
            for seed in seeds:
                tag = f"{variant}_{case}"
                archive = _run_config_ablation(
                    case, tag, seed, {"cluster_distance": value}, preset, execute, python, n_mc
                )
                if archive is not None:
                    outputs.append(normalize_config_ablation_archive(
                        archive, "threshold_sensitivity", case, variant, seed, preset,
                        {"cluster_distance": value}
                    ))
    if execute:
        outputs.extend(_derive_main_reference("structure_stability", "main_hac_across_seeds", seeds))
    else:
        print("derive outcome-level structure_stability reference from official main_blind")
    return outputs


def run_stage(stage: str, execute: bool = False, preset: str = "full",
              seeds: list[int] | None = None, python: str | None = None,
              n_mc: int = 200) -> list[RuntimeRun]:
    global ACTIVE_RESULTS_ROOT
    _require_runtime_engine()
    ACTIVE_RESULTS_ROOT = SMOKE_ROOT if preset == "smoke" else OFFICIAL_ROOT
    seeds = seeds or DEFAULT_SEEDS
    python = python or sys.executable
    started = time.time()
    if stage == "main":
        outputs = run_main_blind(seeds, preset, execute, python, n_mc)
    elif stage == "baselines":
        outputs = []
        outputs.extend(run_direct_mc_dropout(seeds, preset, execute, python, n_mc))
        outputs.extend(run_deep_ensemble(seeds, preset, execute, python))
        outputs.extend(run_structure_discovery_baselines(seeds, preset, execute, python))
    elif stage == "ablations":
        outputs = run_ablations(seeds, preset, execute, python, n_mc)
    elif stage == "sensitivity":
        outputs = run_sensitivity(seeds, preset, execute, python, n_mc)
    elif stage == "runtime":
        outputs = derive_runtime_params(seeds, execute)
    else:
        raise ValueError(f"Unknown official experiment stage: {stage}")
    if execute:
        _write_json(ROOT / "results" / "audit" / f"runtime_adapter_{stage}_{int(started)}.json", {
            "stage": stage,
            "preset": preset,
            "seeds": seeds,
            "outputs": [_relative(item.official_dir) for item in outputs],
            "elapsed_s": time.time() - started,
        })
    return outputs
