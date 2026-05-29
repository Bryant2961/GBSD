"""Official result schema for GBSD paper summaries."""

from __future__ import annotations

SUMMARY_FILES = {
    "all_raw": "summary_all_raw.csv",
    "all_mean_std": "summary_all_mean_std.csv",
    "main_blind": "summary_main_blind.csv",
    "baselines": "summary_baselines.csv",
    "guard_ablation": "summary_guard_ablation.csv",
    "uq_ablation": "summary_uq_ablation.csv",
    "reconstruction_ablation": "summary_reconstruction_ablation.csv",
    "zero_distillation_ablation": "summary_zero_distillation_ablation.csv",
    "runtime_params": "summary_runtime_params.csv",
    "threshold_sensitivity": "summary_threshold_sensitivity.csv",
    "structure_stability": "summary_structure_stability.csv",
}

REQUIRED_METRIC_FIELDS = [
    "protocol_id",
    "experiment",
    "problem",
    "variant",
    "seed",
    "split_id",
    "dense_rL2",
    "structured_rL2",
    "final_rL2",
    "dense_params",
    "structured_trainable_params",
    "structured_effective_params",
    "compression_ratio",
    "coverage95",
    "nll",
    "aiw",
    "error_std_corr",
    "final_source",
    "guard_decision",
    "teacher_training_time_s",
    "student_training_time_s",
    "structured_training_time_s",
    "nu_true",
    "nu_pred",
    "nu_rel_error",
]

REQUIRED_RUN_FILES = [
    "config_resolved.yaml",
    "split_manifest.json",
    "predictions_blind.npz",
    "predictions_guard.npz",
    "guard_decision.json",
    "metrics.json",
    "timing.json",
    "model_size.json",
    "run_manifest.json",
]

FORBIDDEN_OFFICIAL_LABEL_FRAGMENTS = [
    "3.34",
    "3.35",
    "3.36",
    "3.37",
    "3.38",
    "v3_",
    "v3.",
    "legacy",
]
