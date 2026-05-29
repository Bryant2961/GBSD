from gbsd.reporting.schema import SUMMARY_FILES


def test_summary_mapping_contains_required_paper_tables():
    required = {
        "main_blind",
        "baselines",
        "guard_ablation",
        "uq_ablation",
        "reconstruction_ablation",
        "zero_distillation_ablation",
        "runtime_params",
        "threshold_sensitivity",
        "structure_stability",
    }
    assert required.issubset(SUMMARY_FILES)
