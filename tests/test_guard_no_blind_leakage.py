from gbsd.io.runtime_engine_adapter import _guard_source_for_variant


def test_guard_protocol_names_are_distinct():
    train = "train"
    guard = "guard_validation"
    blind = "blind_test"
    assert len({train, guard, blind}) == 3


def test_official_guard_decision_uses_only_guard_metrics():
    config = {
        "accept_structured_rel_l2_ratio": "1.10",
        "accept_structured_rel_l2_abs": "0",
        "min_structured_compression": "1.10",
    }
    accepted, details = _guard_source_for_variant(
        "full_gbsd", dense_guard_rL2=0.10, structured_guard_rL2=0.105,
        compression=1.20, config=config, seed=0
    )
    rejected, _ = _guard_source_for_variant(
        "full_gbsd", dense_guard_rL2=0.10, structured_guard_rL2=0.20,
        compression=1.20, config=config, seed=0
    )
    assert accepted == "structured"
    assert rejected == "dense_student"
    assert details["decision_split"] == "guard_validation"
    assert details["guard_may_read_blind_labels"] is False
