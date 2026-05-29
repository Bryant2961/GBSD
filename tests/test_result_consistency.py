import pandas as pd

from gbsd.reporting.collect import validate_required_columns
from gbsd.reporting.schema import REQUIRED_METRIC_FIELDS


def test_required_schema_accepts_complete_dataframe():
    df = pd.DataFrame([{field: None for field in REQUIRED_METRIC_FIELDS}])
    assert validate_required_columns(df) == []


def test_required_schema_reports_missing_fields():
    df = pd.DataFrame([{"protocol_id": "unified_blind_protocol_v1"}])
    missing = validate_required_columns(df)
    assert "dense_rL2" in missing
    assert "final_rL2" in missing

