from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.report import write_outputs
from chem_ts_corr.screening import residual_corr_scores, risk_flags


def _risks(residual: pd.DataFrame) -> pd.DataFrame:
    return risk_flags(
        ranked=pd.DataFrame([{"variable": "x", "score": 0.1, "lag": 1}]),
        residual=residual,
        stability=pd.DataFrame(),
        diag=pd.DataFrame(),
        roles={"x": "PV"},
        control_columns=[],
    )


@pytest.mark.parametrize(
    ("residual", "expected"),
    [
        (pd.DataFrame({"variable": ["x"], "control_condition_number": [1.2e9]}), True),
        (pd.DataFrame({"variable": ["x"], "condition_number": [1.2e9]}), True),
        (pd.DataFrame({"variable": ["x"], "control_condition_number": [10.0], "condition_number": [1.2e9]}), False),
        (pd.DataFrame({"variable": ["x", "x"], "control_condition_number": [np.nan, 1.2e9]}), False),
    ],
)
def test_residual_condition_number_schema_priority(residual, expected):
    row = _risks(residual).iloc[0]
    assert bool(row["residual_collinearity_flag"]) is expected
    assert ("residual_collinearity" in row["risk_flags"]) is expected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "invalid"])
def test_nonfinite_residual_condition_numbers_do_not_create_risk(value):
    row = _risks(pd.DataFrame({"variable": ["x"], "control_condition_number": [value]})).iloc[0]
    assert bool(row["residual_collinearity_flag"]) is False
    assert "residual_collinearity" not in row["risk_flags"]


def test_residual_output_uses_only_canonical_condition_number_column(tmp_path):
    index = pd.date_range("2026-01-01", periods=40, freq="min")
    frame = pd.DataFrame(
        {"target": np.arange(40, dtype=float), "load": np.arange(40, dtype=float), "x": np.arange(40, dtype=float) * 2},
        index=index,
    )
    residual = residual_corr_scores(frame, "target", ["load"], 3)
    assert "control_condition_number" in residual.columns
    assert "condition_number" not in residual.columns

    write_outputs(
        tmp_path, "target", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
        residual_corr_scores=residual,
    )
    written = pd.read_csv(tmp_path / "residual_corr_scores.csv", encoding="utf-8-sig")
    assert "control_condition_number" in written.columns
    assert "condition_number" not in written.columns
