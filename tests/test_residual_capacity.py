import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr.screening import residual_corr_scores, regime_scores, risk_flags


def test_residual_fit_uses_target_regime_and_applies_coefficients_to_full_timeline():
    index = pd.date_range("2026-01-01", periods=20, freq="5min")
    capacity = pd.Series(np.arange(1, 21, dtype=float), index=index, name="cap")
    fit_mask = pd.Series([True] * 10 + [False] * 10, index=index)
    target = pd.Series(
        np.where(fit_mask, 5.0 + 2.0 * capacity, 5.0 + 10.0 * capacity),
        index=index,
        name="target",
    )

    residual, method, _, used_columns = screening._residualize(
        target, capacity.to_frame(), fit_mask=fit_mask
    )

    assert method == "ols"
    assert used_columns == ["cap"]
    assert residual.loc[fit_mask].abs().max() == pytest.approx(0.0, abs=1e-10)
    assert residual.loc[~fit_mask].to_numpy() == pytest.approx(
        (8.0 * capacity.loc[~fit_mask]).to_numpy()
    )


def test_residual_demean_fallback_uses_only_target_regime_mean():
    index = pd.RangeIndex(8)
    target = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0], index=index)
    constant_control = pd.DataFrame({"cap": 1.0}, index=index)
    fit_mask = pd.Series([True, True, True, True, False, False, False, False], index=index)

    residual, method, _, used_columns = screening._residualize(
        target, constant_control, fit_mask=fit_mask
    )

    assert method == "demean"
    assert used_columns == []
    pd.testing.assert_series_equal(residual, target - 2.5)


def test_residual_corr_and_risk_flags_common_capacity_driver():
    n = 180
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    capacity = np.linspace(0, 5, n)
    # target and candidate mostly share same capacity driver
    target = 2.0 * capacity + 0.02 * np.sin(np.linspace(0, 20, n))
    candidate = 1.8 * capacity + 0.02 * np.cos(np.linspace(0, 20, n))

    frame = pd.DataFrame(
        {
            "target": target,
            "cap": capacity,
            "cand": candidate,
        },
        index=idx,
    )

    lag_scores = compute_lag_scores(frame[["target", "cand"]], target="target", max_lag=4)
    ranked = summarize_best_lags(lag_scores)
    residual = residual_corr_scores(frame, target="target", capacity_columns=["cap"], max_lag=4)
    _, stability = regime_scores(frame[["target", "cand", "cap"]], target="target", capacity_column="cap", max_lag=4)

    diag = pd.DataFrame(
        [
            {
                "variable": "cand",
                "missing_rate": 0.0,
                "saturation_ratio": 0.0,
                "abnormal_jump_ratio": 0.0,
            }
        ]
    )
    risks = risk_flags(
        ranked=ranked,
        residual=residual,
        stability=stability,
        diag=diag,
        roles={"cand": "PV", "target": "Y", "cap": "CAPACITY"},
        control_columns=["cap"],
    )

    assert not residual.empty
    assert {"residual_method", "condition_number", "used_control_columns"}.issubset(residual.columns)
    assert not risks.empty
    row = risks.loc[risks["variable"] == "cand"].iloc[0]
    assert bool(row["common_capacity_driver_flag"]) is True
    assert "common_capacity_driver" in str(row["risk_flags"])
