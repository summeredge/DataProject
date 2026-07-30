from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.lag import LAG_PEAK_QUALITY_COLUMNS, build_lag_peak_quality
from chem_ts_corr.screening import final_ranked_features


def _lag_rows(variable: str, scores: dict[int, float | tuple[float, float]]) -> list[dict]:
    rows = []
    for lag, score in scores.items():
        pearson, spearman = score if isinstance(score, tuple) else (score, score)
        rows.append(
            {
                "variable": variable,
                "lag": lag,
                "abs_pearson": pearson,
                "abs_spearman": spearman,
                "n": 100,
            }
        )
    return rows


def _quality(variable: str, scores: dict[int, float], max_lag: int = 5) -> pd.Series:
    result = build_lag_peak_quality(pd.DataFrame(_lag_rows(variable, scores)), max_lag)
    return result.iloc[0]


def test_empty_input_has_fixed_schema():
    result = build_lag_peak_quality(pd.DataFrame(), max_lag=5)

    assert result.empty
    assert result.columns.tolist() == LAG_PEAK_QUALITY_COLUMNS


def test_clear_single_peak_has_positive_shape_quality():
    peak = _quality("sharp", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})

    assert peak["peak_prominence"] > 0
    assert peak["local_sharpness"] > 0
    assert peak["shape_quality"] > 0
    assert peak["lag_quality"] == pytest.approx(peak["shape_quality"])
    assert bool(peak["lag_boundary_flag"]) is False


def test_broad_peak_scores_below_clear_peak():
    sharp = _quality("sharp", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})
    broad = _quality("broad", {-3: 0.3, -1: 0.75, 0: 0.8, 1: 0.75})

    assert sharp["lag_quality"] > broad["lag_quality"]


def test_competing_peak_scores_below_clear_peak():
    sharp = _quality("sharp", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})
    multi = _quality("multi", {-3: 0.78, -1: 0.2, 0: 0.8, 1: 0.2})

    assert sharp["lag_quality"] > multi["lag_quality"]
    assert multi["peak_prominence"] < sharp["peak_prominence"]


def test_boundary_peak_receives_proportional_penalty():
    interior = _quality("interior", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2}, max_lag=5)
    boundary = _quality("boundary", {2: 0.3, 4: 0.2, 5: 0.8}, max_lag=5)

    assert boundary["lag_quality"] == pytest.approx(boundary["shape_quality"] * 0.75)
    assert boundary["lag_quality"] < interior["lag_quality"]
    assert bool(boundary["lag_boundary_flag"]) is True


def test_proportional_scaling_preserves_quality():
    high = _quality("high", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})
    low = _quality("low", {-3: 0.15, -1: 0.1, 0: 0.4, 1: 0.1})

    for column in ["peak_prominence", "local_sharpness", "shape_quality", "lag_quality"]:
        assert high[column] == pytest.approx(low[column])


def test_equal_independent_peaks_have_zero_quality():
    peak = _quality("equal", {-3: 0.8, -1: 0.2, 0: 0.8, 1: 0.2})

    assert peak["peak_prominence"] == 0
    assert peak["shape_quality"] == 0
    assert peak["lag_quality"] == 0


def test_missing_neighbors_produces_zero_local_sharpness():
    peak = _quality("isolated", {0: 0.8, 3: 0.3})

    assert pd.isna(peak["nearby_score_mean"])
    assert peak["peak_sharpness"] == 0
    assert peak["local_sharpness"] == 0


def test_missing_independent_peak_is_not_treated_as_perfect():
    peak = _quality("local_only", {-1: 0.2, 0: 0.8, 1: 0.2})

    assert pd.isna(peak["second_peak_score"])
    assert peak["peak_prominence"] == 0
    assert peak["shape_quality"] == 0


def test_nan_scores_are_ignored_and_all_nan_variable_is_omitted():
    rows = _lag_rows("valid", {-3: (0.3, np.nan), 0: (np.nan, 0.8), 1: (np.nan, np.nan)})
    rows += _lag_rows("invalid", {0: (np.nan, np.nan), 1: (np.nan, np.nan)})

    result = build_lag_peak_quality(pd.DataFrame(rows), max_lag=5)

    assert result["variable"].tolist() == ["valid"]
    assert result.loc[0, "best_lag"] == 0
    assert result.loc[0, "best_score"] == pytest.approx(0.8)


def test_all_quality_metrics_stay_in_unit_interval():
    rows = []
    rows += _lag_rows("sharp", {-5: 0.1, -3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})
    rows += _lag_rows("broad", {-3: 0.3, -1: 0.79, 0: 0.8, 1: 0.79})
    rows += _lag_rows("multi", {-3: 0.8, -1: 0.1, 0: 0.8, 1: 0.1})

    result = build_lag_peak_quality(pd.DataFrame(rows), max_lag=5)

    for column in ["peak_prominence", "local_sharpness", "shape_quality", "lag_quality"]:
        assert result[column].between(0, 1).all()


def test_input_is_not_modified():
    lag_scores = pd.DataFrame(_lag_rows("x", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2}))
    before = lag_scores.copy(deep=True)

    build_lag_peak_quality(lag_scores, max_lag=5)

    pd.testing.assert_frame_equal(lag_scores, before)


def test_old_lag_quality_formula_is_absent():
    source = Path("chem_ts_corr/lag.py").read_text(encoding="utf-8")

    assert "max(0.0, -peak_sharpness)" not in source
    assert "max(0, -peak_sharpness)" not in source
    assert "lag_quality = best_score -" not in source
    assert "0.70 * peak_prominence" in source
    assert "shape_quality -" not in source
    assert "abs(bl)" not in source
    assert "0.75 if boundary" in source


def test_final_ranked_features_accepts_new_peak_quality_output():
    lag_peak = build_lag_peak_quality(
        pd.DataFrame(_lag_rows("x", {-3: 0.3, -1: 0.2, 0: 0.8, 1: 0.2})),
        max_lag=5,
    )
    ranked = pd.DataFrame(
        [{"variable": "x", "lag": 0, "direction": "同步变化", "score": 0.8}]
    )
    variable_only = pd.DataFrame({"variable": ["x"]})

    result = final_ranked_features(
        ranked,
        residual=variable_only,
        stability=variable_only,
        model_lift=variable_only,
        risks=variable_only,
        lag_peak_quality=lag_peak,
        rolling_corr_scores=variable_only,
    )

    assert {"variable", "lag_quality", "lag_boundary_flag"}.issubset(lag_peak.columns)
    assert result.loc[0, "variable"] == "x"
    assert result.loc[0, "lag_quality"] == pytest.approx(lag_peak.loc[0, "lag_quality"])
