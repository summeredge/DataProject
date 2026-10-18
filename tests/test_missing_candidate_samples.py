from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr.preprocess import preprocess_frame, transform_frame
from chem_ts_corr.service import _innovation_evidence, analyze_numeric_frame


@pytest.mark.parametrize("lag", [-3, 3])
@pytest.mark.parametrize("mode", ["raw", "diff", "detrend", "detrend_diff", "lowpass", "lowpass_diff", "lowpass_detrend"])
def test_unrelated_missing_column_preserves_pair_samples_and_lag(lag, mode):
    rng = np.random.default_rng(123)
    index = pd.date_range("2026-01-01", periods=240, freq="min")
    target = pd.Series(rng.normal(size=len(index)), index=index)
    frame = pd.DataFrame({"target": target, "x": target.shift(-lag)})
    augmented = frame.assign(unrelated=rng.normal(size=len(index)))
    augmented.loc[index[20:180], "unrelated"] = np.nan

    results = []
    for source in (frame, augmented):
        cleaned = preprocess_frame(source, "target", None, 0.0, max_interpolate_gap_points=1)
        assert cleaned.index.equals(index)
        transformed = transform_frame(cleaned, mode, 12, max_interpolate_gap_points=1)
        scores = compute_lag_scores(transformed, "target", 5)
        pair_scores = scores.loc[scores.variable.eq("x")].reset_index(drop=True)
        best = summarize_best_lags(pair_scores)
        innovation = _innovation_evidence(transformed, "target", 5, best, mode)
        results.append((pair_scores, innovation))
    fields = ["variable", "lag", "n", "pearson", "spearman"]
    pd.testing.assert_frame_equal(results[0][0][fields], results[1][0][fields])
    pd.testing.assert_frame_equal(results[0][1], results[1][1])
    assert int(summarize_best_lags(results[1][0]).iloc[0].lag) == lag


@pytest.mark.parametrize("resample_rule", [None, "2min"])
def test_selected_regime_validity_keeps_candidate_and_source_history(tmp_path, resample_rule):
    rng = np.random.default_rng(82)
    index = pd.date_range("2026-01-01", periods=360, freq="min")
    target = pd.Series(rng.normal(size=len(index)), index=index)
    frame = pd.DataFrame({"target": target, "x": target.shift(-2), "load": 0.0})
    frame.loc[index[240:], "load"] = 1.0
    frame.loc[index[:235], "x"] = np.nan
    config = AnalysisConfig(input_path=Path("unused"), time_column="time", target="target",
                            output_dir=tmp_path, max_lag=4, resample_rule=resample_rule,
                            segment_column="load", segment_mode="custom", segment_min=1.0,
                            segment_max=1.0, skip_rolling_corr=True)
    tables = analyze_numeric_frame(frame, config)
    row = tables.ranked_features.set_index("variable").loc["x"]
    assert int(row.lag) == (2 if resample_rule is None else 1)
    expected_n = 120 if resample_rule is None else 60
    scores = tables.lag_scores
    assert int(scores.loc[scores.variable.eq("x") & scores.lag.eq(row.lag), "n"].iloc[0]) == expected_n


def test_missing_target_timestamp_remains_available_as_lag_source():
    index = pd.date_range("2026-01-01", periods=30, freq="min")
    frame = pd.DataFrame({"target": np.arange(30, dtype=float), "x": np.arange(30, dtype=float)}, index=index)
    frame.loc[index[15:23], "target"] = np.nan
    cleaned = preprocess_frame(frame, "target", None, 0.0, max_interpolate_gap_points=1)
    assert cleaned.index.equals(index)
    scores = compute_lag_scores(cleaned, "target", 3)
    assert int(scores.loc[scores.lag.eq(3), "n"].iloc[0]) == 20


def test_regime_minimum_rows_checked_after_finite_interpolation(tmp_path):
    index = pd.date_range("2026-01-01", periods=60, freq="min")
    frame = pd.DataFrame({"target": np.sin(np.arange(60)), "x": np.cos(np.arange(60)),
                          "load": np.nan}, index=index)
    frame.loc[index[:8], "load"] = 1.0
    frame.loc[index[-10:], "load"] = 1.0
    cleaned = preprocess_frame(frame, "target", "2min", 0.7, protected_columns=["load"],
                               segment_column="load", segment_mode="custom",
                               segment_min=1.0, segment_max=1.0)
    assert cleaned["load"].notna().sum() == 14
    config = AnalysisConfig(input_path=Path("unused"), time_column="time", target="target",
                            output_dir=tmp_path, max_lag=3, resample_rule="2min",
                            segment_column="load", segment_mode="custom", segment_min=1.0,
                            segment_max=1.0, skip_rolling_corr=True)
    assert not analyze_numeric_frame(frame, config).ranked_features.empty
