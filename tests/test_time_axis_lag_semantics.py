from __future__ import annotations

import numpy as np
import pandas as pd

from chem_ts_corr.causality import _GrangerDiagnostics, run_granger_tests
from chem_ts_corr.conditional_granger import run_conditional_granger_tests
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr.modeling import build_lag_features
from chem_ts_corr.preprocess import preprocess_frame, segment_by_load
from chem_ts_corr.screening import regime_scores
from chem_ts_corr.time_axis import PHYSICAL_GAP_STARTS_ATTR, lagged_series, sample_period_ns
from chem_ts_corr.xgb_validation import build_xgb_feature_sets


def _best_lag(frame: pd.DataFrame, max_lag: int = 5) -> int:
    result = summarize_best_lags(compute_lag_scores(frame, "target", max_lag))
    return int(result.loc[result["variable"].eq("x"), "lag"].iloc[0])


def _lagged_frame(lag: int, rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    target = pd.Series(rng.normal(size=rows))
    x = target.shift(-lag)
    return pd.DataFrame(
        {"target": target.to_numpy(), "x": x.to_numpy()},
        index=pd.date_range("2026-01-01", periods=rows, freq="5min"),
    ).dropna()


def test_regular_datetime_axis_preserves_true_lag_three():
    assert _best_lag(_lagged_frame(3)) == 3


def test_missing_timestamps_do_not_compress_true_lag_to_one():
    frame = _lagged_frame(3)
    frame = frame.drop(frame.index[50:80:3])

    assert _best_lag(frame) == 3


def test_alternating_regimes_keep_physical_lag_three():
    frame = _lagged_frame(3)
    frame["load"] = np.resize([0.0, 1.0, 2.0], len(frame))

    scores, _ = regime_scores(frame, "target", "load", max_lag=4)

    x_scores = scores.loc[scores["variable"].eq("x")]
    assert set(x_scores["lag"]) == {3}
    assert np.allclose(x_scores["score"], 1.0)


def test_regime_mask_applies_to_target_time_not_source_time():
    frame = _lagged_frame(1)
    frame["load"] = np.resize([0.0, 2.0], len(frame))

    scores, _ = regime_scores(frame, "target", "load", max_lag=2)

    low = scores.loc[(scores["variable"] == "x") & (scores["regime"] == "low")].iloc[0]
    assert int(low["lag"]) == 1
    assert float(low["score"]) == 1.0


def test_lagged_series_does_not_cross_datetime_breakpoint():
    index = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10",
         "2026-01-01 01:00", "2026-01-01 01:05"]
    )
    series = pd.Series(range(len(index)), index=index, dtype=float)

    shifted = lagged_series(series, index, 1, period_ns=5 * 60 * 1_000_000_000)

    assert pd.isna(shifted.loc["2026-01-01 01:00"])
    assert shifted.loc["2026-01-01 01:05"] == 3.0


def test_lagged_series_does_not_cross_forced_physical_breakpoint():
    index = pd.date_range("2026-01-01 08:00", periods=5, freq="5min")
    series = pd.Series(range(len(index)), index=index, dtype=float)
    series.attrs[PHYSICAL_GAP_STARTS_ATTR] = (index[3],)

    shifted = lagged_series(series, index, 1, period_ns=5 * 60 * 1_000_000_000)

    assert pd.isna(shifted.loc[index[3]])
    assert shifted.loc[index[4]] == 3.0


def test_compute_lag_scores_does_not_count_forced_break_pair():
    index = pd.date_range("2026-01-01 08:00", periods=12, freq="5min")
    frame = pd.DataFrame({"target": range(12), "x": range(12)}, index=index)
    frame.attrs[PHYSICAL_GAP_STARTS_ATTR] = (index[6],)

    scores = compute_lag_scores(frame, "target", 1, lag_values=[1])

    assert int(scores.iloc[0]["n"]) == 10


def test_preprocess_dropna_preserves_original_sample_period():
    frame = _lagged_frame(1, rows=80)
    frame.loc[frame.index[19:22], "x"] = np.nan

    cleaned = preprocess_frame(
        frame,
        "target",
        resample_rule=None,
        min_valid_ratio=0.7,
        max_interpolate_gap_points=1,
    )

    assert frame.index[20] not in cleaned.index
    assert sample_period_ns(cleaned) == 5 * 60 * 1_000_000_000
    shifted = lagged_series(cleaned["x"], cleaned.index, 1, period_ns=sample_period_ns(cleaned))
    assert pd.isna(shifted.loc[frame.index[22]])


def test_datetime_lag_direction_matches_existing_row_semantics():
    frame = _lagged_frame(2)
    scores = compute_lag_scores(frame, "target", 2, lag_values=[-2, 0, 2])

    best = scores.loc[scores[["abs_pearson", "abs_spearman"]].max(axis=1).idxmax()]
    assert int(best["lag"]) == 2
    assert set(scores["lag"]) == {-2, 0, 2}


def test_resample_rule_sets_lag_unit_to_resampled_period():
    frame = _lagged_frame(2, rows=120)
    resampled = preprocess_frame(frame, "target", "10min", min_valid_ratio=0.7)

    assert sample_period_ns(resampled) == 10 * 60 * 1_000_000_000


def test_segment_by_load_retains_source_axis_period():
    frame = _lagged_frame(1, rows=120)
    frame["load"] = np.resize([0.0, 1.0, 2.0], len(frame))

    segmented = segment_by_load(frame, "load", "low", None, None)

    assert sample_period_ns(segmented) == 5 * 60 * 1_000_000_000


def test_model_and_xgb_features_do_not_bridge_datetime_gap():
    complete = _lagged_frame(1, rows=120)
    gap_time = complete.index[40]
    after_gap = complete.index[41]
    frame = complete.drop(index=gap_time)

    model_features, _ = build_lag_features(
        frame,
        "target",
        max_lag=1,
        candidate_variables=["x"],
        max_features=2,
        best_lags={"x": 1},
    )
    xgb_features = build_xgb_feature_sets(
        frame,
        "target",
        pd.DataFrame([{"variable": "x", "screening_lag": 1}]),
        max_lag=1,
        baseline_lags=[1],
        candidate_lag_radius=0,
    ).features

    assert after_gap not in model_features.index
    assert after_gap not in xgb_features.index


def test_model_and_xgb_features_do_not_bridge_forced_breakpoint():
    frame = _lagged_frame(1, rows=120)
    forced_start = frame.index[41]
    frame.attrs[PHYSICAL_GAP_STARTS_ATTR] = (forced_start,)

    model_features, _ = build_lag_features(
        frame,
        "target",
        max_lag=1,
        candidate_variables=["x"],
        max_features=2,
        best_lags={"x": 1},
    )
    xgb_features = build_xgb_feature_sets(
        frame,
        "target",
        pd.DataFrame([{"variable": "x", "screening_lag": 1}]),
        max_lag=1,
        baseline_lags=[1],
        candidate_lag_radius=0,
    ).features

    assert forced_start not in model_features.index
    assert forced_start not in xgb_features.index


def test_granger_uses_gap_safe_alignment_for_datetime_breaks():
    frame = _lagged_frame(1, rows=240)
    frame["x"] += np.random.default_rng(7).normal(scale=0.05, size=len(frame))
    frame = frame.drop(index=frame.index[80])
    diagnostics = _GrangerDiagnostics()

    result = run_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=2,
        diagnostics=diagnostics,
    )

    assert result.loc[0, "status"] == "ok"
    assert diagnostics.fallback_count == 2


def test_granger_uses_time_aware_alignment_for_forced_breaks():
    frame = _lagged_frame(1, rows=240)
    frame["x"] += np.random.default_rng(17).normal(scale=0.05, size=len(frame))
    frame.attrs[PHYSICAL_GAP_STARTS_ATTR] = (frame.index[80],)
    diagnostics = _GrangerDiagnostics()

    result = run_granger_tests(frame, target="target", variables=["x"], maxlag=2, diagnostics=diagnostics)

    assert result.loc[0, "status"] == "ok"
    assert diagnostics.fallback_count == 2


def test_conditional_granger_applies_segment_mask_only_at_target_time():
    frame = _lagged_frame(1, rows=240)
    frame["x"] += np.random.default_rng(9).normal(scale=0.05, size=len(frame))
    target_mask = pd.Series(
        np.resize([True, False], len(frame)),
        index=frame.index,
    )

    result = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=1,
        min_rows=40,
        candidate_lags={"x": [1]},
        baseline_maxlag=1,
        target_mask=target_mask,
    ).iloc[0]

    assert result["status"] == "ok"
    assert int(result["best_lag"]) == 1
    assert int(result["n_rows"]) == int(target_mask.sum()) - 1


def test_conditional_granger_excludes_forced_break_lag_rows():
    frame = _lagged_frame(1, rows=240)
    forced_start = frame.index[80]
    frame.attrs[PHYSICAL_GAP_STARTS_ATTR] = (forced_start,)

    result = run_conditional_granger_tests(
        frame,
        target="target",
        variables=["x"],
        maxlag=1,
        min_rows=40,
        candidate_lags={"x": [1]},
        baseline_maxlag=1,
    ).iloc[0]

    assert result["status"] == "ok"
    assert int(result["n_rows"]) == len(frame) - 2
