from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr.preprocess import preprocess_frame, standardize_frame, transform_frame
from chem_ts_corr.time_axis import PHYSICAL_GAP_STARTS_ATTR, preserve_sample_period, sample_period_ns
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


def _historical_only_driver():
    rng = np.random.default_rng(128)
    index = pd.date_range("2026-01-01", periods=120, freq="min")
    source = rng.normal(size=40)
    frame = pd.DataFrame({"target": rng.normal(size=120), "x": np.nan,
                          "load": (np.arange(120) >= 80).astype(float)}, index=index)
    frame.loc[index[:40], "x"] = 1000 + rng.normal(size=40)
    frame.loc[index[40:80], "x"] = source
    frame.loc[index[80:], "target"] = source + rng.normal(scale=0.001, size=40)
    return frame


@pytest.mark.parametrize("lag", [20, 40])
@pytest.mark.parametrize("search_factor", [1, 2])
def test_pre_switch_only_driver_is_admitted_and_scaled_from_pairable_history(tmp_path, lag, search_factor):
    frame = _historical_only_driver()
    frame.loc[frame.index[80:80 + lag], "target"] = frame.loc[frame.index[80 - lag:80], "x"].to_numpy()
    max_lag = lag * search_factor
    if search_factor == 2:
        frame.loc[frame.index[:80 - lag], "x"] = np.nan
    selection = frame.load.eq(1)
    cleaned = preprocess_frame(frame, "target", None, 0.7, protected_columns=["load"],
                               segment_column="load", segment_mode="custom",
                               segment_min=1.0, segment_max=1.0, max_lag=max_lag)
    assert "x" in cleaned
    assert cleaned.index.equals(frame.index)
    assert cleaned.loc[selection, "x"].isna().all()
    scaled = standardize_frame(cleaned, fit_mask=selection, max_lag=max_lag)
    history = scaled.loc[frame.index[80 - lag:80], "x"]
    assert history.notna().all()
    assert history.mean() == pytest.approx(0.0, abs=1e-12)
    assert history.std(ddof=0) == pytest.approx(1.0)
    assert scaled.loc[selection, "x"].isna().all()
    pd.testing.assert_series_equal(scaled.target, standardize_frame(cleaned, fit_mask=selection).target)

    config = AnalysisConfig(input_path=Path("unused"), time_column="time", target="target",
                            output_dir=tmp_path, max_lag=max_lag, segment_column="load",
                            segment_mode="custom", segment_min=1.0, segment_max=1.0,
                            skip_rolling_corr=True)
    row = analyze_numeric_frame(frame, config).ranked_features.set_index("variable").loc["x"]
    assert int(row.lag) == lag
    assert int(row.n) == lag


@pytest.mark.parametrize("blocked_by", ["range", "physical_break", "excluded_window"])
def test_unpairable_history_cannot_supply_admission_or_scaling(blocked_by, tmp_path):
    frame = _historical_only_driver()
    max_lag = 5 if blocked_by == "range" else 40
    if blocked_by == "range":
        frame.loc[frame.index[75:80], "x"] = np.nan
    elif blocked_by == "physical_break":
        frame.attrs[PHYSICAL_GAP_STARTS_ATTR] = (frame.index[80],)
    elif blocked_by == "excluded_window":
        from chem_ts_corr.pipeline import load_analysis_source_frame
        input_path = tmp_path / "history.csv"
        frame.rename_axis("time").reset_index().to_csv(input_path, index=False)
        config = AnalysisConfig(input_path=input_path, time_column="time", target="target",
                                output_dir=tmp_path / "run",
                                exclude_windows=[{"start": str(frame.index[78]), "end": str(frame.index[79])}])
        frame = load_analysis_source_frame(config)
    selection = frame.load.eq(1)
    cleaned = preprocess_frame(frame, "target", None, 0.7, protected_columns=["load"],
                               segment_column="load", segment_mode="custom",
                               segment_min=1.0, segment_max=1.0, max_lag=max_lag)
    assert "x" not in cleaned
    scaled = standardize_frame(frame, fit_mask=selection, max_lag=max_lag)
    assert scaled.x.isna().all()


def test_resampled_history_uses_new_period_even_with_original_period_metadata():
    index = pd.date_range("2026-01-01", periods=60, freq="min")
    frame = pd.DataFrame({"target": np.sin(np.arange(60)), "x": np.nan,
                          "load": (np.arange(60) >= 40).astype(float)}, index=index)
    frame.loc[index[34:36], "x"] = [-1.0, -2.0]
    frame.loc[index[38:40], "x"] = [1.0, 2.0]
    results = []
    for stored_period in (False, True):
        source = frame.copy()
        if stored_period:
            preserve_sample_period(source, 60_000_000_000)
        cleaned = preprocess_frame(source, "target", "2min", 0.7, protected_columns=["load"],
                                   segment_column="load", segment_mode="custom",
                                   segment_min=1.0, segment_max=1.0, max_lag=1)
        assert "x" in cleaned
        assert sample_period_ns(cleaned) == 120_000_000_000
        results.append(cleaned)
    pd.testing.assert_frame_equal(results[0], results[1])


def test_historical_only_driver_is_visible_to_existing_candidate_coverage(tmp_path):
    from dataclasses import replace
    from tests.synthetic_cases.evaluate import run_case
    from tests.synthetic_cases.four_layer_cases import CASES

    base = CASES["true_lagged_driver"](n=120, lag=40)
    case = replace(base, frame=_historical_only_driver().rename(columns={"x": "x_driver"}),
                   spurious_variables=frozenset(),
                   variable_types={"true_driver": frozenset({"x_driver"}), "regime": frozenset({"load"})},
                   metadata={**base.metadata, "max_lag": 80, "segment_column": "load",
                             "segment_mode": "custom", "segment_min": 1.0, "segment_max": 1.0,
                             "skip_rolling_corr": True})
    ranked = run_case(case, tmp_path)
    row = ranked.set_index("variable").loc["x_driver"]
    assert int(row.lag) == 40
    assert int(row.n) == 40
    coverage = ranked.attrs["candidate_coverage"]
    assert coverage["recall"]["ranked_features.csv"] == 1.0
    assert all("x_driver" in values for values in coverage["sets"].values())
