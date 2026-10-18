from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.service import analyze_numeric_frame


@pytest.mark.parametrize("seed", range(10))
def test_short_noise_wide_search_does_not_select_tiny_overlap(seed, tmp_path):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"target": rng.normal(size=40), "noise": rng.normal(size=40)},
                         index=pd.date_range("2026-01-01", periods=40, freq="min"))
    config = AnalysisConfig(input_path=Path("unused"), time_column="time", target="target",
                            output_dir=tmp_path, max_lag=35, skip_rolling_corr=True)
    tables = analyze_numeric_frame(frame, config)
    best = tables.ranked_features.iloc[0]
    peak = tables.lag_peak_quality.iloc[0]
    assert best.n >= 20
    assert best.candidate_grade not in {"A", "B"}
    assert int(best.lag) == int(peak.best_lag)
    assert -20 <= peak.near_peak_lag_min <= peak.near_peak_lag_max <= 20


@pytest.mark.parametrize("lag", [-170, 170])
def test_well_sampled_long_lag_survives_small_relative_overlap(lag):
    rng = np.random.default_rng(43)
    index = pd.date_range("2026-01-01", periods=240, freq="min")
    target = pd.Series(rng.normal(size=len(index)), index=index)
    frame = pd.DataFrame({"target": target, "x": target.shift(-lag)})
    scores = compute_lag_scores(frame, "target", 180)
    best = summarize_best_lags(scores).iloc[0]
    assert best.lag == lag
    assert best.n == 70
    assert build_lag_peak_quality(scores, 180).iloc[0].best_lag == lag


def test_near_peak_and_direction_ignore_unsupported_opposite_peak():
    index = pd.date_range("2026-01-01", periods=80, freq="min")
    target = pd.Series(np.random.default_rng(4).normal(size=80), index=index)
    frame = pd.DataFrame({"target": target, "x": target.shift(-3)})
    scores = compute_lag_scores(frame, "target", 5)
    opposite = scores.lag.eq(-3)
    scores.loc[opposite, ["pearson", "spearman", "abs_pearson", "abs_spearman"]] = 1.0
    scores.loc[opposite, "n"] = 5
    assert summarize_best_lags(scores).iloc[0].lag == 3
    peak = build_lag_peak_quality(scores, 5).iloc[0]
    assert peak.best_lag == 3
    assert peak.near_peak_lag_min == peak.near_peak_lag_max == 3
    assert peak.temporal_direction_status == "variable_leads_supported"


@pytest.mark.parametrize("masked", [False, True])
def test_insufficient_pairs_stay_uncomputable(masked):
    rng = np.random.default_rng(65)
    frame = pd.DataFrame({"target": rng.normal(size=40), "x": rng.normal(size=40)},
                         index=pd.date_range("2026-01-01", periods=40, freq="min"))
    mask = None
    if masked:
        mask = pd.Series(np.arange(40) < 4, index=frame.index)
    else:
        frame.loc[frame.index[4:], "x"] = np.nan
    scores = compute_lag_scores(frame, "target", 35, target_mask=mask)
    assert scores.pearson.isna().all()
    assert scores.spearman.isna().all()
    assert summarize_best_lags(scores).empty
    assert build_lag_peak_quality(scores, 35).empty


def test_support_is_per_variable_and_does_not_remove_real_zero():
    rows = []
    for variable, n in [("short", 8), ("long", 100)]:
        rows.append({"variable": variable, "lag": 0, "n": n,
                     "abs_pearson": 0.0, "abs_spearman": 0.0})
    peak = build_lag_peak_quality(pd.DataFrame(rows), 3)
    assert set(peak.variable) == {"short", "long"}
    assert peak.best_score.eq(0.0).all()
