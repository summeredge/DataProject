from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags


def _fake_scores(frame, target, max_lag, lag_values=None):
    lags = list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values)
    rows = []
    for variable in frame.columns:
        if variable == target:
            continue
        for lag in lags:
            score = 1.0 / (1.0 + abs(lag))
            rows.append(
                {
                    "variable": variable,
                    "lag": lag,
                    "pearson": score,
                    "pearson_p": 0.01,
                    "pearson_r2": score * score,
                    "spearman": score,
                    "spearman_p": 0.01,
                    "spearman_r2": score * score,
                    "n": len(frame),
                    "abs_pearson": score,
                    "abs_spearman": score,
                    "lag_boundary_flag": abs(lag) == max_lag,
                    "pearson_q": 0.01,
                    "spearman_q": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_compute_lag_scores_accepts_explicit_lag_values():
    frame = pd.DataFrame({"target": np.arange(20), "x": np.arange(20)})

    scores = compute_lag_scores(frame, "target", max_lag=10, lag_values=[-2, 0, 3])

    assert scores["lag"].tolist() == [-2, 0, 3]
    assert {"pearson", "spearman", "pearson_q", "spearman_q"}.issubset(scores.columns)


def test_compute_lag_scores_default_still_scans_full_range():
    frame = pd.DataFrame({"target": np.arange(20), "x": np.arange(20)})

    scores = compute_lag_scores(frame, "target", max_lag=10)

    assert scores["lag"].tolist() == list(range(-10, 11))


def test_max_lag_360_local_review_scans_37_points(monkeypatch):
    calls = []

    def counted(frame, target, max_lag, lag_values=None):
        calls.append(list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values))
        return _fake_scores(frame, target, max_lag, lag_values)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    pair = pd.DataFrame({"target": np.arange(400), "x": np.arange(400)})

    best = screening._best_lag_review_scores(pair, "target", max_lag=360, primary_best_lag=0)

    assert int(best.iloc[0]["lag"]) == 0
    assert calls == [list(range(-18, 19))]


@pytest.mark.parametrize("primary_best_lag", [None, np.nan, "bad", 1.5, 11, 10, -10])
def test_invalid_or_global_boundary_primary_lag_uses_full_scan(monkeypatch, primary_best_lag):
    calls = []

    def counted(frame, target, max_lag, lag_values=None):
        calls.append(list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values))
        return _fake_scores(frame, target, max_lag, lag_values)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    pair = pd.DataFrame({"target": np.arange(30), "x": np.arange(30)})

    screening._best_lag_review_scores(pair, "target", max_lag=10, primary_best_lag=primary_best_lag)

    assert calls == [list(range(-10, 11))]


def test_local_boundary_best_falls_back_to_full_scan(monkeypatch):
    calls = []

    def boundary_best(frame, target, max_lag, lag_values=None):
        lags = list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values)
        calls.append(lags)
        scores = _fake_scores(frame, target, max_lag, lags)
        winning_lag = 4 if len(lags) == 21 else max(lags)
        scores.loc[scores["lag"].eq(winning_lag), ["pearson", "spearman", "abs_pearson", "abs_spearman"]] = 2.0
        return scores

    monkeypatch.setattr(screening, "compute_lag_scores", boundary_best)
    pair = pd.DataFrame({"target": np.arange(30), "x": np.arange(30)})

    best = screening._best_lag_review_scores(pair, "target", max_lag=10, primary_best_lag=0)

    assert calls == [list(range(-3, 4)), list(range(-10, 11))]
    assert int(best.iloc[0]["lag"]) == 4


def test_local_best_at_global_boundary_does_not_rescan(monkeypatch):
    calls = []

    def global_boundary_best(frame, target, max_lag, lag_values=None):
        lags = list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values)
        calls.append(lags)
        scores = _fake_scores(frame, target, max_lag, lags)
        scores.loc[scores["lag"].eq(max_lag), ["pearson", "spearman", "abs_pearson", "abs_spearman"]] = 2.0
        return scores

    monkeypatch.setattr(screening, "compute_lag_scores", global_boundary_best)
    pair = pd.DataFrame({"target": np.arange(30), "x": np.arange(30)})

    best = screening._best_lag_review_scores(pair, "target", max_lag=10, primary_best_lag=8)

    assert calls == [list(range(5, 11))]
    assert int(best.iloc[0]["lag"]) == 10


def test_nonpositive_max_lag_reviews_only_zero(monkeypatch):
    calls = []

    def counted(frame, target, max_lag, lag_values=None):
        calls.append([0] if lag_values is None else list(lag_values))
        return _fake_scores(frame, target, 0, [0])

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    pair = pd.DataFrame({"target": np.arange(10), "x": np.arange(10)})

    screening._best_lag_review_scores(pair, "target", max_lag=0, primary_best_lag=None)

    assert calls == [[0]]


def test_local_review_matches_full_scan_when_peak_is_inside_window():
    rng = np.random.default_rng(20260716)
    target = pd.Series(rng.normal(size=300), name="target")
    variable = target.shift(-6).fillna(0).rename("x")
    pair = pd.concat([target, variable], axis=1)
    full = summarize_best_lags(compute_lag_scores(pair, "target", max_lag=20)).iloc[0]

    local = screening._best_lag_review_scores(
        pair,
        "target",
        max_lag=20,
        primary_best_lag=int(full["lag"]),
    ).iloc[0]

    assert int(local["lag"]) == int(full["lag"])
    assert local["direction"] == full["direction"]
    assert float(local["score"]) == pytest.approx(float(full["score"]), abs=1e-12)
    assert float(local[local["method"]]) == pytest.approx(float(full[full["method"]]), abs=1e-12)


def test_residual_and_each_regime_use_local_window(monkeypatch):
    rng = np.random.default_rng(7)
    n = 1200
    target = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "target": target,
            "cap": np.repeat([0.0, 1.0, 2.0], n // 3),
            "x": target + rng.normal(scale=1e-6, size=n),
        }
    )
    original = screening.compute_lag_scores
    calls = []

    def counted(pair, target_name, max_lag, lag_values=None):
        calls.append((pair.columns[-1], list(range(-max_lag, max_lag + 1)) if lag_values is None else list(lag_values)))
        return original(pair, target_name, max_lag, lag_values=lag_values)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)

    screening.residual_corr_scores(frame, "target", ["cap"], 360, best_lags={"x": 0})
    screening.regime_scores(frame, "target", "cap", 360, best_lags={"x": 0, "cap": 0})

    x_calls = [lags for variable, lags in calls if variable == "x"]
    assert len(x_calls) == 4
    assert all(lags == list(range(-18, 19)) for lags in x_calls)


def test_residual_and_regime_outputs_match_full_scan_for_in_window_peak():
    rng = np.random.default_rng(19)
    n = 600
    target = pd.Series(rng.normal(size=n), name="target")
    frame = pd.DataFrame(
        {
            "target": target,
            "cap": np.repeat([0.0, 1.0, 2.0], n // 3),
            "x": target.shift(-4).fillna(0),
        }
    )
    primary = summarize_best_lags(
        compute_lag_scores(frame[["target", "x"]], "target", max_lag=12)
    ).iloc[0]
    best_lags = {"x": int(primary["lag"])}

    residual_full = screening.residual_corr_scores(frame, "target", ["cap"], 12)
    residual_local = screening.residual_corr_scores(
        frame, "target", ["cap"], 12, best_lags=best_lags
    )
    regime_full, stability_full = screening.regime_scores(frame, "target", "cap", 12)
    regime_local, stability_local = screening.regime_scores(
        frame, "target", "cap", 12, best_lags=best_lags
    )

    pd.testing.assert_frame_equal(residual_local, residual_full)
    pd.testing.assert_frame_equal(regime_local, regime_full)
    pd.testing.assert_frame_equal(stability_local, stability_full)
