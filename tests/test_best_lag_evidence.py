import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr import screening


ROLLING_COLUMNS = [
    "variable",
    "best_lag",
    "best_score",
    "rolling_corr_median",
    "rolling_abs_corr_median",
    "rolling_corr_iqr",
    "rolling_sign_consistency",
    "valid_window_count",
    "rolling_stability",
]


def _legacy_rolling_corr_scores(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    max_lag: int,
    window: int | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    rows = []
    window_size = max(12, int(window or min(len(frame), max(24, max_lag * 4))))
    min_points = max(6, int(min_periods or window_size // 2))
    for variable in variables:
        pair = frame[[target, variable]].dropna()
        if len(pair) < max(window_size, max_lag + 10):
            continue
        best = summarize_best_lags(compute_lag_scores(pair, target, max_lag))
        if best.empty:
            continue
        best_row = best.iloc[0]
        best_lag = int(best_row["lag"])
        rolling = (
            pair[variable]
            .shift(best_lag)
            .rolling(window=window_size, min_periods=min_points)
            .corr(pair[target])
            .dropna()
        )
        if rolling.empty:
            continue
        sign_consistency = (
            rolling.apply(lambda value: 1 if value >= 0 else -1)
            .value_counts(normalize=True)
            .max()
        )
        iqr = float(rolling.quantile(0.75) - rolling.quantile(0.25))
        abs_median = float(rolling.abs().median())
        stability = max(
            0.0,
            min(
                1.0,
                abs_median * float(sign_consistency) * (1.0 - min(1.0, iqr)),
            ),
        )
        rows.append(
            {
                "variable": variable,
                "best_lag": best_lag,
                "best_score": float(best_row.get("score", 0.0) or 0.0),
                "rolling_corr_median": float(rolling.median()),
                "rolling_abs_corr_median": abs_median,
                "rolling_corr_iqr": iqr,
                "rolling_sign_consistency": float(sign_consistency),
                "valid_window_count": int(len(rolling)),
                "rolling_stability": stability,
            }
        )
    return pd.DataFrame(rows, columns=ROLLING_COLUMNS)


def _lagged_frame(lags: dict[str, int], n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
    target = pd.Series(rng.normal(size=n))
    data = {"target": target.to_numpy()}
    for variable, lag in lags.items():
        shifted = target.shift(-lag)
        missing = shifted.isna()
        shifted.loc[missing] = rng.normal(size=int(missing.sum()))
        data[variable] = shifted.to_numpy() + rng.normal(scale=1e-4, size=n)
    return pd.DataFrame(data, index=pd.date_range("2026-01-01", periods=n, freq="min"))


def _ranked_for(frame: pd.DataFrame, variables: list[str], max_lag: int) -> pd.DataFrame:
    scores = [
        compute_lag_scores(frame[["target", variable]].dropna(), "target", max_lag)
        for variable in variables
    ]
    return summarize_best_lags(pd.concat(scores, ignore_index=True))


def test_rolling_uses_complete_evidence_without_lag_scan(monkeypatch):
    frame = _lagged_frame({"x1": 3, "x2": -2})
    variables = ["x1", "x2"]
    expected = _legacy_rolling_corr_scores(frame, "target", variables, max_lag=5)
    ranked = _ranked_for(frame, variables, max_lag=5)
    evidence, diagnostics = screening.prepare_best_lag_evidence(
        frame, "target", variables, 5, ranked=ranked, ranked_source_frame=frame
    )

    monkeypatch.setattr(
        screening,
        "compute_lag_scores",
        lambda *args, **kwargs: pytest.fail("complete evidence must avoid a lag scan"),
    )
    actual = screening.rolling_corr_scores(
        frame, "target", variables, 5, best_lag_evidence=evidence
    )

    assert diagnostics == {
        "reused_evidence_count": 2,
        "recomputed_evidence_count": 0,
        "invalid_evidence_count": 0,
    }
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_rolling_scans_only_variable_missing_evidence(monkeypatch):
    frame = _lagged_frame({"x1": 2, "x2": -3})
    variables = ["x1", "x2"]
    expected = _legacy_rolling_corr_scores(frame, "target", variables, max_lag=5)
    ranked = _ranked_for(frame, ["x1"], max_lag=5)
    evidence, _ = screening.prepare_best_lag_evidence(
        frame, "target", ["x1"], 5, ranked=ranked, ranked_source_frame=frame
    )
    original = screening.compute_lag_scores
    calls = []

    def counted(pair, target, max_lag):
        calls.append(pair.columns[-1])
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    actual = screening.rolling_corr_scores(
        frame, "target", variables, 5, best_lag_evidence=evidence
    )

    assert calls == ["x2"]
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_scanned_no_result_evidence_prevents_second_full_scan(monkeypatch):
    n = 80
    frame = pd.DataFrame(
        {"target": np.arange(n, dtype=float), "x": np.ones(n)},
        index=pd.date_range("2026-01-01", periods=n, freq="min"),
    )
    original = screening.compute_lag_scores
    calls = []

    def counted(pair, target, max_lag):
        calls.append(pair.columns[-1])
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    evidence, diagnostics = screening.prepare_best_lag_evidence(
        frame, "target", ["x"], 5, allow_ranked_reuse=False
    )
    actual = screening.rolling_corr_scores(
        frame, "target", ["x"], 5, best_lag_evidence=evidence
    )

    assert calls == ["x"]
    assert evidence["x"]["status"] == "scanned_no_result"
    assert diagnostics["recomputed_evidence_count"] == 1
    assert actual.empty


def test_stale_scanned_no_result_evidence_keeps_single_fallback(monkeypatch):
    n = 80
    frame = pd.DataFrame(
        {"target": np.arange(n, dtype=float), "x": np.ones(n)},
        index=pd.date_range("2026-01-01", periods=n, freq="min"),
    )
    evidence, _ = screening.prepare_best_lag_evidence(
        frame, "target", ["x"], 5, allow_ranked_reuse=False
    )
    evidence["x"]["pair_alignment_key"] = "stale-key"
    original = screening.compute_lag_scores
    calls = 0

    def counted(pair, target, max_lag):
        nonlocal calls
        calls += 1
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    actual = screening.rolling_corr_scores(
        frame, "target", ["x"], 5, best_lag_evidence=evidence
    )

    assert calls == 1
    assert actual.empty


def test_prepare_recomputes_ranked_evidence_for_internal_missing_pair(monkeypatch):
    frame = _lagged_frame({"x": 3})
    ranked_source_frame = frame.copy(deep=True)
    ranked = _ranked_for(ranked_source_frame, ["x"], max_lag=5)
    frame.loc[frame.index[70], "x"] = np.nan
    pair = frame[["target", "x"]].dropna()
    expected_best = summarize_best_lags(compute_lag_scores(pair, "target", 5)).iloc[0]
    original = screening.compute_lag_scores
    calls = []

    def counted(current_pair, target, max_lag):
        calls.append(current_pair.index.copy())
        return original(current_pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    evidence, diagnostics = screening.prepare_best_lag_evidence(
        frame,
        "target",
        ["x"],
        5,
        ranked=ranked,
        ranked_source_frame=ranked_source_frame,
    )

    assert len(calls) == 1
    assert calls[0].equals(pair.index)
    assert evidence["x"]["source"] == "recomputed"
    assert evidence["x"]["best_lag"] == int(expected_best["lag"])
    assert evidence["x"]["best_score"] == float(expected_best["score"])
    assert diagnostics == {
        "reused_evidence_count": 0,
        "recomputed_evidence_count": 1,
        "invalid_evidence_count": 1,
    }


def test_alignment_key_uses_full_order_not_only_row_count():
    frame = _lagged_frame({"x": 1}, n=80)
    first = frame.iloc[:-1][["target", "x"]]
    second = frame.iloc[1:][["target", "x"]]

    assert len(first) == len(second)
    assert screening.pair_alignment_key(first) != screening.pair_alignment_key(second)


def test_ranked_evidence_without_source_frame_fails_closed(monkeypatch):
    frame = _lagged_frame({"x": 2})
    ranked = _ranked_for(frame, ["x"], max_lag=5)
    original = screening.compute_lag_scores
    calls = []

    def counted(pair, target, max_lag):
        calls.append(pair.columns[-1])
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    evidence, diagnostics = screening.prepare_best_lag_evidence(
        frame, "target", ["x"], 5, ranked=ranked, ranked_source_frame=None
    )
    screening.rolling_corr_scores(
        frame, "target", ["x"], 5, best_lag_evidence=evidence
    )

    assert calls == ["x"]
    assert evidence["x"]["source"] == "recomputed"
    assert diagnostics == {
        "reused_evidence_count": 0,
        "recomputed_evidence_count": 1,
        "invalid_evidence_count": 1,
    }


def test_same_length_different_source_index_recomputes_once(monkeypatch):
    ranked_source_frame = _lagged_frame({"x": -2})
    frame = ranked_source_frame.copy(deep=True)
    reordered_index = frame.index.to_list()
    reordered_index[70], reordered_index[71] = reordered_index[71], reordered_index[70]
    frame.index = pd.DatetimeIndex(reordered_index)
    ranked = _ranked_for(ranked_source_frame, ["x"], max_lag=5)
    original = screening.compute_lag_scores
    calls = []

    def counted(pair, target, max_lag):
        calls.append(pair.index.copy())
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    evidence, diagnostics = screening.prepare_best_lag_evidence(
        frame,
        "target",
        ["x"],
        5,
        ranked=ranked,
        ranked_source_frame=ranked_source_frame,
    )

    assert len(frame) == len(ranked_source_frame)
    assert frame.index[0] == ranked_source_frame.index[0]
    assert frame.index[-1] == ranked_source_frame.index[-1]
    assert len(calls) == 1
    assert calls[0].equals(frame.index)
    assert evidence["x"]["source"] == "recomputed"
    assert diagnostics["invalid_evidence_count"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("best_lag", 6),
        ("best_score", np.nan),
        ("max_lag", 4),
        ("pair_alignment_key", "stale-key"),
    ],
)
def test_invalid_evidence_falls_back_once(monkeypatch, field, value):
    frame = _lagged_frame({"x": 2})
    ranked = _ranked_for(frame, ["x"], max_lag=5)
    evidence, _ = screening.prepare_best_lag_evidence(
        frame, "target", ["x"], 5, ranked=ranked, ranked_source_frame=frame
    )
    evidence["x"][field] = value
    original = screening.compute_lag_scores
    calls = 0

    def counted(pair, target, max_lag):
        nonlocal calls
        calls += 1
        return original(pair, target, max_lag)

    monkeypatch.setattr(screening, "compute_lag_scores", counted)
    screening.rolling_corr_scores(
        frame, "target", ["x"], 5, best_lag_evidence=evidence
    )

    assert calls == 1


@pytest.mark.parametrize("expected_lag", [-3, 0, 5])
def test_evidence_preserves_negative_zero_and_boundary_lags(expected_lag):
    frame = _lagged_frame({"x": expected_lag})
    expected = _legacy_rolling_corr_scores(frame, "target", ["x"], max_lag=5)
    ranked = _ranked_for(frame, ["x"], max_lag=5)
    evidence, _ = screening.prepare_best_lag_evidence(
        frame, "target", ["x"], 5, ranked=ranked, ranked_source_frame=frame
    )

    actual = screening.rolling_corr_scores(
        frame, "target", ["x"], 5, best_lag_evidence=evidence
    )

    assert int(actual.iloc[0]["best_lag"]) == expected_lag
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
