from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f

import chem_ts_corr.causality as causality
from scripts import benchmark_granger
from chem_ts_corr.causality import (
    _GrangerDiagnostics,
    _append_r_row,
    _build_augmented_matrix,
    _delete_r_column,
    _downdate_qr_state,
    _fast_granger_lag_statistics,
    _initial_qr_state,
    _lag_column_indices,
    _new_observation_row,
    _restricted_mask_key,
    run_granger_tests,
)


def _reference_lag_statistics(pair: pd.DataFrame, target: str, variable: str, maxlag: int):
    clean = pair[[target, variable]].dropna()
    target_values = clean[target].to_numpy(dtype=float)
    variable_values = clean[variable].to_numpy(dtype=float)
    output = {}
    for lag in range(1, maxlag + 1):
        n = len(target_values)
        response = target_values[lag:]
        target_lags = np.empty((n - lag, lag), dtype=float)
        candidate_lags = np.empty((n - lag, lag), dtype=float)
        for offset in range(1, lag + 1):
            target_lags[:, offset - 1] = target_values[lag - offset : n - offset]
            candidate_lags[:, offset - 1] = variable_values[lag - offset : n - offset]

        restricted = np.empty((n - lag, lag + 1), dtype=float)
        restricted[:, 0] = 1.0
        restricted[:, 1:] = target_lags
        unrestricted = np.empty((n - lag, 2 * lag + 1), dtype=float)
        unrestricted[:, 0] = 1.0
        unrestricted[:, 1 : lag + 1] = target_lags
        unrestricted[:, lag + 1 :] = candidate_lags

        restricted_coef, _, restricted_rank, _ = np.linalg.lstsq(
            restricted, response, rcond=None
        )
        unrestricted_coef, _, unrestricted_rank, _ = np.linalg.lstsq(
            unrestricted, response, rcond=None
        )
        restricted_residual = response - restricted @ restricted_coef
        unrestricted_residual = response - unrestricted @ unrestricted_coef
        restricted_ssr = float(np.dot(restricted_residual, restricted_residual))
        unrestricted_ssr = float(np.dot(unrestricted_residual, unrestricted_residual))
        df_num = int(unrestricted_rank - restricted_rank)
        df_den = int(len(response) - unrestricted_rank)
        centered = response - np.mean(response)
        target_scale = max(float(np.dot(centered, centered)), 1.0)

        statistic = None
        p_value = None
        skipped = True
        if (
            df_num > 0
            and df_den > 0
            and np.isfinite(restricted_ssr)
            and np.isfinite(unrestricted_ssr)
            and unrestricted_ssr > 0
            and unrestricted_ssr > np.finfo(float).eps * target_scale
        ):
            delta = max(0.0, restricted_ssr - unrestricted_ssr)
            candidate_f = (delta / df_num) / (unrestricted_ssr / df_den)
            candidate_p = float(f.sf(candidate_f, df_num, df_den))
            if np.isfinite(candidate_f) and np.isfinite(candidate_p):
                statistic = float(candidate_f)
                p_value = candidate_p
                skipped = False

        output[lag] = {
            "nobs": len(response),
            "restricted_ssr": restricted_ssr,
            "unrestricted_ssr": unrestricted_ssr,
            "restricted_rank": int(restricted_rank),
            "unrestricted_rank": int(unrestricted_rank),
            "df_num": df_num,
            "df_den": df_den,
            "f_statistic": statistic,
            "p_value": p_value,
            "skipped": skipped,
        }
    return output


def _assert_lag_equivalence(
    pair: pd.DataFrame,
    maxlag: int,
    *,
    rtol=1e-7,
    atol=1e-9,
    diagnostics: _GrangerDiagnostics | None = None,
):
    expected = _reference_lag_statistics(pair, "Y", "X", maxlag)
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
    )
    _assert_statistics_match_reference(actual, expected, rtol=rtol, atol=atol)
    return actual, expected


def _assert_statistics_match_reference(actual, expected, *, rtol=1e-7, atol=1e-9):
    assert list(actual) == list(expected)
    for lag in expected:
        observed = actual[lag]
        reference = expected[lag]
        assert observed.nobs == reference["nobs"]
        assert observed.restricted_rank == reference["restricted_rank"]
        assert observed.unrestricted_rank == reference["unrestricted_rank"]
        assert observed.df_num == reference["df_num"]
        assert observed.df_den == reference["df_den"]
        assert observed.restricted_ssr == pytest.approx(
            reference["restricted_ssr"], rel=rtol, abs=atol
        )
        assert observed.unrestricted_ssr == pytest.approx(
            reference["unrestricted_ssr"], rel=rtol, abs=atol
        )
        assert (observed.f_statistic is None) == reference["skipped"]
        if not reference["skipped"]:
            assert observed.f_statistic == pytest.approx(
                reference["f_statistic"], rel=rtol, abs=atol
            )
            assert observed.p_value == pytest.approx(reference["p_value"], rel=rtol, abs=atol)


def _ordinary_pair(seed: int = 1, rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Y": rng.normal(size=rows), "X": rng.normal(size=rows)})


def _high_collinearity_pair() -> pd.DataFrame:
    target = np.random.default_rng(15).normal(size=180)
    candidate = target + np.random.default_rng(16).normal(scale=1e-10, size=180)
    return pd.DataFrame({"Y": target, "X": candidate})


def _near_rank_deficient_pair() -> pd.DataFrame:
    rows = 180
    target = np.random.default_rng(18).normal(size=rows)
    candidate = np.linspace(-1.0, 1.0, rows)
    candidate += np.random.default_rng(19).normal(scale=1e-10, size=rows)
    return pd.DataFrame({"Y": target, "X": candidate})


def _reference_predictive_contribution(pair: pd.DataFrame, lag: int) -> float:
    data = pd.DataFrame(
        {
            "target": pair["Y"],
            "candidate": pair["X"].shift(lag),
            "target_lag_1": pair["Y"].shift(1),
        }
    ).dropna()
    response = data["target"].to_numpy(dtype=float)

    def rmse(columns):
        values = data[columns].to_numpy(dtype=float)
        matrix = np.column_stack([np.ones(len(values)), values])
        coef = np.linalg.lstsq(matrix, response, rcond=None)[0]
        return float(np.sqrt(np.mean((response - matrix @ coef) ** 2)))

    base = rmse(["target_lag_1"])
    full = rmse(["target_lag_1", "candidate"])
    return max(0.0, (base - full) / base) if base > 0 else 0.0


@pytest.mark.parametrize(
    "pair,maxlag",
    [
        (_ordinary_pair(), 8),
        (
            pd.DataFrame(
                {
                    "Y": 1e9 + np.random.default_rng(2).normal(size=180),
                    "X": np.random.default_rng(3).normal(size=180),
                }
            ),
            4,
        ),
        (
            pd.DataFrame(
                {
                    "Y": np.random.default_rng(4).normal(size=180),
                    "X": np.arange(180, dtype=float),
                }
            ),
            5,
        ),
        (
            pd.DataFrame(
                {
                    "Y": np.random.default_rng(5).normal(size=180),
                    "X": np.ones(180),
                }
            ),
            4,
        ),
        (_high_collinearity_pair(), 4),
    ],
    ids=[
        "ordinary",
        "large_offset",
        "rank_deficient_lags",
        "constant_candidate",
        "high_collinearity",
    ],
)
def test_reverse_scan_matches_independent_reference(pair, maxlag):
    _assert_lag_equivalence(pair, maxlag)


def test_longer_reverse_scan_matches_independent_reference():
    _assert_lag_equivalence(_ordinary_pair(seed=17, rows=320), 25)


@pytest.mark.parametrize(
    "pair",
    [_high_collinearity_pair(), _near_rank_deficient_pair()],
    ids=["high_collinearity", "near_rank_deficient"],
)
def test_unstable_ssr_guard_does_not_rebuild_valid_qr(pair):
    diagnostics = _GrangerDiagnostics()

    _assert_lag_equivalence(pair, 12, diagnostics=diagnostics)

    assert diagnostics.fallback_count > 0
    assert diagnostics.qr_rebuild_count == 0
    assert diagnostics.matrix_build_count <= 26


@pytest.mark.parametrize("restricted", [True, False])
def test_single_guard_update_failure_recovers_fast_path(monkeypatch, restricted):
    pair = _ordinary_pair(seed=28, rows=220)
    diagnostics = _GrangerDiagnostics()
    original = causality._downdate_ssr_guard
    injected = {"done": False}

    def fail_once(guard, target_values, variable_values, lag, *, restricted):
        if not injected["done"] and restricted == fail_once.restricted and lag == 10:
            injected["done"] = True
            raise FloatingPointError("injected guard update failure")
        return original(
            guard,
            target_values,
            variable_values,
            lag,
            restricted=restricted,
        )

    fail_once.restricted = restricted
    monkeypatch.setattr(causality, "_downdate_ssr_guard", fail_once)
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        10,
        diagnostics=diagnostics,
    )
    expected = _reference_lag_statistics(pair, "Y", "X", 10)

    assert injected["done"]
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 1
    assert diagnostics.qr_rebuild_count == 0
    assert diagnostics.matrix_build_count == 3
    assert actual[9].fallback_used
    assert not actual[8].fallback_used
    for lag in actual:
        assert actual[lag].f_statistic == pytest.approx(
            expected[lag]["f_statistic"],
            rel=1e-7,
            abs=1e-9,
        )
        assert actual[lag].p_value == pytest.approx(
            expected[lag]["p_value"],
            rel=1e-7,
            abs=1e-9,
        )


def test_unrestricted_initial_guard_failure_recovers_fast_path(monkeypatch):
    maxlag = 10
    pair = _ordinary_pair(seed=29, rows=220)
    diagnostics = _GrangerDiagnostics()
    original = causality._try_initial_ssr_guard
    calls = []

    def fail_once(state):
        calls.append(state.shape[0])
        if state.shape == (2 * maxlag + 2, 2 * maxlag + 2) and calls.count(
            2 * maxlag + 2
        ) == 1:
            return None
        return original(state)

    monkeypatch.setattr(causality, "_try_initial_ssr_guard", fail_once)
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
    )
    expected = _reference_lag_statistics(pair, "Y", "X", maxlag)

    assert actual[maxlag].fallback_used
    assert not actual[maxlag - 1].fallback_used
    assert 2 * (maxlag - 1) + 2 in calls
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 1
    assert diagnostics.qr_rebuild_count == 0
    _assert_statistics_match_reference(actual, expected)


def test_restricted_initial_guard_failure_recovers_and_reuses_cache(monkeypatch):
    maxlag = 8
    frame = _ordinary_pair(seed=30, rows=220)
    frame["X2"] = np.random.default_rng(31).normal(size=len(frame))
    diagnostics = _GrangerDiagnostics()
    restricted_cache = {}
    mask_key = _restricted_mask_key("Y", np.ones(len(frame), dtype=bool))
    original = causality._try_initial_ssr_guard
    failed = {"done": False}
    restricted_shapes = []

    def fail_once(state):
        if state.shape[0] <= maxlag + 2:
            restricted_shapes.append(state.shape[0])
        if state.shape == (maxlag + 2, maxlag + 2) and not failed["done"]:
            failed["done"] = True
            return None
        return original(state)

    monkeypatch.setattr(causality, "_try_initial_ssr_guard", fail_once)
    first = _fast_granger_lag_statistics(
        frame[["Y", "X"]],
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
        restricted_cache=restricted_cache,
        mask_key=mask_key,
    )
    second = _fast_granger_lag_statistics(
        frame[["Y", "X2"]],
        "Y",
        "X2",
        maxlag,
        diagnostics=diagnostics,
        restricted_cache=restricted_cache,
        mask_key=mask_key,
    )

    assert failed["done"]
    assert maxlag + 1 in restricted_shapes
    assert first[maxlag].fallback_used
    assert not first[maxlag - 1].fallback_used
    assert second[maxlag].fallback_used
    assert diagnostics.restricted_cache_entries == 1
    assert diagnostics.initial_qr_count == 3
    assert diagnostics.lstsq_count == 1
    assert diagnostics.fallback_count == 2
    _assert_statistics_match_reference(
        first,
        _reference_lag_statistics(frame[["Y", "X"]], "Y", "X", maxlag),
    )
    _assert_statistics_match_reference(
        second,
        _reference_lag_statistics(frame[["Y", "X2"]], "Y", "X2", maxlag),
    )


def test_unrestricted_initial_qr_failure_rebuilds_lower_lag(monkeypatch):
    maxlag = 8
    pair = _ordinary_pair(seed=32, rows=220)
    diagnostics = _GrangerDiagnostics()
    original = causality._initial_qr_state
    failed = {"done": False}

    def fail_once(target_values, variable_values, lag, *, restricted, diagnostics):
        state = original(
            target_values,
            variable_values,
            lag,
            restricted=restricted,
            diagnostics=diagnostics,
        )
        if not restricted and lag == maxlag and not failed["done"]:
            failed["done"] = True
            raise np.linalg.LinAlgError("injected initial QR failure")
        return state

    monkeypatch.setattr(causality, "_initial_qr_state", fail_once)
    monkeypatch.setattr(
        causality,
        "_legacy_lag_statistics_path",
        lambda *args, **kwargs: pytest.fail("whole-path legacy fallback is forbidden"),
    )
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
    )
    expected = _reference_lag_statistics(pair, "Y", "X", maxlag)

    assert failed["done"]
    assert actual[maxlag].fallback_used
    assert not actual[maxlag - 1].fallback_used
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 1
    assert diagnostics.qr_rebuild_count == 1
    assert diagnostics.lstsq_count < 2 * maxlag
    _assert_statistics_match_reference(actual, expected)


def test_restricted_qr_rebuild_retries_each_lower_lag(monkeypatch):
    maxlag = 8
    pair = _ordinary_pair(seed=33, rows=220)
    diagnostics = _GrangerDiagnostics()
    original_initial = causality._initial_qr_state
    original_rebuild = causality._rebuild_qr_state
    initial_failed = {"done": False}
    rebuild_failures = {"count": 0}
    rebuild_lags = []

    def fail_initial(target_values, variable_values, lag, *, restricted, diagnostics):
        state = original_initial(
            target_values,
            variable_values,
            lag,
            restricted=restricted,
            diagnostics=diagnostics,
        )
        if restricted and lag == maxlag and not initial_failed["done"]:
            initial_failed["done"] = True
            raise np.linalg.LinAlgError("injected initial QR failure")
        return state

    def fail_two_rebuilds(
        target_values,
        variable_values,
        lag,
        *,
        restricted,
        diagnostics,
    ):
        state = original_rebuild(
            target_values,
            variable_values,
            lag,
            restricted=restricted,
            diagnostics=diagnostics,
        )
        if restricted:
            rebuild_lags.append(lag)
            if rebuild_failures["count"] < 2:
                rebuild_failures["count"] += 1
                raise np.linalg.LinAlgError("injected rebuild failure")
        return state

    monkeypatch.setattr(causality, "_initial_qr_state", fail_initial)
    monkeypatch.setattr(causality, "_rebuild_qr_state", fail_two_rebuilds)
    monkeypatch.setattr(
        causality,
        "_legacy_lag_statistics_path",
        lambda *args, **kwargs: pytest.fail("whole-path legacy fallback is forbidden"),
    )
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
    )
    expected = _reference_lag_statistics(pair, "Y", "X", maxlag)

    assert initial_failed["done"]
    assert rebuild_lags[:3] == [maxlag - 1, maxlag - 2, maxlag - 3]
    assert diagnostics.fallback_count == 3
    assert diagnostics.lstsq_count == 3
    assert diagnostics.qr_rebuild_count == 3
    assert all(actual[lag].fallback_used for lag in [maxlag, maxlag - 1, maxlag - 2])
    assert not actual[maxlag - 3].fallback_used
    _assert_statistics_match_reference(actual, expected)


def test_known_lag_signal_matches_reference_and_final_result():
    rng = np.random.default_rng(6)
    rows = 220
    x = rng.normal(size=rows)
    y = np.zeros(rows)
    for index in range(3, rows):
        y[index] = 0.4 * y[index - 1] + 0.8 * x[index - 3] + rng.normal(scale=0.2)
    pair = pd.DataFrame({"Y": y, "X": x})

    actual, expected = _assert_lag_equivalence(pair, 7)
    expected_valid = {lag: row for lag, row in expected.items() if not row["skipped"]}
    expected_best = min(expected_valid, key=lambda lag: expected_valid[lag]["p_value"])
    observed_best = min(actual, key=lambda lag: actual[lag].p_value)
    result = run_granger_tests(pair, "Y", ["X"], 7).iloc[0]

    assert observed_best == expected_best
    assert int(result["best_granger_lag"]) == expected_best
    assert float(result["min_p_value"]) == pytest.approx(expected[expected_best]["p_value"])
    assert float(result["f_statistic"]) == pytest.approx(
        expected[expected_best]["f_statistic"]
    )
    assert float(result["predictive_contribution"]) == pytest.approx(
        _reference_predictive_contribution(pair, expected_best)
    )


@pytest.mark.parametrize("kind", ["same_as_target", "near_perfect"])
def test_degenerate_predictors_match_skip_behavior(kind):
    rng = np.random.default_rng(7)
    rows = 160
    if kind == "same_as_target":
        y = rng.normal(size=rows)
        x = y.copy()
        maxlag = 4
    else:
        x = np.arange(rows, dtype=float)
        y = np.zeros(rows)
        y[1:] = x[:-1]
        maxlag = 1
    _assert_lag_equivalence(pd.DataFrame({"Y": y, "X": x}), maxlag)


def test_public_runner_preserves_pairwise_missing_rows_and_insufficient_status():
    pair = _ordinary_pair(rows=100)
    pair.loc[[5, 11], "X"] = np.nan
    cleaned = pair[["Y", "X"]].dropna()
    actual, _ = _assert_lag_equivalence(cleaned, 5)
    direct = _fast_granger_lag_statistics(pair, "Y", "X", 5)
    assert [asdict(direct[lag]) for lag in direct] == [asdict(actual[lag]) for lag in actual]

    insufficient = run_granger_tests(pair.head(20), "Y", ["X"], 5)
    assert insufficient.iloc[0]["status"] == "skipped: insufficient rows"


@pytest.mark.parametrize("restricted", [True, False])
def test_r_only_state_matches_augmented_gram_through_reverse_updates(restricted):
    rng = np.random.default_rng(8)
    y = rng.normal(size=90)
    x = rng.normal(size=90)
    diagnostics = _GrangerDiagnostics()
    candidate = None if restricted else x
    state = _initial_qr_state(
        y,
        candidate,
        6,
        restricted=restricted,
        diagnostics=diagnostics,
    )

    for lag in range(6, 0, -1):
        augmented = _build_augmented_matrix(y, candidate, lag, restricted=restricted)
        columns = lag + 2 if restricted else 2 * lag + 2
        assert state.shape == (columns, columns)
        assert state.T @ state == pytest.approx(augmented.T @ augmented, rel=1e-10, abs=1e-10)
        assert augmented[:, -1].tolist() == y[lag:].tolist()
        if lag > 1:
            state = _downdate_qr_state(
                state,
                y,
                candidate,
                lag,
                restricted=restricted,
            )


def test_column_deletion_and_row_append_each_preserve_gram_contract():
    rng = np.random.default_rng(9)
    y = rng.normal(size=70)
    x = rng.normal(size=70)
    lag = 5
    augmented = _build_augmented_matrix(y, x, lag, restricted=False)
    state = np.linalg.qr(augmented, mode="r")
    x_index, y_index = _lag_column_indices(lag, restricted=False)
    assert x_index is not None

    without_x = _delete_r_column(state, x_index)
    augmented_without_x = np.delete(augmented, x_index, axis=1)
    assert without_x.T @ without_x == pytest.approx(
        augmented_without_x.T @ augmented_without_x, rel=1e-11, abs=1e-11
    )

    without_y = _delete_r_column(without_x, y_index)
    augmented_without_y = np.delete(augmented_without_x, y_index, axis=1)
    assert without_y.T @ without_y == pytest.approx(
        augmented_without_y.T @ augmented_without_y, rel=1e-11, abs=1e-11
    )

    new_row = _new_observation_row(y, x, lag - 1, restricted=False)
    updated = _append_r_row(without_y, new_row)
    expected = np.vstack([augmented_without_y, new_row])
    assert updated.T @ updated == pytest.approx(expected.T @ expected, rel=1e-11, abs=1e-11)
    assert new_row[-1] == y[lag - 1]


def test_well_conditioned_path_uses_one_r_only_qr_per_state(monkeypatch):
    calls = []
    original_qr = np.linalg.qr

    def counted_qr(matrix, mode="reduced"):
        calls.append((matrix.shape, mode))
        return original_qr(matrix, mode=mode)

    monkeypatch.setattr(np.linalg, "qr", counted_qr)
    diagnostics = _GrangerDiagnostics()
    frame = _ordinary_pair(rows=180)
    frame["X2"] = np.random.default_rng(10).normal(size=len(frame))
    result = run_granger_tests(
        frame,
        "Y",
        ["X", "X2"],
        8,
        diagnostics=diagnostics,
    )

    assert len(result) == 2
    assert diagnostics.initial_qr_count == 3
    assert diagnostics.matrix_build_count == 3
    assert diagnostics.restricted_cache_entries == 1
    assert diagnostics.qr_rebuild_count == 0
    assert diagnostics.fallback_count == 0
    assert diagnostics.lstsq_count == 0
    assert len(calls) == 3
    assert all(mode == "r" for _, mode in calls)


def test_restricted_cache_uses_complete_missing_mask():
    frame = _ordinary_pair(rows=180)
    frame["X2"] = np.random.default_rng(11).normal(size=len(frame))
    frame.loc[[5, 9], "X"] = np.nan
    frame.loc[[6, 10], "X2"] = np.nan
    diagnostics = _GrangerDiagnostics()

    run_granger_tests(frame, "Y", ["X", "X2"], 6, diagnostics=diagnostics)

    assert frame["X"].notna().sum() == frame["X2"].notna().sum()
    assert diagnostics.restricted_cache_entries == 2
    assert diagnostics.initial_qr_count == 4


def test_restricted_lstsq_fallback_is_reused_for_same_mask():
    rng = np.random.default_rng(18)
    frame = pd.DataFrame(
        {
            "Y": 1e9 + rng.normal(size=160),
            "X1": rng.normal(size=160),
            "X2": rng.normal(size=160),
        }
    )
    diagnostics = _GrangerDiagnostics()

    run_granger_tests(frame, "Y", ["X1", "X2"], 3, diagnostics=diagnostics)

    assert diagnostics.restricted_cache_entries == 1
    assert diagnostics.fallback_count == 6
    assert diagnostics.lstsq_count == 9


def test_restricted_cache_key_includes_target_name():
    mask = np.array([True, False, True, True])
    assert _restricted_mask_key("Y1", mask) != _restricted_mask_key("Y2", mask)
    assert _restricted_mask_key("Y1", mask) != _restricted_mask_key(
        "Y1", np.array([True, True, False, True])
    )


def test_single_statistical_fallback_does_not_rebuild_or_pollute_later_lags(monkeypatch):
    original = causality._statistics_from_r
    injected = {"done": False}

    def fail_one_unrestricted_state(state, nobs, *, target_scale=1.0):
        if not injected["done"] and state.shape == (8, 8) and target_scale == 1.0:
            injected["done"] = True
            return None
        return original(state, nobs, target_scale=target_scale)

    monkeypatch.setattr(causality, "_statistics_from_r", fail_one_unrestricted_state)
    diagnostics = _GrangerDiagnostics()
    pair = _ordinary_pair(rows=180)
    actual = _fast_granger_lag_statistics(pair, "Y", "X", 6, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 6)

    assert injected["done"]
    assert diagnostics.qr_rebuild_count == 0
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 1
    assert actual[3].fallback_used
    assert not actual[2].fallback_used
    assert actual[2].f_statistic == pytest.approx(expected[2]["f_statistic"], rel=1e-7)


@pytest.mark.parametrize("failure", ["delete", "append", "nonfinite"])
def test_damaged_qr_state_rebuilds_and_later_lags_match_reference(monkeypatch, failure):
    pair = _ordinary_pair(seed=12, rows=180)
    diagnostics = _GrangerDiagnostics()
    injected = {"done": False}

    if failure == "delete":
        original = causality._delete_r_column

        def fail_once(state, column):
            if not injected["done"] and state.shape == (14, 14):
                injected["done"] = True
                raise FloatingPointError("injected delete failure")
            return original(state, column)

        monkeypatch.setattr(causality, "_delete_r_column", fail_once)
    elif failure == "append":
        original = causality._append_r_row

        def fail_once(state, row):
            if not injected["done"] and state.shape == (12, 12):
                injected["done"] = True
                raise FloatingPointError("injected append failure")
            return original(state, row)

        monkeypatch.setattr(causality, "_append_r_row", fail_once)
    else:
        original = causality._downdate_qr_state

        def fail_once(state, target_values, variable_values, lag, *, restricted):
            updated = original(
                state,
                target_values,
                variable_values,
                lag,
                restricted=restricted,
            )
            if not injected["done"] and not restricted:
                injected["done"] = True
                updated[0, 0] = np.nan
            return updated

        monkeypatch.setattr(causality, "_downdate_qr_state", fail_once)

    actual = _fast_granger_lag_statistics(pair, "Y", "X", 6, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 6)

    assert injected["done"]
    assert diagnostics.qr_rebuild_count == 1
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 1
    assert actual[5].fallback_used
    assert not actual[4].fallback_used
    for lag in range(1, 7):
        if not expected[lag]["skipped"]:
            assert actual[lag].f_statistic == pytest.approx(
                expected[lag]["f_statistic"], rel=1e-7, abs=1e-9
            )


@pytest.mark.parametrize("column", ["Y", "X"])
def test_nonfinite_pair_value_preserves_no_valid_lag_status(column):
    frame = _ordinary_pair(seed=21, rows=100)
    frame.loc[50, column] = np.inf

    result = run_granger_tests(frame, "Y", ["X"], 5)

    assert result.iloc[0]["status"] == "skipped: no valid lag tests"


def test_unused_trailing_candidate_inf_keeps_valid_lag_results():
    pair = _ordinary_pair(seed=25, rows=100)
    pair.loc[len(pair) - 1, "X"] = np.inf

    actual = _fast_granger_lag_statistics(pair, "Y", "X", 5)
    expected = _reference_lag_statistics(pair, "Y", "X", 5)

    assert all(actual[lag].f_statistic is not None for lag in actual)
    for lag in actual:
        assert actual[lag].f_statistic == pytest.approx(expected[lag]["f_statistic"])


def test_both_initial_qr_failures_rebuild_lower_lag(monkeypatch):
    pair = _ordinary_pair(seed=26, rows=180)
    diagnostics = _GrangerDiagnostics()

    def fail_initial_qr(*args, **kwargs):
        raise np.linalg.LinAlgError("injected initial QR failure")

    monkeypatch.setattr(causality, "_initial_qr_state", fail_initial_qr)
    actual = _fast_granger_lag_statistics(pair, "Y", "X", 6, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 6)

    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 2
    assert diagnostics.qr_rebuild_count == 2
    assert actual[6].fallback_used
    assert not actual[5].fallback_used
    for lag in actual:
        assert actual[lag].f_statistic == pytest.approx(expected[lag]["f_statistic"])
        assert actual[lag].p_value == pytest.approx(expected[lag]["p_value"])


def test_lstsq_failure_skips_only_current_lag_and_scan_continues(monkeypatch):
    pair = _ordinary_pair(seed=22, rows=180)
    diagnostics = _GrangerDiagnostics()
    original_statistics = causality._statistics_from_r
    original_lstsq = np.linalg.lstsq
    injected = {"fallback": False, "solver": False}

    def force_one_fallback(state, nobs, *, target_scale=1.0):
        if not injected["fallback"] and state.shape == (8, 8) and target_scale == 1.0:
            injected["fallback"] = True
            return None
        return original_statistics(state, nobs, target_scale=target_scale)

    def fail_one_solver(matrix, response, rcond=None):
        if not injected["solver"] and matrix.shape[1] == 7:
            injected["solver"] = True
            raise np.linalg.LinAlgError("injected lstsq failure")
        return original_lstsq(matrix, response, rcond=rcond)

    monkeypatch.setattr(causality, "_statistics_from_r", force_one_fallback)
    monkeypatch.setattr(np.linalg, "lstsq", fail_one_solver)
    actual = _fast_granger_lag_statistics(pair, "Y", "X", 6, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 6)

    assert injected == {"fallback": True, "solver": True}
    assert actual[3].f_statistic is None
    assert not actual[2].fallback_used
    assert actual[2].f_statistic == pytest.approx(expected[2]["f_statistic"], rel=1e-7)
    assert diagnostics.qr_rebuild_count == 0
    assert diagnostics.fallback_count == 1
    assert diagnostics.lstsq_count == 2


@pytest.mark.parametrize("corruption", ["wrong_dimension", "lower_triangle", "response_diagonal"])
def test_finite_qr_state_damage_triggers_rebuild(monkeypatch, corruption):
    pair = _ordinary_pair(seed=23, rows=180)
    diagnostics = _GrangerDiagnostics()
    original = causality._downdate_qr_state
    injected = {"done": False}

    def corrupt_once(state, target_values, variable_values, lag, *, restricted):
        updated = original(
            state,
            target_values,
            variable_values,
            lag,
            restricted=restricted,
        )
        if not injected["done"] and not restricted and lag == 6:
            injected["done"] = True
            if corruption == "wrong_dimension":
                return updated[:-1, :-1]
            if corruption == "lower_triangle":
                updated[-1, 0] = 0.25
            else:
                updated[-1, -1] *= 0.5
        return updated

    monkeypatch.setattr(causality, "_downdate_qr_state", corrupt_once)
    actual = _fast_granger_lag_statistics(pair, "Y", "X", 6, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 6)

    assert injected["done"]
    assert diagnostics.qr_rebuild_count == 1
    assert diagnostics.fallback_count == 1
    assert actual[5].fallback_used
    assert not actual[4].fallback_used
    assert actual[5].f_statistic == pytest.approx(expected[5]["f_statistic"], rel=1e-7)
    assert actual[1].f_statistic == pytest.approx(expected[1]["f_statistic"], rel=1e-7)


@pytest.mark.parametrize("maxlag", [15, 32])
def test_every_state_gram_check_detects_norm_preserving_damage(monkeypatch, maxlag):
    pair = _ordinary_pair(seed=24, rows=400)
    diagnostics = _GrangerDiagnostics()
    original = causality._downdate_qr_state
    injected = {"done": False}

    def corrupt_cross_term(state, target_values, variable_values, lag, *, restricted):
        updated = original(
            state,
            target_values,
            variable_values,
            lag,
            restricted=restricted,
        )
        if not injected["done"] and not restricted and lag == maxlag:
            injected["done"] = True
            updated[0, -1] *= -1.0
        return updated

    monkeypatch.setattr(causality, "_downdate_qr_state", corrupt_cross_term)
    actual = _fast_granger_lag_statistics(
        pair,
        "Y",
        "X",
        maxlag,
        diagnostics=diagnostics,
    )
    expected = _reference_lag_statistics(pair, "Y", "X", maxlag)

    assert injected["done"]
    assert diagnostics.qr_rebuild_count == 1
    assert diagnostics.fallback_count == 1
    assert actual[maxlag - 1].fallback_used
    for lag in actual:
        assert actual[lag].p_value == pytest.approx(expected[lag]["p_value"], rel=1e-7)


@pytest.mark.parametrize("restricted", [True, False])
def test_ssr_guard_detects_damage_outside_sampled_gram_terms(monkeypatch, restricted):
    pair = _ordinary_pair(seed=27, rows=400)
    diagnostics = _GrangerDiagnostics()
    original = causality._downdate_qr_state
    injected = {"done": False}

    def corrupt_response(state, target_values, variable_values, lag, *, restricted):
        updated = original(
            state,
            target_values,
            variable_values,
            lag,
            restricted=restricted,
        )
        if injected["done"] or restricted != corrupt_response.restricted or lag != 15:
            return updated

        next_lag = lag - 1
        audited = [0, 1, next_lag]
        if not restricted:
            audited.extend([next_lag + 1, 2 * next_lag])
        basis, _ = np.linalg.qr(updated[:, audited], mode="reduced")
        response = updated[:, -1].copy()
        projection = basis @ (basis.T @ response)
        residual = response - projection
        direction = np.zeros_like(response)
        direction[-1] = 1.0
        direction -= basis @ (basis.T @ direction)
        direction -= residual * (np.dot(residual, direction) / np.dot(residual, residual))
        direction /= np.linalg.norm(direction)
        angle = 0.6
        updated[:, -1] = (
            projection
            + np.cos(angle) * residual
            + np.sin(angle) * np.linalg.norm(residual) * direction
        )

        assert np.linalg.norm(updated[:, -1]) == pytest.approx(np.linalg.norm(response))
        assert not np.isclose(updated[-1, -1] ** 2, response[-1] ** 2)
        assert causality._gram_cross_audit_matches(
            updated,
            target_values,
            variable_values,
            next_lag,
            restricted=restricted,
        )
        injected["done"] = True
        return updated

    corrupt_response.restricted = restricted
    monkeypatch.setattr(causality, "_downdate_qr_state", corrupt_response)
    actual = _fast_granger_lag_statistics(pair, "Y", "X", 15, diagnostics=diagnostics)
    expected = _reference_lag_statistics(pair, "Y", "X", 15)

    assert injected["done"]
    assert diagnostics.qr_rebuild_count == 1
    assert diagnostics.fallback_count == 1
    assert actual[14].fallback_used
    for lag in actual:
        assert actual[lag].f_statistic == pytest.approx(
            expected[lag]["f_statistic"],
            rel=1e-7,
            abs=1e-9,
        )
        assert actual[lag].p_value == pytest.approx(
            expected[lag]["p_value"],
            rel=1e-7,
            abs=1e-9,
        )


def test_benchmark_json_uses_peak_working_set_schema(monkeypatch, capsys):
    args = SimpleNamespace(
        rows=100,
        variables=1,
        maxlag=2,
        seed=1,
        case="well_conditioned",
        worker=None,
        compare_old=True,
        old_timeout=10.0,
    )

    def fake_worker(_args, implementation, timeout=None):
        assert timeout is None or timeout == 10.0
        return {
            "elapsed_seconds": 1.0 if implementation == "new" else 2.0,
            "peak_working_set_mb": 10.0 if implementation == "new" else 20.0,
        }

    monkeypatch.setattr(benchmark_granger, "_arguments", lambda: args)
    monkeypatch.setattr(benchmark_granger, "_run_worker", fake_worker)
    benchmark_granger.main()
    result = json.loads(capsys.readouterr().out)

    assert result["peak_working_set_mb"] == 10.0
    assert result["old_peak_working_set_mb"] == 20.0
    assert "peak_memory_mb" not in result
