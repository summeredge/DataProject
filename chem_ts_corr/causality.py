from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f

from chem_ts_corr.common import benjamini_hochberg


_GRANGER_COLUMNS = [
    "variable",
    "status",
    "best_granger_lag",
    "min_p_value",
    "f_statistic",
    "predictive_contribution",
    "interpretation",
    "fdr_q_value",
]


@dataclass
class _GrangerDiagnostics:
    initial_qr_count: int = 0
    qr_rebuild_count: int = 0
    restricted_cache_entries: int = 0
    matrix_build_count: int = 0
    fallback_count: int = 0
    lstsq_count: int = 0


@dataclass(frozen=True)
class _ModelStatistics:
    nobs: int
    ssr: float
    rank: int
    target_scale: float
    fallback_used: bool = False


@dataclass(frozen=True)
class _LagStatistics:
    lag: int
    nobs: int
    restricted_ssr: float
    unrestricted_ssr: float
    restricted_rank: int
    unrestricted_rank: int
    df_num: int
    df_den: int
    f_statistic: float | None
    p_value: float | None
    skipped_reason: str | None
    fallback_used: bool


_RestrictedCacheKey = tuple[str, int, bytes]
_RestrictedPath = dict[int, _ModelStatistics]


def run_granger_tests(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    maxlag: int,
    *,
    diagnostics: _GrangerDiagnostics | None = None,
) -> pd.DataFrame:
    active_diagnostics = diagnostics if diagnostics is not None else _GrangerDiagnostics()
    restricted_cache: dict[_RestrictedCacheKey, _RestrictedPath] = {}
    rows: list[dict[str, float | int | str | None]] = []
    for variable in variables:
        if variable == target:
            continue

        try:
            mask = frame[target].notna() & frame[variable].notna()
            pair = frame[[target, variable]].dropna()
            mask_key = _restricted_mask_key(target, mask.to_numpy(dtype=bool))
        except Exception as exc:
            rows.append({"variable": variable, "status": f"failed: {exc}", "min_p_value": None})
            continue

        if maxlag <= 0:
            rows.append(
                {
                    "variable": variable,
                    "status": "skipped: no valid lag tests",
                    "min_p_value": None,
                }
            )
            continue

        if len(pair) < max(30, maxlag * 5):
            rows.append(
                {"variable": variable, "status": "skipped: insufficient rows", "min_p_value": None}
            )
            continue

        try:
            lag_results = _fast_granger_ssr_ftests(
                pair,
                target,
                variable,
                maxlag,
                diagnostics=active_diagnostics,
                restricted_cache=restricted_cache,
                mask_key=mask_key,
            )
        except Exception as exc:
            rows.append({"variable": variable, "status": f"failed: {exc}", "min_p_value": None})
            continue

        if not lag_results:
            rows.append(
                {"variable": variable, "status": "skipped: no valid lag tests", "min_p_value": None}
            )
            continue

        best_lag = min(lag_results, key=lambda lag: lag_results[lag][1])
        f_statistic, min_p_value = lag_results[best_lag]
        rows.append(
            {
                "variable": variable,
                "status": "ok",
                "best_granger_lag": best_lag,
                "min_p_value": min_p_value,
                "f_statistic": f_statistic,
                "predictive_contribution": _predictive_contribution(
                    pair[target], pair[variable], best_lag
                ),
                "interpretation": "predictive validation only; not a causal conclusion",
            }
        )

    result_frame = pd.DataFrame(rows)
    if "min_p_value" in result_frame.columns:
        result_frame["fdr_q_value"] = benjamini_hochberg(result_frame["min_p_value"])
        result_frame = result_frame.sort_values("fdr_q_value", na_position="last")
    return result_frame.reindex(columns=_GRANGER_COLUMNS)


def _fast_granger_ssr_ftests(
    pair: pd.DataFrame,
    target: str,
    variable: str,
    maxlag: int,
    *,
    diagnostics: _GrangerDiagnostics | None = None,
    restricted_cache: dict[_RestrictedCacheKey, _RestrictedPath] | None = None,
    mask_key: _RestrictedCacheKey | None = None,
) -> dict[int, tuple[float, float]]:
    lag_statistics = _fast_granger_lag_statistics(
        pair,
        target,
        variable,
        maxlag,
        diagnostics=diagnostics,
        restricted_cache=restricted_cache,
        mask_key=mask_key,
    )
    return {
        lag: (float(stats.f_statistic), float(stats.p_value))
        for lag, stats in lag_statistics.items()
        if stats.f_statistic is not None and stats.p_value is not None
    }


def _fast_granger_lag_statistics(
    pair: pd.DataFrame,
    target: str,
    variable: str,
    maxlag: int,
    *,
    diagnostics: _GrangerDiagnostics | None = None,
    restricted_cache: dict[_RestrictedCacheKey, _RestrictedPath] | None = None,
    mask_key: _RestrictedCacheKey | None = None,
) -> dict[int, _LagStatistics]:
    if maxlag <= 0:
        return {}
    if target not in pair.columns or variable not in pair.columns:
        raise KeyError(f"missing required columns: {target}, {variable}")

    active_diagnostics = diagnostics if diagnostics is not None else _GrangerDiagnostics()
    active_cache = restricted_cache if restricted_cache is not None else {}
    clean_pair = pair[[target, variable]].dropna()
    target_values = clean_pair[target].to_numpy(dtype=float)
    variable_values = clean_pair[variable].to_numpy(dtype=float)
    if len(target_values) <= maxlag:
        return {}
    if not np.isfinite(target_values).all() or not np.isfinite(variable_values).all():
        return _legacy_lag_statistics_path(
            target_values,
            variable_values,
            maxlag,
            active_diagnostics,
        )

    target_square_prefix = _squared_prefix(target_values)
    variable_square_prefix = _squared_prefix(variable_values)

    key = mask_key or _restricted_mask_key(target, np.ones(len(target_values), dtype=bool))
    if key not in active_cache:
        active_cache[key] = _restricted_statistics_path(
            target_values,
            maxlag,
            active_diagnostics,
        )
        active_diagnostics.restricted_cache_entries = len(active_cache)
    restricted_path = active_cache[key]

    state = _try_build_valid_qr_state(
        target_values=target_values,
        variable_values=variable_values,
        lag=maxlag,
        restricted=False,
        diagnostics=active_diagnostics,
        target_square_prefix=target_square_prefix,
        variable_square_prefix=variable_square_prefix,
        rebuild=False,
    )
    ssr_guard = _try_valid_ssr_guard(state)
    descending: dict[int, _LagStatistics] = {}
    force_fallback = False
    guard_failure_streak = 0
    for lag in range(maxlag, 0, -1):
        restricted_stats = restricted_path[lag]
        fallback_used = restricted_stats.fallback_used

        if state is None or force_fallback or ssr_guard is None:
            unrestricted_stats = _safe_lstsq_model_statistics(
                target_values,
                variable_values,
                lag,
                restricted=False,
                diagnostics=active_diagnostics,
            )
            force_fallback = False
            fallback_used = True
        else:
            unrestricted_stats = _statistics_from_r(state, len(target_values) - lag)
            if unrestricted_stats is None:
                unrestricted_stats = _safe_lstsq_model_statistics(
                    target_values,
                    variable_values,
                    lag,
                    restricted=False,
                    diagnostics=active_diagnostics,
                )
                fallback_used = True

        if _needs_joint_fallback(restricted_stats, unrestricted_stats):
            if not restricted_stats.fallback_used:
                restricted_stats = _safe_lstsq_model_statistics(
                    target_values,
                    variable_values,
                    lag,
                    restricted=True,
                    diagnostics=active_diagnostics,
                )
                restricted_path[lag] = restricted_stats
            if not unrestricted_stats.fallback_used:
                unrestricted_stats = _safe_lstsq_model_statistics(
                    target_values,
                    variable_values,
                    lag,
                    restricted=False,
                    diagnostics=active_diagnostics,
                )
            fallback_used = True

        lag_stats = _combine_model_statistics(
            lag,
            restricted_stats,
            unrestricted_stats,
            fallback_used,
        )
        descending[lag] = lag_stats
        if fallback_used:
            active_diagnostics.fallback_count += 1

        if lag > 1:
            next_lag = lag - 1
            if state is None:
                state = _try_build_valid_qr_state(
                    target_values=target_values,
                    variable_values=variable_values,
                    lag=next_lag,
                    restricted=False,
                    diagnostics=active_diagnostics,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=variable_square_prefix,
                    rebuild=True,
                )
                ssr_guard = _try_valid_ssr_guard(state)
                force_fallback = False
                guard_failure_streak = 0
                continue

            guard_was_available = ssr_guard is not None
            next_guard = _try_downdate_ssr_guard(
                ssr_guard,
                target_values,
                variable_values,
                lag,
                restricted=False,
            )
            try:
                state = _downdate_qr_state(
                    state,
                    target_values,
                    variable_values,
                    lag,
                    restricted=False,
                )
                if not _qr_state_is_valid(
                    state,
                    expected_columns=2 * next_lag + 2,
                    target_values=target_values,
                    variable_values=variable_values,
                    lag=next_lag,
                    restricted=False,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=variable_square_prefix,
                    ssr_guard=None,
                ):
                    raise FloatingPointError("invalid QR state")
                if next_guard is not None and not _ssr_guard_matches(state, next_guard):
                    raise FloatingPointError("QR state does not match SSR guard")
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                state = _try_build_valid_qr_state(
                    target_values=target_values,
                    variable_values=variable_values,
                    lag=next_lag,
                    restricted=False,
                    diagnostics=active_diagnostics,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=variable_square_prefix,
                    rebuild=True,
                )
                ssr_guard = _try_valid_ssr_guard(state)
                force_fallback = state is not None
                guard_failure_streak = 0
            else:
                guard_update_failed = guard_was_available and next_guard is None
                if guard_update_failed:
                    guard_failure_streak += 1
                    force_fallback = True
                    if guard_failure_streak == 1:
                        next_guard = _try_valid_ssr_guard(state)
                elif not guard_was_available and guard_failure_streak == 0:
                    next_guard = _try_valid_ssr_guard(state)
                elif next_guard is not None:
                    guard_failure_streak = 0
                ssr_guard = next_guard

    return {lag: descending[lag] for lag in range(1, maxlag + 1)}


def _restricted_statistics_path(
    target_values: np.ndarray,
    maxlag: int,
    diagnostics: _GrangerDiagnostics,
) -> _RestrictedPath:
    target_square_prefix = _squared_prefix(target_values)
    state = _try_build_valid_qr_state(
        target_values=target_values,
        variable_values=None,
        lag=maxlag,
        restricted=True,
        diagnostics=diagnostics,
        target_square_prefix=target_square_prefix,
        variable_square_prefix=None,
        rebuild=False,
    )
    ssr_guard = _try_valid_ssr_guard(state)
    descending: _RestrictedPath = {}
    force_fallback = False
    guard_failure_streak = 0
    for lag in range(maxlag, 0, -1):
        nobs = len(target_values) - lag
        target_scale = _target_variation_scale(target_values[lag:])
        if state is None or force_fallback or ssr_guard is None:
            stats = _safe_lstsq_model_statistics(
                target_values,
                None,
                lag,
                restricted=True,
                diagnostics=diagnostics,
            )
            force_fallback = False
        else:
            stats = _statistics_from_r(state, nobs, target_scale=target_scale)
            if stats is None:
                stats = _safe_lstsq_model_statistics(
                    target_values,
                    None,
                    lag,
                    restricted=True,
                    diagnostics=diagnostics,
                )
        descending[lag] = stats

        if lag > 1:
            next_lag = lag - 1
            if state is None:
                state = _try_build_valid_qr_state(
                    target_values=target_values,
                    variable_values=None,
                    lag=next_lag,
                    restricted=True,
                    diagnostics=diagnostics,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=None,
                    rebuild=True,
                )
                ssr_guard = _try_valid_ssr_guard(state)
                force_fallback = False
                guard_failure_streak = 0
                continue

            guard_was_available = ssr_guard is not None
            next_guard = _try_downdate_ssr_guard(
                ssr_guard,
                target_values,
                None,
                lag,
                restricted=True,
            )
            try:
                state = _downdate_qr_state(
                    state,
                    target_values,
                    None,
                    lag,
                    restricted=True,
                )
                if not _qr_state_is_valid(
                    state,
                    expected_columns=next_lag + 2,
                    target_values=target_values,
                    variable_values=None,
                    lag=next_lag,
                    restricted=True,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=None,
                    ssr_guard=None,
                ):
                    raise FloatingPointError("invalid QR state")
                if next_guard is not None and not _ssr_guard_matches(state, next_guard):
                    raise FloatingPointError("QR state does not match SSR guard")
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                state = _try_build_valid_qr_state(
                    target_values=target_values,
                    variable_values=None,
                    lag=next_lag,
                    restricted=True,
                    diagnostics=diagnostics,
                    target_square_prefix=target_square_prefix,
                    variable_square_prefix=None,
                    rebuild=True,
                )
                ssr_guard = _try_valid_ssr_guard(state)
                force_fallback = state is not None
                guard_failure_streak = 0
            else:
                guard_update_failed = guard_was_available and next_guard is None
                if guard_update_failed:
                    guard_failure_streak += 1
                    force_fallback = True
                    if guard_failure_streak == 1:
                        next_guard = _try_valid_ssr_guard(state)
                elif not guard_was_available and guard_failure_streak == 0:
                    next_guard = _try_valid_ssr_guard(state)
                elif next_guard is not None:
                    guard_failure_streak = 0
                ssr_guard = next_guard
    return {lag: descending[lag] for lag in range(1, maxlag + 1)}


def _initial_qr_state(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
    diagnostics: _GrangerDiagnostics,
) -> np.ndarray:
    matrix = _build_augmented_matrix(target_values, variable_values, lag, restricted=restricted)
    diagnostics.matrix_build_count += 1
    diagnostics.initial_qr_count += 1
    return _r_only_qr(matrix)


def _rebuild_qr_state(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
    diagnostics: _GrangerDiagnostics,
) -> np.ndarray:
    matrix = _build_augmented_matrix(target_values, variable_values, lag, restricted=restricted)
    diagnostics.matrix_build_count += 1
    diagnostics.qr_rebuild_count += 1
    return _r_only_qr(matrix)


def _try_build_valid_qr_state(
    *,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    restricted: bool,
    diagnostics: _GrangerDiagnostics,
    target_square_prefix: np.ndarray,
    variable_square_prefix: np.ndarray | None,
    rebuild: bool,
) -> np.ndarray | None:
    try:
        if rebuild:
            state = _rebuild_qr_state(
                target_values,
                variable_values,
                lag,
                restricted=restricted,
                diagnostics=diagnostics,
            )
        else:
            state = _initial_qr_state(
                target_values,
                variable_values,
                lag,
                restricted=restricted,
                diagnostics=diagnostics,
            )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None

    expected_columns = lag + 2 if restricted else 2 * lag + 2
    if not _qr_state_is_valid(
        state,
        expected_columns=expected_columns,
        target_values=target_values,
        variable_values=variable_values,
        lag=lag,
        restricted=restricted,
        target_square_prefix=target_square_prefix,
        variable_square_prefix=variable_square_prefix,
        ssr_guard=None,
    ):
        return None
    return state


def _try_valid_ssr_guard(state: np.ndarray | None) -> np.ndarray | None:
    if state is None:
        return None
    guard = _try_initial_ssr_guard(state)
    if guard is None or not _ssr_guard_matches(state, guard):
        return None
    return guard


def _r_only_qr(matrix: np.ndarray) -> np.ndarray:
    r = np.linalg.qr(matrix, mode="r")
    return np.asarray(r, dtype=float)


def _build_augmented_matrix(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> np.ndarray:
    if lag <= 0:
        raise ValueError("lag must be positive")
    n = len(target_values)
    nobs = n - lag
    columns = lag + 2 if restricted else 2 * lag + 2
    if nobs < columns:
        raise ValueError("insufficient rows for R-only QR state")
    if not restricted and variable_values is None:
        raise ValueError("unrestricted state requires candidate values")

    matrix = np.empty((nobs, columns), dtype=float, order="F")
    matrix[:, 0] = 1.0
    for offset in range(1, lag + 1):
        matrix[:, offset] = target_values[lag - offset : n - offset]
    if not restricted:
        assert variable_values is not None
        for offset in range(1, lag + 1):
            matrix[:, lag + offset] = variable_values[lag - offset : n - offset]
    matrix[:, -1] = target_values[lag:]
    return matrix


def _downdate_qr_state(
    state: np.ndarray,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> np.ndarray:
    x_index, y_index = _lag_column_indices(lag, restricted=restricted)
    updated = state
    if x_index is not None:
        updated = _delete_r_column(updated, x_index)
    updated = _delete_r_column(updated, y_index)
    new_row = _new_observation_row(
        target_values,
        variable_values,
        lag - 1,
        restricted=restricted,
    )
    return _append_r_row(updated, new_row)


def _lag_column_indices(lag: int, *, restricted: bool) -> tuple[int | None, int]:
    if lag <= 1:
        raise ValueError("no lower positive lag state")
    if restricted:
        return None, lag
    return 2 * lag, lag


def _delete_r_column(r: np.ndarray, column: int) -> np.ndarray:
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("R state must be square")
    if column < 0 or column >= r.shape[1] - 1:
        raise ValueError("cannot delete response or out-of-range column")

    work = np.delete(r, column, axis=1)
    for row in range(column, work.shape[1]):
        _rotate_rows(work, row, row + 1, row)
    return work[:-1, :]


def _append_r_row(r: np.ndarray, row: np.ndarray) -> np.ndarray:
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("R state must be square")
    if row.shape != (r.shape[1],):
        raise ValueError("new row does not match R state")

    updated = r.copy()
    trailing = np.asarray(row, dtype=float).copy()
    for column in range(updated.shape[1]):
        a = float(updated[column, column])
        b = float(trailing[column])
        radius = float(np.hypot(a, b))
        if radius == 0.0:
            continue
        cosine = a / radius
        sine = b / radius
        upper = updated[column, column:].copy()
        lower = trailing[column:].copy()
        updated[column, column:] = cosine * upper + sine * lower
        trailing[column:] = -sine * upper + cosine * lower
        trailing[column] = 0.0
    return updated


def _rotate_rows(matrix: np.ndarray, first: int, second: int, start: int) -> None:
    a = float(matrix[first, start])
    b = float(matrix[second, start])
    if b == 0.0:
        return
    radius = float(np.hypot(a, b))
    if radius == 0.0 or not np.isfinite(radius):
        raise FloatingPointError("invalid Givens rotation")
    cosine = a / radius
    sine = b / radius
    upper = matrix[first, start:].copy()
    lower = matrix[second, start:].copy()
    matrix[first, start:] = cosine * upper + sine * lower
    matrix[second, start:] = -sine * upper + cosine * lower
    matrix[second, start] = 0.0


def _new_observation_row(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> np.ndarray:
    if lag <= 0:
        raise ValueError("lag must be positive")
    if not restricted and variable_values is None:
        raise ValueError("unrestricted row requires candidate values")

    columns = lag + 2 if restricted else 2 * lag + 2
    row = np.empty(columns, dtype=float)
    row[0] = 1.0
    row[1 : lag + 1] = target_values[lag - 1 :: -1][:lag]
    if not restricted:
        assert variable_values is not None
        row[lag + 1 : 2 * lag + 1] = variable_values[lag - 1 :: -1][:lag]
    row[-1] = target_values[lag]
    return row


def _statistics_from_r(
    state: np.ndarray,
    nobs: int,
    *,
    target_scale: float = 1.0,
) -> _ModelStatistics | None:
    if (
        state.ndim != 2
        or state.shape[0] != state.shape[1]
        or state.shape[0] < 2
        or not np.isfinite(state).all()
    ):
        return None
    predictor_r = state[:-1, :-1]
    diagonal = np.abs(np.diag(predictor_r))
    if len(diagonal) == 0 or not np.isfinite(diagonal).all():
        return None
    largest = float(np.max(diagonal))
    smallest = float(np.min(diagonal))
    column_norms = np.sqrt(np.sum(predictor_r * predictor_r, axis=0))
    relative_pivots = np.divide(
        diagonal,
        column_norms,
        out=np.zeros_like(diagonal),
        where=column_norms > 0,
    )
    threshold = np.sqrt(np.finfo(float).eps)
    if (
        largest == 0.0
        or smallest / largest <= threshold
        or float(np.min(relative_pivots)) <= threshold
    ):
        return None

    ssr = float(state[-1, -1] ** 2)
    if not np.isfinite(ssr):
        return None
    return _ModelStatistics(
        nobs=nobs,
        ssr=ssr,
        rank=predictor_r.shape[1],
        target_scale=target_scale,
    )


def _qr_state_is_valid(
    state: np.ndarray,
    *,
    expected_columns: int,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    restricted: bool,
    target_square_prefix: np.ndarray,
    variable_square_prefix: np.ndarray | None,
    ssr_guard: np.ndarray | None,
) -> bool:
    if (
        state.ndim != 2
        or state.shape != (expected_columns, expected_columns)
        or not np.isfinite(state).all()
    ):
        return False

    scale = max(float(np.max(np.abs(state))), 1.0)
    lower_error = float(np.max(np.abs(np.tril(state, k=-1))))
    if lower_error > 256.0 * np.finfo(float).eps * expected_columns * scale:
        return False

    expected_norms = _expected_augmented_column_norms(
        target_values,
        variable_values,
        lag,
        restricted=restricted,
        target_square_prefix=target_square_prefix,
        variable_square_prefix=variable_square_prefix,
    )
    observed_norms = np.sum(state * state, axis=0)
    if not np.allclose(observed_norms, expected_norms, rtol=1e-8, atol=1e-8):
        return False
    if ssr_guard is not None and not _ssr_guard_matches(state, ssr_guard):
        return False
    if not _gram_cross_audit_matches(
        state,
        target_values,
        variable_values,
        lag,
        restricted=restricted,
    ):
        return False
    return True


def _try_initial_ssr_guard(state: np.ndarray) -> np.ndarray | None:
    try:
        return _initial_ssr_guard(state)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None


def _initial_ssr_guard(state: np.ndarray) -> np.ndarray:
    inverse_r = np.linalg.solve(state, np.eye(state.shape[0], dtype=float))
    guard = inverse_r @ inverse_r.T
    if not np.isfinite(guard).all():
        raise FloatingPointError("invalid initial SSR guard")
    return (guard + guard.T) * 0.5


def _try_downdate_ssr_guard(
    guard: np.ndarray | None,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> np.ndarray | None:
    if guard is None:
        return None
    try:
        return _downdate_ssr_guard(
            guard,
            target_values,
            variable_values,
            lag,
            restricted=restricted,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None


def _downdate_ssr_guard(
    guard: np.ndarray,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> np.ndarray:
    x_index, y_index = _lag_column_indices(lag, restricted=restricted)
    updated = guard
    if x_index is not None:
        updated = _delete_inverse_gram_column(updated, x_index)
    updated = _delete_inverse_gram_column(updated, y_index)
    row = _new_observation_row(
        target_values,
        variable_values,
        lag - 1,
        restricted=restricted,
    )
    return _append_inverse_gram_row(updated, row)


def _delete_inverse_gram_column(inverse_gram: np.ndarray, column: int) -> np.ndarray:
    pivot = float(inverse_gram[column, column])
    scale = max(float(np.max(np.abs(np.diag(inverse_gram)))), 1.0)
    if not np.isfinite(pivot) or pivot <= np.finfo(float).eps * scale:
        raise FloatingPointError("invalid inverse Gram deletion pivot")
    column_values = np.delete(inverse_gram[:, column], column)
    reduced = np.delete(np.delete(inverse_gram, column, axis=0), column, axis=1)
    updated = reduced - np.outer(column_values, column_values) / pivot
    if not np.isfinite(updated).all():
        raise FloatingPointError("invalid inverse Gram deletion")
    return (updated + updated.T) * 0.5


def _append_inverse_gram_row(inverse_gram: np.ndarray, row: np.ndarray) -> np.ndarray:
    projected = inverse_gram @ row
    denominator = 1.0 + float(np.dot(row, projected))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        raise FloatingPointError("invalid inverse Gram row update")
    updated = inverse_gram - np.outer(projected, projected) / denominator
    if not np.isfinite(updated).all():
        raise FloatingPointError("invalid inverse Gram row update")
    return (updated + updated.T) * 0.5


def _ssr_guard_matches(state: np.ndarray, guard: np.ndarray) -> bool:
    if guard.shape != state.shape or not np.isfinite(guard).all():
        return False
    inverse_ssr = float(guard[-1, -1])
    if inverse_ssr <= 0.0 or not np.isfinite(inverse_ssr):
        return False
    expected_ssr = 1.0 / inverse_ssr
    observed_ssr = float(state[-1, -1] ** 2)
    return bool(np.isclose(observed_ssr, expected_ssr, rtol=1e-7, atol=1e-9))


def _squared_prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))


def _expected_augmented_column_norms(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
    target_square_prefix: np.ndarray,
    variable_square_prefix: np.ndarray | None,
) -> np.ndarray:
    n = len(target_values)
    columns = lag + 2 if restricted else 2 * lag + 2
    expected = np.empty(columns, dtype=float)
    expected[0] = n - lag
    for offset in range(1, lag + 1):
        expected[offset] = (
            target_square_prefix[n - offset] - target_square_prefix[lag - offset]
        )
    if not restricted:
        if variable_values is None or variable_square_prefix is None:
            raise ValueError("unrestricted audit requires candidate values")
        for offset in range(1, lag + 1):
            expected[lag + offset] = (
                variable_square_prefix[n - offset] - variable_square_prefix[lag - offset]
            )
    expected[-1] = target_square_prefix[n] - target_square_prefix[lag]
    return expected


def _gram_cross_audit_matches(
    state: np.ndarray,
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
) -> bool:
    response_index = state.shape[1] - 1
    pairs = {(0, response_index), (1, response_index), (lag, response_index)}
    if not restricted:
        pairs.update(
            {
                (lag + 1, response_index),
                (2 * lag, response_index),
                (1, lag + 1),
            }
        )
    for first, second in pairs:
        observed = float(np.dot(state[:, first], state[:, second]))
        expected = _augmented_column_dot(
            target_values,
            variable_values,
            lag,
            first,
            second,
            restricted=restricted,
        )
        if not np.isclose(observed, expected, rtol=1e-8, atol=1e-8):
            return False
    return True


def _augmented_column_dot(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    first: int,
    second: int,
    *,
    restricted: bool,
) -> float:
    first_values = _augmented_column_values(
        target_values,
        variable_values,
        lag,
        first,
        restricted=restricted,
    )
    second_values = _augmented_column_values(
        target_values,
        variable_values,
        lag,
        second,
        restricted=restricted,
    )
    if first_values is None:
        return float(np.sum(second_values))
    if second_values is None:
        return float(np.sum(first_values))
    return float(np.dot(first_values, second_values))


def _augmented_column_values(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    column: int,
    *,
    restricted: bool,
) -> np.ndarray | None:
    n = len(target_values)
    response_index = lag + 1 if restricted else 2 * lag + 1
    if column == 0:
        return None
    if column == response_index:
        return target_values[lag:]
    if column <= lag:
        return target_values[lag - column : n - column]
    if restricted or variable_values is None:
        raise ValueError("invalid augmented column")
    offset = column - lag
    return variable_values[lag - offset : n - offset]


def _needs_joint_fallback(
    restricted: _ModelStatistics,
    unrestricted: _ModelStatistics,
) -> bool:
    if not np.isfinite(restricted.ssr) or not np.isfinite(unrestricted.ssr):
        return True
    tolerance = 128.0 * np.finfo(float).eps * max(
        restricted.ssr,
        unrestricted.ssr,
        restricted.target_scale,
        1.0,
    )
    if unrestricted.ssr > restricted.ssr + tolerance:
        return True
    return _is_near_perfect_fit_from_scale(unrestricted.ssr, restricted.target_scale)


def _combine_model_statistics(
    lag: int,
    restricted: _ModelStatistics,
    unrestricted: _ModelStatistics,
    fallback_used: bool,
) -> _LagStatistics:
    nobs = restricted.nobs
    df_num = unrestricted.rank - restricted.rank
    df_den = nobs - unrestricted.rank
    skipped_reason: str | None = None
    f_statistic: float | None = None
    p_value: float | None = None

    if df_num <= 0:
        skipped_reason = "no effective restrictions"
    elif df_den <= 0:
        skipped_reason = "insufficient degrees of freedom"
    elif (
        not np.isfinite(restricted.ssr)
        or not np.isfinite(unrestricted.ssr)
        or unrestricted.ssr <= 0
        or _is_near_perfect_fit_from_scale(unrestricted.ssr, restricted.target_scale)
    ):
        skipped_reason = "invalid or near-perfect fit"
    else:
        ssr_delta = max(0.0, restricted.ssr - unrestricted.ssr)
        candidate_f = (ssr_delta / df_num) / (unrestricted.ssr / df_den)
        candidate_p = float(f.sf(candidate_f, df_num, df_den))
        if np.isfinite(candidate_f) and np.isfinite(candidate_p):
            f_statistic = float(candidate_f)
            p_value = candidate_p
        else:
            skipped_reason = "non-finite statistic"

    return _LagStatistics(
        lag=lag,
        nobs=nobs,
        restricted_ssr=restricted.ssr,
        unrestricted_ssr=unrestricted.ssr,
        restricted_rank=restricted.rank,
        unrestricted_rank=unrestricted.rank,
        df_num=df_num,
        df_den=df_den,
        f_statistic=f_statistic,
        p_value=p_value,
        skipped_reason=skipped_reason,
        fallback_used=fallback_used,
    )


def _lstsq_model_statistics(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
    diagnostics: _GrangerDiagnostics,
) -> _ModelStatistics:
    y, y_lags, x_lags = _lagged_arrays(
        target_values,
        target_values if variable_values is None else variable_values,
        lag,
    )
    predictors = y_lags if restricted else np.column_stack([y_lags, x_lags])
    diagnostics.matrix_build_count += 1
    ssr, rank = _ols_ssr_and_rank(predictors, y, diagnostics=diagnostics)
    return _ModelStatistics(
        nobs=len(y),
        ssr=ssr,
        rank=rank,
        target_scale=_target_variation_scale(y),
        fallback_used=True,
    )


def _safe_lstsq_model_statistics(
    target_values: np.ndarray,
    variable_values: np.ndarray | None,
    lag: int,
    *,
    restricted: bool,
    diagnostics: _GrangerDiagnostics,
) -> _ModelStatistics:
    try:
        return _lstsq_model_statistics(
            target_values,
            variable_values,
            lag,
            restricted=restricted,
            diagnostics=diagnostics,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return _invalid_model_statistics(target_values, lag)


def _invalid_model_statistics(target_values: np.ndarray, lag: int) -> _ModelStatistics:
    return _ModelStatistics(
        nobs=max(0, len(target_values) - lag),
        ssr=float("nan"),
        rank=0,
        target_scale=1.0,
        fallback_used=True,
    )


def _legacy_lag_statistics_path(
    target_values: np.ndarray,
    variable_values: np.ndarray,
    maxlag: int,
    diagnostics: _GrangerDiagnostics,
) -> dict[int, _LagStatistics]:
    output: dict[int, _LagStatistics] = {}
    for lag in range(1, maxlag + 1):
        restricted_stats = _safe_lstsq_model_statistics(
            target_values,
            variable_values,
            lag,
            restricted=True,
            diagnostics=diagnostics,
        )
        unrestricted_stats = _safe_lstsq_model_statistics(
            target_values,
            variable_values,
            lag,
            restricted=False,
            diagnostics=diagnostics,
        )
        output[lag] = _combine_model_statistics(
            lag,
            restricted_stats,
            unrestricted_stats,
            fallback_used=True,
        )
        diagnostics.fallback_count += 1
    return output


def _target_variation_scale(y: np.ndarray) -> float:
    if not np.isfinite(y).all():
        return 1.0
    centered = y - np.mean(y)
    return max(float(np.dot(centered, centered)), 1.0)


def _restricted_mask_key(target: str, mask: np.ndarray) -> _RestrictedCacheKey:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    return target, int(mask.size), packed.tobytes()


def _lagged_arrays(
    target_values: np.ndarray,
    variable_values: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    n = len(target_values)
    if n <= lag:
        return (
            np.empty(0, dtype=float),
            np.empty((0, lag), dtype=float),
            np.empty((0, lag), dtype=float),
        )

    y = target_values[lag:]
    y_lags = np.column_stack([target_values[lag - i : n - i] for i in range(1, lag + 1)])
    x_lags = np.column_stack([variable_values[lag - i : n - i] for i in range(1, lag + 1)])
    return y, y_lags, x_lags


def _ols_ssr_and_rank(
    x: np.ndarray,
    y: np.ndarray,
    *,
    diagnostics: _GrangerDiagnostics | None = None,
) -> tuple[float, int]:
    matrix = _add_intercept(x)
    if diagnostics is not None:
        diagnostics.lstsq_count += 1
    coef, _residuals, rank, _singular_values = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coef
    ssr = float(np.dot(residual, residual))
    return ssr, int(rank)


def _add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _is_near_perfect_fit(ssr: float, y: np.ndarray) -> bool:
    return _is_near_perfect_fit_from_scale(ssr, _target_variation_scale(y))


def _is_near_perfect_fit_from_scale(ssr: float, target_scale: float) -> bool:
    return ssr <= np.finfo(float).eps * max(target_scale, 1.0)


def _predictive_contribution(target: pd.Series, variable: pd.Series, lag: int) -> float:
    data = pd.DataFrame(
        {
            "target": target,
            "candidate": variable.shift(lag),
            "target_lag_1": target.shift(1),
        }
    ).dropna()
    if len(data) < 10:
        return 0.0
    y = data["target"].to_numpy()
    base = _linear_rmse(data[["target_lag_1"]], y)
    full = _linear_rmse(data[["target_lag_1", "candidate"]], y)
    return max(0.0, (base - full) / base) if base > 0 else 0.0


def _linear_rmse(x: pd.DataFrame, y: object) -> float:
    matrix = _add_intercept(x.to_numpy())
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    pred = matrix @ coef
    return float(np.sqrt(np.mean((y - pred) ** 2)))
