from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_XGB_TOP_N = 8
DEFAULT_BASELINE_LAGS = (1, 2, 5, 10, 30, 60)
DEFAULT_CANDIDATE_LAG_RADIUS = 2
DEFAULT_OUTER_SPLITS = 3
DEFAULT_VALIDATION_FRACTION = 0.15

AUTO_ALLOWED_RECOMMENDATIONS = frozenset(
    {
        "priority_review",
        "priority_review_with_statistical_limit",
        "secondary_review",
        "secondary_review_with_statistical_limit",
        "risk_limited_review",
    }
)
_AUTO_EXCLUDED_RISK_TOKEN_ORDER = (
    "strong_formula_leakage",
    "poor_data_quality",
    "target_leads_variable",
)
AUTO_EXCLUDED_RISK_TOKENS = frozenset(_AUTO_EXCLUDED_RISK_TOKEN_ORDER)
AUTO_EXCLUDED_CANDIDATE_CLASSES = frozenset(
    {"downstream_response", "formula_or_derived", "poor_quality"}
)

XGB_CANDIDATE_COLUMNS = (
    "candidate_order",
    "variable",
    "selection_source",
    "force_included",
    "auto_eligible",
    "auto_exclusion_reasons",
    "final_rank",
    "final_recommendation",
    "screening_lag",
    "candidate_class",
    "risk_flags",
    "recommended_use",
    "variable_role",
)


@dataclass(frozen=True)
class XGBFeatureSets:
    features: pd.DataFrame
    target: pd.Series
    m0_features: tuple[str, ...]
    m1_features: tuple[str, ...]
    m2_features: tuple[str, ...]
    candidate_feature_map: dict[str, tuple[str, ...]]
    max_used_lag: int


@dataclass(frozen=True)
class XGBTimeSplit:
    fold: int
    train_slice: slice
    validation_slice: slice
    test_slice: slice
    gap: int


def build_xgb_candidate_pool(
    final_review_summary: pd.DataFrame,
    ranked_features: pd.DataFrame | None = None,
    *,
    target: str,
    top_n: int = DEFAULT_XGB_TOP_N,
    whitelist: list[str] | None = None,
    control_columns: list[str] | None = None,
) -> pd.DataFrame:
    message = "XGBoost fourth-level validation requires final_review_summary"
    if final_review_summary is None or final_review_summary.empty:
        raise ValueError(f"{message}: input is empty")
    if "variable" not in final_review_summary.columns:
        raise ValueError(f"{message}: missing variable column")
    if "final_recommendation" not in final_review_summary.columns:
        raise ValueError(f"{message}: missing final_recommendation column")

    summary = final_review_summary.copy(deep=True)
    ranked = ranked_features.copy(deep=True) if ranked_features is not None else pd.DataFrame()
    summary = summary.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)
    ranked_by_variable = _rows_by_variable(ranked)
    controls = {_text(value) for value in control_columns or [] if _text(value)}

    rows: list[dict[str, object]] = []
    for _, source in summary.iterrows():
        row = _candidate_metadata(source.to_dict(), ranked_by_variable)
        reasons = _auto_exclusion_reasons(row, target=target, control_columns=controls)
        row["auto_eligible"] = not reasons
        row["auto_exclusion_reasons"] = ";".join(reasons)
        row["_source_order"] = len(rows)
        rows.append(row)

    candidates = pd.DataFrame(rows)
    rank_sort = pd.to_numeric(candidates["final_rank"], errors="coerce")
    candidates["_rank_missing"] = rank_sort.isna()
    candidates["_rank_sort"] = rank_sort
    eligible = candidates[candidates["auto_eligible"]].sort_values(
        ["_rank_missing", "_rank_sort", "_source_order"], kind="mergesort"
    )
    selected = eligible.head(max(0, int(top_n))).copy()
    selected["selection_source"] = "final_review"
    selected["force_included"] = False

    selected_rows = selected.to_dict("records")
    selected_positions = {_text(row.get("variable")): index for index, row in enumerate(selected_rows)}
    summary_by_variable = _rows_by_variable(summary)
    for variable in _clean_whitelist(whitelist):
        if variable == target:
            continue
        if variable in selected_positions:
            row = selected_rows[selected_positions[variable]]
            row["selection_source"] = "final_review+whitelist"
            row["force_included"] = True
            continue

        source = summary_by_variable.get(variable, {"variable": variable})
        row = _candidate_metadata(source, ranked_by_variable)
        reasons = _auto_exclusion_reasons(row, target=target, control_columns=controls)
        row["auto_eligible"] = not reasons
        row["auto_exclusion_reasons"] = ";".join(reasons)
        row["selection_source"] = "whitelist"
        row["force_included"] = True
        selected_positions[variable] = len(selected_rows)
        selected_rows.append(row)

    if not selected_rows:
        return pd.DataFrame(columns=XGB_CANDIDATE_COLUMNS)
    result = pd.DataFrame(selected_rows)
    result["candidate_order"] = range(1, len(result) + 1)
    return result.loc[:, XGB_CANDIDATE_COLUMNS].reset_index(drop=True)


def _auto_exclusion_reasons(
    row: pd.Series | dict[str, object],
    *,
    target: str,
    control_columns: set[str],
) -> tuple[str, ...]:
    variable = _text(row.get("variable"))
    reasons: list[str] = []
    if not variable:
        reasons.append("empty_variable")
    if variable == target:
        reasons.append("target_variable")
    if _text(row.get("final_recommendation")) not in AUTO_ALLOWED_RECOMMENDATIONS:
        reasons.append("recommendation_not_eligible")
    lag = _number(row.get("screening_lag"))
    if lag is None or lag <= 0:
        reasons.append("non_positive_screening_lag")
    if variable in control_columns:
        reasons.append("control_variable")
    if _text(row.get("recommended_use")) == "control_variable_reference":
        reasons.append("control_reference")
    candidate_class = _text(row.get("candidate_class"))
    if candidate_class in AUTO_EXCLUDED_CANDIDATE_CLASSES:
        reasons.append(candidate_class)
    risk_tokens = _risk_tokens(row.get("risk_flags"))
    reasons.extend(token for token in _AUTO_EXCLUDED_RISK_TOKEN_ORDER if token in risk_tokens)
    return tuple(reasons)


def prepare_xgb_validation_frame(
    frame: pd.DataFrame,
    target: str,
    candidate_pool: pd.DataFrame,
    control_columns: list[str] | None = None,
) -> pd.DataFrame:
    if target not in frame.columns:
        raise ValueError(f"target column not found: {target}")
    source = frame.copy(deep=True)
    pool = candidate_pool.copy(deep=True) if candidate_pool is not None else pd.DataFrame()
    candidate_variables = pool["variable"].tolist() if "variable" in pool.columns else []
    requested = [target, *(control_columns or []), *candidate_variables]
    columns = [column for column in _stable_unique(requested) if column in source.columns]
    prepared = source.loc[:, columns].copy()
    for column in prepared.columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    if not prepared.index.is_monotonic_increasing:
        prepared = prepared.sort_index(kind="mergesort")
    return prepared


def normalize_positive_lags(lags: Sequence[int], max_lag: int) -> tuple[int, ...]:
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    valid = {
        int(lag)
        for lag in lags
        if isinstance(lag, Integral) and not isinstance(lag, bool) and 1 <= int(lag) <= max_lag
    }
    return tuple(sorted(valid))


def candidate_lag_window(
    best_lag: object,
    max_lag: int,
    radius: int = DEFAULT_CANDIDATE_LAG_RADIUS,
) -> tuple[int, ...]:
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    lag = _integer(best_lag)
    if lag is None or lag <= 0:
        return ()
    return tuple(range(max(1, lag - radius), min(max_lag, lag + radius) + 1))


def build_xgb_feature_sets(
    frame: pd.DataFrame,
    target: str,
    candidate_pool: pd.DataFrame,
    *,
    control_columns: list[str] | None = None,
    max_lag: int,
    baseline_lags: Sequence[int] = DEFAULT_BASELINE_LAGS,
    candidate_lag_radius: int = DEFAULT_CANDIDATE_LAG_RADIUS,
) -> XGBFeatureSets:
    prepared = prepare_xgb_validation_frame(frame, target, candidate_pool, control_columns)
    pool = candidate_pool.copy(deep=True) if candidate_pool is not None else pd.DataFrame()
    baseline = normalize_positive_lags([1, *baseline_lags], max_lag)
    controls = [
        column
        for column in _stable_unique(control_columns or [])
        if column != target and column in prepared.columns
    ]

    feature_data: dict[str, pd.Series] = {}
    used_lags: list[int] = []

    def add_feature(variable: str, lag: int) -> str:
        name = f"{variable}__lag_{lag}"
        if name not in feature_data:
            feature_data[name] = prepared[variable].shift(lag)
            used_lags.append(lag)
        return name

    m0 = (add_feature(target, 1),)
    m1_list = [add_feature(target, lag) for lag in baseline]
    for variable in controls:
        m1_list.extend(add_feature(variable, lag) for lag in baseline)
    m1 = tuple(_stable_unique(m1_list))

    m2_list = list(m1)
    candidate_map: dict[str, tuple[str, ...]] = {}
    if not pool.empty and "variable" in pool.columns:
        pool["_candidate_order"] = pd.to_numeric(
            pool.get("candidate_order", pd.Series(index=pool.index, dtype=float)), errors="coerce"
        )
        pool["_source_order"] = range(len(pool))
        pool = pool.sort_values(["_candidate_order", "_source_order"], kind="mergesort", na_position="last")
        for _, row in pool.iterrows():
            variable = _text(row.get("variable"))
            if not variable or variable in candidate_map:
                continue
            added: list[str] = []
            if variable in prepared.columns:
                for lag in candidate_lag_window(row.get("screening_lag"), max_lag, candidate_lag_radius):
                    name = f"{variable}__lag_{lag}"
                    if name not in m2_list:
                        add_feature(variable, lag)
                        m2_list.append(name)
                        added.append(name)
            candidate_map[variable] = tuple(added)

    m2 = tuple(m2_list)
    all_features = pd.DataFrame(feature_data, index=prepared.index).loc[:, m2]
    combined = pd.concat([all_features, prepared[target].rename("__target__")], axis=1).dropna()
    features = combined.loc[:, m2]
    target_series = combined["__target__"].rename(target)
    return XGBFeatureSets(
        features=features,
        target=target_series,
        m0_features=m0,
        m1_features=m1,
        m2_features=m2,
        candidate_feature_map=candidate_map,
        max_used_lag=max(used_lags),
    )


def build_expanding_time_splits(
    n_samples: int,
    *,
    n_splits: int = DEFAULT_OUTER_SPLITS,
    gap: int = 0,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    min_train_rows: int = 100,
    min_validation_rows: int = 30,
    min_test_rows: int = 30,
) -> list[XGBTimeSplit]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    for name, value in [
        ("min_train_rows", min_train_rows),
        ("min_validation_rows", min_validation_rows),
        ("min_test_rows", min_test_rows),
    ]:
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    test_size = n_samples // (n_splits + 1)
    if test_size < min_test_rows:
        raise ValueError(f"test_size {test_size} is below min_test_rows {min_test_rows}")
    first_test_start = n_samples - n_splits * test_size
    splits: list[XGBTimeSplit] = []
    for fold in range(n_splits):
        test_start = first_test_start + fold * test_size
        test_end = n_samples if fold == n_splits - 1 else test_start + test_size
        validation_end = test_start - gap
        validation_size = max(min_validation_rows, int(validation_end * validation_fraction))
        validation_start = validation_end - validation_size
        train_end = validation_start - gap

        if train_end < min_train_rows:
            raise ValueError(
                f"fold {fold} train rows {max(0, train_end)} are below min_train_rows {min_train_rows}"
            )
        if validation_start < 0 or validation_end - validation_start < min_validation_rows:
            raise ValueError(f"fold {fold} has insufficient validation rows")
        if test_end - test_start < min_test_rows:
            raise ValueError(f"fold {fold} has insufficient test rows")
        if train_end > validation_start or validation_end > test_start:
            raise ValueError(f"fold {fold} contains overlapping time slices")
        if validation_start - train_end != gap or test_start - validation_end != gap:
            raise ValueError(f"fold {fold} does not preserve the requested gap")

        splits.append(
            XGBTimeSplit(
                fold=fold,
                train_slice=slice(0, train_end),
                validation_slice=slice(validation_start, validation_end),
                test_slice=slice(test_start, test_end),
                gap=gap,
            )
        )
    return splits


def _candidate_metadata(
    source: dict[str, object], ranked_by_variable: dict[str, dict[str, object]]
) -> dict[str, object]:
    variable = _text(source.get("variable"))
    ranked = ranked_by_variable.get(variable, {})
    return {
        "variable": variable,
        "final_rank": source.get("final_rank"),
        "final_recommendation": source.get("final_recommendation"),
        "screening_lag": _coalesce(source.get("screening_lag"), ranked.get("lag")),
        "candidate_class": _coalesce(source.get("candidate_class"), ranked.get("candidate_class")),
        "risk_flags": _coalesce(source.get("risk_flags"), ranked.get("risk_flags"), ""),
        "recommended_use": _coalesce(source.get("recommended_use"), ranked.get("recommended_use")),
        "variable_role": _coalesce(
            ranked.get("variable_role"), ranked.get("role"), source.get("variable_role")
        ),
    }


def _rows_by_variable(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty or "variable" not in frame.columns:
        return {}
    rows: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        variable = _text(row.get("variable"))
        if variable not in rows:
            rows[variable] = row.to_dict()
    return rows


def _risk_tokens(value: object) -> set[str]:
    text = _text(value)
    return {token.strip() for token in text.split(";") if token.strip()}


def _clean_whitelist(values: list[str] | None) -> list[str]:
    return _stable_unique(_text(value) for value in values or [] if _text(value))


def _stable_unique(values: Sequence[object] | object) -> list:
    result: list = []
    seen: set = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _coalesce(*values: object) -> object:
    for value in values:
        try:
            if pd.notna(value):
                return value
        except (TypeError, ValueError):
            continue
    return pd.NA


def _text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)
