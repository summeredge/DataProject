from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd

from chem_ts_corr.feature_alignment import (
    fit_tabular_model,
    predict_tabular_model,
)

from chem_ts_corr.time_axis import lagged_series, physical_gap_starts, sample_period_ns

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DEFAULT_XGB_TOP_N = 8
MAX_XGB_AUTO_TOP_N = 10
MAX_XGB_TOTAL_CANDIDATES = 12
MAX_XGB_LAG_POINTS = 5000
DEFAULT_BASELINE_LAGS = (1, 2, 5, 10, 30, 60)
DEFAULT_CANDIDATE_LAG_RADIUS = 2
DEFAULT_OUTER_SPLITS = 3
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "n_jobs": -1,
    "random_state": 42,
}
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_XGB_MIN_TRAIN_ROWS = 100
DEFAULT_XGB_MIN_VALIDATION_ROWS = 30
DEFAULT_XGB_MIN_TEST_ROWS = 30
XGB_TRAIN_MODEL_NAMES = frozenset({"M0", "M1", "M2", "CANDIDATE"})
XGB_VALIDATION_STATUS = frozenset(
    {
        "validated_incremental_signal",
        "weak_incremental_value",
        "redundant_with_baseline",
        "unstable_out_of_time",
        "insufficient_features",
    }
)

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
    "severe_data_quality",
    "target_leads_variable",
)
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
    "candidate_priority_rank",
    "candidate_priority_score",
    "candidate_priority_tier",
    "candidate_pool_rank",
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


@dataclass(frozen=True)
class XGBFoldMetric:
    fold: int
    model_name: str
    train_rows: int
    validation_rows: int
    test_rows: int
    best_iteration: int | None
    rmse: float
    mae: float
    r2: float

    def __post_init__(self) -> None:
        if self.model_name not in XGB_TRAIN_MODEL_NAMES:
            raise ValueError("model_name must be one of M0, M1, M2, CANDIDATE")


@dataclass(frozen=True)
class XGBValidationResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    provenance: XGBValidationProvenance | None = None


@dataclass(frozen=True)
class XGBValidationProvenance:
    m1_features: tuple[str, ...]
    split_signature: tuple[tuple[object, ...], ...]
    parameter_signature: tuple[tuple[str, str], ...]
    early_stopping_rounds: int
    data_fingerprint: str


@dataclass(frozen=True)
class CandidateUpliftMetric:
    variable: str
    fold: int
    train_rows: int
    validation_rows: int
    test_rows: int
    rmse: float
    mae: float
    r2: float
    baseline_rmse: float
    baseline_mae: float
    rmse_improvement_pct: float
    mae_improvement_pct: float
    best_iteration: int | None


@dataclass(frozen=True)
class CandidateUpliftSummary:
    variable: str
    fold_count: int
    positive_rmse_fold_count: int
    positive_mae_fold_count: int
    positive_rmse_fold_ratio: float
    median_rmse_improvement_pct: float
    median_mae_improvement_pct: float
    mean_rmse_improvement_pct: float
    mean_mae_improvement_pct: float
    worst_fold_rmse_improvement_pct: float
    validation_status: str

    def __post_init__(self) -> None:
        if self.validation_status not in XGB_VALIDATION_STATUS:
            raise ValueError("unknown candidate uplift validation_status")


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
    resolved_top_n = validate_xgb_top_n(top_n)

    summary = final_review_summary.copy(deep=True)
    ranked = ranked_features.copy(deep=True) if ranked_features is not None else pd.DataFrame()
    normalized_target = _text(target)
    summary["variable"] = summary["variable"].map(_text)
    summary = summary[summary["variable"].ne("")]
    summary = summary.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)
    ranked_by_variable = _rows_by_variable(ranked)
    controls = {_text(value) for value in control_columns or [] if _text(value)}

    rows: list[dict[str, object]] = []
    for _, source in summary.iterrows():
        row = _candidate_metadata(source.to_dict(), ranked_by_variable)
        reasons = _auto_exclusion_reasons(
            row, target=normalized_target, control_columns=controls
        )
        row["auto_eligible"] = not reasons
        row["auto_exclusion_reasons"] = ";".join(reasons)
        row["_source_order"] = len(rows)
        rows.append(row)

    selected_rows: list[dict[str, object]] = []
    if rows:
        candidates = pd.DataFrame(rows)
        candidates["_candidate_priority_sort"] = pd.to_numeric(
            candidates["candidate_priority_rank"], errors="coerce"
        )
        candidates["_final_rank_sort"] = pd.to_numeric(
            candidates["final_rank"], errors="coerce"
        )
        candidates["_candidate_pool_sort"] = pd.to_numeric(
            candidates["candidate_pool_rank"], errors="coerce"
        )
        all_ranks_missing = candidates[
            ["_candidate_priority_sort", "_final_rank_sort", "_candidate_pool_sort"]
        ].isna().all(axis=1)
        candidates["_fallback_source_order"] = candidates["_source_order"].where(
            all_ranks_missing
        )
        eligible = candidates[candidates["auto_eligible"]].sort_values(
            [
                "_candidate_priority_sort", "_final_rank_sort", "_candidate_pool_sort",
                "_fallback_source_order", "variable",
            ],
            na_position="last",
            kind="mergesort",
        )
        selected = eligible.head(resolved_top_n).copy()
        selected["selection_source"] = "final_review"
        selected["force_included"] = False
        selected_rows = selected.to_dict("records")

    selected_positions = {_text(row.get("variable")): index for index, row in enumerate(selected_rows)}
    summary_by_variable = _rows_by_variable(summary)
    for variable in _clean_whitelist(whitelist):
        if variable == normalized_target:
            continue
        if variable in selected_positions:
            row = selected_rows[selected_positions[variable]]
            row["selection_source"] = "final_review+whitelist"
            row["force_included"] = True
            continue

        source = summary_by_variable.get(variable, {"variable": variable})
        row = _candidate_metadata(source, ranked_by_variable)
        reasons = _auto_exclusion_reasons(
            row, target=normalized_target, control_columns=controls
        )
        row["auto_eligible"] = not reasons
        row["auto_exclusion_reasons"] = ";".join(reasons)
        row["selection_source"] = "whitelist"
        row["force_included"] = True
        selected_positions[variable] = len(selected_rows)
        selected_rows.append(row)

    if not selected_rows:
        return pd.DataFrame(columns=XGB_CANDIDATE_COLUMNS)
    result = pd.DataFrame(selected_rows)
    result["variable"] = result["variable"].map(_text)
    if result["variable"].eq("").any() or not result["variable"].is_unique:
        raise ValueError("XGB candidate variables must be non-empty and unique")
    _validate_total_xgb_candidate_count(result["variable"])
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
    if candidate_class in AUTO_EXCLUDED_CANDIDATE_CLASSES and not _is_legacy_poor_quality(row):
        reasons.append(candidate_class)
    risk_tokens = _risk_tokens(row.get("risk_flags"))
    reasons.extend(token for token in _AUTO_EXCLUDED_RISK_TOKEN_ORDER if token in risk_tokens)
    return tuple(reasons)


def _is_legacy_poor_quality(row: pd.Series | dict[str, object]) -> bool:
    risk_tokens = _risk_tokens(row.get("risk_flags"))
    return (
        _text(row.get("candidate_class")) == "poor_quality"
        and "poor_data_quality" in risk_tokens
        and "severe_data_quality" not in risk_tokens
    )


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
    max_lag = validate_xgb_max_lag(max_lag)
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
    max_lag = validate_xgb_max_lag(max_lag)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    lag = _integer(best_lag)
    if lag is None or lag <= 0:
        return ()
    return tuple(range(max(1, lag - radius), min(max_lag, lag + radius) + 1))


def resolve_xgb_max_used_lag(
    candidate_pool: pd.DataFrame,
    *,
    max_lag: int,
    baseline_lags: Sequence[int] = DEFAULT_BASELINE_LAGS,
    candidate_lag_radius: int = DEFAULT_CANDIDATE_LAG_RADIUS,
    available_columns: Sequence[str] | None = None,
) -> int:
    """Resolve the largest positive lag actually used by XGB feature sets.

    This is the fold ``gap`` value. It reuses ``normalize_positive_lags`` and
    ``candidate_lag_window`` so the result always matches the lag set that
    ``build_xgb_feature_sets`` would construct for the same inputs.
    """
    used = set(normalize_positive_lags([1, *baseline_lags], max_lag))
    available = None if available_columns is None else {_text(value) for value in available_columns}
    if (
        candidate_pool is not None
        and not candidate_pool.empty
        and "variable" in candidate_pool.columns
    ):
        for _, row in candidate_pool.iterrows():
            variable = _text(row.get("variable"))
            if available is not None and variable not in available:
                continue
            used.update(
                candidate_lag_window(
                    row.get("screening_lag"), max_lag, candidate_lag_radius
                )
            )
    return max(used) if used else 1


def validate_xgb_max_lag(max_lag: object) -> int:
    if isinstance(max_lag, bool) or not isinstance(max_lag, Integral):
        raise ValueError("max_lag must be an integer between 1 and 5000")
    resolved = int(max_lag)
    if not 1 <= resolved <= MAX_XGB_LAG_POINTS:
        raise ValueError("max_lag must be an integer between 1 and 5000")
    return resolved


def validate_xgb_top_n(top_n: object) -> int:
    if isinstance(top_n, bool) or not isinstance(top_n, Integral):
        raise ValueError("top_n must be an integer between 1 and 10")
    resolved = int(top_n)
    if not 1 <= resolved <= MAX_XGB_AUTO_TOP_N:
        raise ValueError("top_n must be an integer between 1 and 10")
    return resolved


def _validate_total_xgb_candidate_count(variables: Sequence[str]) -> None:
    if len(variables) > MAX_XGB_TOTAL_CANDIDATES:
        raise ValueError("XGB total candidate count including whitelist must not exceed 12")


def build_xgb_feature_sets(
    frame: pd.DataFrame,
    target: str,
    candidate_pool: pd.DataFrame,
    *,
    control_columns: list[str] | None = None,
    max_lag: int,
    baseline_lags: Sequence[int] = DEFAULT_BASELINE_LAGS,
    candidate_lag_radius: int = DEFAULT_CANDIDATE_LAG_RADIUS,
    target_mask: pd.Series | None = None,
) -> XGBFeatureSets:
    period_ns = sample_period_ns(frame)
    forced_starts = physical_gap_starts(frame)
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
            feature_data[name] = lagged_series(
                prepared[variable], prepared.index, lag, period_ns=period_ns, forced_starts=forced_starts
            )
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
    combined = pd.concat([all_features, prepared[target].rename("__target__")], axis=1)
    if target_mask is not None:
        resolved_mask = target_mask.reindex(combined.index).fillna(False).astype(bool)
        combined = combined.loc[resolved_mask]
    combined = combined.dropna()
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
    min_train_rows: int = DEFAULT_XGB_MIN_TRAIN_ROWS,
    min_validation_rows: int = DEFAULT_XGB_MIN_VALIDATION_ROWS,
    min_test_rows: int = DEFAULT_XGB_MIN_TEST_ROWS,
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


def train_xgb_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    fold: int,
    model_name: str,
    params: dict[str, object] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> tuple[XGBFoldMetric, np.ndarray]:
    _require_xgb_dependency()
    if model_name not in XGB_TRAIN_MODEL_NAMES:
        raise ValueError("model_name must be one of M0, M1, M2, CANDIDATE")
    _require_non_empty_partition(X_train, y_train, fold, "train")
    _require_non_empty_partition(X_valid, y_valid, fold, "validation")
    _require_non_empty_partition(X_test, y_test, fold, "test")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1")

    model_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    model_params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBRegressor(**model_params)
    fit_tabular_model(
        model,
        X_train,
        y_train,
        aligned_frames=(X_valid, X_test),
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    prediction = np.asarray(predict_tabular_model(model, X_test), dtype=float)
    truth = np.asarray(y_test, dtype=float)
    if prediction.shape != truth.shape:
        raise ValueError("XGB prediction length does not match test labels")

    best_iteration = getattr(model, "best_iteration", None)
    metric = XGBFoldMetric(
        fold=fold,
        model_name=model_name,
        train_rows=len(X_train),
        validation_rows=len(X_valid),
        test_rows=len(X_test),
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        rmse=float(np.sqrt(np.mean((truth - prediction) ** 2))),
        mae=float(np.mean(np.abs(truth - prediction))),
        r2=_r2_score(truth, prediction),
    )
    return metric, prediction


def run_xgb_time_validation(
    feature_sets: XGBFeatureSets,
    splits: list[XGBTimeSplit],
    *,
    params: dict[str, object] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> XGBValidationResult:
    _require_xgb_dependency()
    if not splits:
        raise ValueError("No time splits provided")
    if feature_sets.features.empty or not feature_sets.m2_features:
        raise ValueError("No valid XGB features available")
    if not feature_sets.features.index.equals(feature_sets.target.index):
        raise ValueError("XGB features and target must use the same index")

    model_features = {
        "M0": feature_sets.m0_features,
        "M1": feature_sets.m1_features,
        "M2": feature_sets.m2_features,
    }
    available = set(feature_sets.features.columns)
    if any(not columns or not set(columns).issubset(available) for columns in model_features.values()):
        raise ValueError("No valid XGB features available")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split in splits:
        target_train = feature_sets.target.iloc[split.train_slice]
        target_valid = feature_sets.target.iloc[split.validation_slice]
        target_test = feature_sets.target.iloc[split.test_slice]
        fold_predictions = pd.DataFrame(
            {
                "fold": split.fold,
                "timestamp_index": target_test.index,
                "y_true": target_test.to_numpy(),
            }
        )
        for model_name in ("M0", "M1", "M2"):
            columns = list(model_features[model_name])
            model_frame = feature_sets.features.loc[:, columns]
            metric, prediction = train_xgb_fold(
                model_frame.iloc[split.train_slice],
                target_train,
                model_frame.iloc[split.validation_slice],
                target_valid,
                model_frame.iloc[split.test_slice],
                target_test,
                fold=split.fold,
                model_name=model_name,
                params=params,
                early_stopping_rounds=early_stopping_rounds,
            )
            metric_rows.append(asdict(metric))
            fold_predictions[f"{model_name}_prediction"] = prediction
        prediction_frames.append(fold_predictions)

    metric_columns = [
        "fold", "model_name", "train_rows", "validation_rows", "test_rows",
        "best_iteration", "rmse", "mae", "r2",
    ]
    fold_metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    fold_metrics["_model_order"] = fold_metrics["model_name"].map({"M0": 0, "M1": 1, "M2": 2})
    fold_metrics = fold_metrics.sort_values(["fold", "_model_order"], kind="mergesort")
    fold_metrics = fold_metrics.drop(columns="_model_order").reset_index(drop=True)
    summary = _summarize_xgb_metrics(fold_metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.loc[
        :, ["fold", "timestamp_index", "y_true", "M0_prediction", "M1_prediction", "M2_prediction"]
    ]
    provenance = _xgb_validation_provenance(
        feature_sets, splits, params, early_stopping_rounds
    )
    return XGBValidationResult(
        fold_metrics=fold_metrics,
        summary=summary,
        predictions=predictions,
        provenance=provenance,
    )


def run_candidate_uplift_validation(
    feature_sets: XGBFeatureSets,
    splits: list[XGBTimeSplit],
    candidate_pool: pd.DataFrame,
    *,
    baseline_result: XGBValidationResult | None = None,
    params: dict[str, object] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = candidate_pool.copy(deep=True) if candidate_pool is not None else pd.DataFrame()
    candidates = _bounded_uplift_candidates(pool)
    _require_xgb_dependency()
    if not splits:
        raise ValueError("No time splits provided")
    if feature_sets.features.empty or not feature_sets.m1_features:
        raise ValueError("No valid XGB features available")
    if not feature_sets.features.index.equals(feature_sets.target.index):
        raise ValueError("XGB features and target must use the same index")

    available = set(feature_sets.features.columns)
    m1_features = tuple(feature_sets.m1_features)
    if not set(m1_features).issubset(available):
        raise ValueError("No valid XGB features available")

    valid_features: dict[str, tuple[str, ...]] = {}
    invalid_variables: list[str] = []
    for variable in candidates:
        mapped = feature_sets.candidate_feature_map.get(variable, ())
        added = tuple(
            feature for feature in _stable_unique(mapped) if feature in available and feature not in m1_features
        )
        if added:
            valid_features[variable] = added
        else:
            invalid_variables.append(variable)

    metric_rows: list[dict[str, object]] = []
    baseline_cache = (
        _baseline_metrics_from_result(
            baseline_result,
            feature_sets,
            splits,
            params,
            early_stopping_rounds,
        )
        if valid_features
        else {}
    )
    for split in splits:
        if not valid_features:
            break
        X_m1 = feature_sets.features.loc[:, m1_features]
        y_train = feature_sets.target.iloc[split.train_slice]
        y_valid = feature_sets.target.iloc[split.validation_slice]
        y_test = feature_sets.target.iloc[split.test_slice]
        if split.fold not in baseline_cache:
            baseline_metric, _ = train_xgb_fold(
                X_m1.iloc[split.train_slice],
                y_train,
                X_m1.iloc[split.validation_slice],
                y_valid,
                X_m1.iloc[split.test_slice],
                y_test,
                fold=split.fold,
                model_name="M1",
                params=params,
                early_stopping_rounds=early_stopping_rounds,
            )
            baseline_cache[split.fold] = baseline_metric

        for variable, added in valid_features.items():
            candidate_columns = (*m1_features, *added)
            candidate_frame = feature_sets.features.loc[:, candidate_columns]
            candidate_metric, _ = train_xgb_fold(
                candidate_frame.iloc[split.train_slice],
                y_train,
                candidate_frame.iloc[split.validation_slice],
                y_valid,
                candidate_frame.iloc[split.test_slice],
                y_test,
                fold=split.fold,
                model_name="CANDIDATE",
                params=params,
                early_stopping_rounds=early_stopping_rounds,
            )
            baseline = baseline_cache[split.fold]
            metric_rows.append(
                asdict(
                    CandidateUpliftMetric(
                        variable=variable,
                        fold=split.fold,
                        train_rows=candidate_metric.train_rows,
                        validation_rows=candidate_metric.validation_rows,
                        test_rows=candidate_metric.test_rows,
                        rmse=candidate_metric.rmse,
                        mae=candidate_metric.mae,
                        r2=candidate_metric.r2,
                        baseline_rmse=baseline.rmse,
                        baseline_mae=baseline.mae,
                        rmse_improvement_pct=_improvement_pct(baseline.rmse, candidate_metric.rmse),
                        mae_improvement_pct=_improvement_pct(baseline.mae, candidate_metric.mae),
                        best_iteration=candidate_metric.best_iteration,
                    )
                )
            )

    metric_columns = list(CandidateUpliftMetric.__dataclass_fields__)
    metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    summary = summarize_candidate_uplift(metrics)
    if invalid_variables:
        summary = pd.DataFrame(
            [
                *summary.to_dict("records"),
                *(asdict(_insufficient_uplift_summary(variable)) for variable in invalid_variables),
            ],
            columns=CandidateUpliftSummary.__dataclass_fields__,
        )
    if not summary.empty:
        summary["_candidate_order"] = summary["variable"].map(
            {variable: order for order, variable in enumerate(candidates)}
        )
        summary = summary.sort_values(
            ["median_rmse_improvement_pct", "_candidate_order"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        ).drop(columns="_candidate_order").reset_index(drop=True)
    return metrics, summary


def _xgb_validation_provenance(
    feature_sets: XGBFeatureSets,
    splits: list[XGBTimeSplit],
    params: dict[str, object] | None,
    early_stopping_rounds: int,
) -> XGBValidationProvenance:
    m1_features = tuple(feature_sets.m1_features)
    available = set(feature_sets.features.columns)
    if not m1_features or not set(m1_features).issubset(available):
        raise ValueError("No valid XGB features available")
    effective_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    parameter_signature = tuple(
        sorted((str(key), _parameter_value_signature(value)) for key, value in effective_params.items())
    )
    split_signature = tuple(
        (
            split.fold,
            split.gap,
            split.train_slice.start,
            split.train_slice.stop,
            split.train_slice.step,
            split.validation_slice.start,
            split.validation_slice.stop,
            split.validation_slice.step,
            split.test_slice.start,
            split.test_slice.stop,
            split.test_slice.step,
        )
        for split in splits
    )
    fingerprint = _xgb_data_fingerprint(
        feature_sets.features.loc[:, m1_features], feature_sets.target
    )
    return XGBValidationProvenance(
        m1_features=m1_features,
        split_signature=split_signature,
        parameter_signature=parameter_signature,
        early_stopping_rounds=early_stopping_rounds,
        data_fingerprint=fingerprint,
    )


def _xgb_data_fingerprint(features: pd.DataFrame, target: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(repr(tuple((str(column), str(features[column].dtype)) for column in features)).encode())
    digest.update(pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes())
    digest.update(repr((str(target.name), str(target.dtype))).encode())
    digest.update(pd.util.hash_pandas_object(target, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _parameter_value_signature(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        items = sorted((str(key), _parameter_value_signature(item)) for key, item in value.items())
        return repr(tuple(items))
    if isinstance(value, (list, tuple)):
        return repr(tuple(_parameter_value_signature(item) for item in value))
    return f"{type(value).__name__}:{value!r}"


def _baseline_metrics_from_result(
    baseline_result: XGBValidationResult | None,
    feature_sets: XGBFeatureSets,
    splits: list[XGBTimeSplit],
    params: dict[str, object] | None,
    early_stopping_rounds: int,
) -> dict[int, XGBFoldMetric]:
    if baseline_result is None:
        return {}
    expected_provenance = _xgb_validation_provenance(
        feature_sets, splits, params, early_stopping_rounds
    )
    if baseline_result.provenance != expected_provenance:
        raise ValueError("baseline_result provenance does not match current XGB validation inputs")
    frame = baseline_result.fold_metrics.copy(deep=True)
    required = {
        "fold", "model_name", "train_rows", "validation_rows", "test_rows",
        "best_iteration", "rmse", "mae", "r2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"baseline_result fold_metrics missing columns: {', '.join(sorted(missing))}")
    predictions = baseline_result.predictions.copy(deep=True)
    prediction_columns = {"fold", "timestamp_index", "y_true", "M1_prediction"}
    missing_predictions = prediction_columns.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(
            "baseline_result predictions missing columns: "
            + ", ".join(sorted(missing_predictions))
        )

    cache: dict[int, XGBFoldMetric] = {}
    fold_values = pd.to_numeric(frame["fold"], errors="coerce")
    prediction_folds = pd.to_numeric(predictions["fold"], errors="coerce")
    for split in splits:
        rows = frame[frame["model_name"].astype(str).eq("M1") & fold_values.eq(split.fold)]
        if len(rows) != 1:
            raise ValueError(f"baseline_result requires exactly one M1 metric for fold {split.fold}")
        row = rows.iloc[0]
        expected_rows = (
            len(feature_sets.target.iloc[split.train_slice]),
            len(feature_sets.target.iloc[split.validation_slice]),
            len(feature_sets.target.iloc[split.test_slice]),
        )
        actual_rows = (
            _required_int(row.get("train_rows"), "train_rows", split.fold),
            _required_int(row.get("validation_rows"), "validation_rows", split.fold),
            _required_int(row.get("test_rows"), "test_rows", split.fold),
        )
        if actual_rows != expected_rows:
            raise ValueError(f"baseline_result row counts do not match fold {split.fold} slices")
        fold_predictions = predictions[prediction_folds.eq(split.fold)]
        expected_target = feature_sets.target.iloc[split.test_slice]
        if fold_predictions["timestamp_index"].tolist() != expected_target.index.tolist():
            raise ValueError(f"baseline_result test index does not match fold {split.fold}")
        baseline_truth = pd.to_numeric(fold_predictions["y_true"], errors="coerce").to_numpy()
        current_truth = pd.to_numeric(expected_target, errors="coerce").to_numpy()
        if not np.array_equal(baseline_truth, current_truth, equal_nan=True):
            raise ValueError(f"baseline_result y_true does not match fold {split.fold}")
        baseline_prediction = pd.to_numeric(
            fold_predictions["M1_prediction"], errors="coerce"
        ).to_numpy()
        recomputed_rmse = float(np.sqrt(np.mean((baseline_truth - baseline_prediction) ** 2)))
        recomputed_mae = float(np.mean(np.abs(baseline_truth - baseline_prediction)))
        recomputed_r2 = _r2_score(baseline_truth, baseline_prediction)
        metric_rmse = _required_float(row.get("rmse"), "rmse", split.fold)
        metric_mae = _required_float(row.get("mae"), "mae", split.fold)
        metric_r2 = _required_float(row.get("r2"), "r2", split.fold)
        for field, metric_value, recomputed_value in [
            ("rmse", metric_rmse, recomputed_rmse),
            ("mae", metric_mae, recomputed_mae),
            ("r2", metric_r2, recomputed_r2),
        ]:
            if not np.isclose(metric_value, recomputed_value, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"baseline_result {field} does not match M1_prediction for fold {split.fold}"
                )
        cache[split.fold] = XGBFoldMetric(
            fold=split.fold,
            model_name="M1",
            train_rows=actual_rows[0],
            validation_rows=actual_rows[1],
            test_rows=actual_rows[2],
            best_iteration=_integer(row.get("best_iteration")),
            rmse=metric_rmse,
            mae=metric_mae,
            r2=metric_r2,
        )
    return cache


def summarize_candidate_uplift(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = list(CandidateUpliftSummary.__dataclass_fields__)
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "variable", "rmse_improvement_pct", "mae_improvement_pct",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"candidate uplift metrics missing columns: {', '.join(sorted(missing))}")

    source = metrics.copy(deep=True)
    rows: list[dict[str, object]] = []
    for variable in _stable_unique(source["variable"].astype(str).tolist()):
        group = source[source["variable"].astype(str).eq(variable)]
        rmse_values = pd.to_numeric(group["rmse_improvement_pct"], errors="coerce").dropna()
        mae_values = pd.to_numeric(group["mae_improvement_pct"], errors="coerce").dropna()
        fold_count = int(len(group))
        positive_rmse = int((rmse_values > 0).sum())
        negative_rmse = int((rmse_values < 0).sum())
        positive_mae = int((mae_values > 0).sum())
        ratio = positive_rmse / fold_count if fold_count else 0.0
        median_rmse = float(rmse_values.median())
        median_mae = float(mae_values.median())
        mean_rmse = float(rmse_values.mean())
        mean_mae = float(mae_values.mean())
        worst_rmse = float(rmse_values.min())
        status = _candidate_validation_status(
            fold_count=fold_count,
            positive_rmse_fold_count=positive_rmse,
            negative_rmse_fold_count=negative_rmse,
            positive_rmse_fold_ratio=ratio,
            median_rmse_improvement_pct=median_rmse,
            median_mae_improvement_pct=median_mae,
            worst_fold_rmse_improvement_pct=worst_rmse,
        )
        rows.append(
            asdict(
                CandidateUpliftSummary(
                    variable=variable,
                    fold_count=fold_count,
                    positive_rmse_fold_count=positive_rmse,
                    positive_mae_fold_count=positive_mae,
                    positive_rmse_fold_ratio=ratio,
                    median_rmse_improvement_pct=median_rmse,
                    median_mae_improvement_pct=median_mae,
                    mean_rmse_improvement_pct=mean_rmse,
                    mean_mae_improvement_pct=mean_mae,
                    worst_fold_rmse_improvement_pct=worst_rmse,
                    validation_status=status,
                )
            )
        )
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(
        "median_rmse_improvement_pct", ascending=False, kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _candidate_validation_status(
    *,
    fold_count: int,
    positive_rmse_fold_count: int,
    negative_rmse_fold_count: int,
    positive_rmse_fold_ratio: float,
    median_rmse_improvement_pct: float,
    median_mae_improvement_pct: float,
    worst_fold_rmse_improvement_pct: float,
) -> str:
    if (
        positive_rmse_fold_count > 0
        and negative_rmse_fold_count > 0
        and worst_fold_rmse_improvement_pct < 0
    ):
        return "unstable_out_of_time"
    if (
        fold_count >= 2
        and positive_rmse_fold_ratio >= 0.67
        and median_rmse_improvement_pct > 0
        and median_mae_improvement_pct >= 0
    ):
        return "validated_incremental_signal"
    if median_rmse_improvement_pct > 0:
        return "weak_incremental_value"
    return "redundant_with_baseline"


def _insufficient_uplift_summary(variable: str) -> CandidateUpliftSummary:
    return CandidateUpliftSummary(
        variable=variable,
        fold_count=0,
        positive_rmse_fold_count=0,
        positive_mae_fold_count=0,
        positive_rmse_fold_ratio=0.0,
        median_rmse_improvement_pct=float("nan"),
        median_mae_improvement_pct=float("nan"),
        mean_rmse_improvement_pct=float("nan"),
        mean_mae_improvement_pct=float("nan"),
        worst_fold_rmse_improvement_pct=float("nan"),
        validation_status="insufficient_features",
    )


def _bounded_uplift_candidates(candidate_pool: pd.DataFrame) -> list[str]:
    if candidate_pool.empty or "variable" not in candidate_pool.columns:
        return []
    source = candidate_pool.copy(deep=True)
    source["_source_order"] = range(len(source))
    source["_candidate_order"] = pd.to_numeric(
        source.get("candidate_order", pd.Series(index=source.index, dtype=float)),
        errors="coerce",
    )
    source = source.sort_values(
        ["_candidate_order", "_source_order"], kind="mergesort", na_position="last"
    )
    selected: list[str] = []
    automatic_count = 0
    for _, row in source.iterrows():
        variable = _text(row.get("variable"))
        if not variable or variable in selected:
            continue
        force_included = row.get("force_included") is True or str(
            row.get("force_included", "")
        ).strip().lower() in {"1", "true", "yes"}
        if force_included or automatic_count < MAX_XGB_AUTO_TOP_N:
            selected.append(variable)
            if not force_included:
                automatic_count += 1
    _validate_total_xgb_candidate_count(selected)
    return selected


def _summarize_xgb_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in ("M0", "M1", "M2"):
        metrics = fold_metrics[fold_metrics["model_name"].eq(model_name)]
        rows.append(
            {
                "model_name": model_name,
                "mean_rmse": float(metrics["rmse"].mean()),
                "median_rmse": float(metrics["rmse"].median()),
                "mean_mae": float(metrics["mae"].mean()),
                "median_mae": float(metrics["mae"].median()),
                "mean_r2": float(metrics["r2"].mean()),
                "fold_count": int(len(metrics)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["M2_vs_M1_rmse_improvement_pct"] = np.nan
    summary["M2_vs_M1_mae_improvement_pct"] = np.nan
    indexed = summary.set_index("model_name")
    m1_rmse = float(indexed.loc["M1", "mean_rmse"])
    m2_rmse = float(indexed.loc["M2", "mean_rmse"])
    m1_mae = float(indexed.loc["M1", "mean_mae"])
    m2_mae = float(indexed.loc["M2", "mean_mae"])
    m2_mask = summary["model_name"].eq("M2")
    summary.loc[m2_mask, "M2_vs_M1_rmse_improvement_pct"] = _improvement_pct(m1_rmse, m2_rmse)
    summary.loc[m2_mask, "M2_vs_M1_mae_improvement_pct"] = _improvement_pct(m1_mae, m2_mae)
    return summary


def _require_xgb_dependency() -> None:
    if XGBRegressor is None:
        raise RuntimeError(
            'xgboost is not installed.\nInstall optional dependency:\npip install -e ".[xgb]"'
        )


def _require_non_empty_partition(
    features: pd.DataFrame, target: pd.Series, fold: int, partition: str
) -> None:
    if len(features) == 0 or len(target) == 0:
        raise ValueError(f"fold {fold} has no {partition} rows")
    if len(features) != len(target):
        raise ValueError(f"fold {fold} {partition} features and target row counts differ")


def _r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    residual_sum = float(np.sum((truth - prediction) ** 2))
    total_sum = float(np.sum((truth - np.mean(truth)) ** 2))
    if total_sum == 0:
        return 1.0 if residual_sum == 0 else 0.0
    return 1.0 - residual_sum / total_sum


def _improvement_pct(baseline: float, candidate: float) -> float:
    if baseline == 0 or not np.isfinite(baseline) or not np.isfinite(candidate):
        return float("nan")
    return (baseline - candidate) / baseline * 100.0


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
        "candidate_priority_rank": _coalesce(
            ranked.get("candidate_priority_rank"), source.get("candidate_priority_rank")
        ),
        "candidate_priority_score": _coalesce(
            ranked.get("candidate_priority_score"), source.get("candidate_priority_score")
        ),
        "candidate_priority_tier": _coalesce(
            ranked.get("candidate_priority_tier"), source.get("candidate_priority_tier")
        ),
        "candidate_pool_rank": _coalesce(
            ranked.get("candidate_pool_rank"), source.get("candidate_pool_rank")
        ),
    }


def _rows_by_variable(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty or "variable" not in frame.columns:
        return {}
    rows: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        variable = _text(row.get("variable"))
        if variable and variable not in rows:
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


def _required_int(value: object, field: str, fold: int) -> int:
    number = _integer(value)
    if number is None:
        raise ValueError(f"baseline_result {field} is invalid for fold {fold}")
    return number


def _required_float(value: object, field: str, fold: int) -> float:
    number = _number(value)
    if number is None:
        raise ValueError(f"baseline_result {field} is invalid for fold {fold}")
    return number
