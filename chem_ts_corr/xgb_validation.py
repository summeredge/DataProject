from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DEFAULT_XGB_TOP_N = 8
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
        if self.model_name not in {"M0", "M1", "M2"}:
            raise ValueError("model_name must be one of M0, M1, M2")


@dataclass(frozen=True)
class XGBValidationResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame


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
    if model_name not in {"M0", "M1", "M2"}:
        raise ValueError("model_name must be one of M0, M1, M2")
    _require_non_empty_partition(X_train, y_train, fold, "train")
    _require_non_empty_partition(X_valid, y_valid, fold, "validation")
    _require_non_empty_partition(X_test, y_test, fold, "test")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1")

    model_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    model_params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    prediction = np.asarray(model.predict(X_test), dtype=float)
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
    return XGBValidationResult(fold_metrics=fold_metrics, summary=summary, predictions=predictions)


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
