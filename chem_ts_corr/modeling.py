from __future__ import annotations

import numpy as np
import pandas as pd


def build_lag_features(
    frame: pd.DataFrame,
    target: str,
    max_lag: int,
    candidate_variables: list[str],
    max_features: int,
    best_lags: dict[str, int] | None = None,
    lag_mode: str = "best_only",
) -> tuple[pd.DataFrame, pd.Series]:
    feature_parts: list[pd.Series] = []
    for variable in candidate_variables:
        if variable == target:
            continue
        selected_lags = _selected_lags(best_lags.get(variable) if best_lags else None, max_lag, lag_mode)
        for lag in selected_lags:
            feature_parts.append(frame[variable].shift(lag).rename(f"{variable}__lag_{lag}"))

    if not feature_parts:
        return pd.DataFrame(index=frame.index), frame[target]
    features = pd.concat(feature_parts, axis=1)
    if features.shape[1] > max_features:
        features = features.iloc[:, :max_features]

    dataset = pd.concat([features, frame[target].rename(target)], axis=1).dropna()
    return dataset.drop(columns=[target]), dataset[target]


def fit_explainable_model(
    frame: pd.DataFrame,
    target: str,
    max_lag: int,
    candidate_variables: list[str],
    max_features: int,
    random_state: int,
    best_lags: dict[str, int] | None = None,
    lag_mode: str = "best_only",
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return pd.DataFrame(), {"model_status": "skipped: scikit-learn is not installed"}

    x, y = build_lag_features(
        frame,
        target,
        max_lag,
        candidate_variables,
        max_features,
        best_lags,
        lag_mode=lag_mode,
    )
    if x.empty or len(x) < 30:
        return pd.DataFrame(), {"model_status": "skipped: insufficient rows"}

    test_size = 0.25 if len(x) >= 80 else 0.2
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics: dict[str, float | str] = {
        "model_status": "ok",
        "r2_holdout": float(r2_score(y_test, predictions)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
    }

    importance = _try_shap_importance(model, x_train, random_state)
    if importance is None:
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "importance": model.feature_importances_,
                "method": "random_forest_feature_importance",
            }
        )

    importance["variable"] = importance["feature"].str.replace(r"__lag_\d+$", "", regex=True)
    importance["lag"] = importance["feature"].str.extract(r"__lag_(\d+)$").astype(float)
    return importance.sort_values("importance", ascending=False).reset_index(drop=True), metrics


def _try_shap_importance(
    model: object, x_train: pd.DataFrame, random_state: int
) -> pd.DataFrame | None:
    try:
        import shap  # type: ignore
    except Exception:
        return None

    sample = x_train.sample(min(len(x_train), 500), random_state=random_state)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    mean_abs = np.abs(values).mean(axis=0)
    return pd.DataFrame(
        {"feature": sample.columns, "importance": mean_abs, "method": "mean_abs_shap"}
    )


def _selected_lags(best_lag: int | None, max_lag: int, lag_mode: str) -> list[int]:
    if lag_mode == "best_only":
        return [_best_only_lag(best_lag, max_lag)]
    if lag_mode == "nearby":
        return _nearby_lags(best_lag, max_lag)
    raise ValueError('lag_mode must be "best_only" or "nearby"')


def _best_only_lag(best_lag: int | None, max_lag: int) -> int:
    if best_lag is None or pd.isna(best_lag):
        return 0
    return min(max_lag, max(0, int(abs(best_lag))))


def _nearby_lags(best_lag: int | None, max_lag: int, radius: int = 2) -> list[int]:
    if best_lag is None or pd.isna(best_lag):
        return list(range(0, min(max_lag, 6) + 1))
    center = max(0, int(abs(best_lag)))
    lower = max(0, center - radius)
    upper = min(max_lag, center + radius)
    return list(range(lower, upper + 1))
