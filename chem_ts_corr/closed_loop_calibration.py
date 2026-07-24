from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


CALIBRATION_RESULT_COLUMNS = [
    "variable",
    "auto_closed_loop_probability",
    "calibration_status",
    "training_label",
    "label_source",
    "prediction_time",
]
FEATURE_COLUMNS = [
    "best_lag",
    "lag_quality",
    "lag_boundary_flag",
    "rolling_stability",
    "regime_consistency_score",
    "regime_sign_consistency",
    "regime_lag_consistency",
    "model_lift_score",
    "prediction_score",
    "closed_loop_suspect_flag",
    "risk_count",
    "diagnosis_status_score",
    "confidence_level_score",
]
POSITIVE_LABEL = "engineering_input_closed_loop"
NEGATIVE_LABEL = "engineering_input_not_closed_loop"


def build_training_labels(
    manual_closed_loop_variables: list[str] | None,
    manual_non_closed_loop_variables: list[str] | None,
) -> pd.DataFrame:
    """Create labels only from explicit human closed-loop decisions."""
    labels: dict[str, tuple[str, str]] = {}
    for variable in manual_closed_loop_variables or []:
        labels[str(variable)] = (POSITIVE_LABEL, "manual_closed_loop")
    for variable in manual_non_closed_loop_variables or []:
        variable = str(variable)
        labels[variable] = (NEGATIVE_LABEL, "manual_non_closed_loop") if variable not in labels else ("unknown", "unknown")
    return pd.DataFrame(
        [{"variable": variable, "training_label": label, "label_source": source} for variable, (label, source) in labels.items()],
        columns=["variable", "training_label", "label_source"],
    )


def build_calibration_features(
    ranked_features: pd.DataFrame,
    risk_flags: pd.DataFrame | None,
    lag_peak_quality: pd.DataFrame | None,
    rolling_corr_scores: pd.DataFrame | None,
    regime_scores: pd.DataFrame | None,
    model_lift_scores: pd.DataFrame | None,
    auto_diagnosis: pd.DataFrame | None,
) -> pd.DataFrame:
    features = ranked_features[["variable"]].drop_duplicates().copy() if "variable" in ranked_features.columns else pd.DataFrame(columns=["variable"])
    sources = [risk_flags, lag_peak_quality, rolling_corr_scores, regime_scores, model_lift_scores]
    for source in sources:
        if source is not None and not source.empty and "variable" in source.columns:
            features = features.merge(source.drop_duplicates("variable"), on="variable", how="left", suffixes=("", "_source"))
    if auto_diagnosis is not None and not auto_diagnosis.empty and "mv_variable" in auto_diagnosis.columns:
        diagnosis = auto_diagnosis.rename(columns={"mv_variable": "variable"}).drop_duplicates("variable")
        features = features.merge(diagnosis[[column for column in ["variable", "diagnosis_status", "confidence_level"] if column in diagnosis.columns]], on="variable", how="left")
    for column in ["lag_boundary_flag", "closed_loop_suspect_flag"]:
        features[column] = features.get(column, pd.Series(False, index=features.index)).eq(True).astype(float)
    features["diagnosis_status_score"] = features.get("diagnosis_status", pd.Series("", index=features.index)).map({"confirmed": 1.0, "possible": 0.5, "not_supported": 0.0}).fillna(0.0)
    features["confidence_level_score"] = features.get("confidence_level", pd.Series("", index=features.index)).map({"high": 1.0, "medium": 0.5, "low": 0.0}).fillna(0.0)
    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return features[["variable", *FEATURE_COLUMNS]]


def run_closed_loop_calibration(
    output_dir: Path,
    manual_closed_loop_variables: list[str] | None,
    manual_non_closed_loop_variables: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Train and apply a shadow-only logistic calibration model from saved results."""
    read = lambda name: _read_csv(output_dir / name)
    features = build_calibration_features(
        read("ranked_features.csv"), read("risk_flags.csv"), read("lag_peak_quality.csv"),
        read("rolling_corr_scores.csv"), read("regime_scores.csv"), read("model_lift_scores.csv"),
        read("auto_closed_loop_diagnosis.csv"),
    )
    labels = build_training_labels(
        manual_closed_loop_variables,
        manual_non_closed_loop_variables,
    )
    merged = features.merge(labels, on="variable", how="left")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    labelled = merged[merged["training_label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    metrics: dict[str, object] = {
        "evaluation_mode": "training_only",
        "sample_count": int(len(labelled)),
        "positive_count": int((labelled["training_label"] == POSITIVE_LABEL).sum()),
        "negative_count": int((labelled["training_label"] == NEGATIVE_LABEL).sum()),
    }
    model: dict[str, object]
    if labelled["training_label"].nunique() < 2:
        status = "insufficient_training_labels"
        probabilities = pd.Series(np.nan, index=merged.index)
        metrics.update({"status": status, "accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None})
        model = {"status": status, "feature_list": FEATURE_COLUMNS, "training_time": timestamp, **metrics}
    else:
        x_train = labelled[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = (labelled["training_label"] == POSITIVE_LABEL).astype(int).to_numpy()
        estimator = LogisticRegression(random_state=0, max_iter=200).fit(x_train, y_train)
        probabilities = pd.Series(estimator.predict_proba(merged[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1], index=merged.index)
        train_probability = estimator.predict_proba(x_train)[:, 1]
        train_prediction = (train_probability >= 0.5).astype(int)
        metrics.update({
            "status": "trained",
            "accuracy": float(accuracy_score(y_train, train_prediction)),
            "precision": float(precision_score(y_train, train_prediction, zero_division=0)),
            "recall": float(recall_score(y_train, train_prediction, zero_division=0)),
            "f1": float(f1_score(y_train, train_prediction, zero_division=0)),
            "auc": float(roc_auc_score(y_train, train_probability)) if len(y_train) >= 4 else None,
        })
        model = {
            "status": "trained",
            "feature_list": FEATURE_COLUMNS,
            "coefficients": dict(zip(FEATURE_COLUMNS, estimator.coef_[0].tolist(), strict=True)),
            "intercept": float(estimator.intercept_[0]),
            "training_time": timestamp,
            **metrics,
        }
        status = "calibrated"
    results = pd.DataFrame({
        "variable": merged["variable"],
        "auto_closed_loop_probability": probabilities,
        "calibration_status": status,
        "training_label": merged["training_label"].fillna("unknown"),
        "label_source": merged["label_source"].fillna("unknown"),
        "prediction_time": timestamp,
    }, columns=CALIBRATION_RESULT_COLUMNS)
    results.to_csv(output_dir / "closed_loop_calibration_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "closed_loop_calibration_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "closed_loop_calibration_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, model, metrics


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
