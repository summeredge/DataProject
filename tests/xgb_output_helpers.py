"""Complete the four fold files in mocked XGB success bundles, without training."""

import json
from pathlib import Path

import pandas as pd

from chem_ts_corr.xgb_validation import (
    CANDIDATE_FOLD_METRICS_COLUMNS,
    CandidateUpliftMetric,
    XGBFoldMetric,
    XGBTimeSplit,
    build_candidate_fold_metrics,
    build_xgb_fold_context,
)


def write_fold_outputs(directory: Path, improvements=None) -> None:
    model = pd.read_csv(directory / "xgb_model_summary.csv")
    uplift = pd.read_csv(directory / "xgb_candidate_uplift.csv")
    fold_count = int(model.iloc[0]["fold_count"])
    time = pd.date_range("2026-01-01", periods=400, freq="min")
    splits = [XGBTimeSplit(fold, slice(0, 100 + fold * 40),
                          slice(112 + fold * 40, 142 + fold * 40),
                          slice(154 + fold * 40, 194 + fold * 40), 12)
              for fold in range(fold_count)]
    indexes = {split.fold: (time[split.train_slice], time[split.validation_slice], time[split.test_slice])
               for split in splits}
    context = build_xgb_fold_context(indexes, splits, max_used_lag=12)
    metrics, candidates, predictions = [], [], []
    baseline = model[model["model_name"].eq("M1")].iloc[0]
    for coverage in context.to_dict("records"):
        fold = coverage["fold"]
        for row in model.to_dict("records"):
            metrics.append({"fold": fold, "model_name": row["model_name"],
                            **{key: coverage[key] for key in ("train_rows", "validation_rows", "test_rows")},
                            "best_iteration": 1, "rmse": row["mean_rmse"], "mae": row["mean_mae"],
                            "r2": row["mean_r2"]})
        for row in uplift.to_dict("records"):
            if row["validation_status"] == "insufficient_features":
                continue
            rmse, mae = (improvements[row["variable"]][fold] if improvements is not None
                         else (row["mean_rmse_improvement_pct"], row["mean_mae_improvement_pct"]))
            candidates.append({"variable": row["variable"], "fold": fold,
                               **{key: coverage[key] for key in ("train_rows", "validation_rows", "test_rows")},
                               "rmse": baseline["mean_rmse"] * (1 - rmse / 100),
                               "mae": baseline["mean_mae"] * (1 - mae / 100), "r2": 0,
                               "baseline_rmse": baseline["mean_rmse"], "baseline_mae": baseline["mean_mae"],
                               "rmse_improvement_pct": rmse, "mae_improvement_pct": mae, "best_iteration": 1})
        predictions.extend({"fold": fold, "timestamp_index": timestamp, "y_true": 1,
                            "M0_prediction": 1, "M1_prediction": 1, "M2_prediction": 1}
                           for timestamp in indexes[fold][2])
    context.to_csv(directory / "xgb_fold_context.csv", index=False)
    pd.DataFrame(metrics, columns=XGBFoldMetric.__dataclass_fields__).to_csv(
        directory / "xgb_fold_metrics.csv", index=False)
    details = build_candidate_fold_metrics(
        pd.DataFrame(candidates, columns=CandidateUpliftMetric.__dataclass_fields__), indexes, fold_context=context)
    assert list(details.columns) == list(CANDIDATE_FOLD_METRICS_COLUMNS)
    details.to_csv(directory / "xgb_candidate_fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(directory / "xgb_predictions.csv", index=False)
    path = directory / "xgb_validation_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary.setdefault("fold_count", fold_count)
    path.write_text(json.dumps(summary), encoding="utf-8")
