import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import service
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.service import analyze_numeric_frame


def test_analyze_numeric_frame_metrics_include_max_lag(tmp_path):
    n = 80
    frame = pd.DataFrame(
        {
            "target": np.sin(np.arange(n) / 5),
            "x1": np.cos(np.arange(n) / 5),
            "x2": np.arange(n, dtype=float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=7,
        top_k=2,
        enable_model=False,
    )

    tables = analyze_numeric_frame(frame, config)

    assert tables.metrics["max_lag"] == 7.0
    assert tables.metrics["top_k"] == 2.0


def test_primary_segment_mask_keeps_cross_segment_lag_source(tmp_path):
    rows = 240
    rng = np.random.default_rng(20260720)
    target = pd.Series(rng.normal(size=rows))
    frame = pd.DataFrame(
        {
            "target": target.to_numpy(),
            "x": target.shift(-1).to_numpy(),
            "load": np.resize([0.0, 1.0], rows),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="5min"),
    ).dropna()
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        segment_column="load",
        segment_mode="custom",
        segment_min=0.0,
        segment_max=0.0,
        max_lag=2,
        top_k=1,
        enable_model=False,
        skip_model_lift=True,
        skip_rolling_corr=True,
    )

    tables = analyze_numeric_frame(frame, config)
    best = tables.lag_scores.loc[tables.lag_scores["variable"].eq("x")]
    best = best.loc[best[["abs_pearson", "abs_spearman"]].max(axis=1).idxmax()]

    assert int(best["lag"]) == 1
    assert float(best["abs_pearson"]) == pytest.approx(1.0)
    assert tables.metrics["rows_after_segment"] == 120.0
    assert tables.metrics["rows_after_preprocess"] == 120.0


def test_analyze_numeric_frame_reports_initial_stage_progress(tmp_path):
    n = 80
    frame = pd.DataFrame(
        {
            "target": np.sin(np.arange(n) / 5),
            "x1": np.cos(np.arange(n) / 5),
            "x2": np.arange(n, dtype=float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=4,
        top_k=2,
        enable_model=False,
        skip_model_lift=True,
        skip_rolling_corr=True,
    )
    progress_messages: list[str] = []

    tables = analyze_numeric_frame(frame, config, progress_callback=progress_messages.append)

    assert "预处理中" in progress_messages
    assert "正在计算滞后相关" in progress_messages
    assert "正在计算 V5 支持证据" in progress_messages
    assert "正在生成候选排序" in progress_messages
    assert tables.model_lift_scores.empty
    assert tables.rolling_corr_scores.empty
    assert "skip_model_lift" not in tables.metrics
    assert "skip_rolling_corr" not in tables.metrics


def test_initial_stage_executes_only_formal_v5_support_not_followup_analyses(tmp_path, monkeypatch):
    from chem_ts_corr import screening

    n = 100
    frame = pd.DataFrame(
        {"target": np.sin(np.arange(n) / 5), "x1": np.cos(np.arange(n) / 5)},
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv", time_column="timestamp", target="target",
        output_dir=tmp_path, max_lag=4, top_k=2,
    )
    calls: list[str] = []
    for name in ["prepare_best_lag_evidence", "residual_corr_scores", "regime_scores", "rolling_corr_scores", "model_lift_scores"]:
        monkeypatch.setattr(screening, name, lambda *args, _name=name, **kwargs: calls.append(_name))

    tables = analyze_numeric_frame(frame, config)

    assert calls == ["prepare_best_lag_evidence", "rolling_corr_scores"]
    assert tables.model_lift_scores.empty
    assert tables.rolling_corr_scores.empty
    assert "model_status" not in tables.metrics
    assert "granger_status" not in tables.metrics


def test_innovation_peak_in_opposite_direction_is_not_verified(monkeypatch):
    captured: dict[str, list[int]] = {}
    raw_ranked = pd.DataFrame(
        [
            {
                "variable": "x",
                "lag": 1,
                "direction": "variable leads target",
                "score": 0.90,
                "method": "pearson",
                "pearson": 0.90,
                "spearman": 0.80,
            }
        ]
    )

    def capture_scan(frame, target, max_lag, lag_values=None, target_mask=None):
        captured["lag_values"] = list(lag_values or [])
        return pd.DataFrame({"placeholder": [1]})

    monkeypatch.setattr(service, "compute_lag_scores", capture_scan)
    monkeypatch.setattr(
        service,
        "summarize_best_lags",
        lambda scores: pd.DataFrame(
            [
                {
                    "variable": "x",
                    "lag": -1,
                    "direction": "target leads variable",
                    "score": 0.85,
                    "method": "pearson",
                    "pearson": 0.85,
                    "spearman": 0.75,
                }
            ]
        ),
    )

    result = service._innovation_evidence(
        pd.DataFrame({"target": range(20), "x": range(20)}),
        "target",
        6,
        raw_ranked,
        "raw",
    ).iloc[0]

    assert captured["lag_values"] == [-1, 0, 1, 2, 3]
    assert result["innovation_lag"] == -1
    assert result["innovation_direction"] == "target leads variable"
    assert result["innovation_sign"] == 1
    assert result["innovation_status"] == "innovation_lag_conflict"
    assert pd.isna(result["innovation_score"])


def test_innovation_difference_does_not_cross_physical_gap(monkeypatch):
    complete_index = pd.date_range("2026-01-01", periods=20, freq="5min")
    frame = pd.DataFrame(
        {"target": np.arange(20, dtype=float), "x": np.arange(20, dtype=float)},
        index=complete_index,
    ).drop(index=complete_index[10])
    after_gap = complete_index[11]
    captured: dict[str, pd.Index] = {}
    raw_ranked = pd.DataFrame(
        [{"variable": "x", "lag": 1, "direction": "variable leads target", "score": 0.9}]
    )

    def capture_scan(frame, *args, **kwargs):
        captured["index"] = frame.index
        return pd.DataFrame({"placeholder": [1]})

    monkeypatch.setattr(service, "compute_lag_scores", capture_scan)
    monkeypatch.setattr(
        service,
        "summarize_best_lags",
        lambda scores: pd.DataFrame(
            [{"variable": "x", "lag": 1, "direction": "variable leads target", "score": 0.8}]
        ),
    )

    service._innovation_evidence(frame, "target", 3, raw_ranked, "raw")

    assert after_gap not in captured["index"]


def test_initial_screening_keeps_innovation_evidence_in_score(tmp_path, monkeypatch):
    n = 100
    target = np.sin(np.arange(n) / 5)
    frame = pd.DataFrame(
        {"target": target, "conflict": target, "verified": np.random.default_rng(42).normal(size=n)},
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv", time_column="timestamp", target="target",
        output_dir=tmp_path, max_lag=4, top_k=1,
    )
    calls: list[str] = []
    original_innovation = service._innovation_evidence

    def capture_innovation(*args, **kwargs):
        calls.append("innovation")
        return original_innovation(*args, **kwargs)

    monkeypatch.setattr(service, "_innovation_evidence", capture_innovation)
    tables = analyze_numeric_frame(frame, config)

    assert calls == ["innovation"]
    assert not tables.ranked_features.empty
    assert "innovation_score" in tables.ranked_features.columns


@pytest.mark.parametrize(
    "extra_columns",
    [
        {},
        {"constant": np.ones(80)},
    ],
    ids=["target_only", "all_candidates_filtered"],
)
def test_analyze_numeric_frame_returns_empty_schema_when_no_candidates(tmp_path, extra_columns):
    frame = pd.DataFrame(
        {"target": np.sin(np.arange(80) / 5), **extra_columns},
        index=pd.date_range("2025-01-01", periods=80, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=4,
        top_k=2,
        enable_model=False,
    )

    tables = analyze_numeric_frame(frame, config)

    assert tables.ranked_features.empty
    assert {"variable", "driver_rank", "driver_priority_score"}.issubset(
        tables.ranked_features.columns
    )
    assert tables.lag_scores.empty
    assert {"variable", "lag", "abs_pearson", "abs_spearman"}.issubset(
        tables.lag_scores.columns
    )
