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


def test_analyze_numeric_frame_reports_progress_and_skip_flags(tmp_path):
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
    assert "已跳过模型提升评分" in progress_messages
    assert "已跳过滚动稳定性评分" in progress_messages
    assert "正在生成候选排序" in progress_messages
    assert set(tables.model_lift_scores["status"]) == {"skipped: user disabled model lift scoring"}
    assert set(tables.rolling_corr_scores["valid_window_count"]) == {0}
    assert tables.metrics["skip_model_lift"] == "True"
    assert tables.metrics["skip_rolling_corr"] == "True"


def test_analyze_numeric_frame_passes_primary_lag_evidence_to_rolling(tmp_path, monkeypatch):
    from chem_ts_corr import screening

    n = 100
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
    )
    captured = {}
    original_prepare = screening.prepare_best_lag_evidence
    original_residual = screening.residual_corr_scores
    original_regime = screening.regime_scores
    original_rolling = screening.rolling_corr_scores

    def capture_prepare(frame, *args, ranked_source_frame=None, **kwargs):
        captured["frame"] = frame
        captured["ranked_source_frame"] = ranked_source_frame
        return original_prepare(
            frame,
            *args,
            ranked_source_frame=ranked_source_frame,
            **kwargs,
        )

    def capture_rolling(*args, best_lag_evidence=None, **kwargs):
        captured["evidence"] = best_lag_evidence
        return original_rolling(
            *args,
            best_lag_evidence=best_lag_evidence,
            **kwargs,
        )

    def capture_residual(*args, best_lags=None, **kwargs):
        captured["residual_best_lags"] = best_lags
        return original_residual(*args, best_lags=best_lags, **kwargs)

    def capture_regime(*args, best_lags=None, **kwargs):
        captured["regime_best_lags"] = best_lags
        return original_regime(*args, best_lags=best_lags, **kwargs)

    monkeypatch.setattr(screening, "prepare_best_lag_evidence", capture_prepare)
    monkeypatch.setattr(screening, "residual_corr_scores", capture_residual)
    monkeypatch.setattr(screening, "regime_scores", capture_regime)
    monkeypatch.setattr(screening, "rolling_corr_scores", capture_rolling)
    monkeypatch.setattr(
        screening,
        "compute_lag_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rolling path must reuse the primary lag search")
        ),
    )

    analyze_numeric_frame(frame, config)

    assert captured["evidence"]
    expected_best_lags = {
        variable: item["best_lag"] for variable, item in captured["evidence"].items()
    }
    assert captured["residual_best_lags"] == expected_best_lags
    assert captured["regime_best_lags"] == expected_best_lags
    assert captured["frame"] is captured["ranked_source_frame"]
    assert set(captured["evidence"]) == {"x1", "x2"}
    assert {item["source"] for item in captured["evidence"].values()} == {"ranked"}
    assert all("best_score" in item for item in captured["evidence"].values())
    assert all(
        item["pair_alignment_key"]
        == screening.pair_alignment_key(captured["frame"][["target", variable]].dropna())
        for variable, item in captured["evidence"].items()
    )


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

    def capture_scan(frame, target, max_lag, lag_values=None):
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


def test_preselection_keeps_high_raw_association_when_innovation_conflicts(tmp_path, monkeypatch):
    from chem_ts_corr import screening

    n = 100
    rng = np.random.default_rng(42)
    target = np.sin(np.arange(n) / 5)
    frame = pd.DataFrame(
        {
            "target": target,
            "conflict": target,
            "verified": rng.normal(size=n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )
    config = AnalysisConfig(
        input_path=tmp_path / "unused.csv",
        time_column="timestamp",
        target="target",
        output_dir=tmp_path,
        max_lag=4,
        top_k=1,
        enable_model=False,
        skip_rolling_corr=True,
    )
    captured: dict[str, list[str]] = {}

    def fake_innovation(frame, target, max_lag, raw_ranked, preprocess_mode):
        rows = []
        for _, row in raw_ranked.iterrows():
            conflict = row["variable"] == "conflict"
            rows.append(
                {
                    "variable": row["variable"],
                    "innovation_score": np.nan if conflict else row["score"],
                    "innovation_lag": row["lag"],
                    "innovation_direction": row["direction"],
                    "innovation_sign": 1,
                    "innovation_status": (
                        "innovation_lag_conflict" if conflict else "innovation_verified"
                    ),
                }
            )
        return pd.DataFrame(rows, columns=service.INNOVATION_COLUMNS)

    def capture_lift(frame, target, candidate_variables, max_lag, **kwargs):
        captured["candidate_variables"] = list(candidate_variables)
        return pd.DataFrame(
            [{"variable": variable, "status": "non_predictive_lag"} for variable in candidate_variables]
        )

    monkeypatch.setattr(service, "_innovation_evidence", fake_innovation)
    monkeypatch.setattr(screening, "model_lift_scores", capture_lift)

    analyze_numeric_frame(frame, config)

    assert captured["candidate_variables"] == ["conflict"]


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
