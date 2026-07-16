import numpy as np
import pandas as pd

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
