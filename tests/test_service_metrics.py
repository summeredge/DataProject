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
