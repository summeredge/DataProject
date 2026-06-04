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
