from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import _causal_best_lags


def _config(tmp_path: Path, *, mode: str) -> AnalysisConfig:
    return AnalysisConfig(
        input_path=tmp_path / "input.csv",
        time_column="time",
        target="target",
        output_dir=tmp_path,
        preprocess_mode=mode,
        lowpass_tau_minutes=7.5,
        diff_interval_minutes=5.0 if mode == "lowpass_diff" else None,
        detrend_window=8,
        max_interpolate_gap_points=2,
    )


def _write(config: AnalysisConfig, frame: pd.DataFrame) -> None:
    table = frame.copy()
    table.insert(0, config.time_column, frame.index)
    table.to_csv(config.input_path, index=False, encoding=config.encoding)


def test_causal_secondary_lowpass_detrend_prefix_ignores_future_suffix(
    tmp_path, monkeypatch
):
    from chem_ts_corr import preprocess

    monkeypatch.setattr(preprocess, "standardize_frame", lambda frame, fit_mask=None: frame)
    index = pd.date_range("2026-01-01", periods=60, freq="min")
    base = pd.DataFrame(
        {
            "target": np.sin(np.arange(60) / 5),
            "predictor": np.cos(np.arange(60) / 7),
        },
        index=index,
    )
    changed = base.copy()
    changed.loc[index[45]:, "predictor"] = 1_000_000.0
    config = _config(tmp_path, mode="lowpass_detrend")

    _write(config, base)
    web._clear_scaled_frame_cache()
    first = web._scaled_frame_for_secondary_causal(config)
    _write(config, changed)
    web._clear_scaled_frame_cache()
    second = web._scaled_frame_for_secondary_causal(config)

    pd.testing.assert_frame_equal(first.loc[: index[44]], second.loc[: index[44]])


def test_causal_secondary_predictor_gap_uses_only_past_value(tmp_path):
    index = pd.date_range("2026-01-01", periods=12, freq="min")
    predictor = np.arange(12, dtype=float)
    predictor[0] = 1.0
    predictor[1] = np.nan
    predictor[2] = 1000.0
    config = _config(tmp_path, mode="raw")
    _write(
        config,
        pd.DataFrame(
            {"target": np.arange(12, dtype=float), "predictor": predictor},
            index=index,
        ),
    )

    scaled = web._scaled_frame_for_secondary_causal(config)

    assert scaled.loc[index[1], "predictor"] == pytest.approx(
        scaled.loc[index[0], "predictor"]
    )


def test_formal_runners_use_causal_helper_not_legacy_helper():
    source = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")
    runners = [
        "run_enhanced_screening_for_active_branch",
        "run_granger_for_active_branch",
        "run_model_for_active_branch",
        "run_causal_review_for_active_branch",
    ]
    for name in runners:
        body = source.split(f"def {name}", 1)[1].split("\ndef ", 1)[0]
        assert "_scaled_frame_for_secondary_causal(" in body
        assert "_scaled_frame_for_secondary(" not in body


def test_causal_lag_recalculation_preserves_negative_direction():
    index = pd.date_range("2026-01-01", periods=40, freq="min")
    target = pd.Series(np.sin(np.arange(40) / 2.3), index=index)
    frame = pd.DataFrame({"target": target, "predictor": target.shift(2)}, index=index)

    best_lags = _causal_best_lags(
        frame,
        "target",
        ["predictor"],
        3,
        target_mask=None,
    )

    assert best_lags["predictor"] == -2
