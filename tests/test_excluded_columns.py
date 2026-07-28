from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import drop_excluded_columns
from chem_ts_corr.pipeline import run_analysis


def _input_frame(rows: int = 120) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "target": np.sin(values / 8) + values / 200,
            "keep_a": np.sin((values - 2) / 8),
            "keep_b": np.cos(values / 11),
            "drop_a": np.sin((values - 1) / 8),
            "drop_b": np.cos((values - 3) / 11),
        }
    )


def _config(tmp_path: Path, *, excluded_columns: list[str]) -> AnalysisConfig:
    input_path = tmp_path / "input.csv"
    _input_frame().to_csv(input_path, index=False, encoding="utf-8-sig")
    return AnalysisConfig(
        input_path=input_path,
        time_column="time",
        target="target",
        output_dir=tmp_path / "run",
        max_lag=3,
        top_k=10,
        excluded_columns=excluded_columns,
    )


def test_drop_excluded_columns_is_ordered_non_mutating_and_preserves_attrs():
    frame = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    frame.attrs["source"] = "original"

    result = drop_excluded_columns(frame, [" B ", "", "B", "C"])

    assert list(result.columns) == ["A"]
    assert list(frame.columns) == ["A", "B", "C"]
    assert result is not frame
    assert result.attrs == {"source": "original"}


def test_drop_excluded_columns_empty_selection_returns_independent_copy():
    frame = pd.DataFrame({"A": [1], "B": [2]})

    result = drop_excluded_columns(frame, [])

    pd.testing.assert_frame_equal(result, frame)
    assert result is not frame


def test_drop_excluded_columns_rejects_protected_and_unknown_columns():
    frame = pd.DataFrame({"target": [1], "A": [2]})

    with pytest.raises(ValueError, match="剔除列与受保护参数冲突：target"):
        drop_excluded_columns(frame, ["target"], protected_columns=["target"])
    with pytest.raises(ValueError, match="剔除列不存在：missing"):
        drop_excluded_columns(frame, ["missing"])


def test_main_analysis_removes_columns_from_every_variable_result(tmp_path: Path):
    config = _config(tmp_path, excluded_columns=["drop_a", "drop_b"])

    run_analysis(config)

    for filename in [
        "diagnostics.csv",
        "lag_scores.csv",
        "residual_corr_scores.csv",
        "regime_scores.csv",
        "risk_flags.csv",
        "ranked_features.csv",
        "model_lift_scores.csv",
        "rolling_corr_scores.csv",
    ]:
        result = pd.read_csv(config.output_dir / filename, encoding="utf-8-sig")
        assert "excluded_columns" not in result.columns
        if "variable" in result.columns:
            assert not {"drop_a", "drop_b"}.intersection(result["variable"].astype(str))


def test_all_web_reload_paths_keep_excluded_columns_out(tmp_path: Path):
    config = _config(tmp_path, excluded_columns=["drop_a", "drop_b"])

    numeric = web._numeric_frame(config)
    secondary = web._prepared_frame_for_secondary(config)
    validation = web._prepared_frame_for_validation(config)
    scaled = web._scaled_frame_for_secondary(config)

    for frame in [numeric, secondary, validation, scaled]:
        assert "drop_a" not in frame.columns
        assert "drop_b" not in frame.columns
        assert {"target", "keep_a", "keep_b"}.issubset(frame.columns)


def test_explicit_secondary_columns_cannot_reintroduce_excluded_column(tmp_path: Path):
    config = _config(tmp_path, excluded_columns=["drop_a"])

    with pytest.raises(ValueError, match="二次验证补充变量.*drop_a"):
        web._secondary_config_from_form(
            config, {"secondary_include_variables": "keep_a,drop_a"}
        )
    with pytest.raises(ValueError, match="受保护参数冲突：drop_a"):
        web._prepared_frame_for_secondary(config, protected_columns=["drop_a"])


def test_xgb_whitelist_cannot_reintroduce_excluded_column(tmp_path: Path):
    config = _config(tmp_path, excluded_columns=["drop_a"])

    with pytest.raises(ValueError, match="XGBoost 白名单.*drop_a"):
        web._ensure_columns_not_excluded(config, ["keep_a", "drop_a"], "XGBoost 白名单")


def test_primary_request_validation_rejects_protected_and_overbroad_exclusions(
    tmp_path: Path,
):
    config = _config(tmp_path, excluded_columns=[])
    common = {
        "input_path": config.input_path,
        "encoding": config.encoding,
        "time_column": "time",
        "target": "target",
        "segment_column": None,
        "capacity_columns": [],
        "residual_control_columns": [],
        "force_include_variables": [],
    }

    with pytest.raises(ValueError, match="剔除列不能同时作为目标列：target"):
        web._validate_analysis_excluded_columns(
            **common, excluded_columns=["target"]
        )
    with pytest.raises(ValueError, match="剔除后至少需要保留一个可分析数值候选列"):
        web._validate_analysis_excluded_columns(
            **common,
            excluded_columns=["keep_a", "keep_b", "drop_a", "drop_b"],
        )


def test_scaled_frame_cache_key_isolated_by_excluded_columns(tmp_path: Path):
    config_a = _config(tmp_path, excluded_columns=[])
    config_b = AnalysisConfig(
        **{**config_a.__dict__, "excluded_columns": ["drop_a"]}
    )
    config_c = AnalysisConfig(
        **{**config_a.__dict__, "excluded_columns": ["drop_b"]}
    )
    config_b_same = AnalysisConfig(
        **{**config_a.__dict__, "excluded_columns": ["drop_a"]}
    )

    assert web._scaled_frame_cache_key(config_a) != web._scaled_frame_cache_key(config_b)
    assert web._scaled_frame_cache_key(config_b) != web._scaled_frame_cache_key(config_c)
    assert web._scaled_frame_cache_key(config_b) == web._scaled_frame_cache_key(config_b_same)


def test_run_config_round_trip_preserves_excluded_columns(tmp_path: Path):
    config = _config(tmp_path, excluded_columns=["drop_a", "drop_b"])

    web._write_run_config(config.output_dir, config, "file-id")
    restored = web._read_run_config(config.output_dir)

    assert restored.excluded_columns == ["drop_a", "drop_b"]


def test_second_initial_analysis_with_one_of_eleven_variables_excluded_keeps_alignment(
    tmp_path: Path,
):
    rows = 180
    rng = np.random.default_rng(20260722)
    variables = {
        f"x{index}": np.cumsum(rng.normal(size=rows))
        for index in range(11)
    }
    target = sum(
        (index + 1) / 11 * np.roll(values, 1)
        for index, values in enumerate(variables.values())
    ) + rng.normal(scale=0.05, size=rows)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "target": target,
            **variables,
        }
    )
    input_path = tmp_path / "eleven_variables.csv"
    frame.to_csv(input_path, index=False, encoding="utf-8-sig")

    common = dict(
        input_path=input_path,
        time_column="time",
        target="target",
        max_lag=3,
        top_k=11,
        enable_model=True,
        skip_rolling_corr=False,
    )
    first = AnalysisConfig(
        **common,
        output_dir=tmp_path / "all_variables",
        excluded_columns=[],
    )
    second = AnalysisConfig(
        **common,
        output_dir=tmp_path / "one_excluded",
        excluded_columns=["x10"],
    )

    run_analysis(first)
    run_analysis(second)

    ranked = pd.read_csv(second.output_dir / "ranked_features.csv", encoding="utf-8-sig")
    assert not ranked.empty
    assert "x10" not in set(ranked["variable"].astype(str))
    assert web._scaled_frame_cache_key(first) != web._scaled_frame_cache_key(second)
    assert (second.output_dir / "model_lift_scores.csv").read_text(encoding="utf-8-sig").splitlines() == ["variable"]
