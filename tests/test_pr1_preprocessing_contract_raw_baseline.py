from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.config import (
    CONTRACT_PREPROCESS_MODES,
    NOT_IMPLEMENTED_PREPROCESS_MODES,
    NOT_WIRED_ANALYSIS_PREPROCESS_MODES,
    SUPPORTED_PREPROCESS_MODES,
    AnalysisConfig,
)
from chem_ts_corr.preprocess import transform_frame, transform_frame_causal
from chem_ts_corr.report import write_outputs
from chem_ts_corr.service import analyze_numeric_frame


# 固定合成数据上的 Raw 主筛查回归基线（PR-1）。
# 生成方式与 test_initial_screening_contract.py 的完整初筛用例一致：
# target = sin(t/7)，候选变量 = sin((t+i+1)/7)，控制变量 = cos((t+i+1)/7)，
# 共 120 行、1 分钟采样，max_lag=3、top_k=15。
RAW_BASELINE_VARIABLES = [
    "candidate_0",
    "candidate_1",
    "control_7",
    "candidate_2",
    "candidate_3",
    "candidate_4",
    "control_6",
    "control_5",
    "control_4",
    "control_3",
    "control_2",
    "control_1",
    "control_0",
]

RAW_BASELINE = {
    "final_score": {
        "candidate_0": 0.9928057204912689,
        "candidate_1": 0.9928057204912689,
        "control_7": 0.9928057204912689,
        "candidate_2": 0.9928057204912689,
        "candidate_3": 0.9830826444525599,
        "candidate_4": 0.954698946775768,
        "control_6": 0.25,
        "control_5": 0.25,
        "control_4": 0.25,
        "control_3": 0.25,
        "control_2": 0.25,
        "control_1": 0.25,
        "control_0": 0.2460085982371364,
    },
    "driver_rank": {
        "candidate_0": 1,
        "candidate_1": 2,
        "control_7": 3,
        "candidate_2": 4,
        "candidate_3": 5,
        "candidate_4": 6,
        "control_6": 7,
        "control_5": 8,
        "control_4": 9,
        "control_3": 10,
        "control_2": 11,
        "control_1": 12,
        "control_0": 13,
    },
    "lag": {
        "candidate_0": 1,
        "candidate_1": 2,
        "control_7": -3,
        "candidate_2": 3,
        "candidate_3": 3,
        "candidate_4": 3,
        "control_6": -3,
        "control_5": -3,
        "control_4": -3,
        "control_3": -3,
        "control_2": -3,
        "control_1": -3,
        "control_0": -3,
    },
    "candidate_grade": {
        "candidate_0": "A",
        "candidate_1": "A",
        "control_7": "A",
        "candidate_2": "A",
        "candidate_3": "A",
        "candidate_4": "A",
        "control_6": "E",
        "control_5": "E",
        "control_4": "E",
        "control_3": "E",
        "control_2": "E",
        "control_1": "E",
        "control_0": "E",
    },
    "risk_flags": {
        "candidate_0": "redundant_proxy",
        "candidate_1": "redundant_proxy",
        "control_7": "lag_boundary",
        "candidate_2": "redundant_proxy;lag_boundary",
        "candidate_3": "lag_boundary",
        "candidate_4": "lag_boundary",
        "control_6": "target_leads_variable;lag_boundary",
        "control_5": "target_leads_variable;lag_boundary",
        "control_4": "target_leads_variable;lag_boundary",
        "control_3": "target_leads_variable;lag_boundary",
        "control_2": "target_leads_variable;lag_boundary",
        "control_1": "target_leads_variable;lag_boundary",
        "control_0": "target_leads_variable;lag_boundary",
    },
}

# 推荐顺序与来源按 prioritize_recommended_candidates 的实际输出锁定。
RAW_RECOMMENDED = [
    {"variable": "candidate_2", "candidate_source": "raw_and_residual", "candidate_pool_rank": 3},
    {"variable": "candidate_0", "candidate_source": "raw_and_residual", "candidate_pool_rank": 1},
    {"variable": "candidate_3", "candidate_source": "raw_and_residual", "candidate_pool_rank": 4},
    {"variable": "candidate_1", "candidate_source": "raw_and_residual", "candidate_pool_rank": 2},
    {"variable": "candidate_4", "candidate_source": "raw_and_residual", "candidate_pool_rank": 5},
]

# top_k=5 时固定 Raw 排名前五项（含控制参考变量 control_7）。
RAW_TOP_K_5 = [
    "candidate_0",
    "candidate_1",
    "control_7",
    "candidate_2",
    "candidate_3",
]

NON_FINITE_VALUES = [float("nan"), float("inf"), -float("inf")]


def _raw_frame() -> pd.DataFrame:
    rows = 120
    time = np.arange(rows, dtype=float)
    controls = [f"control_{index}" for index in range(8)]
    candidates = [f"candidate_{index}" for index in range(5)]
    return pd.DataFrame(
        {
            "target": np.sin(time / 7),
            **{name: np.sin((time + index + 1) / 7) for index, name in enumerate(candidates)},
            **{name: np.cos((time + index + 1) / 7) for index, name in enumerate(controls)},
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="min"),
    )


def _raw_config(tmp_path: Path, **overrides) -> AnalysisConfig:
    controls = [f"control_{index}" for index in range(8)]
    kwargs = {
        "input_path": tmp_path / "input.csv",
        "time_column": "time",
        "target": "target",
        "output_dir": tmp_path,
        "max_lag": 3,
        "top_k": 15,
        "residual_control_columns": controls,
        "force_include_variables": [],
        "enable_model": False,
        "skip_model_lift": True,
        "skip_rolling_corr": True,
    }
    kwargs.update(overrides)
    return AnalysisConfig(**kwargs)


def test_config_exposes_new_preprocessing_fields_with_defaults():
    config = AnalysisConfig(Path("input.csv"), "time", "target", Path("out"))

    assert config.lowpass_tau_minutes == 5.0
    assert config.diff_interval_minutes is None


@pytest.mark.parametrize("value", [0.0, -1.0, -0.001])
def test_config_rejects_nonpositive_lowpass_tau_minutes(value: float):
    with pytest.raises(ValueError, match="lowpass_tau_minutes"):
        AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), lowpass_tau_minutes=value
        )


@pytest.mark.parametrize("value", [0.0, -1.5])
def test_config_rejects_nonpositive_diff_interval_minutes(value: float):
    with pytest.raises(ValueError, match="diff_interval_minutes"):
        AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), diff_interval_minutes=value
        )


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_config_rejects_nonfinite_lowpass_tau_minutes(value: float):
    with pytest.raises(ValueError, match="lowpass_tau_minutes"):
        AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), lowpass_tau_minutes=value
        )


@pytest.mark.parametrize("value", NON_FINITE_VALUES)
def test_config_rejects_nonfinite_diff_interval_minutes(value: float):
    with pytest.raises(ValueError, match="diff_interval_minutes"):
        AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), diff_interval_minutes=value
        )


@pytest.mark.parametrize("value", [0.5, 5.0, 30.0])
def test_config_accepts_finite_positive_preprocessing_parameters(value: float):
    config = AnalysisConfig(
        Path("input.csv"),
        "time",
        "target",
        Path("out"),
        lowpass_tau_minutes=value,
        diff_interval_minutes=value,
    )

    assert config.lowpass_tau_minutes == value
    assert config.diff_interval_minutes == value


def test_config_accepts_none_diff_interval_minutes():
    config = AnalysisConfig(
        Path("input.csv"), "time", "target", Path("out"), diff_interval_minutes=None
    )

    assert config.diff_interval_minutes is None


def test_config_represents_contract_modes_and_keeps_legacy_modes():
    assert CONTRACT_PREPROCESS_MODES == {"raw", "lowpass", "lowpass_detrend", "lowpass_diff"}
    assert NOT_IMPLEMENTED_PREPROCESS_MODES == {"lowpass", "lowpass_detrend", "lowpass_diff"}
    assert {"detrend", "diff", "detrend_diff"} <= SUPPORTED_PREPROCESS_MODES

    for mode in sorted(SUPPORTED_PREPROCESS_MODES):
        config = AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), preprocess_mode=mode
        )
        assert config.preprocess_mode == mode


def test_config_rejects_unknown_preprocess_mode():
    with pytest.raises(ValueError, match="Unknown preprocess mode"):
        AnalysisConfig(
            Path("input.csv"), "time", "target", Path("out"), preprocess_mode="bogus"
        )


@pytest.mark.parametrize("mode", sorted(NOT_WIRED_ANALYSIS_PREPROCESS_MODES))
def test_lowpass_modes_run_in_transform_frame_and_causal_paths(mode: str):
    frame = _raw_frame().iloc[:20]

    transformed = transform_frame(frame, mode, 24)
    causal = transform_frame_causal(frame, mode, 24)

    assert transformed.columns.tolist() == frame.columns.tolist()
    assert not transformed.empty
    assert causal.columns.tolist() == frame.columns.tolist()
    assert not causal.empty


@pytest.mark.parametrize("mode", sorted(NOT_IMPLEMENTED_PREPROCESS_MODES))
def test_lowpass_modes_are_not_exposed_by_official_web_or_cli(mode: str):
    from chem_ts_corr import cli, web

    web_source = Path(web.__file__).read_text(encoding="utf-8")
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")

    assert f'<option value="{mode}">' not in web_source
    assert f'"{mode}"' not in cli_source


@pytest.mark.parametrize("mode", sorted(NOT_IMPLEMENTED_PREPROCESS_MODES))
def test_lowpass_modes_do_not_enter_analysis_flow(tmp_path: Path, mode: str):
    config = _raw_config(tmp_path, preprocess_mode=mode)

    with pytest.raises(ValueError, match="analysis/screening flow"):
        analyze_numeric_frame(_raw_frame(), config)


def test_raw_main_screening_regression_baseline(tmp_path: Path):
    config = _raw_config(tmp_path)
    tables = analyze_numeric_frame(_raw_frame(), config)
    ranked = tables.ranked_features
    recommended = tables.recommended_candidates

    assert ranked["variable"].tolist() == RAW_BASELINE_VARIABLES
    assert ranked["final_score"].dropna().diff().dropna().le(1e-15).all()
    assert ranked["driver_rank"].tolist() == list(range(1, len(ranked) + 1))

    indexed = ranked.set_index("variable")
    for variable in RAW_BASELINE_VARIABLES:
        row = indexed.loc[variable]
        assert row["final_score"] == pytest.approx(
            RAW_BASELINE["final_score"][variable], rel=0, abs=1e-9
        )
        assert int(row["driver_rank"]) == RAW_BASELINE["driver_rank"][variable]
        assert int(row["lag"]) == RAW_BASELINE["lag"][variable]
        assert str(row["candidate_grade"]) == RAW_BASELINE["candidate_grade"][variable]
        assert str(row["risk_flags"]) == RAW_BASELINE["risk_flags"][variable]
        lag = int(row["lag"])
        if lag > 0:
            assert str(row["direction"]) == "变量领先目标"
        elif lag < 0:
            assert str(row["direction"]) == "变量滞后目标"
        else:
            assert str(row["direction"]) == "同步变化"

    assert ranked.head(config.top_k)["variable"].tolist() == RAW_BASELINE_VARIABLES

    assert recommended["variable"].tolist() == [
        item["variable"] for item in RAW_RECOMMENDED
    ]
    for expected in RAW_RECOMMENDED:
        row = recommended.set_index("variable").loc[expected["variable"]]
        assert str(row["candidate_source"]) == expected["candidate_source"]
        assert int(row["candidate_pool_rank"]) == expected["candidate_pool_rank"]
        assert bool(row["selected_by_raw"])


def test_raw_ranked_csv_key_fields_and_order(tmp_path: Path):
    config = _raw_config(tmp_path)
    tables = analyze_numeric_frame(_raw_frame(), config)
    write_outputs(
        tmp_path,
        config.target,
        tables.ranked_features,
        tables.lag_scores,
        tables.granger_tests,
        tables.importance,
        tables.metrics,
        diagnostics=tables.diagnostics,
        risk_flags=tables.risk_flags,
        lag_peak_quality=tables.lag_peak_quality,
        residual_corr_scores=tables.residual_corr_scores,
        recommended_candidates=tables.recommended_candidates,
    )

    csv = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    key_fields = {
        "variable",
        "driver_rank",
        "final_score",
        "lag",
        "direction",
        "pearson",
        "spearman",
        "lag_quality",
        "data_quality_score",
        "risk_flags",
        "candidate_grade",
        "recommended_use",
    }
    assert key_fields <= set(csv.columns)
    assert csv["variable"].tolist() == RAW_BASELINE_VARIABLES
    assert csv["driver_rank"].tolist() == list(range(1, len(csv) + 1))
    assert csv["final_score"].dropna().diff().dropna().le(1e-15).all()
    assert {str(value) for value in csv["risk_flags"]} == set(RAW_BASELINE["risk_flags"].values())


def test_new_config_defaults_do_not_change_raw_output(tmp_path: Path):
    frame = _raw_frame()
    default_tables = analyze_numeric_frame(frame, _raw_config(tmp_path))
    explicit_tables = analyze_numeric_frame(
        frame,
        _raw_config(tmp_path, lowpass_tau_minutes=5.0, diff_interval_minutes=None),
    )

    pd.testing.assert_frame_equal(
        default_tables.ranked_features,
        explicit_tables.ranked_features,
        check_exact=False,
    )
    assert default_tables.ranked_features["variable"].tolist() == RAW_BASELINE_VARIABLES


@pytest.mark.parametrize("mode", ["raw", "detrend", "diff", "detrend_diff"])
def test_new_fields_do_not_affect_legacy_mode_outputs(tmp_path: Path, mode: str):
    frame = _raw_frame()
    baseline = analyze_numeric_frame(
        frame, _raw_config(tmp_path / "baseline", preprocess_mode=mode)
    ).ranked_features
    varied = analyze_numeric_frame(
        frame,
        _raw_config(
            tmp_path / "varied",
            preprocess_mode=mode,
            lowpass_tau_minutes=30.0,
            diff_interval_minutes=1.0,
        ),
    ).ranked_features

    pd.testing.assert_frame_equal(baseline, varied, check_exact=False)


def test_raw_top_k_5_truncation_boundary(tmp_path: Path):
    config = _raw_config(tmp_path, top_k=5)
    tables = analyze_numeric_frame(_raw_frame(), config)
    ranked = tables.ranked_features
    recommended = tables.recommended_candidates

    # 完整 ranked_features 仍保留全部变量，driver_rank 与 final_score 排序不变。
    assert ranked["variable"].tolist() == RAW_BASELINE_VARIABLES
    assert ranked["driver_rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["final_score"].dropna().diff().dropna().le(1e-15).all()
    indexed = ranked.set_index("variable")
    for variable in RAW_BASELINE_VARIABLES:
        assert indexed.loc[variable, "final_score"] == pytest.approx(
            RAW_BASELINE["final_score"][variable], rel=0, abs=1e-9
        )

    # Top-K 截断边界：前五项及顺序固定，第六名不在 Top-K。
    assert ranked.head(config.top_k)["variable"].tolist() == RAW_TOP_K_5
    assert not set(RAW_TOP_K_5) & {"candidate_4"}
    assert int(indexed.loc["candidate_4", "driver_rank"]) == 6

    # 实际 Top-K 消费方：metrics 透传 top_k，候选池 raw 通道按 top_k 截断。
    assert tables.metrics["top_k"] == pytest.approx(5.0)
    eligible = ranked.loc[ranked["variable_role"] != "residual_control"]
    assert eligible.head(config.top_k)["variable"].tolist() == [
        "candidate_0",
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "candidate_4",
    ]
    assert set(recommended["variable"]) == set(eligible.head(config.top_k)["variable"])
    assert recommended["variable"].tolist() == [
        item["variable"] for item in RAW_RECOMMENDED
    ]
    for expected in RAW_RECOMMENDED:
        row = recommended.set_index("variable").loc[expected["variable"]]
        assert str(row["candidate_source"]) == expected["candidate_source"]
        assert int(row["candidate_pool_rank"]) == expected["candidate_pool_rank"]
        assert int(row["raw_candidate_rank"]) == expected["candidate_pool_rank"]
        assert int(row["raw_candidate_rank"]) <= config.top_k

    # 结果表入口：写出的 recommended_candidates.csv 与内存推荐结果一致。
    write_outputs(
        tmp_path,
        config.target,
        tables.ranked_features,
        tables.lag_scores,
        tables.granger_tests,
        tables.importance,
        tables.metrics,
        diagnostics=tables.diagnostics,
        risk_flags=tables.risk_flags,
        lag_peak_quality=tables.lag_peak_quality,
        residual_corr_scores=tables.residual_corr_scores,
        recommended_candidates=tables.recommended_candidates,
    )
    csv_recommended = pd.read_csv(
        tmp_path / "recommended_candidates.csv", encoding="utf-8-sig"
    )
    pd.testing.assert_series_equal(
        csv_recommended["variable"],
        recommended["variable"],
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        csv_recommended["candidate_pool_rank"],
        recommended["candidate_pool_rank"],
        check_dtype=False,
    )
