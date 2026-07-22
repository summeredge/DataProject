from __future__ import annotations

import inspect

import pandas as pd
import pytest

from chem_ts_corr.lag import summarize_best_lags
from chem_ts_corr.screening import final_ranked_features
from chem_ts_corr.web import (
    INDEX_HTML,
    _correlation_direction,
    _with_correlation_display_fields,
)


def _lag_row(
    lag: int,
    pearson: float,
    spearman: float,
    *,
    pearson_q: float,
    spearman_q: float,
    lag_boundary_flag: bool = False,
) -> dict[str, object]:
    return {
        "variable": "x",
        "lag": lag,
        "pearson": pearson,
        "pearson_p": pearson_q / 2,
        "pearson_r2": pearson**2,
        "spearman": spearman,
        "spearman_p": spearman_q / 2,
        "spearman_r2": spearman**2,
        "n": 80 - abs(lag),
        "abs_pearson": abs(pearson),
        "abs_spearman": abs(spearman),
        "lag_boundary_flag": lag_boundary_flag,
        "pearson_q": pearson_q,
        "spearman_q": spearman_q,
        "p_value": min(pearson_q, spearman_q) / 2,
        "corr_q_value": min(pearson_q, spearman_q),
        "p_value_status": "ok",
    }


def _final(ranked: pd.DataFrame, *, top_k: int | None = None) -> pd.DataFrame:
    variables = ranked["variable"].tolist()
    empty = pd.DataFrame(columns=["variable"])
    risks = pd.DataFrame(
        [{"variable": variable, "risk_flags": "", "data_quality_score": 1.0} for variable in variables]
    )
    return final_ranked_features(
        ranked,
        empty,
        empty,
        empty,
        risks,
        empty,
        empty,
        top_k=top_k,
    )


def test_web_correlation_fields_come_from_one_unified_best_lag_row():
    scores = pd.DataFrame(
        [
            _lag_row(2, 0.95, 0.20, pearson_q=0.011, spearman_q=0.31),
            _lag_row(
                4,
                0.10,
                -0.96,
                pearson_q=0.42,
                spearman_q=0.024,
                lag_boundary_flag=True,
            ),
        ]
    )

    best = summarize_best_lags(scores)
    candidate = _with_correlation_display_fields(_final(best)).iloc[0]

    assert candidate["lag"] == 4
    assert candidate["pearson"] == pytest.approx(0.10)
    assert candidate["spearman"] == pytest.approx(-0.96)
    assert candidate["method"] == "spearman"
    assert candidate["corr_q_value"] == pytest.approx(0.024)
    assert candidate["dominant_corr"] == pytest.approx(-0.96)
    assert candidate["correlation_direction"] == "负向"
    assert candidate["dominant_corr"] != candidate["raw_corr"]
    assert bool(candidate["lag_boundary_flag"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.051, "正向"),
        (0.050, "方向较弱"),
        (-0.050, "方向较弱"),
        (-0.051, "负向"),
        (float("nan"), "未计算"),
    ],
)
def test_correlation_direction_has_fixed_weak_direction_boundaries(value, expected):
    assert _correlation_direction(value) == expected


@pytest.mark.parametrize(
    ("pearson", "spearman", "method", "expected_corr", "expected_direction"),
    [
        (-0.40, 0.70, "spearman", 0.70, "正向"),
        (-0.80, -0.50, "pearson", -0.80, "负向"),
    ],
)
def test_dominant_correlation_uses_the_signed_value_for_the_selected_method(
    pearson, spearman, method, expected_corr, expected_direction
):
    display = _with_correlation_display_fields(
        pd.DataFrame(
            [{"variable": "x", "pearson": pearson, "spearman": spearman, "method": method}]
        )
    )

    assert display.loc[0, "dominant_corr"] == pytest.approx(expected_corr)
    assert display.loc[0, "correlation_direction"] == expected_direction


@pytest.mark.parametrize(
    ("lag", "time_relationship", "dominant_corr", "correlation_direction"),
    [
        (5, "变量领先目标", -0.8, "负向"),
        (0, "同步变化", -0.8, "负向"),
        (-5, "变量滞后目标", 0.8, "正向"),
    ],
)
def test_time_relationship_and_correlation_direction_remain_independent(
    lag, time_relationship, dominant_corr, correlation_direction
):
    display = _with_correlation_display_fields(
        pd.DataFrame(
            [
                {
                    "variable": "x",
                    "lag": lag,
                    "direction": time_relationship,
                    "pearson": dominant_corr,
                    "spearman": -dominant_corr,
                    "method": "pearson",
                }
            ]
        )
    )

    assert display.loc[0, "direction"] == time_relationship
    assert display.loc[0, "correlation_direction"] == correlation_direction


def test_correlation_display_fields_do_not_change_scores_or_top_k_order():
    base = pd.DataFrame(
        [
            {"variable": "a", "score": 0.82, "lag": 1, "innovation_score": 0.70},
            {"variable": "b", "score": 0.75, "lag": 2, "innovation_score": 0.72},
            {"variable": "c", "score": 0.60, "lag": 3, "innovation_score": 0.58},
        ]
    )
    enriched = base.assign(
        pearson=[-0.82, 0.70, -0.60],
        spearman=[-0.78, 0.75, -0.55],
        method=["pearson", "spearman", "pearson"],
        pearson_p=[0.01, 0.02, 0.03],
        spearman_p=[0.02, 0.01, 0.04],
        pearson_q=[0.02, 0.04, 0.05],
        spearman_q=[0.04, 0.02, 0.06],
        corr_q_value=[0.02, 0.02, 0.05],
        pearson_r2=[0.82**2, 0.70**2, 0.60**2],
        spearman_r2=[0.78**2, 0.75**2, 0.55**2],
        n=[100, 99, 98],
    )

    before = _final(base, top_k=2)
    after = _final(enriched, top_k=2)

    assert after["variable"].tolist() == before["variable"].tolist()
    pd.testing.assert_series_equal(after["driver_rank"], before["driver_rank"])
    pd.testing.assert_series_equal(
        after["driver_priority_score"], before["driver_priority_score"]
    )
    pd.testing.assert_series_equal(after["final_score"], before["final_score"])
    for field in ["raw_corr", "association_score", "correlation_evidence_score"]:
        pd.testing.assert_series_equal(after[field], before[field])


def test_correlation_display_fields_keep_old_result_payloads_compatible():
    old_ranked = pd.DataFrame([{"variable": "x", "raw_corr": 0.8}])

    display = _with_correlation_display_fields(old_ranked)

    assert pd.isna(display.loc[0, "dominant_corr"])
    assert display.loc[0, "correlation_direction"] == "未计算"
    pd.testing.assert_frame_equal(
        display.drop(columns=["dominant_corr", "correlation_direction"]), old_ranked
    )


def test_web_candidate_tables_use_only_requested_correlation_core_columns():
    candidate_columns = INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("}", 1)[0]
    overview_columns = INDEX_HTML.split("overviewTop:", 1)[1].split("],", 1)[0]

    for field in ["pearson", "spearman", "method", "correlation_direction"]:
        assert f'"{field}"' in candidate_columns
    for field in [
        "corr_q_value",
        "n",
        "pearson_p",
        "spearman_p",
        "pearson_q",
        "spearman_q",
        "pearson_r2",
        "spearman_r2",
    ]:
        assert f'"{field}"' not in candidate_columns
    for field in ["pearson", "spearman", "method", "correlation_direction"]:
        assert f'"{field}"' in overview_columns


def test_candidate_table_labels_separate_time_relationship_and_correlation_direction():
    core_columns = INDEX_HTML.split("function coreCandidateColumns()", 1)[1].split("}", 1)[0]
    labels = INDEX_HTML.split("function columnLabel(column)", 1)[1].split(
        "function resetUI", 1
    )[0]

    expected_order = [
        "variable",
        "driver_rank",
        "driver_priority_score",
        "pearson",
        "spearman",
        "method",
        "correlation_direction",
        "lag",
        "direction",
        "candidate_class",
        "risk_flags",
        "recommended_use",
    ]
    positions = [core_columns.index(f'"{field}"') for field in expected_order]
    assert positions == sorted(positions)
    assert 'direction: "时间关系"' in labels
    assert 'correlation_direction: "相关方向"' in labels


def test_candidate_detail_has_structured_correlation_evidence_and_labels():
    required = [
        "相关性证据",
        "CORRELATION_OVERVIEW_COLUMNS",
        "CORRELATION_DETAIL_COLUMNS",
        '<details class="correlation-evidence-details">',
        "展开 P/Q、R² 与样本数",
        "大样本与时序自相关下，P/Q 值、R² 和样本数仅供参考，不参与评分、筛选、排序或颜色强调。",
        'lag: "最佳滞后点"',
        'direction: "滞后方向"',
        'pearson: "Pearson 相关系数"',
        'spearman: "Spearman 相关系数"',
        'method: "主导相关方法"',
        'dominant_corr: "主导相关系数"',
        'pearson_p: "Pearson P 值"',
        'spearman_p: "Spearman P 值"',
        'pearson_q: "Pearson Q 值"',
        'spearman_q: "Spearman Q 值"',
        'corr_q_value: "主导方法 Q 值"',
        'pearson_r2: "Pearson R²"',
        'spearman_r2: "Spearman ρ²"',
        'n: "有效样本数"',
        'lag_boundary_flag: "是否触及滞后边界"',
        "方向性解释",
        "directionalityTimeExplanation",
        "directionalitySummary",
        "timeRelationshipExplanation",
        "correlationDirectionExplanation",
        "innovationDirectionExplanation",
        "correlationConsistencyMessage",
        "时间领先和正负相关只表示当前数据中的时序关联，不等于因果方向。",
    ]
    for marker in required:
        assert marker in INDEX_HTML
    assert "THREE_DECIMAL_CORRELATION_COLUMNS" in INDEX_HTML
    assert "SIGNIFICANCE_COLUMNS" in INDEX_HTML
    assert "scoreValue.toFixed(3)" in INDEX_HTML
    assert "scoreValue.toPrecision(3)" in INDEX_HTML
    assert 'return value ? "是" : "否";' in INDEX_HTML


def test_directionality_detail_maps_all_innovation_statuses_to_chinese_explanations():
    required = [
        "innovation_verified",
        "原始值与变化量的方向和滞后基本一致。",
        "innovation_sign_conflict",
        "原始值与变化量方向冲突，可能存在共同趋势、闭环调节、工况混合或异常点影响。",
        "innovation_lag_conflict",
        "原始值与变化量的滞后关系不一致，动态关系可能不稳定。",
        "innovation_sign_unknown",
        "变化量方向无法可靠判断。",
        "not_computed",
        "未完成变化量方向验证。",
        '"0": "方向较弱"',
    ]
    for marker in required:
        assert marker in INDEX_HTML


def test_p_q_r2_and_sample_size_are_only_in_collapsed_detail_without_highlighting():
    detail = INDEX_HTML.split("function renderScreeningScoreDetails", 1)[1].split(
        "function selectTableRow", 1
    )[0]
    collapsed = detail.split('<details class="correlation-evidence-details">', 1)[1].split(
        "</details>", 1
    )[0]
    detail_columns = INDEX_HTML.split("const CORRELATION_DETAIL_COLUMNS", 1)[1].split(
        "];", 1
    )[0]
    table_class = INDEX_HTML.split("function tableCellClass", 1)[1].split(
        "function translateDisplayValue", 1
    )[0]

    for field in [
        "pearson_p",
        "spearman_p",
        "pearson_q",
        "spearman_q",
        "corr_q_value",
        "pearson_r2",
        "spearman_r2",
        "n",
    ]:
        assert f'"{field}"' in detail_columns
        assert f'"{field}"' not in table_class
    assert "renderFields(CORRELATION_DETAIL_COLUMNS)" in collapsed
    assert detail.count("renderFields(CORRELATION_DETAIL_COLUMNS)") == 1
    assert '<details class="correlation-evidence-details" open>' not in detail


def test_display_only_implementation_has_no_rescan_or_score_dependency():
    display_source = inspect.getsource(_with_correlation_display_fields)
    scoring_source = inspect.getsource(final_ranked_features).split(
        'final["association_score"] =', 1
    )[1]

    assert "compute_lag_scores" not in display_source
    assert "summarize_best_lags" not in display_source
    assert "raw_corr" not in display_source
    assert "abs(" not in display_source
    for field in [
        "pearson",
        "spearman",
        "pearson_p",
        "spearman_p",
        "pearson_q",
        "spearman_q",
        "corr_q_value",
        "pearson_r2",
        "spearman_r2",
        "n",
    ]:
        assert f'final["{field}"]' not in scoring_source
