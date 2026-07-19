from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.near_miss import build_near_miss_candidates
from chem_ts_corr.report import build_markdown_summary
from chem_ts_corr.screening import final_ranked_features


def _frame(values: dict[str, object] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=["variable"]) if values is None else pd.DataFrame([values])


def _result(
    raw: float = 0.8,
    *,
    innovation: object = None,
    residual: object = None,
    rolling: object = None,
    lag_quality: object = None,
    model_lift: object = None,
    regime: object = None,
    risk_flags: str = "",
) -> pd.Series:
    ranked = pd.DataFrame([{"variable": "x", "score": raw, "innovation_score": innovation, "lag": 1, "direction": "变量领先目标"}])
    residual_frame = _frame() if residual is None else _frame({"variable": "x", "residual_corr": residual})
    rolling_frame = _frame() if rolling is None else _frame({"variable": "x", "rolling_stability": rolling})
    lag_frame = _frame() if lag_quality is None else _frame({"variable": "x", "lag_quality": lag_quality})
    lift_frame = _frame() if model_lift is None else _frame({"variable": "x", "model_lift": model_lift})
    stability = _frame() if regime is None else _frame({"variable": "x", "regime_stability_final": regime})
    risks = _frame({"variable": "x", "risk_flags": risk_flags})
    return final_ranked_features(
        ranked, residual_frame, stability, lift_frame, risks, lag_frame, rolling_frame
    ).iloc[0]


def test_legacy_score_fields_are_removed_from_output_schema():
    row = _result()

    assert "raw_corr_score" not in row.index
    assert "residual_corr_score" not in row.index
    for field in ["association_score", "independent_signal_score", "correlation_evidence_score"]:
        assert field in row.index


@pytest.mark.parametrize(("raw", "expected"), [(-0.2, 0.0), (0.8, 0.8), (1.4, 1.0)])
def test_association_score_is_clipped(raw: float, expected: float):
    assert _result(raw)["association_score"] == pytest.approx(expected)


def test_present_residual_uses_formal_independent_signal():
    row = _result(residual=0.3)

    assert row["independent_signal_score"] == pytest.approx(0.3)
    assert row["residual_status"] == "ok"
    assert row["correlation_evidence_status"] == "independent_verified"


def test_missing_residual_remains_nan_with_association_only_fallback():
    row = _result()

    assert pd.isna(row["residual_corr"])
    assert pd.isna(row["independent_signal_score"])
    assert row["residual_status"] == "not_computed"
    assert row["correlation_evidence_status"] == "association_only"
    assert row["correlation_evidence_score"] == pytest.approx(row["association_score"])


def test_zero_residual_is_valid():
    row = _result(residual=0.0)

    assert row["independent_signal_score"] == 0.0
    assert row["residual_status"] == "ok"


@pytest.mark.parametrize(
    ("field", "status"),
    [("rolling_stability", "rolling_status"), ("lag_quality", "lag_quality_status"), ("model_lift_score", "model_lift_status")],
)
def test_missing_optional_evidence_remains_nan(field: str, status: str):
    row = _result()

    assert pd.isna(row[field])
    assert row[status] == "not_computed"


def test_real_zero_optional_evidence_is_valid():
    row = _result(rolling=0.0, lag_quality=0.0, model_lift=0.0)

    for field, status in [
        ("rolling_stability", "rolling_status"),
        ("lag_quality", "lag_quality_status"),
        ("model_lift_score", "model_lift_status"),
    ]:
        assert row[field] == 0.0
        assert row[status] == "ok"


def test_real_half_optional_evidence_is_preserved():
    row = _result(rolling=0.5, lag_quality=0.5)

    assert row["rolling_stability"] == pytest.approx(0.5)
    assert row["rolling_status"] == "ok"
    assert row["lag_quality"] == pytest.approx(0.5)
    assert row["lag_quality_status"] == "ok"


def test_v2_complete_balanced_evidence_preserves_common_scale():
    row = _result(
        raw=0.8, innovation=0.8, rolling=0.8, lag_quality=0.8, model_lift=0.8
    )

    assert row["score_method"] == "industrial_robust_v2"
    assert row["evidence_completeness"] == 1.0
    assert row["evidence_score"] == pytest.approx(0.8)


def test_direction_risk_changes_driver_priority_without_reducing_evidence():
    row = _result(raw=0.9, innovation=0.9, rolling=0.9, lag_quality=0.9, model_lift=0.9, risk_flags="target_leads_variable")

    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == 1.0
    assert row["risk_cap_reason"] == ""
    assert row["final_score"] == pytest.approx(0.9)
    assert row["candidate_grade"] == "A"
    assert row["driver_rank"] == 1


def test_ranking_and_topk_stay_driver_rank_based():
    ranked = pd.DataFrame([
        {"variable": "a", "score": 0.9, "lag": -1},
        {"variable": "b", "score": 0.58, "lag": 1},
    ])
    risks = pd.DataFrame([
        {"variable": "a", "risk_flags": "target_leads_variable"},
        {"variable": "b", "risk_flags": ""},
    ])
    empty = _frame()
    ranked["innovation_score"] = ranked["score"]
    complete = ranked[["variable", "score"]]
    model = complete.rename(columns={"score": "model_lift_score"}).assign(status="ok")
    lag = complete.rename(columns={"score": "lag_quality"})
    rolling = complete.rename(columns={"score": "rolling_stability"})
    result = final_ranked_features(ranked, empty, empty, model, risks, lag, rolling)
    top = final_ranked_features(ranked, empty, empty, model, risks, lag, rolling, top_k=1)

    assert result["variable"].tolist() == ["b", "a"]
    assert result["final_score"].tolist() == pytest.approx([0.58, 0.9])
    assert result["driver_priority_score"].tolist() == pytest.approx([0.58, 0.9 * 0.45])
    assert result["driver_rank"].tolist() == [1, 2]
    assert top["variable"].tolist() == ["b"]
    assert top.loc[0, "driver_rank"] == 1


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "variable": "x", "driver_rank": 1, "driver_priority_score": 0.8,
        "final_score": 0.8, "evidence_score": 0.8, "association_score": 0.8,
        "independent_signal_score": np.nan, "correlation_evidence_score": 0.8,
        "correlation_evidence_status": "association_only", "regime_stability_final": np.nan,
        "regime_status": "not_computed", "rolling_stability": np.nan,
        "rolling_status": "not_computed", "lag_quality": np.nan,
        "lag_quality_status": "not_computed", "model_lift_score": np.nan,
        "model_lift_status": "not_computed", "risk_penalty": 0.0, "risk_score_cap": 1.0,
        "recommended_use": "manual_review_required",
    }])


def test_report_uses_formal_fields_and_blank_missing_evidence():
    markdown = build_markdown_summary("target", _summary_frame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())
    section = markdown.split("## 评分分解 Top 15", 1)[1].split("## 预测候选", 1)[0]

    for field in ["association_score", "independent_signal_score", "correlation_evidence_score"]:
        assert field in section
    assert "raw_corr_score" not in section
    assert "residual_corr_score" not in section
    table = [line for line in section.splitlines() if line.startswith("|")]
    headers = [cell.strip() for cell in table[0].strip("|").split("|")]
    values = [cell.strip() for cell in table[2].strip("|").split("|")]
    row = dict(zip(headers, values))
    for field in ["independent_signal_score", "rolling_stability", "lag_quality", "model_lift_score"]:
        assert row[field] == ""


def test_report_risk_and_regime_explanations_are_current():
    markdown = build_markdown_summary("target", _summary_frame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    for text in ["工业稳健 V2", "缺失证据降低完整度", "risk_score_cap", "工况覆盖度"]:
        assert text in markdown
    assert "剩余已计算项按原始权重重归一" not in markdown


def test_web_uses_formal_score_labels_and_keeps_driver_sort():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    for marker in [
        'association_score: "原始关联规范化得分"',
        'independent_signal_score: "独立残差信号得分"',
        'correlation_evidence_score: "关联证据综合得分"',
        'table: { column: "driver_rank", direction: "asc" }',
    ]:
        assert marker in source
    assert "raw_corr_score:" not in source
    assert "residual_corr_score:" not in source


def test_near_miss_does_not_fabricate_missing_residual():
    lag_scores = pd.DataFrame([{"variable": "x", "lag": 1, "score": 0.9}])
    result = build_near_miss_candidates(lag_scores, pd.DataFrame())

    assert pd.isna(result.loc[0, "independent_signal_score"])
    assert "residual_signal" not in result.loc[0, "near_miss_reason"]


def test_json_serialization_keeps_missing_scores_as_null():
    frame = pd.DataFrame([_result()])
    record = json.loads(frame.to_json(orient="records"))[0]

    assert record["independent_signal_score"] is None
    assert record["rolling_stability"] is None
    assert "raw_corr_score" not in record


def test_all_inputs_are_not_modified():
    frames = [
        pd.DataFrame([{"variable": "x", "score": 0.8, "lag": 1}]),
        _frame(), _frame(), _frame(), _frame({"variable": "x", "risk_flags": ""}), _frame(), _frame(),
    ]
    before = [frame.copy(deep=True) for frame in frames]

    final_ranked_features(*frames)

    for actual, expected in zip(frames, before):
        pd.testing.assert_frame_equal(actual, expected)


def test_formal_scores_remain_in_unit_interval():
    row = _result(raw=2.0, residual=-1.0, rolling=2.0, lag_quality=-1.0, model_lift=2.0)

    for field in [
        "association_score", "independent_signal_score", "correlation_evidence_score",
        "rolling_stability", "lag_quality", "model_lift_score", "evidence_score", "final_score",
    ]:
        assert 0 <= row[field] <= 1


def test_old_score_fields_and_display_fallbacks_are_absent_from_formal_code():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("chem_ts_corr").glob("*.py")
    )
    for forbidden in [
        '"raw_corr_score"', '"residual_corr_score"', "display_residual", "display_rolling",
        "display_lagq", "display_lift", "rolling_raw.fillna(0.5)", "lagq_raw.fillna(0.5)",
        "lift_raw.fillna(0.0)", 'residual_raw.fillna(final["raw_corr_score"])',
    ]:
        assert forbidden not in source
