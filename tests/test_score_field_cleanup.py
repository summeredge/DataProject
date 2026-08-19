from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.near_miss import build_near_miss_candidates
from chem_ts_corr.report import build_markdown_summary, write_outputs
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
    lag_frame = _frame() if lag_quality is None else _frame({
        "variable": "x", "lag_quality": lag_quality,
        "temporal_direction_status": "variable_leads_supported",
    })
    lift_frame = _frame() if model_lift is None else _frame({"variable": "x", "model_lift": model_lift})
    stability = _frame() if regime is None else _frame({"variable": "x", "regime_stability_final": regime})
    risks = _frame({"variable": "x", "risk_flags": risk_flags})
    return final_ranked_features(
        ranked, residual_frame, stability, lift_frame, risks, lag_frame, rolling_frame
    ).iloc[0]


def test_legacy_and_followup_fields_are_removed_from_initial_output_schema():
    row = _result()

    for field in [
        "raw_corr_score", "residual_corr_score", "independent_signal_score",
        "residual_status", "regime_stability_final", "rolling_stability",
        "model_lift_score", "model_lift_status", "prediction_score",
    ]:
        assert field not in row.index
    for field in [
        "association_score", "correlation_evidence_score", "lag_quality", "final_score",
        "evidence_coverage_status", "evidence_missing_items", "stability_score",
    ]:
        assert field in row.index


@pytest.mark.parametrize(("raw", "expected"), [(-0.2, 0.0), (0.8, 0.8), (1.4, 1.0)])
def test_association_score_is_clipped(raw: float, expected: float):
    assert _result(raw)["association_score"] == pytest.approx(expected)


def test_present_residual_does_not_change_correlation_evidence_status():
    assert _result(residual=0.3)["correlation_evidence_status"] == "association_only"


def test_missing_residual_remains_missing_with_association_only_fallback():
    row = _result()

    assert row["correlation_evidence_status"] == "association_only"
    assert row["correlation_evidence_score"] == pytest.approx(row["association_score"])


def test_zero_residual_is_explanatory_only():
    row = _result(residual=0.0)

    assert row["correlation_evidence_status"] == "association_only"
    assert row["correlation_evidence_score"] == pytest.approx(0.8)


def test_missing_lag_quality_remains_missing():
    row = _result()

    assert pd.isna(row["lag_quality"])
    assert row["lag_quality_status"] == "not_computed"


def test_real_zero_lag_quality_is_valid():
    row = _result(lag_quality=0.0)

    assert row["lag_quality"] == 0.0
    assert row["lag_quality_status"] == "ok"


def test_v5_base_score_uses_association_with_complete_core_evidence():
    row = _result(raw=0.8, innovation=0.8, lag_quality=0.8)

    assert row["score_method"] == "initial_association_temporal_v5"
    assert row["evidence_completeness"] == pytest.approx(1.0)
    assert row["evidence_score"] == pytest.approx(0.8)


def test_current_runtime_score_method_assignment_is_v5():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")

    assert 'final["score_method"] = V5_SCORE_METHOD' in source
    assert 'final["score_method"] = "industrial_robust_v2"' not in source
    assert source.count('final["score_method"] = ') == 1


def test_ranked_features_csv_preserves_initial_schema_and_exports_v5(tmp_path: Path):
    ranked = pd.DataFrame([_result()])
    expected_columns = list(ranked.columns)

    write_outputs(tmp_path, "target", ranked, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    exported = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    assert list(exported.columns) == expected_columns
    assert exported["score_method"].tolist() == ["initial_association_temporal_v5"]
    assert "prediction_score" not in exported.columns


def test_not_computed_model_lift_is_not_exposed_by_initial_output():
    ranked = pd.DataFrame([{"variable": "x", "score": 0.8, "innovation_score": 0.8, "lag": 1}])
    model_lift = _frame({"variable": "x", "model_lift_score": 0.8, "status": "not_computed"})
    row = final_ranked_features(
        ranked, _frame(), _frame(), model_lift, _frame({"variable": "x", "risk_flags": ""}),
        _frame({"variable": "x", "lag_quality": 0.8}), _frame({"variable": "x", "rolling_stability": 0.8}),
    ).iloc[0]

    assert "prediction_score" not in row.index
    assert "model_lift_status" not in row.index


def test_direction_risk_does_not_change_final_score_or_primary_order():
    row = _result(raw=0.9, innovation=0.9, lag_quality=0.9, risk_flags="target_leads_variable")

    assert row["risk_penalty"] == 0.0
    assert row["risk_score_cap"] == 1.0
    assert row["final_score"] == pytest.approx(0.9)
    assert row["driver_priority_factor"] == 1.0
    assert row["driver_priority_score"] == row["final_score"]


def test_ranking_and_topk_use_final_score_descending():
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

    assert result["variable"].tolist() == ["a", "b"]
    assert result["final_score"].tolist() == pytest.approx([0.9, 0.58])
    assert result["driver_priority_score"].tolist() == pytest.approx(result["final_score"])
    assert result["driver_priority_factor"].tolist() == pytest.approx([1.0, 1.0])
    assert result["driver_rank"].tolist() == [1, 2]
    assert top["variable"].tolist() == ["a", "b"]


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "variable": "x", "final_score": 0.8, "candidate_grade": "B",
        "evidence_completeness": 0.75, "data_quality_score": 0.999768,
        "evidence_confidence": 0.999768, "evidence_score": 0.8,
        "association_score": 0.8, "correlation_evidence_score": 0.8,
        "innovation_score": 0.8, "lag_quality": np.nan,
        "risk_flags": "", "recommended_use": "manual_review_required",
    }])


def test_report_uses_initial_score_fields_without_four_layer_explanations():
    markdown = build_markdown_summary("target", _summary_frame(), pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())

    assert "# 初步筛选摘要：target" in markdown
    assert "final_score" in markdown
    for field in ["layer1_association_status", "four_layer_coverage_status", "candidate_summary"]:
        assert field not in markdown
    assert "工业稳健 V2" not in markdown


def test_web_uses_final_score_labels_and_hides_followup_fields():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")

    assert 'final_score: "初步筛选得分"' in source
    assert 'table: { column: "final_score", direction: "desc" }' in source
    assert 'table: { column: "driver_rank", direction: "asc" }' not in source
    assert 'layer1_association_status: "Layer 1 关联状态"' not in source
    assert 'candidate_summary: "候选解释"' not in source


def test_web_primary_tables_use_initial_fields():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    main_columns = source.split("function coreCandidateColumns()", 1)[1].split("}", 1)[0]
    overview_columns = source.split("overviewTop:", 1)[1].split("],", 1)[0]

    for columns in [main_columns, overview_columns]:
        assert '"final_score"' in columns
        assert "driver_priority_score" not in columns
        assert "candidate_class" not in columns
        assert "evidence_coverage_status" not in columns


def test_web_score_precision_set_excludes_p_and_q_values():
    source = Path("chem_ts_corr/web.py").read_text(encoding="utf-8")
    score_columns = source.split("const THREE_DECIMAL_SCORE_COLUMNS", 1)[1].split("]);", 1)[0]

    for field in ["final_score", "evidence_completeness", "evidence_confidence", "data_quality_score", "evidence_strength", "evidence_score"]:
        assert field in score_columns
    assert "p_value" not in score_columns
    assert "q_value" not in score_columns


def test_output_scores_keep_raw_precision_for_csv_export():
    row = _result(raw=0.999768, innovation=np.nan, lag_quality=0.935306)

    assert row["final_score"] != round(row["final_score"], 3)


def test_near_miss_does_not_fabricate_missing_residual():
    lag_scores = pd.DataFrame([{"variable": "x", "lag": 1, "score": 0.9}])
    result = build_near_miss_candidates(lag_scores, pd.DataFrame())

    assert pd.isna(result.loc[0, "independent_signal_score"])
    assert "residual_signal" not in result.loc[0, "near_miss_reason"]


def test_json_serialization_keeps_initial_fields_and_no_followup_scores():
    record = json.loads(pd.DataFrame([_result()]).to_json(orient="records"))[0]

    assert "final_score" in record
    assert "prediction_score" not in record
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
    row = _result(raw=2.0, residual=-1.0, lag_quality=2.0, model_lift=2.0)

    for field in ["association_score", "correlation_evidence_score", "lag_quality", "evidence_score", "final_score"]:
        assert 0 <= row[field] <= 1


def test_old_score_fields_and_display_fallbacks_are_absent_from_formal_code():
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("chem_ts_corr").glob("*.py"))
    for forbidden in [
        '"raw_corr_score"', '"residual_corr_score"', "display_residual", "display_rolling",
        "display_lagq", "display_lift", "rolling_raw.fillna(0.5)", "lagq_raw.fillna(0.5)",
        "lift_raw.fillna(0.0)", 'residual_raw.fillna(final["raw_corr_score"])',
    ]:
        assert forbidden not in source
