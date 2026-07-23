from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.causal_review import build_causal_review_candidates
from chem_ts_corr.report import build_markdown_summary
from chem_ts_corr.screening import final_ranked_features


def _ranked(rows: list[tuple[str, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"variable": variable, "score": score, "lag": lag, "direction": ""}
            for variable, score, lag in rows
        ]
    )
    frame["innovation_score"] = frame["score"]
    return frame


def _run(
    rows: list[tuple[str, float, int]],
    *,
    risk_flags: dict[str, str] | None = None,
    evidence: pd.DataFrame | None = None,
    top_k: int | None = None,
    force_include_variables: list[str] | None = None,
    control_columns: list[str] | None = None,
) -> pd.DataFrame:
    ranked = _ranked(rows)
    complete = ranked[["variable", "score"]]
    empty = pd.DataFrame(columns=["variable"])
    risks = pd.DataFrame(
        [{"variable": variable, "risk_flags": flags} for variable, flags in (risk_flags or {}).items()],
        columns=["variable", "risk_flags"],
    )
    return final_ranked_features(
        ranked,
        empty,
        empty,
        complete.rename(columns={"score": "model_lift_score"}).assign(status="ok"),
        risks,
        complete.rename(columns={"score": "lag_quality"}),
        complete.rename(columns={"score": "rolling_stability"}),
        force_include_variables=force_include_variables,
        top_k=top_k,
        control_columns=control_columns,
        closed_loop_evidence=evidence,
    )


def _evidence(variable: str, level: str) -> pd.DataFrame:
    return pd.DataFrame([{"variable": variable, "closed_loop_evidence_level": level}])


def test_old_calls_and_explicit_none_are_identical():
    rows = [("up", 0.8, 1), ("down", 0.9, -1)]
    first = _run(rows, risk_flags={"down": "target_leads_variable"})
    second = _run(rows, risk_flags={"down": "target_leads_variable"}, evidence=None)

    pd.testing.assert_frame_equal(first, second)


def test_confirmed_closed_loop_downgrades_upstream_without_changing_statistics():
    baseline = _run([("up", 0.9, 1)]).iloc[0]
    result = _run([("up", 0.9, 1)], evidence=_evidence("up", "confirmed")).iloc[0]

    for field in ["evidence_score", "final_score", "risk_flags", "risk_level", "candidate_grade"]:
        pd.testing.assert_series_equal(result[[field]], baseline[[field]])
    assert result["candidate_class"] == "closed_loop_related"
    assert result["driver_priority_factor"] == pytest.approx(0.55)
    assert result["driver_priority_score"] == pytest.approx(result["final_score"] * 0.55)
    assert result["driver_priority_score"] < baseline["driver_priority_score"]
    assert result["recommended_use"] == "closed_loop_confirmed"
    assert result["recommended_action"] == "已确认闭环控制关系，不作为上游驱动优先候选"


def test_confirmed_closed_loop_never_increases_lower_base_factor():
    baseline = _run([("down", 0.9, -1)], risk_flags={"down": "target_leads_variable"}).iloc[0]
    result = _run(
        [("down", 0.9, -1)],
        risk_flags={"down": "target_leads_variable"},
        evidence=_evidence("down", "confirmed"),
    ).iloc[0]

    assert baseline["driver_priority_factor"] == pytest.approx(0.45)
    assert result["candidate_class"] == "closed_loop_related"
    assert result["driver_priority_factor"] == pytest.approx(0.45)
    assert result["driver_priority_score"] == pytest.approx(baseline["driver_priority_score"])


def test_rejected_status_does_not_change_ranking_or_recommendation():
    baseline = _run([("up", 0.8, 1)])
    result = _run([("up", 0.8, 1)], evidence=_evidence("up", "rejected"))

    pd.testing.assert_frame_equal(baseline, result)


def test_conflict_keeps_scores_and_ranks_but_changes_recommendation():
    flags = {"down": "closed_loop_suspect;target_leads_variable"}
    baseline = _run([("down", 0.8, -1)], risk_flags=flags).iloc[0]
    result = _run([("down", 0.8, -1)], risk_flags=flags, evidence=_evidence("down", "conflict")).iloc[0]

    for field in ["risk_flags", "candidate_class", "driver_priority_factor", "driver_priority_score", "driver_rank", "final_score"]:
        assert result[field] == baseline[field]
    assert result["recommended_use"] == "closed_loop_conflict"
    assert result["recommended_action"] == "人工确认非闭环，但自动判断存在闭环嫌疑，需人工复核"


def test_suspected_status_keeps_existing_automatic_behavior():
    flags = {"down": "closed_loop_suspect;target_leads_variable"}
    baseline = _run([("down", 0.8, -1)], risk_flags=flags)
    result = _run([("down", 0.8, -1)], risk_flags=flags, evidence=_evidence("down", "suspected"))

    pd.testing.assert_frame_equal(baseline, result)


def test_top_k_is_selected_after_confirmed_closed_loop_adjustment():
    result = _run(
        [("a", 0.9, 1), ("b", 0.7, 1), ("c", 0.6, 1)],
        evidence=_evidence("a", "confirmed"),
        top_k=2,
    )

    assert result["variable"].tolist() == ["b", "c"]
    assert result["driver_rank"].tolist() == [1, 2]


def test_force_include_keeps_confirmed_closed_loop_global_rank():
    result = _run(
        [("a", 0.9, 1), ("b", 0.7, 1), ("c", 0.6, 1)],
        evidence=_evidence("a", "confirmed"),
        top_k=2,
        force_include_variables=["a"],
    ).set_index("variable")

    assert bool(result.loc["a", "force_included"])
    assert result.loc["a", "driver_rank"] == 3
    assert result.loc["a", "recommended_use"] == "closed_loop_confirmed"


def test_confirmed_control_variable_remains_excluded_from_regular_top_k():
    result = _run(
        [("ctrl", 0.9, 1), ("up", 0.8, 1)],
        evidence=_evidence("ctrl", "confirmed"),
        top_k=1,
        control_columns=["ctrl"],
    )

    assert result["variable"].tolist() == ["up"]


def test_evidence_is_built_only_in_service_and_pipeline_writes_it():
    service = Path("chem_ts_corr/service.py").read_text(encoding="utf-8")
    pipeline = Path("chem_ts_corr/pipeline.py").read_text(encoding="utf-8")

    assert service.count("build_closed_loop_evidence(") == 1
    assert "build_closed_loop_evidence" not in pipeline
    assert "tables.closed_loop_evidence.to_csv" in pipeline


@pytest.mark.parametrize("recommended_use", ["closed_loop_confirmed", "closed_loop_conflict"])
def test_closed_loop_recommendations_are_reported_as_low_priority_review_only(recommended_use: str):
    ranked = pd.DataFrame(
        [{
            "variable": "loop", "final_score": 0.9, "candidate_grade": "A",
            "recommended_use": recommended_use, "recommended_action": "人工复核",
            "lag": 1, "direction": "", "raw_corr": 0.9, "risk_level": "none",
            "risk_flags": "", "force_included": False,
        }]
    )

    summary = build_markdown_summary("target", ranked, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame())
    review = build_causal_review_candidates(ranked)

    assert "## 不建议作为因果结论的变量" in summary
    assert "## 强初筛候选\n\n无可用结果。" in summary
    assert "## 预测候选\n\n无可用结果。" in summary
    assert review.loc[0, "review_tier"] == "tier_4"
