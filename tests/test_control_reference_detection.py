from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.report import build_markdown_summary, write_outputs
from chem_ts_corr.screening import (
    CONTROL_REFERENCE_COLUMNS,
    build_recommended_candidates,
    detect_auto_control_reference,
    final_ranked_features,
)


def _ranked_features(
    variables: list[str],
    *,
    control_columns: list[str] | None = None,
    capacity_columns: list[str] | None = None,
    segment_column: str | None = None,
) -> pd.DataFrame:
    ranked = pd.DataFrame(
        [
            {
                "variable": variable,
                "score": 0.8,
                "innovation_score": 0.8,
                "lag": 1,
                "direction": "变量领先目标",
            }
            for variable in variables
        ]
    )
    evidence = ranked[["variable", "score"]]
    empty = pd.DataFrame(columns=["variable"])
    return final_ranked_features(
        ranked,
        empty,
        empty,
        evidence.rename(columns={"score": "model_lift_score"}).assign(status="ok"),
        empty,
        evidence.rename(columns={"score": "lag_quality"}),
        evidence.rename(columns={"score": "rolling_stability"}),
        control_columns=control_columns,
        capacity_columns=capacity_columns,
        segment_column=segment_column,
    )


@pytest.mark.parametrize(
    ("variable", "expected_type", "expected_source"),
    [
        ("FIC400002.SV", "pid_setpoint", "tag_suffix_sv"),
        ("TIC330003.SP", "pid_setpoint", "tag_suffix_sp"),
        ("LIC330004.MV", "pid_output", "tag_suffix_mv"),
        ("fic400002.sv", "pid_setpoint", "tag_suffix_sv"),
        ("TIC330003.Sp", "pid_setpoint", "tag_suffix_sp"),
        ("FIC400002_SV", "pid_setpoint", "tag_suffix_sv"),
        ("FIC400002-SV", "pid_setpoint", "tag_suffix_sv"),
        ("FIC400002:SV", "pid_setpoint", "tag_suffix_sv"),
        ("  FIC400002.SV  ", "pid_setpoint", "tag_suffix_sv"),
    ],
)
def test_detect_auto_control_reference_suffixes(
    variable: str, expected_type: str, expected_source: str
):
    assert detect_auto_control_reference(variable) == (True, expected_type, expected_source)


@pytest.mark.parametrize(
    "variable",
    [
        "FIC421002.PV",
        "TIC330003.PV",
        "PIC340004.PV",
        "LIC400001",
        "AI400014.PV",
        "SV_ANALYZER.PV",
        "FIC_SV_VALUE.PV",
        None,
    ],
)
def test_detect_auto_control_reference_does_not_infer_from_prefix(variable: object):
    assert detect_auto_control_reference(variable) == (False, "", "")


def test_explicit_control_configuration_has_priority_over_suffix_role():
    result = _ranked_features(["FIC400002.SV"], control_columns=["FIC400002.SV"])
    row = result.iloc[0]

    assert bool(row["is_auto_control_reference"])
    assert bool(row["is_control_reference"])
    assert row["control_reference_type"] == "residual_control"
    assert row["control_reference_source"] == "configured_residual_control"
    assert row["variable_role"] == "residual_control"


def test_auto_control_reference_fields_do_not_change_scores_or_complete_order():
    result = _ranked_features(["normal.SV", "normal.PV", "normal.MV"])
    indexed = result.set_index("variable")

    for column in [
        "association_score",
        "evidence_strength",
        "evidence_score",
        "final_score",
        "candidate_grade",
    ]:
        assert indexed[column].nunique(dropna=False) == 1
    assert result["variable"].tolist() == ["normal.MV", "normal.PV", "normal.SV"]
    assert result["driver_rank"].tolist() == [1, 2, 3]


def test_auto_control_references_follow_candidate_exclusion_switch():
    ranked = _ranked_features(["ordinary.PV", "loop.SV", "loop.SP", "loop.MV"])

    excluded = build_recommended_candidates(ranked, top_k=None, exclude_control_columns=True)
    allowed = build_recommended_candidates(ranked, top_k=None, exclude_control_columns=False)

    assert excluded["variable"].tolist() == ["ordinary.PV"]
    assert set(allowed["variable"]) == {"ordinary.PV", "loop.SV", "loop.SP", "loop.MV"}
    assert set(ranked["variable"]) == {"ordinary.PV", "loop.SV", "loop.SP", "loop.MV"}
    assert not (set(CONTROL_REFERENCE_COLUMNS) & set(excluded.columns))
    assert not (set(CONTROL_REFERENCE_COLUMNS) & set(allowed.columns))


def test_control_references_are_reported_without_prefix_based_pv_false_positive():
    ranked = _ranked_features(["FIC400002.SV", "LIC330004.MV", "FIC421002.PV"])
    candidates = build_recommended_candidates(ranked, top_k=None, exclude_control_columns=True)

    markdown = build_markdown_summary(
        "target", ranked, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), candidates
    )
    section = markdown.split("## 控制/负荷参考变量", 1)[1].split("## 相关性线索", 1)[0]
    strong = markdown.split("## 强初筛候选", 1)[1].split("## 控制/负荷参考变量", 1)[0]

    assert "FIC400002.SV" in section
    assert "LIC330004.MV" in section
    assert "FIC421002.PV" not in section
    assert "FIC400002.SV" not in strong
    assert "LIC330004.MV" not in strong


def test_ranked_csv_appends_reference_fields_without_changing_recommended_schema(
    tmp_path: Path,
):
    ranked = _ranked_features(["ordinary.PV", "loop.SV"])
    recommended = build_recommended_candidates(
        ranked, top_k=None, exclude_control_columns=True
    )
    existing_columns = ranked.columns[: -len(CONTROL_REFERENCE_COLUMNS)].tolist()

    write_outputs(
        tmp_path,
        "target",
        ranked,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        recommended_candidates=recommended,
    )

    written_ranked = pd.read_csv(tmp_path / "ranked_features.csv", encoding="utf-8-sig")
    written_recommended = pd.read_csv(
        tmp_path / "recommended_candidates.csv", encoding="utf-8-sig"
    )
    assert written_ranked.columns[: -len(CONTROL_REFERENCE_COLUMNS)].tolist() == existing_columns
    assert written_ranked.columns[-len(CONTROL_REFERENCE_COLUMNS) :].tolist() == list(
        CONTROL_REFERENCE_COLUMNS
    )
    assert not (set(CONTROL_REFERENCE_COLUMNS) & set(written_recommended.columns))
    ordinary = written_ranked.set_index("variable").loc["ordinary.PV"]
    assert pd.isna(ordinary["control_reference_type"])
    assert pd.isna(ordinary["control_reference_source"])
