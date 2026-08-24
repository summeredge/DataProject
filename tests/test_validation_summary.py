from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.validation_summary import (
    VALIDATION_SUMMARY_COLUMNS,
    build_validation_summary,
    build_validation_summary_from_output_dir,
    write_validation_summary,
)


def test_validation_summary_has_only_the_frozen_five_columns():
    summary = build_validation_summary(
        pd.DataFrame([{"variable": "x", "final_score": 0.9, "driver_rank": 1}]),
        enhanced_validation_summary=pd.DataFrame(
            [{"variable": "x", "status": "ok", "model_lift": 0.2}]
        ),
    )

    assert list(summary.columns) == VALIDATION_SUMMARY_COLUMNS
    assert "validation_score" not in summary.columns
    assert "validation_rank" not in summary.columns


def test_initial_only_run_marks_validation_as_not_run_without_conclusions():
    summary = build_validation_summary(
        pd.DataFrame(
            [
                {"variable": "x", "final_score": 0.9, "driver_rank": 1},
                {"variable": "y", "final_score": 0.8, "driver_rank": 2},
            ]
        )
    )

    assert summary["validation_status"].tolist() == ["not_run", "not_run"]
    assert summary["evidence_consistency"].tolist() == ["not_run", "not_run"]
    assert summary["supporting_methods"].tolist() == ["", ""]
    assert all("not_run" in value for value in summary["limiting_factors"])
    assert not (summary["validation_status"] == "supported").any()
    assert list(summary.columns) == VALIDATION_SUMMARY_COLUMNS


def test_missing_secondary_methods_remain_not_run_and_not_computed():
    summary = build_validation_summary(
        pd.DataFrame([{"variable": "x"}, {"variable": "y"}]),
        enhanced_validation_summary=pd.DataFrame(
            [
                {"variable": "x", "status": "ok", "model_lift": 0.2},
                {"variable": "y", "status": "skipped: insufficient rows"},
            ]
        ),
    ).set_index("variable")

    assert summary.loc["x", "validation_status"] == "limited"
    assert summary.loc["x", "evidence_consistency"] == "partial"
    assert summary.loc["x", "supporting_methods"] == "enhanced_screening"
    assert "granger:not_run" in summary.loc["x", "limiting_factors"]
    assert "model_explanation:not_run" in summary.loc["x", "limiting_factors"]

    assert summary.loc["y", "validation_status"] == "not_computed"
    assert summary.loc["y", "evidence_consistency"] == "not_computed"
    assert summary.loc["y", "supporting_methods"] == ""
    assert "enhanced_screening:skipped" in summary.loc["y", "limiting_factors"]
    assert "granger:not_run" in summary.loc["y", "limiting_factors"]


def test_failed_missing_uncomputable_and_zero_evidence_keep_distinct_semantics():
    summary = build_validation_summary(
        pd.DataFrame(
            [{"variable": name} for name in [
                "failed", "skipped", "missing", "zero", "zero_model", "uncomputable"
            ]]
        ),
        enhanced_validation_summary=pd.DataFrame(
            [
                {"variable": "failed", "status": "failed: solver"},
                {"variable": "skipped", "status": "skipped: insufficient rows"},
                {"variable": "zero", "status": "ok", "model_lift": 0.0},
                {"variable": "uncomputable", "status": "not_computed"},
            ]
        ),
        model_variable_importance=pd.DataFrame(
            [{"variable": "zero_model", "max_importance": 0.0}]
        ),
    ).set_index("variable")

    assert "enhanced_screening:failed" in summary.loc["failed", "limiting_factors"]
    assert "enhanced_screening:skipped" in summary.loc["skipped", "limiting_factors"]
    assert "enhanced_screening:variable_missing" in summary.loc["missing", "limiting_factors"]
    assert "enhanced_screening:zero_evidence" in summary.loc["zero", "limiting_factors"]
    assert "enhanced_screening:not_computed" in summary.loc["uncomputable", "limiting_factors"]
    assert "model_explanation:zero_evidence" in summary.loc["zero_model", "limiting_factors"]

    assert summary.loc["zero", "supporting_methods"] == ""
    assert summary.loc["zero", "validation_status"] == "limited"
    assert summary.loc["zero", "evidence_consistency"] == "partial"
    assert summary.loc["zero_model", "supporting_methods"] == ""


def test_granger_failure_skip_and_zero_contribution_are_not_support():
    summary = build_validation_summary(
        pd.DataFrame([{"variable": name} for name in ["failed", "skipped", "zero"]]),
        granger_tests=pd.DataFrame(
            [
                {"variable": "failed", "status": "failed: solver"},
                {"variable": "skipped", "status": "skipped: insufficient rows"},
                {
                    "variable": "zero",
                    "status": "ok",
                    "predictive_contribution": 0.0,
                },
            ]
        ),
    ).set_index("variable")

    assert "granger:failed" in summary.loc["failed", "limiting_factors"]
    assert "granger:skipped" in summary.loc["skipped", "limiting_factors"]
    assert "granger:zero_evidence" in summary.loc["zero", "limiting_factors"]
    assert summary.loc["zero", "supporting_methods"] == ""


@pytest.mark.parametrize(
    ("contribution", "limiting_factor"),
    [
        (None, "granger:missing"),
        (np.nan, "granger:missing"),
        (np.inf, "granger:not_computed"),
        (-np.inf, "granger:not_computed"),
        (-0.2, "granger:computed_no_support"),
        (0.0, "granger:zero_evidence"),
    ],
)
def test_granger_nonpositive_or_missing_contribution_never_supports(
    contribution: float | None, limiting_factor: str
):
    summary = build_validation_summary(
        pd.DataFrame([{"variable": "x"}]),
        granger_tests=pd.DataFrame(
            [{
                "variable": "x",
                "status": "ok",
                "predictive_contribution": contribution,
            }]
        ),
    )

    row = summary.iloc[0]
    assert row["supporting_methods"] == ""
    assert limiting_factor in row["limiting_factors"]


def test_granger_positive_finite_contribution_supports():
    summary = build_validation_summary(
        pd.DataFrame([{"variable": "x"}]),
        granger_tests=pd.DataFrame(
            [{"variable": "x", "status": "ok", "predictive_contribution": 0.2}]
        ),
    )

    assert summary.iloc[0]["supporting_methods"] == "granger"


def test_output_directory_summary_is_read_only_for_initial_screening_files(tmp_path: Path):
    ranked_path = tmp_path / "ranked_features.csv"
    ranked = pd.DataFrame(
        [
            {"variable": "x", "final_score": 0.9, "driver_rank": 1},
            {"variable": "y", "final_score": 0.8, "driver_rank": 2},
        ]
    )
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    before = ranked_path.read_bytes()
    pd.DataFrame([{"variable": "x", "status": "ok", "model_lift": 0.2}]).to_csv(
        tmp_path / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig"
    )

    summary = write_validation_summary(tmp_path)
    stored = pd.read_csv(tmp_path / "validation_summary.csv", encoding="utf-8-sig")

    assert list(stored.columns) == VALIDATION_SUMMARY_COLUMNS
    assert list(summary.columns) == VALIDATION_SUMMARY_COLUMNS
    assert (tmp_path / "ranked_features.csv").read_bytes() == before
    assert set(stored.columns).isdisjoint({"final_score", "driver_rank", "validation_score", "validation_rank"})


def test_missing_output_files_are_not_promoted_to_validation_evidence(tmp_path: Path):
    pd.DataFrame([{"variable": "x", "final_score": 0.9}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )

    summary = build_validation_summary_from_output_dir(tmp_path)

    assert summary.loc[0, "validation_status"] == "not_run"
    assert summary.loc[0, "evidence_consistency"] == "not_run"
    assert not (tmp_path / "validation_summary.csv").exists()


def test_initial_payload_exposes_only_not_run_summary_rows(tmp_path: Path):
    pd.DataFrame([{"variable": "x", "final_score": 0.9}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )
    (tmp_path / "summary.md").write_text("# 初步筛选摘要\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = web._build_result_payload("run", tmp_path, config)

    assert list(payload["validationSummary"][0]) == VALIDATION_SUMMARY_COLUMNS
    assert payload["validationSummary"][0]["validation_status"] == "not_run"
    assert payload["validationSummary"][0]["supporting_methods"] == ""
    assert "not_run" in payload["validationSummary"][0]["limiting_factors"]
