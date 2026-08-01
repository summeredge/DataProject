from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr.ranking_baseline import (
    RISK_FLAG_COLUMNS,
    _optional_bool,
    evaluate_ranking_baseline,
    evaluate_run_directory,
    main,
)


def _ranked(count: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "variable": f"x{index + 1}",
                "final_score": float(count - index) / count,
                "candidate_grade": "A" if index < 3 else "B" if index < 6 else "C",
                "recommended_use": "candidate",
                "recommended_action": "review",
                "lag": index,
                "direction": "variable_leads_target",
                "risk_flags": "",
            }
        )
    return pd.DataFrame(rows)


def test_current_rank_uses_input_order_not_final_score():
    ranked = pd.DataFrame(
        [
            {"variable": "low", "final_score": 0.1},
            {"variable": "high", "final_score": 0.9},
        ]
    )

    detail, _ = evaluate_ranking_baseline(ranked)

    assert detail["variable"].tolist() == ["low", "high"]
    assert detail["current_rank"].tolist() == [1, 2]


def test_inputs_are_not_modified():
    ranked = pd.DataFrame([{"variable": "x1", "risk_flags": "target_leads_variable"}])
    risks = pd.DataFrame([{"variable": "x1", "common_capacity_driver_flag": True}])
    expectations = pd.DataFrame(
        [{"variable": "x1", "expected_class": "reasonable_driver", "reason": "ok"}]
    )
    ranked_before = ranked.copy(deep=True)
    risks_before = risks.copy(deep=True)
    expectations_before = expectations.copy(deep=True)

    evaluate_ranking_baseline(ranked, risks, expectations)

    pd.testing.assert_frame_equal(ranked, ranked_before)
    pd.testing.assert_frame_equal(risks, risks_before)
    pd.testing.assert_frame_equal(expectations, expectations_before)


def test_explicit_risk_column_wins_over_risk_text():
    ranked = pd.DataFrame(
        [
            {
                "variable": "x1",
                "risk_flags": "target_leads_variable",
                "target_leads_variable_flag": False,
            }
        ]
    )

    detail, _ = evaluate_ranking_baseline(ranked)

    assert bool(detail.loc[0, "target_leads_variable_flag"]) is False


@pytest.mark.parametrize(("column", "token"), RISK_FLAG_COLUMNS.items())
def test_risk_priority_is_applied_per_row_for_all_columns(column: str, token: str):
    ranked = pd.DataFrame(
        [
            {"variable": "ranked_false", "candidate_grade": "A", "risk_flags": token, column: False},
            {"variable": "ranked_true", "candidate_grade": "A", "risk_flags": "", column: True},
            {"variable": "file_true", "candidate_grade": "A", "risk_flags": "", column: pd.NA},
            {"variable": "text_true", "candidate_grade": "A", "risk_flags": token, column: pd.NA},
            {"variable": "all_missing", "candidate_grade": "A", "risk_flags": "", column: ""},
        ]
    )
    risks = pd.DataFrame(
        [
            {"variable": "ranked_false", column: True},
            {"variable": "ranked_true", column: False},
            {"variable": "file_true", column: True},
            {"variable": "text_true", column: pd.NA},
            {"variable": "all_missing", column: None},
        ]
    )

    detail, metrics = evaluate_ranking_baseline(ranked, risks, cutoffs=(5,))

    assert detail[column].tolist() == [False, True, True, True, False]
    assert detail[column].dtype == bool
    metric_key = token.removesuffix("_suspect") + "_count"
    if token == "target_leads_variable":
        metric_key = "target_leads_count"
    elif token == "common_capacity_driver":
        metric_key = "common_capacity_count"
    assert metrics["cutoffs"]["5"][metric_key] == 3
    if column != "lag_boundary_flag":
        ab_key = f"ab_{metric_key}"
        assert metrics["cutoffs"]["5"][ab_key] == 3


@pytest.mark.parametrize("value", ["unknown", "not_computed", "missing", "", None, pd.NA])
def test_optional_bool_rejects_unknown_and_missing_values(value: object):
    assert _optional_bool(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), (1, True), (1.0, True), (0, False), (0.0, False),
     (" YES ", True), ("y", True), ("false", False), ("N", False), (2, None)],
)
def test_optional_bool_accepts_only_known_values(value: object, expected: bool | None):
    assert _optional_bool(value) is expected


def test_risk_text_uses_exact_tokens():
    ranked = pd.DataFrame(
        [
            {"variable": "prefixed", "risk_flags": "not_target_leads_variable"},
            {"variable": "suffixed", "risk_flags": "target_leads_variable_extra"},
            {"variable": "comma", "risk_flags": "foo,target_leads_variable"},
            {"variable": "exact", "risk_flags": " target_leads_variable ; other "},
        ]
    )

    detail, _ = evaluate_ranking_baseline(ranked)

    assert detail["target_leads_variable_flag"].tolist() == [False, False, False, True]


def test_risk_text_fallback_is_parsed_when_explicit_columns_are_missing():
    ranked = pd.DataFrame(
        [{"variable": "x1", "risk_flags": "target_leads_variable;common_capacity_driver"}]
    )

    detail, _ = evaluate_ranking_baseline(ranked)

    row = detail.iloc[0]
    assert bool(row["target_leads_variable_flag"]) is True
    assert bool(row["common_capacity_driver_flag"]) is True


def test_top10_top20_metrics_count_hits_risks_and_ab_risks():
    ranked = _ranked(20)
    ranked.loc[0, "risk_flags"] = "target_leads_variable"
    ranked.loc[1, "risk_flags"] = "common_capacity_driver"
    ranked.loc[10, "risk_flags"] = "strong_formula_leakage"
    ranked.loc[11, "risk_flags"] = "poor_data_quality"
    ranked.loc[12, "risk_flags"] = "severe_data_quality"
    ranked.loc[13, "risk_flags"] = "lag_boundary"
    expectations = pd.DataFrame(
        [
            {"variable": "x1", "expected_class": "reasonable_driver"},
            {"variable": "x11", "expected_class": "reasonable_driver"},
            {"variable": "x2", "expected_class": "implausible_driver"},
            {"variable": "x12", "expected_class": "implausible_driver"},
        ]
    )

    _, metrics = evaluate_ranking_baseline(ranked, expectations=expectations)

    top10 = metrics["cutoffs"]["10"]
    top20 = metrics["cutoffs"]["20"]
    assert top10["known_reasonable_hits"] == 1
    assert top10["known_implausible_hits"] == 1
    assert top10["target_leads_count"] == 1
    assert top10["common_capacity_count"] == 1
    assert top10["ab_target_leads_count"] == 1
    assert top10["ab_common_capacity_count"] == 1
    assert top10["ab_strong_formula_leakage_count"] == 0
    assert top10["ab_poor_data_quality_count"] == 0
    assert top20["known_reasonable_hits"] == 2
    assert top20["known_implausible_hits"] == 2
    assert top20["strong_formula_leakage_count"] == 1
    assert top20["poor_data_quality_count"] == 1
    assert top20["severe_data_quality_count"] == 1
    assert top20["data_quality_risk_count"] == 2
    assert top20["lag_boundary_count"] == 1


def test_quality_risk_flags_are_separate_and_not_double_counted():
    ranked = pd.DataFrame([
        {"variable": "poor", "risk_flags": "poor_data_quality"},
        {"variable": "severe", "risk_flags": "severe_data_quality"},
        {"variable": "both", "risk_flags": "poor_data_quality;severe_data_quality"},
        {"variable": "both", "risk_flags": "severe_data_quality"},
    ])

    detail, metrics = evaluate_ranking_baseline(ranked, cutoffs=(4,))
    top = metrics["cutoffs"]["4"]

    assert detail["poor_data_quality_flag"].tolist() == [True, False, True, False]
    assert detail["severe_data_quality_flag"].tolist() == [False, True, True, True]
    assert top["poor_data_quality_count"] == 2
    assert top["severe_data_quality_count"] == 2
    assert top["data_quality_risk_count"] == 3


def test_all_ab_risk_metrics_exclude_c_d_e_grades():
    tokens = [
        "target_leads_variable",
        "common_capacity_driver",
        "strong_formula_leakage",
        "poor_data_quality",
    ]
    ranked = pd.DataFrame(
        [
            {"variable": f"ab{index}", "candidate_grade": "A" if index % 2 == 0 else "B", "risk_flags": token}
            for index, token in enumerate(tokens)
        ]
        + [
            {"variable": f"non_ab{index}", "candidate_grade": grade, "risk_flags": token}
            for grade in ["C", "D", "E"]
            for index, token in enumerate(tokens)
        ]
    )

    _, metrics = evaluate_ranking_baseline(ranked, cutoffs=(20,))

    top = metrics["cutoffs"]["20"]
    assert top["ab_target_leads_count"] == 1
    assert top["ab_common_capacity_count"] == 1
    assert top["ab_strong_formula_leakage_count"] == 1
    assert top["ab_poor_data_quality_count"] == 1


def test_duplicate_ranked_variables_preserve_rows_and_unique_expectation_hits():
    ranked = pd.DataFrame(
        [
            {"variable": "x1", "candidate_grade": "A", "risk_flags": "target_leads_variable"},
            {"variable": "x1", "candidate_grade": "A", "risk_flags": "target_leads_variable"},
            {"variable": "x2", "candidate_grade": "C", "risk_flags": ""},
        ]
    )
    expectations = pd.DataFrame(
        [{"variable": "x1", "expected_class": "implausible_driver"}]
    )

    detail, metrics = evaluate_ranking_baseline(ranked, expectations=expectations, cutoffs=(3,))

    assert detail["variable"].tolist() == ["x1", "x1", "x2"]
    assert detail["current_rank"].tolist() == [1, 2, 3]
    assert metrics["duplicate_ranked_variable_count"] == 1
    assert metrics["duplicate_ranked_variables"] == ["x1"]
    top = metrics["cutoffs"]["3"]
    assert top["known_implausible_hits"] == 1
    assert top["target_leads_count"] == 2
    assert top["ab_target_leads_count"] == 1


def test_ab_data_quality_counts_are_unique_by_variable():
    ranked = pd.DataFrame(
        [
            {"variable": "poor", "candidate_grade": "A", "risk_flags": "poor_data_quality"},
            {"variable": "poor", "candidate_grade": "B", "risk_flags": "poor_data_quality"},
            {"variable": "severe", "candidate_grade": "A", "risk_flags": "severe_data_quality"},
            {"variable": "severe", "candidate_grade": "B", "risk_flags": "severe_data_quality"},
            {"variable": "both", "candidate_grade": "A", "risk_flags": "poor_data_quality"},
            {"variable": "both", "candidate_grade": "B", "risk_flags": "severe_data_quality"},
        ]
    )

    _, metrics = evaluate_ranking_baseline(ranked, cutoffs=(6,))
    top = metrics["cutoffs"]["6"]

    assert top["ab_poor_data_quality_count"] == 2
    assert top["ab_severe_data_quality_count"] == 2
    assert top["poor_data_quality_count"] == 2
    assert top["severe_data_quality_count"] == 2
    assert top["data_quality_risk_count"] == 3


def test_duplicate_risk_rows_align_by_variable_occurrence():
    ranked = pd.DataFrame(
        [
            {"variable": "x1", "target_leads_variable_flag": pd.NA},
            {"variable": "x1", "target_leads_variable_flag": pd.NA},
        ]
    )
    risks = pd.DataFrame(
        [
            {"variable": "x1", "target_leads_variable_flag": False},
            {"variable": "x1", "target_leads_variable_flag": True},
        ]
    )

    detail, metrics = evaluate_ranking_baseline(ranked, risks, cutoffs=(2,))

    assert detail["variable"].tolist() == ["x1", "x1"]
    assert detail["current_rank"].tolist() == [1, 2]
    assert detail["target_leads_variable_flag"].tolist() == [False, True]
    assert metrics["cutoffs"]["2"]["target_leads_count"] == 1


def test_recall_of_expected_and_found_are_distinct_when_expected_variable_is_missing():
    ranked = pd.DataFrame([{"variable": "found"}])
    expectations = pd.DataFrame(
        [
            {"variable": "found", "expected_class": "reasonable_driver"},
            {"variable": "missing", "expected_class": "reasonable_driver"},
        ]
    )

    _, metrics = evaluate_ranking_baseline(ranked, expectations=expectations, cutoffs=(1,))

    top1 = metrics["cutoffs"]["1"]
    assert top1["known_reasonable_recall_of_expected"] == 0.5
    assert top1["known_reasonable_recall_of_found"] == 1.0
    assert metrics["expectations"]["missing_variables"] == ["missing"]


def test_zero_reasonable_denominators_return_none():
    ranked = pd.DataFrame([{"variable": "x1"}])
    expectations = pd.DataFrame([{"variable": "x1", "expected_class": "implausible_driver"}])

    _, metrics = evaluate_ranking_baseline(ranked, expectations=expectations, cutoffs=(1,))

    top1 = metrics["cutoffs"]["1"]
    assert top1["known_reasonable_recall_of_expected"] is None
    assert top1["known_reasonable_recall_of_found"] is None


def test_illegal_expected_class_raises_clear_error():
    expectations = pd.DataFrame([{"variable": "x1", "expected_class": "important"}])

    with pytest.raises(ValueError, match="important"):
        evaluate_ranking_baseline(pd.DataFrame([{"variable": "x1"}]), expectations=expectations)


def test_duplicate_expectation_variable_raises_clear_error():
    expectations = pd.DataFrame(
        [
            {"variable": "x1", "expected_class": "neutral"},
            {"variable": "x1", "expected_class": "reasonable_driver"},
        ]
    )

    with pytest.raises(ValueError, match="x1"):
        evaluate_ranking_baseline(pd.DataFrame([{"variable": "x1"}]), expectations=expectations)


def test_missing_ranked_features_file_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="未找到 ranked_features.csv"):
        evaluate_run_directory(tmp_path)


def test_missing_optional_risk_file_still_completes(tmp_path: Path):
    pd.DataFrame([{"variable": "x1"}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )

    paths = evaluate_run_directory(tmp_path)

    assert paths["variables"].exists()
    assert paths["metrics"].exists()
    assert paths["markdown"].exists()


def test_cli_writes_outputs_and_standard_json(tmp_path: Path):
    pd.DataFrame([{"variable": "x1", "final_score": 0.9}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )
    output_dir = tmp_path / "out"

    assert main(["--run-dir", str(tmp_path), "--output-dir", str(output_dir)]) == 0

    variables = output_dir / "ranking_baseline_variables.csv"
    metrics = output_dir / "ranking_baseline_metrics.json"
    markdown = output_dir / "ranking_baseline.md"
    assert variables.exists()
    assert metrics.exists()
    assert markdown.exists()
    assert json.loads(metrics.read_text(encoding="utf-8"))["ranked_row_count"] == 1


def test_utf8_bom_csv_is_read(tmp_path: Path):
    pd.DataFrame([{"variable": "x1"}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )

    paths = evaluate_run_directory(tmp_path)
    detail = pd.read_csv(paths["variables"], encoding="utf-8-sig")

    assert detail.loc[0, "variable"] == "x1"


def test_static_module_does_not_call_scoring_algorithms():
    source = Path("chem_ts_corr/ranking_baseline.py").read_text(encoding="utf-8")

    for forbidden in [
        "final_ranked_features(",
        "risk_flags(",
        "build_lag_peak_quality(",
        "compute_lag_scores(",
        "model_lift_scores(",
        "rolling_corr_scores(",
        "run_granger_tests(",
        "fit_explainable_model(",
    ]:
        assert forbidden not in source

    assert "return bool(text)" not in source
    assert "def _optional_bool(" in source
    assert ".combine_first(" in source
    assert "token.lower() in parts" in source
    assert ".fillna(False)" in source
    assert ".astype(bool)" in source
