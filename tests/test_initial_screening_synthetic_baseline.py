from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.synthetic_cases.evaluate import (
    KEY_FIELDS,
    REPORT_TOP_K,
    evaluate_noise_false_positives,
    evaluate_rank_stability,
    metrics,
    run_case,
    stability_contract_checks,
)
from tests.synthetic_cases.four_layer_cases import CASES, SyntheticCase
from tests.synthetic_cases.generate_baseline import (
    BASELINE_PATH,
    STABILITY_SCENARIOS,
    build_case_report,
    build_full_baseline_report,
)


def _indexed(case_name: str, tmp_path: Path):
    case = CASES[case_name]()
    ranked = run_case(case, tmp_path)
    return case, ranked, ranked.set_index("variable")


@pytest.mark.parametrize("name", list(CASES))
def test_generators_are_deterministic_and_metadata_complete(name: str):
    first = CASES[name]()
    second = CASES[name]()
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert isinstance(first.frame.index, pd.DatetimeIndex)
    assert first.frame.index.is_monotonic_increasing
    assert first.frame.index.is_unique
    assert first.target in first.frame
    assert first.true_drivers.isdisjoint(first.spurious_variables)
    declared = set().union(*first.variable_types.values())
    candidates = set(first.frame) - {first.target}
    assert declared == candidates
    typed = [variable for values in first.variable_types.values() for variable in values]
    assert len(typed) == len(set(typed))
    assert first.true_drivers | first.spurious_variables <= candidates
    assert set(first.lags) | set(first.directions) <= candidates
    assert first.true_drivers <= set(first.lags)
    assert first.true_drivers <= set(first.directions)
    assert set(first.reference_map) <= first.spurious_variables
    assert set(first.reference_map.values()) <= first.true_drivers
    assert {"scenario", "seed", "n", "noise"} <= set(first.metadata)


def test_reference_maps_identify_each_spurious_relationship():
    assert CASES["common_driver"]().reference_map == {"x_common": "z_driver"}
    assert CASES["collinear_proxy"]().reference_map == {"x2_proxy": "x1_driver"}
    assert CASES["mixed_evidence"]().reference_map == {
        "x_common": "z_driver",
        "x_proxy": "x_driver",
    }
    assert CASES["model_incremental_validation"]().reference_map == {
        "x_proxy": "x_incremental"
    }


def _proxy_metric_case(reference_flags: str, proxy_flags: str):
    case = SyntheticCase(
        frame=pd.DataFrame(), target="target", true_drivers=frozenset({"reference"}),
        spurious_variables=frozenset({"proxy"}), lags={}, directions={},
        variable_types={"proxy": frozenset({"proxy"})},
        reference_map={"proxy": "reference"}, metadata={"scenario": "unit"},
    )
    ranked = pd.DataFrame([
        {"variable": "reference", "final_rank": 1, "final_score": 0.80, "candidate_grade": "A", "risk_flags": reference_flags, "corr_q_value": 0.01},
        {"variable": "proxy", "final_rank": 2, "final_score": 0.70, "candidate_grade": "B", "risk_flags": proxy_flags, "corr_q_value": 0.01},
    ])
    return metrics(case, ranked)


def test_proxy_metrics_do_not_treat_unresolved_group_as_identified():
    result = _proxy_metric_case("redundant_proxy", "redundant_proxy")
    detail = result["proxy_separation_details"][0]
    assert detail["group_detected"] is True
    assert detail["resolution_status"] == "unresolved_group"
    assert detail["separated"] is False
    assert result["redundancy_group_detection_rate"] == 1.0
    assert result["proxy_identification_rate"] == 0.0
    assert result["proxy_separation_rate"] == 0.0


def test_proxy_metrics_require_reference_to_remain_unflagged():
    result = _proxy_metric_case("", "redundant_proxy")
    detail = result["proxy_separation_details"][0]
    assert detail["resolution_status"] == "identified_proxy"
    assert detail["separated"] is True
    assert result["proxy_identification_rate"] == 1.0


def test_proxy_metrics_record_conflicting_assignment():
    result = _proxy_metric_case("redundant_proxy", "")
    detail = result["proxy_separation_details"][0]
    assert detail["resolution_status"] == "conflicting_assignment"
    assert detail["separated"] is False


@pytest.mark.parametrize("name", ["collinear_proxy", "mixed_evidence", "model_incremental_validation"])
def test_redundancy_resolution_is_invariant_to_column_order(name: str, tmp_path: Path):
    case = CASES[name]()
    candidates = [column for column in case.frame if column != case.target]
    orders = [
        list(case.frame.columns),
        list(reversed(candidates)) + [case.target],
        list(np.random.default_rng(809).permutation(case.frame.columns)),
    ]
    fields = ["variable", "final_score", "candidate_grade", "recommended_use", "risk_flags"]
    results = []
    for index, columns in enumerate(orders):
        reordered = replace(case, frame=case.frame.loc[:, columns])
        ranked = run_case(reordered, tmp_path / f"order-{index}")
        ranked["redundant_proxy_flag"] = ranked["risk_flags"].str.contains(
            r"(?:^|;)redundant_proxy(?:;|$)", regex=True
        )
        results.append(
            ranked.reindex(columns=[*fields, "redundant_proxy_flag"])
            .sort_values("variable")
            .reset_index(drop=True)
        )
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0], check_exact=True)


@pytest.mark.parametrize("name", list(CASES))
def test_two_complete_initial_runs_are_identical(name: str, tmp_path: Path):
    first_case = CASES[name]()
    second_case = CASES[name]()
    first = run_case(first_case, tmp_path / "first")
    second = run_case(second_case, tmp_path / "second")
    pd.testing.assert_frame_equal(
        first.reindex(columns=KEY_FIELDS),
        second.reindex(columns=KEY_FIELDS),
        check_exact=True,
    )
    assert metrics(first_case, first) == metrics(second_case, second)
    assert build_case_report(first_case, first) == build_case_report(second_case, second)


def test_initial_contract_fields_are_score_ranked(tmp_path: Path):
    case, _, ranked = _indexed("true_lagged_driver", tmp_path)
    assert ranked["final_score"].is_monotonic_decreasing
    assert ranked["final_rank"].tolist() == list(range(1, len(ranked) + 1))
    row = ranked.loc["x_driver"]
    assert int(row["final_rank"]) <= 3
    assert abs(int(row["lag"]) - case.lags["x_driver"]) <= 1
    assert row["candidate_class"] == "upstream_driver_candidate"
    assert row["driver_priority_factor"] == 1.0
    assert row["driver_priority_score"] == row["final_score"]


def test_downstream_response_preserves_direction_and_grade_cap(tmp_path: Path):
    case, _, ranked = _indexed("downstream_response", tmp_path)
    row = ranked.loc["x_downstream"]
    assert int(row["lag"]) == case.lags["x_downstream"] < 0
    assert "target_leads_variable" in str(row["risk_flags"])
    assert row["candidate_class"] == "downstream_response"
    assert row["candidate_grade"] not in {"A", "B"}


def test_common_driver_and_collinear_proxy_signals_remain_visible(tmp_path: Path):
    case, raw, ranked = _indexed("common_driver", tmp_path)
    assert {"z_driver", "x_common"} <= set(ranked.index)
    assert int(ranked.loc["z_driver", "final_rank"]) < int(ranked.loc["x_common", "final_rank"])
    assert metrics(case, raw)["common_driver_details"]

    case, raw, ranked = _indexed("collinear_proxy", tmp_path / "collinear")
    assert "redundant_proxy" in str(ranked.loc["x1_driver", "risk_flags"])
    assert "redundant_proxy" in str(ranked.loc["x2_proxy", "risk_flags"])
    proxy_metrics = metrics(case, raw)
    assert proxy_metrics["redundancy_group_detection_rate"] == 1.0
    assert proxy_metrics["proxy_identification_rate"] == 0.0
    assert proxy_metrics["proxy_separation_rate"] == 0.0
    assert proxy_metrics["proxy_separation_details"][0]["resolution_status"] == "unresolved_group"


def test_nonlinear_scenario_is_initial_only(tmp_path: Path):
    case = CASES["nonlinear_stable_driver"]()
    lag = case.lags["x_nonlinear"]
    aligned = case.frame["x_nonlinear"].shift(lag)
    valid = pd.DataFrame({"x": aligned, "y": case.frame["target"]}).dropna()
    assert abs(valid["x"].corr(valid["y"], method="pearson")) < 0.15
    assert valid["x"].pow(2).corr(valid["y"]) > 0.90
    _, _, ranked = _indexed("nonlinear_stable_driver", tmp_path)
    assert "prediction_score" not in ranked.columns
    assert "model_lift_status" not in ranked.columns
    assert "x_nonlinear" in ranked.index


def test_regime_review_is_not_executed_by_initial_screening(tmp_path: Path):
    case, _, ranked = _indexed("regime_sign_reversal", tmp_path)
    lag = case.lags["x_reversal"]
    shifted = case.frame["x_reversal"].shift(lag)
    low = case.frame["load"] < 0
    high = case.frame["load"] > 0
    assert shifted[low].corr(case.frame.loc[low, "target"]) < -0.8
    assert shifted[high].corr(case.frame.loc[high, "target"]) > 0.8
    assert "regime_stability_final" not in ranked.columns
    assert "x_reversal" in ranked.index


def test_outlier_and_lag_boundary_quality_signals_are_preserved(tmp_path: Path):
    _, _, ranked = _indexed("outlier_driven_correlation", tmp_path)
    row = ranked.loc["x_outlier"]
    assert row["candidate_grade"] not in {"A", "B"}
    assert float(row["data_quality_score"]) <= 1.0

    case, _, ranked = _indexed("lag_boundary_artifact", tmp_path / "boundary")
    row = ranked.loc["x_boundary"]
    assert int(row["lag"]) == int(case.metadata["max_lag"])
    assert bool(row["lag_boundary_flag"])
    assert "lag_boundary" in str(row["risk_flags"])
    assert row["candidate_grade"] not in {"A", "B"}


def test_noise_only_controls_false_positives(tmp_path: Path):
    summary = evaluate_noise_false_positives(CASES["noise_only"], tmp_path)
    assert summary["seeds"] == [109, 110, 111, 112, 113]
    assert summary["high_grade_false_positive_rate"] == 0.0
    assert summary["false_positive_run_count"] == 0
    assert summary["maximum_false_positive_grade"] not in {"A", "B"}


def test_model_incremental_case_keeps_model_followup_independent(tmp_path: Path):
    case, raw, ranked = _indexed("model_incremental_validation", tmp_path)
    assert "model_lift_status" not in ranked.columns
    assert "prediction_score" not in ranked.columns
    detail = metrics(case, raw)["proxy_separation_details"][0]
    assert detail["resolution_status"] in {"identified_proxy", "unresolved_group"}


def test_mixed_evidence_keeps_core_risk_signals_and_no_followup_fields(tmp_path: Path):
    case, raw, ranked = _indexed("mixed_evidence", tmp_path)
    indexed = ranked
    for variable in ["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]:
        assert variable in indexed.index
    assert all(name in indexed.index for name in case.true_drivers)
    assert all(int(indexed.loc[name, "lag"]) > 0 for name in case.true_drivers)
    assert "target_leads_variable" in str(indexed.loc["x_downstream", "risk_flags"])
    assert metrics(case, raw)["common_driver_details"]
    assert "prediction_score" not in indexed.columns


@pytest.mark.parametrize("name", sorted(STABILITY_SCENARIOS))
def test_rank_stability_is_reproducible(name: str, tmp_path: Path):
    first = evaluate_rank_stability(CASES[name], tmp_path / "first")
    second = evaluate_rank_stability(CASES[name], tmp_path / "second")
    if name == "noise_only":
        first["multi_seed_false_positives"] = evaluate_noise_false_positives(CASES[name], tmp_path / "first-noise")
        second["multi_seed_false_positives"] = evaluate_noise_false_positives(CASES[name], tmp_path / "second-noise")
    assert first == second
    assert all(passed for _, passed, _ in stability_contract_checks(name, first))


def _assert_recursive(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_recursive(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_recursive(actual_item, expected_item)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_committed_initial_baseline_matches_actual_report(tmp_path: Path):
    actual = build_full_baseline_report(tmp_path)
    committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    _assert_recursive(actual, committed)
    assert set(committed) == set(CASES)
    for report in committed.values():
        assert report["top_k"] == REPORT_TOP_K
        assert report["top_k_results"] == [evidence["variable"] for evidence in report["key_evidence"][:REPORT_TOP_K]]
        assert all(set(evidence) == set(KEY_FIELDS) for evidence in report["key_evidence"])
        assert report["passed"] == (not report["failed_expectations"])


def test_initial_source_has_no_four_layer_explanation_dependency():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    assert "evidence_explanations" not in source
    assert "layer1_association_status" not in source
    assert "four_layer_coverage_status" not in source
    assert "add_evidence_explanations" not in source


def test_statistical_screening_contracts_remain_in_source():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    assert "abs(best_lag)" not in source
    assert "target_leads_variable" in source
    assert "lag_boundary" in source
    assert "_redundant_proxy_variables" in source
    assert "robust_outlier_ratio" in source
    assert "__squared" in source
    assert "def order_initial_candidates" in source
    assert '"_initial_final_score"' in source
