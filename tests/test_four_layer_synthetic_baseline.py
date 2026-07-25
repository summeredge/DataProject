from __future__ import annotations

import json
import subprocess
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


PR_8B_BASE = "a16335d0305c26cb752b9e02a11264989355a221"


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
    changed = CASES[name](
        n=int(first.metadata["n"]) + 8,
        noise=float(first.metadata["noise"]) * 1.1,
    )
    assert changed.frame.shape == (first.frame.shape[0] + 8, first.frame.shape[1])
    noise_changed = CASES[name](noise=float(first.metadata["noise"]) * 1.1)
    assert not noise_changed.frame.equals(first.frame)


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


def _proxy_metric_case(reference_flags: str, proxy_flags: str, *, proxy_rank: int = 2, proxy_grade: str = "B"):
    case = SyntheticCase(
        frame=pd.DataFrame(), target="target", true_drivers=frozenset({"reference"}),
        spurious_variables=frozenset({"proxy"}), lags={}, directions={},
        variable_types={"proxy": frozenset({"proxy"})},
        reference_map={"proxy": "reference"}, metadata={"scenario": "unit"},
    )
    ranked = pd.DataFrame([
        {"variable": "reference", "driver_rank": 1, "candidate_grade": "A", "driver_priority_factor": 1.0, "driver_priority_score": 0.80, "risk_flags": reference_flags, "corr_q_value": 0.01},
        {"variable": "proxy", "driver_rank": proxy_rank, "candidate_grade": proxy_grade, "driver_priority_factor": 0.80, "driver_priority_score": 0.70, "risk_flags": proxy_flags, "corr_q_value": 0.01},
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
    assert result["proxy_separation_rate"] == result["proxy_identification_rate"]


def test_proxy_metrics_record_not_detected_relationship():
    result = _proxy_metric_case("", "")
    detail = result["proxy_separation_details"][0]
    assert detail["group_detected"] is False
    assert detail["resolution_status"] == "not_detected"
    assert detail["separated"] is False


def test_proxy_metrics_record_conflicting_assignment():
    result = _proxy_metric_case("redundant_proxy", "")
    detail = result["proxy_separation_details"][0]
    assert detail["group_detected"] is True
    assert detail["resolution_status"] == "conflicting_assignment"
    assert detail["separated"] is False


@pytest.mark.parametrize(
    "name", ["collinear_proxy", "mixed_evidence", "model_incremental_validation"]
)
def test_redundancy_resolution_is_invariant_to_column_order(name: str, tmp_path: Path):
    case = CASES[name]()
    candidates = [column for column in case.frame if column != case.target]
    orders = [
        list(case.frame.columns),
        list(reversed(candidates)) + [case.target],
        list(np.random.default_rng(809).permutation(case.frame.columns)),
    ]
    fields = [
        "layer3_independence_status",
        "candidate_class",
        "driver_priority_factor",
        "driver_priority_score",
        "driver_rank",
        "candidate_grade",
        "recommended_use",
    ]
    results = []
    for index, columns in enumerate(orders):
        reordered = replace(case, frame=case.frame.loc[:, columns])
        ranked = run_case(reordered, tmp_path / f"order-{index}")
        ranked["redundant_proxy_flag"] = ranked["risk_flags"].str.contains(
            r"(?:^|;)redundant_proxy(?:;|$)", regex=True
        )
        results.append(
            ranked.reindex(columns=["variable", "redundant_proxy_flag", *fields])
            .sort_values("variable")
            .reset_index(drop=True)
        )
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0], check_exact=True)


@pytest.mark.parametrize("name", list(CASES))
def test_two_complete_production_runs_are_identical(name: str, tmp_path: Path):
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


def test_true_lagged_driver_contract(tmp_path: Path):
    case, _, ranked = _indexed("true_lagged_driver", tmp_path)
    row = ranked.loc["x_driver"]
    assert int(row["driver_rank"]) <= 3
    assert abs(int(row["lag"]) - case.lags["x_driver"]) <= 1
    assert int(row["lag"]) > 0
    assert row["candidate_class"] == "upstream_driver_candidate"
    assert "target_leads_variable" not in str(row["risk_flags"])


def test_downstream_response_contract(tmp_path: Path):
    case, _, ranked = _indexed("downstream_response", tmp_path)
    row = ranked.loc["x_downstream"]
    assert int(row["lag"]) == case.lags["x_downstream"] < 0
    assert "target_leads_variable" in str(row["risk_flags"])
    assert row["candidate_class"] == "downstream_response"
    assert row["candidate_grade"] not in {"A", "B"}
    assert row["layer2_temporal_status"] == "conflicting"


def test_common_driver_contract(tmp_path: Path):
    case, raw, ranked = _indexed("common_driver", tmp_path)
    assert {"z_driver", "x_common"} <= set(ranked.index)
    assert pd.notna(ranked.loc["x_common", "independent_signal_score"])
    assert "common_capacity_driver" in str(ranked.loc["x_common", "risk_flags"])
    assert int(ranked.loc["z_driver", "driver_rank"]) < int(
        ranked.loc["x_common", "driver_rank"]
    )
    assert metrics(case, raw)["common_driver_suppression_rate"] == 1.0
    assert ranked.loc["x_common", "layer3_independence_status"] == "not_supported"


def test_collinear_proxy_group_is_detected_but_not_resolved(tmp_path: Path):
    case, raw, ranked = _indexed("collinear_proxy", tmp_path)
    true = ranked.loc["x1_driver"]
    proxy = ranked.loc["x2_proxy"]
    assert case.frame["x1_driver"].corr(case.frame["x2_proxy"]) > 0.99
    assert "redundant_proxy" in str(true["risk_flags"])
    assert "redundant_proxy" in str(proxy["risk_flags"])
    assert true["layer3_independence_status"] == "not_supported"
    assert proxy["layer3_independence_status"] == "not_supported"
    assert true["candidate_class"] == proxy["candidate_class"]
    assert true["driver_priority_factor"] == proxy["driver_priority_factor"]
    detail = metrics(case, raw)["proxy_separation_details"][0]
    assert metrics(case, raw)["redundancy_group_detection_rate"] == 1.0
    assert metrics(case, raw)["proxy_identification_rate"] == 0.0
    assert metrics(case, raw)["proxy_separation_rate"] == 0.0
    assert detail["resolution_status"] == "unresolved_group"
    assert detail["separated"] is False


def test_noise_and_spurious_false_positive_rates_are_distinct(tmp_path: Path):
    common_case = CASES["common_driver"]()
    common_ranked = run_case(common_case, tmp_path / "common")
    common_metrics = metrics(common_case, common_ranked)
    assert common_metrics["common_driver_suppression_rate"] is not None

    collinear_case = CASES["collinear_proxy"]()
    collinear_ranked = run_case(collinear_case, tmp_path / "collinear")
    collinear_metrics = metrics(collinear_case, collinear_ranked)
    proxy_grade = collinear_ranked.set_index("variable").loc[
        "x2_proxy", "candidate_grade"
    ]
    assert collinear_metrics["noise_high_grade_false_positive_rate"] == 0.0
    assert collinear_metrics["spurious_high_grade_false_positive_rate"] == (
        1.0 if proxy_grade in {"A", "B"} else 0.0
    )

    mixed_case = CASES["mixed_evidence"]()
    mixed_ranked = run_case(mixed_case, tmp_path / "mixed")
    mixed_metrics = metrics(mixed_case, mixed_ranked)
    indexed = mixed_ranked.set_index("variable")
    noise_high = indexed.loc["noise", "candidate_grade"] in {"A", "B"}
    spurious_high = [
        indexed.loc[name, "candidate_grade"] in {"A", "B"}
        for name in mixed_case.spurious_variables
    ]
    assert mixed_metrics["noise_high_grade_false_positive_rate"] == float(noise_high)
    assert mixed_metrics["spurious_high_grade_false_positive_rate"] == sum(
        spurious_high
    ) / len(spurious_high)
    assert collinear_metrics["proxy_separation_rate"] is not None
    assert mixed_metrics["common_driver_suppression_rate"] is not None
    assert mixed_metrics["proxy_separation_rate"] is not None
    assert set(collinear_metrics["proxy_separation_details"][0]) == {
        "variable",
        "reference",
        "variable_rank",
        "reference_rank",
        "rank_gap",
        "variable_grade",
        "reference_grade",
        "score_gap",
        "risk_flags",
        "reference_risk_flags",
        "group_detected",
        "resolution_status",
        "separated",
    }
    assert set(mixed_metrics["common_driver_details"][0]) == {
        "variable",
        "reference",
        "variable_rank",
        "reference_rank",
        "rank_gap",
        "variable_grade",
        "reference_grade",
        "score_gap",
        "risk_flags",
        "suppressed",
    }


def test_nonlinear_scenario_data_contract():
    case = CASES["nonlinear_stable_driver"]()
    expected_noise = {f"noise_{index:02d}" for index in range(8)}
    assert case.variable_types == {
        "true_driver": frozenset({"x_nonlinear"}),
        "noise": frozenset(expected_noise),
    }
    assert len(set(case.frame) - {case.target}) == 9
    lag = case.lags["x_nonlinear"]
    aligned = case.frame["x_nonlinear"].shift(lag)
    valid = pd.DataFrame({"x": aligned, "y": case.frame["target"]}).dropna()
    assert abs(valid["x"].corr(valid["y"], method="pearson")) < 0.15
    assert abs(valid["x"].corr(valid["y"], method="spearman")) < 0.15
    assert valid["x"].pow(2).corr(valid["y"]) > 0.90


def test_nonlinear_true_lag_contract(tmp_path: Path):
    case, _, ranked = _indexed("nonlinear_stable_driver", tmp_path)
    row = ranked.loc["x_nonlinear"]
    noise_scores = pd.to_numeric(
        ranked.loc[
            ranked.index.isin(case.variable_types["noise"]), "prediction_score"
        ],
        errors="coerce",
    ).fillna(0.0)
    assert int(row["driver_rank"]) <= 3
    assert row["candidate_grade"] in {"A", "B", "C"}
    assert float(row["prediction_score"]) > 0.05
    assert float(row["prediction_score"]) >= float(noise_scores.max()) + 0.05
    assert "low_model_lift" not in str(row["risk_flags"])
    assert row["layer4_model_status"] == "supported"


def test_regime_sign_reversal_contract(tmp_path: Path):
    case, _, ranked = _indexed("regime_sign_reversal", tmp_path)
    lag = case.lags["x_reversal"]
    shifted = case.frame["x_reversal"].shift(lag)
    low = case.frame["load"] < 0
    high = case.frame["load"] > 0
    low_corr = shifted[low].corr(case.frame.loc[low, "target"])
    high_corr = shifted[high].corr(case.frame.loc[high, "target"])
    assert low_corr < -0.8
    assert high_corr > 0.8
    row = ranked.loc["x_reversal"]
    assert bool(row["regime_sign_reversal_flag"])
    assert float(row["regime_sign_consistency"]) < 0.2
    assert float(row["regime_stability_final"]) < 0.2
    assert float(row["stability_score"]) < 0.35
    assert "unstable_across_regimes" in str(row["risk_flags"])
    assert row["candidate_grade"] not in {"A", "B"}
    assert row["stability_status"] == "conflicting"
    assert row["data_quality_status"] == "supported"
    assert int(row["driver_rank"]) >= 1


def test_outlier_scenario_data_contract():
    case = CASES["outlier_driven_correlation"]()
    outliers = case.metadata["outlier_indices"]
    full = abs(case.frame["x_outlier"].corr(case.frame["target"]))
    clean = case.frame.drop(case.frame.index[outliers])
    clean_corr = abs(clean["x_outlier"].corr(clean["target"]))
    assert full > 0.75
    assert clean_corr < case.metadata["clean_expected_correlation_max"]
    assert full - clean_corr > 0.60


def test_outlier_production_contract(tmp_path: Path):
    _, _, ranked = _indexed("outlier_driven_correlation", tmp_path)
    row = ranked.loc["x_outlier"]
    assert row["candidate_grade"] not in {"A", "B"}
    assert float(row["data_quality_score"]) <= 1.0
    assert (
        "poor_data_quality" in str(row["risk_flags"])
        or "unstable_over_time" in str(row["risk_flags"])
        or float(row["stability_score"]) < 0.35
    )


def test_lag_boundary_contract(tmp_path: Path):
    case, _, ranked = _indexed("lag_boundary_artifact", tmp_path)
    row = ranked.loc["x_boundary"]
    assert int(row["lag"]) == int(case.metadata["max_lag"])
    assert bool(row["lag_boundary_flag"])
    assert "lag_boundary" in str(row["risk_flags"])
    assert row["candidate_grade"] not in {"A", "B"}
    assert row["layer2_temporal_status"] == "partially_supported"


def test_noise_only_fdr_and_multi_seed_false_positives(tmp_path: Path):
    summary = evaluate_noise_false_positives(
        CASES["noise_only"],
        tmp_path,
    )
    assert summary["seeds"] == [109, 110, 111, 112, 113]
    assert summary["high_grade_false_positive_rate"] == 0.0
    assert summary["false_positive_run_count"] == 0
    assert summary["maximum_false_positive_grade"] not in {"A", "B"}
    for run in summary["runs"]:
        assert run["variable_count"] >= 30
        assert run["noise_top_5_rate"] <= 5 / 30
        assert run["significant_q_count"] <= 3
        assert not run["high_grade_variables"]


def test_model_incremental_validation_runs_layer_4(tmp_path: Path):
    _, _, ranked = _indexed("model_incremental_validation", tmp_path)
    true = ranked.loc["x_incremental"]
    noise = ranked.loc["noise"]
    assert str(true["model_lift_status"]).startswith("ok")
    assert pd.notna(true["prediction_score"])
    assert pd.isna(noise["prediction_score"]) or float(true["prediction_score"]) > float(
        noise["prediction_score"]
    )


def test_model_incremental_proxy_separation(tmp_path: Path):
    case, raw, ranked = _indexed("model_incremental_validation", tmp_path)
    proxy = ranked.loc["x_proxy"]
    assert proxy["layer3_independence_status"] == "not_supported"
    detail = metrics(case, raw)["proxy_separation_details"][0]
    assert detail["resolution_status"] in {"identified_proxy", "unresolved_group"}


def test_mixed_evidence_true_drivers_and_noise(tmp_path: Path):
    case, _, ranked = _indexed("mixed_evidence", tmp_path)
    for variable in ["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]:
        assert variable in ranked.index
    assert all(int(ranked.loc[name, "driver_rank"]) <= 3 for name in case.true_drivers)
    assert all(int(ranked.loc[name, "lag"]) > 0 for name in case.true_drivers)
    assert all(
        "target_leads_variable" not in str(ranked.loc[name, "risk_flags"])
        for name in case.true_drivers
    )
    assert ranked.loc["noise", "candidate_grade"] not in {"A", "B"}
    assert int(ranked.loc["noise", "driver_rank"]) > int(
        ranked.loc["x_driver", "driver_rank"]
    )
    assert pd.notna(ranked.loc["x_driver", "prediction_score"])


def test_mixed_evidence_downstream_contract(tmp_path: Path):
    _, _, ranked = _indexed("mixed_evidence", tmp_path)
    assert "target_leads_variable" in str(ranked.loc["x_downstream", "risk_flags"])
    assert ranked.loc["x_downstream", "candidate_grade"] not in {"A", "B"}
    assert ranked.loc["x_downstream", "layer2_temporal_status"] == "conflicting"


def test_mixed_evidence_common_driver_contract(tmp_path: Path):
    case, raw, ranked = _indexed("mixed_evidence", tmp_path)
    assert "common_capacity_driver" in str(ranked.loc["x_common", "risk_flags"])
    assert int(ranked.loc["x_common", "driver_rank"]) > min(
        int(ranked.loc[name, "driver_rank"]) for name in ["x_driver", "z_driver"]
    )
    assert metrics(case, raw)["common_driver_suppression_rate"] == 1.0


def test_mixed_evidence_proxy_contract(tmp_path: Path):
    case, raw, ranked = _indexed("mixed_evidence", tmp_path)
    proxy = ranked.loc["x_proxy"]
    driver = ranked.loc["x_driver"]
    assert "redundant_proxy" in str(proxy["risk_flags"])
    assert "redundant_proxy" in str(driver["risk_flags"])
    assert proxy["layer3_independence_status"] == "not_supported"
    assert driver["layer3_independence_status"] == "not_supported"
    assert proxy["candidate_class"] == driver["candidate_class"]
    assert proxy["driver_priority_factor"] == driver["driver_priority_factor"]
    detail = metrics(case, raw)["proxy_separation_details"][0]
    assert metrics(case, raw)["redundancy_group_detection_rate"] == 1.0
    assert metrics(case, raw)["proxy_identification_rate"] == 0.0
    assert metrics(case, raw)["proxy_separation_rate"] == 0.0
    assert detail["resolution_status"] == "unresolved_group"


@pytest.mark.parametrize("name", sorted(STABILITY_SCENARIOS))
def test_rank_stability_is_reproducible(name: str, tmp_path: Path):
    first = evaluate_rank_stability(CASES[name], tmp_path / "first")
    second = evaluate_rank_stability(CASES[name], tmp_path / "second")
    if name == "noise_only":
        first["multi_seed_false_positives"] = evaluate_noise_false_positives(
            CASES[name], tmp_path / "first-noise"
        )
        second["multi_seed_false_positives"] = evaluate_noise_false_positives(
            CASES[name], tmp_path / "second-noise"
        )
    assert first == second
    assert len(first["candidate_missing_counts"]) == 3
    assert all(
        set(item) == {"label", "missing_from_variant", "new_in_variant"}
        for item in first["candidate_missing_counts"]
    )
    assert all(
        passed for _, passed, _ in stability_contract_checks(name, first)
    )


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


def test_committed_baseline_matches_actual_production_report(tmp_path: Path):
    actual = build_full_baseline_report(tmp_path)
    committed = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    _assert_recursive(actual, committed)
    assert set(committed) == set(CASES)
    for report in committed.values():
        required = {
            "scenario",
            "seed",
            "samples",
            "parameters",
            "variable_types",
            "reference_map",
            "top_k",
            "top_k_results",
            "key_evidence",
            "metrics",
            "stability_metrics",
            "passed",
            "failure_reason",
            "failed_expectations",
            "passed_expectations",
        }
        assert set(report) == required
        assert report["top_k"] == REPORT_TOP_K
        assert len(report["top_k_results"]) <= REPORT_TOP_K
        assert report["top_k_results"] == [
            evidence["variable"] for evidence in report["key_evidence"][:REPORT_TOP_K]
        ]
        assert all(set(evidence) == set(KEY_FIELDS) for evidence in report["key_evidence"])
        assert report["passed"] == (not report["failed_expectations"])
        assert bool(report["failure_reason"]) == (not report["passed"])
        if not report["passed"]:
            assert any(
                layer in report["failure_reason"]
                for layer in [
                    "layer_1",
                    "layer_2",
                    "layer_3",
                    "layer_4",
                    "stability",
                    "mixed",
                ]
            )


def test_pr_8d_only_changes_audited_explanation_production_paths():
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", PR_8B_BASE],
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    changed = tracked + untracked
    production = {
        path
        for path in changed
        if path.startswith("chem_ts_corr/") and path.endswith(".py")
    }
    assert production <= {
        "chem_ts_corr/auto_" + "closed" + "_loop.py",
        "chem_ts_corr/causal_review_service.py",
        "chem_ts_corr/closed" + "_loop.py",
        "chem_ts_corr/closed" + "_loop_calibration.py",
        "chem_ts_corr/config.py",
        "chem_ts_corr/evidence_explanations.py",
        "chem_ts_corr/final_review_summary.py",
        "chem_ts_corr/llm_report.py",
        "chem_ts_corr/pipeline.py",
        "chem_ts_corr/ranking_baseline.py",
        "chem_ts_corr/report.py",
        "chem_ts_corr/screening.py",
        "chem_ts_corr/service.py",
        "chem_ts_corr/web.py",
    }


def test_pr_8c_source_contracts_cover_each_repaired_ranking_path():
    source = Path("chem_ts_corr/screening.py").read_text(encoding="utf-8")
    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("chem_ts_corr").rglob("*.py")
    )
    assert "abs(best_lag)" not in production_source
    assert "target_leads_variable" in source
    assert "lag_boundary" in source
    assert "_redundant_proxy_variables" in source
    assert "robust_outlier_ratio" in source
    assert "__squared" in source
    assert 'INDUSTRIAL_SCORE_COMPONENTS = ("association", "prediction", "stability", "lag_quality")' in source
    for field in [
        "layer1_association_status",
        "layer2_temporal_status",
        "layer3_independence_status",
        "layer4_model_status",
        "stability_status",
        "data_quality_status",
        "evidence_missing_items",
        "evidence_conflict_items",
    ]:
        assert field in source
