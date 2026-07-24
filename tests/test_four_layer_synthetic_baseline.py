from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic_cases.evaluate import KEY_FIELDS, metrics, run_case
from tests.synthetic_cases.four_layer_cases import CASES


@pytest.mark.parametrize("name", list(CASES))
def test_generators_are_deterministic_and_complete(name: str):
    first, second = CASES[name](), CASES[name]()
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert isinstance(first.frame.index, pd.DatetimeIndex)
    assert first.target in first.frame
    assert first.metadata["seed"]
    assert first.true_drivers.isdisjoint(first.spurious_variables)
    assert first.true_drivers | first.spurious_variables <= set(first.frame.columns)
    assert set(first.lags) | set(first.directions) <= set(first.frame.columns)
    assert set(first.true_drivers) <= set(first.lags) | ({"z_driver"} if name == "common_driver" else set())
    assert {"n", "noise", "scenario"} <= set(first.metadata)
    changed = CASES[name](n=first.metadata["n"] + 8, noise=float(first.metadata["noise"]) * 1.1)
    assert changed.frame.shape[0] == first.frame.shape[0] + 8


def test_true_lagged_driver_has_rank_direction_and_lag(tmp_path):
    case = CASES["true_lagged_driver"](); ranked = run_case(case, tmp_path)
    row = ranked.set_index("variable").loc["x_driver"]
    assert int(row.driver_rank) <= 3
    assert abs(int(row.lag) - case.lags["x_driver"]) <= 1
    assert "target_leads_variable" not in str(row.risk_flags)
    assert row.candidate_grade in {"A", "B", "C"}


@pytest.mark.xfail(strict=True, reason="target_leads_variable is flagged but does not cap candidate_grade; follow-up should bind temporal direction to confidence")
def test_downstream_response_is_not_highest_confidence(tmp_path):
    row = run_case(CASES["downstream_response"](), tmp_path).set_index("variable").loc["x_downstream"]
    assert "target_leads_variable" in str(row.risk_flags)
    assert row.candidate_grade not in {"A", "B"}


@pytest.mark.xfail(strict=True, reason="lag_boundary is flagged but does not cap candidate_grade; follow-up should distinguish boundary artifact from confirmed lag")
def test_lag_boundary_is_explicit_and_not_high_grade(tmp_path):
    row = run_case(CASES["lag_boundary_artifact"](), tmp_path).set_index("variable").loc["x_boundary"]
    assert "lag_boundary" in str(row.risk_flags)
    assert row.candidate_grade not in {"A", "B"}


def test_collinear_proxy_is_not_indistinguishable_from_true_driver(tmp_path):
    ranked = run_case(CASES["collinear_proxy"](), tmp_path).set_index("variable")
    assert ranked.loc["x1_driver", "driver_rank"] < ranked.loc["x2_proxy", "driver_rank"]


@pytest.mark.parametrize("name", ["common_driver", "mixed_evidence"])
@pytest.mark.xfail(strict=True, reason="common_capacity_driver is not detected when the control is also a ranked candidate; follow-up should preserve independent/common-load evidence")
def test_full_production_entry_suppresses_common_driver_without_hiding_true_driver(tmp_path, name):
    case = CASES[name](); ranked = run_case(case, tmp_path).set_index("variable")
    assert "common_capacity_driver" in str(ranked.loc["x_common", "risk_flags"])
    assert min(ranked.loc[v, "driver_rank"] for v in case.true_drivers if v in ranked.index) < ranked.loc["x_common", "driver_rank"]


@pytest.mark.xfail(strict=True, reason="layer 1 is Pearson/Spearman lag screening only; weak-Pearson U-shaped driver needs nonlinear/model evidence")
def test_nonlinear_driver_is_not_rejected_by_linear_screening(tmp_path):
    case = CASES["nonlinear_stable_driver"]()
    assert abs(case.frame["target"].corr(case.frame["x_nonlinear"], method="pearson")) < 0.15
    row = run_case(case, tmp_path).set_index("variable").loc["x_nonlinear"]
    assert row.driver_rank <= 3 and row.candidate_grade in {"A", "B", "C"}


def test_regime_reversal_is_not_high_stability(tmp_path):
    row = run_case(CASES["regime_sign_reversal"](), tmp_path).set_index("variable").loc["x_reversal"]
    assert "unstable_across_regimes" in str(row.risk_flags)


def test_outlier_correlation_is_downgraded(tmp_path):
    row = run_case(CASES["outlier_driven_correlation"](), tmp_path).set_index("variable").loc["x_outlier"]
    assert row.candidate_grade not in {"A", "B"}


def test_noise_only_has_no_high_grade_false_positive(tmp_path):
    ranked = run_case(CASES["noise_only"](), tmp_path)
    assert not ranked["candidate_grade"].isin(["A", "B"]).any()


def test_baseline_report_is_reproducible(tmp_path):
    report = {}
    for name, factory in CASES.items():
        case = factory(); ranked = run_case(case, tmp_path / name)
        report[name] = {"seed": case.metadata["seed"], "n": case.metadata["n"], "parameters": case.metadata, "top_k": ranked["variable"].tolist(), "key_evidence": ranked[[c for c in KEY_FIELDS if c in ranked]].to_dict("records"), "metrics": metrics(case, ranked)}
    path = tmp_path / "four_layer_ranking_baseline.json"; path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_committed_baseline_report_has_all_scenarios_and_required_fields():
    report = json.loads(Path("tests/baselines/four_layer_ranking_baseline.json").read_text(encoding="utf-8"))
    assert set(report) == set(CASES)
    for entry in report.values():
        assert {"scenario", "seed", "samples", "parameters", "top_k_results", "metrics", "passed", "failure_reason"} <= set(entry)


def test_pr_8b_does_not_modify_production_scoring_files():
    import subprocess
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD^", "HEAD"], text=True).splitlines()
    assert not set(changed).intersection({"chem_ts_corr/screening.py", "chem_ts_corr/service.py", "chem_ts_corr/config.py"})
