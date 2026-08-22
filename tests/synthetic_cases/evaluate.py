from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.service import analyze_numeric_frame

from .four_layer_cases import SyntheticCase


REPORT_TOP_K = 5
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
COMMON_RISK_TOKENS = (
    "common_capacity_driver",
    "independence",
    "residual",
    "collinearity",
    "redundancy",
    "proxy",
)

KEY_FIELDS = [
    "variable",
    "final_score",
    "candidate_grade",
    "recommended_use",
    "lag",
    "direction",
    "risk_flags",
    "regime_sign_reversal_flag",
    "regime_sign_consistency",
    "regime_stability_final",
    "stability_score",
    "lag_quality",
    "lag_boundary_flag",
    "data_quality_score",
    "evidence_score",
    "innovation_score",
    "pearson_q",
    "spearman_q",
    "corr_q_value",
]


def run_case(case: SyntheticCase, output_dir: Path) -> pd.DataFrame:
    metadata = case.metadata
    config = AnalysisConfig(
        input_path=output_dir / "unused.csv",
        time_column="time",
        target=case.target,
        output_dir=output_dir,
        max_lag=int(metadata.get("max_lag", 6)),
        top_k=int(metadata.get("top_k", 20)),
        skip_model_lift=bool(metadata.get("skip_model_lift", True)),
        skip_rolling_corr=bool(metadata.get("skip_rolling_corr", False)),
        segment_column=metadata.get("segment_column"),
        residual_control_columns=metadata.get("residual_control_columns"),
        exclude_control_columns_from_candidates=False,
    )
    tables = analyze_numeric_frame(case.frame, config)
    ranked = tables.ranked_features.copy()
    regime = tables.regime_scores
    extra = ["variable", "regime_sign_reversal_flag"]
    if (
        "regime_sign_reversal_flag" not in ranked.columns
        and not regime.empty
        and set(extra) <= set(regime)
    ):
        flags = regime[extra].drop_duplicates("variable")
        ranked = ranked.merge(flags, on="variable", how="left")
    ranked = ranked.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    ranked["final_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _typed(case: SyntheticCase, kind: str) -> frozenset[str]:
    return case.variable_types.get(kind, frozenset())


def _average_rank(index: pd.DataFrame, variables: frozenset[str]) -> float | None:
    found = [float(index.loc[name, "final_rank"]) for name in variables if name in index.index]
    return float(np.mean(found)) if found else None


def _relationship_details(
    case: SyntheticCase,
    index: pd.DataFrame,
    kind: str,
    *,
    separated: bool,
) -> list[dict[str, object]]:
    details = []
    for variable in sorted(_typed(case, kind)):
        reference = case.reference_map.get(variable)
        if reference is None or variable not in index.index or reference not in index.index:
            continue
        row = index.loc[variable]
        reference_row = index.loc[reference]
        variable_rank = int(row["final_rank"])
        reference_rank = int(reference_row["final_rank"])
        score_gap = float(reference_row["final_score"]) - float(row["final_score"])
        risk_flags = str(row["risk_flags"])
        reference_risk_flags = str(reference_row["risk_flags"])
        tokens = COMMON_RISK_TOKENS
        if separated:
            proxy_redundant = "redundant_proxy" in risk_flags
            reference_redundant = "redundant_proxy" in reference_risk_flags
            if proxy_redundant and not reference_redundant:
                identified = variable_rank > reference_rank and (
                    GRADE_ORDER[str(row["candidate_grade"])]
                    > GRADE_ORDER[str(reference_row["candidate_grade"])]
                    or score_gap >= 0.05
                )
                resolution_status = (
                    "identified_proxy" if identified else "conflicting_assignment"
                )
            elif proxy_redundant and reference_redundant:
                resolution_status = "unresolved_group"
            elif reference_redundant:
                resolution_status = "conflicting_assignment"
            else:
                resolution_status = "not_detected"
            group_detected = resolution_status != "not_detected"
            detail = {
                "variable": variable,
                "reference": reference,
                "variable_rank": variable_rank,
                "reference_rank": reference_rank,
                "rank_gap": variable_rank - reference_rank,
                "variable_grade": str(row["candidate_grade"]),
                "reference_grade": str(reference_row["candidate_grade"]),
                "score_gap": score_gap,
                "risk_flags": risk_flags,
                "reference_risk_flags": reference_risk_flags,
                "group_detected": group_detected,
                "resolution_status": resolution_status,
                "separated": resolution_status == "identified_proxy",
            }
            details.append(detail)
            continue
        else:
            accepted = variable_rank > reference_rank and (
                GRADE_ORDER[str(row["candidate_grade"])]
                > GRADE_ORDER[str(reference_row["candidate_grade"])]
                or score_gap >= 0.05
                or any(token in risk_flags for token in tokens)
            )
        detail = {
            "variable": variable,
            "reference": reference,
            "variable_rank": variable_rank,
            "reference_rank": reference_rank,
            "rank_gap": variable_rank - reference_rank,
            "variable_grade": str(row["candidate_grade"]),
            "reference_grade": str(reference_row["candidate_grade"]),
            "score_gap": score_gap,
            "risk_flags": risk_flags,
        }
        detail["separated" if separated else "suppressed"] = accepted
        details.append(detail)
    return details


def metrics(case: SyntheticCase, ranked: pd.DataFrame) -> dict[str, object]:
    index = ranked.set_index("variable")
    true_ranks = {
        name: int(index.loc[name, "final_rank"])
        for name in case.true_drivers
        if name in index.index
    }
    spurious = case.spurious_variables.intersection(index.index)
    noise = _typed(case, "noise").intersection(index.index)
    high = index["candidate_grade"].isin(["A", "B"])
    common_details = _relationship_details(
        case, index, "common_driver_proxy", separated=False
    )
    proxy_details = _relationship_details(case, index, "proxy", separated=True)
    return {
        "top_1_hit": any(rank <= 1 for rank in true_ranks.values()),
        "top_3_recall": sum(rank <= 3 for rank in true_ranks.values())
        / max(1, len(case.true_drivers)),
        "top_5_recall": sum(rank <= 5 for rank in true_ranks.values())
        / max(1, len(case.true_drivers)),
        "true_driver_average_rank": (
            float(np.mean(list(true_ranks.values()))) if true_ranks else None
        ),
        "downstream_average_rank": _average_rank(index, _typed(case, "downstream")),
        "common_driver_average_rank": _average_rank(
            index, _typed(case, "common_driver_proxy")
        ),
        "proxy_average_rank": _average_rank(index, _typed(case, "proxy")),
        "common_driver_suppression_rate": (
            sum(detail["suppressed"] for detail in common_details) / len(common_details)
            if common_details
            else None
        ),
        "proxy_separation_rate": (
            sum(detail["resolution_status"] == "identified_proxy" for detail in proxy_details)
            / len(proxy_details)
            if proxy_details
            else None
        ),
        "proxy_identification_rate": (
            sum(detail["resolution_status"] == "identified_proxy" for detail in proxy_details)
            / len(proxy_details)
            if proxy_details
            else None
        ),
        "redundancy_group_detection_rate": (
            sum(detail["group_detected"] for detail in proxy_details) / len(proxy_details)
            if proxy_details
            else None
        ),
        "common_driver_details": common_details,
        "proxy_separation_details": proxy_details,
        "lag_identification_error": {
            name: abs(int(index.loc[name, "lag"]) - lag)
            for name, lag in case.lags.items()
            if name in index.index
        },
        "noise_high_grade_false_positive_rate": (
            float(high.reindex(list(noise)).mean()) if noise else 0.0
        ),
        "spurious_high_grade_false_positive_rate": (
            float(high.reindex(list(spurious)).mean()) if spurious else 0.0
        ),
        "noise_top_5_rate": (
            sum(int(index.loc[name, "final_rank"]) <= 5 for name in noise)
            / max(1, len(noise))
        ),
        "significant_q_count": int(
            pd.to_numeric(ranked.get("corr_q_value"), errors="coerce").le(0.05).sum()
        ),
    }


def evaluate_rank_stability(
    factory: Callable[..., SyntheticCase],
    output_root: Path,
    *,
    top_k: int = REPORT_TOP_K,
) -> dict[str, object]:
    base = factory()
    variants = [
        ("base", {}),
        ("seed_plus_1", {"seed": int(base.metadata["seed"]) + 1}),
        ("noise_plus_10pct", {"noise": float(base.metadata["noise"]) * 1.1}),
        ("samples_plus_10pct", {"n": int(round(int(base.metadata["n"]) * 1.1))}),
    ]
    runs: list[dict[str, object]] = []
    for label, parameters in variants:
        case = factory(**parameters)
        ranked = run_case(case, output_root / label)
        ranks = ranked.set_index("variable")["final_rank"].astype(float).to_dict()
        runs.append(
            {
                "label": label,
                "parameters": {
                    "seed": case.metadata["seed"],
                    "noise": case.metadata["noise"],
                    "n": case.metadata["n"],
                },
                "top_k": ranked.head(top_k)["variable"].tolist(),
                "ranks": {key: int(value) for key, value in ranks.items()},
                "true_driver_grades": {
                    name: ranked.set_index("variable").loc[name, "candidate_grade"]
                    for name in case.true_drivers
                    if name in ranks
                },
            }
        )
    base_top = set(runs[0]["top_k"])
    overlaps = [
        len(base_top.intersection(run["top_k"])) / max(1, len(base_top.union(run["top_k"])))
        for run in runs[1:]
    ]
    base_variables = set(runs[0]["ranks"])
    missing_rank = None
    correlations: list[float] = []
    candidate_missing_counts = []
    candidate_presence = []
    true_driver_presence = []
    for run in runs[1:]:
        run_variables = set(run["ranks"])
        union_variables = sorted(base_variables.union(run_variables))
        pair_missing_rank = len(union_variables) + 1
        missing_rank = max(missing_rank or 0, pair_missing_rank)
        base_vector = [
            runs[0]["ranks"].get(variable, pair_missing_rank)
            for variable in union_variables
        ]
        variant_vector = [
            run["ranks"].get(variable, pair_missing_rank)
            for variable in union_variables
        ]
        if len(union_variables) < 2:
            correlation = 1.0 if base_vector == variant_vector else 0.0
        else:
            correlation = pd.Series(base_vector, dtype=float).corr(
                pd.Series(variant_vector, dtype=float), method="spearman"
            )
            correlation = (
                float(correlation)
                if pd.notna(correlation)
                else (1.0 if base_vector == variant_vector else 0.0)
            )
        correlations.append(correlation)
        candidate_presence.append(
            len(base_variables.intersection(run_variables)) / max(1, len(base_variables))
        )
        if base.true_drivers:
            true_driver_presence.extend(
                driver in run_variables for driver in base.true_drivers
            )
        candidate_missing_counts.append(
            {
                "label": run["label"],
                "missing_from_variant": sorted(base_variables - run_variables),
                "new_in_variant": sorted(run_variables - base_variables),
            }
        )
    true_ranks = [
        run["ranks"].get(driver, int(missing_rank or 1))
        for run in runs
        for driver in base.true_drivers
    ]
    base_grades = runs[0]["true_driver_grades"]
    grade_matches = [
        run["true_driver_grades"].get(driver) == base_grades.get(driver)
        for run in runs[1:]
        for driver in base.true_drivers
    ]
    return {
        "top_k_overlap_mean": float(np.mean(overlaps)),
        "top_k_overlap_min": float(np.min(overlaps)),
        "true_driver_rank_mean": float(np.mean(true_ranks)) if true_ranks else None,
        "true_driver_rank_std": float(np.std(true_ranks)) if true_ranks else None,
        "true_driver_rank_max": int(max(true_ranks)) if true_ranks else None,
        "true_driver_rank_range": (
            int(max(true_ranks) - min(true_ranks)) if true_ranks else None
        ),
        "top_1_hit_rate": (
            sum(rank == 1 for rank in true_ranks) / len(true_ranks) if true_ranks else 0.0
        ),
        "top_3_recall_mean": (
            sum(rank <= 3 for rank in true_ranks) / len(true_ranks)
            if true_ranks
            else 0.0
        ),
        "spearman_rank_stability": float(np.mean(correlations)),
        "grade_consistency": (
            sum(grade_matches) / len(grade_matches) if grade_matches else 1.0
        ),
        "candidate_presence_rate": float(np.mean(candidate_presence)),
        "true_driver_presence_rate": (
            float(np.mean(true_driver_presence)) if true_driver_presence else None
        ),
        "missing_candidate_rank": missing_rank,
        "candidate_missing_counts": candidate_missing_counts,
        "perturbations": runs,
    }


def evaluate_noise_false_positives(
    factory: Callable[..., SyntheticCase],
    output_root: Path,
    *,
    seeds: tuple[int, ...] = (109, 110, 111, 112, 113),
) -> dict[str, object]:
    runs = []
    all_grades: list[str] = []
    for seed in seeds:
        case = factory(seed=seed)
        ranked = run_case(case, output_root / str(seed))
        values = metrics(case, ranked)
        grades = ranked["candidate_grade"].astype(str).tolist()
        all_grades.extend(grades)
        runs.append(
            {
                "seed": seed,
                "samples": case.metadata["n"],
                "noise": case.metadata["noise"],
                "variable_count": len(case.variable_types["noise"]),
                "top_k": ranked.head(5)["variable"].tolist(),
                "high_grade_false_positive_rate": values[
                    "noise_high_grade_false_positive_rate"
                ],
                "noise_top_5_rate": values["noise_top_5_rate"],
                "significant_q_count": values["significant_q_count"],
                "high_grade_variables": ranked.loc[
                    ranked["candidate_grade"].isin(["A", "B"]), "variable"
                ].tolist(),
            }
        )
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    maximum_grade = min(all_grades, key=lambda grade: grade_order.get(grade, 99))
    return {
        "seeds": list(seeds),
        "high_grade_false_positive_rate": float(
            np.mean([run["high_grade_false_positive_rate"] for run in runs])
        ),
        "false_positive_run_count": sum(
            bool(run["high_grade_variables"]) for run in runs
        ),
        "maximum_false_positive_grade": maximum_grade,
        "runs": runs,
    }


def stability_contract_checks(
    scenario: str, stability: dict[str, object] | None
) -> list[tuple[str, bool, str]]:
    if stability is None:
        return []
    if scenario == "true_lagged_driver":
        return [
            ("stable_true_driver_rank", int(stability["true_driver_rank_max"]) <= 3, "stability true driver leaves Top-3"),
            ("stable_top_1_hit", float(stability["top_1_hit_rate"]) >= 0.75, "stability Top-1 hit rate is below 0.75"),
            ("stable_top_3_recall", float(stability["top_3_recall_mean"]) == 1.0, "stability Top-3 recall is not 1.0"),
            ("stable_true_presence", float(stability["true_driver_presence_rate"]) == 1.0, "stability true driver is missing under perturbation"),
            ("stable_candidate_presence", float(stability["candidate_presence_rate"]) >= 0.80, "stability candidate presence is below 0.80"),
            ("stable_top_k_overlap", float(stability["top_k_overlap_min"]) >= 0.50, "stability Top-K overlap is below 0.50"),
            ("stable_spearman", float(stability["spearman_rank_stability"]) >= 0.50, "stability Spearman rank agreement is below 0.50"),
        ]
    if scenario == "mixed_evidence":
        return [
            ("stable_mixed_recall", float(stability["top_3_recall_mean"]) >= 0.50, "stability mixed Top-3 recall is below 0.50"),
            ("stable_mixed_true_presence", float(stability["true_driver_presence_rate"]) == 1.0, "stability mixed true driver is missing under perturbation"),
            ("stable_mixed_candidate_presence", float(stability["candidate_presence_rate"]) >= 0.70, "stability mixed candidate presence is below 0.70"),
            ("stable_mixed_top_k_overlap", float(stability["top_k_overlap_min"]) >= 0.40, "stability mixed Top-K overlap is below 0.40"),
            ("stable_mixed_spearman", float(stability["spearman_rank_stability"]) >= 0.40, "stability mixed Spearman rank agreement is below 0.40"),
        ]
    if scenario == "collinear_proxy":
        return [
            ("stable_proxy_candidate_presence", float(stability["candidate_presence_rate"]) == 1.0, "stability collinear candidate presence is not 1.0"),
            ("stable_proxy_top_k_overlap", float(stability["top_k_overlap_min"]) >= 0.50, "stability collinear Top-K overlap is below 0.50"),
            ("stable_proxy_spearman", float(stability["spearman_rank_stability"]) >= 0.30, "stability collinear Spearman rank agreement is below 0.30"),
        ]
    if scenario == "noise_only":
        summary = stability.get("multi_seed_false_positives", {})
        return [
            ("noise_top_1_hit", float(stability["top_1_hit_rate"]) == 0.0, "stability noise scenario has a true-driver Top-1 hit"),
            ("noise_false_positive_rate", float(summary.get("high_grade_false_positive_rate", 1.0)) == 0.0, "stability noise high-grade false-positive rate is nonzero"),
            ("noise_false_positive_runs", int(summary.get("false_positive_run_count", 1)) == 0, "stability noise has high-grade false-positive runs"),
            ("noise_maximum_grade", summary.get("maximum_false_positive_grade") not in {"A", "B"}, "stability noise maximum false-positive grade is A/B"),
        ]
    return []


def _row(ranked: pd.DataFrame, variable: str) -> pd.Series | None:
    indexed = ranked.set_index("variable")
    return indexed.loc[variable] if variable in indexed.index else None


def evaluate_case_expectations(
    case: SyntheticCase,
    ranked: pd.DataFrame,
    stability: dict[str, object] | None = None,
) -> dict[str, list[str] | bool | str]:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, failure: str) -> None:
        checks.append((name, bool(passed), failure))

    scenario = str(case.metadata["scenario"])
    if scenario == "true_lagged_driver":
        row = _row(ranked, "x_driver")
        add("x_driver_top_3", row is not None and int(row["final_rank"]) <= 3, "x_driver missing from initial Top-3")
        add("x_driver_lag", row is not None and abs(int(row["lag"]) - case.lags["x_driver"]) <= 1, "x_driver lag exceeds tolerance")
        add("x_driver_direction", row is not None and "target_leads_variable" not in str(row["risk_flags"]), "x_driver incorrectly flagged target_leads_variable")
    elif scenario == "downstream_response":
        row = _row(ranked, "x_downstream")
        add("downstream_flag", row is not None and "target_leads_variable" in str(row["risk_flags"]), "target_leads_variable missing")
        add("downstream_class", row is not None and row["candidate_class"] == "downstream_response", "candidate_class is not downstream_response")
        add("downstream_grade_cap", row is not None and row["candidate_grade"] not in {"A", "B"}, "target_leads_variable does not cap candidate_grade")
    elif scenario == "common_driver":
        z = _row(ranked, "z_driver")
        common = _row(ranked, "x_common")
        add("both_candidates_visible", z is not None and common is not None, "comparison candidates are not both visible")
        add("common_driver_candidate", common is not None, "x_common is missing")
        add("true_driver_precedes_proxy", z is not None and common is not None and int(z["final_rank"]) < int(common["final_rank"]), "x_common outranks z_driver")
        add("common_driver_suppressed", metrics(case, ranked)["common_driver_suppression_rate"] == 1.0, "common_driver_suppression_rate is below 1.0")
    elif scenario == "collinear_proxy":
        true = _row(ranked, "x1_driver")
        proxy = _row(ranked, "x2_proxy")
        proxy_metrics = metrics(case, ranked)
        detail = proxy_metrics["proxy_separation_details"][0] if proxy_metrics["proxy_separation_details"] else {}
        add("redundancy_group_detected", proxy_metrics["redundancy_group_detection_rate"] == 1.0, "collinear group was not detected")
        add("proxy_group_unresolved", detail.get("resolution_status") == "unresolved_group", "collinear group is incorrectly treated as identified proxy")
        add("proxy_identification_rate", proxy_metrics["proxy_identification_rate"] == 0.0 and proxy_metrics["proxy_separation_rate"] == 0.0, "unresolved proxy group contributes to identification rate")
        add("conservative_group_class", true is not None and proxy is not None and true["candidate_class"] == proxy["candidate_class"], "unresolved group does not share conservative class")
    elif scenario == "nonlinear_stable_driver":
        row = _row(ranked, "x_nonlinear")
        add("nonlinear_candidate_present", row is not None, "nonlinear candidate is missing from initial screening")
        add("nonlinear_followup_not_executed", "prediction_score" not in ranked.columns and "model_lift_status" not in ranked.columns, "initial screening exposes model follow-up fields")
    elif scenario == "regime_sign_reversal":
        row = _row(ranked, "x_reversal")
        add("regime_candidate_present", row is not None, "regime reversal candidate is missing from initial screening")
        add("regime_followup_not_executed", "regime_stability_final" not in ranked.columns or ranked["regime_stability_final"].isna().all(), "initial screening exposes executed regime stability")
    elif scenario == "outlier_driven_correlation":
        row = _row(ranked, "x_outlier")
        risk = "" if row is None else str(row["risk_flags"])
        robust_signal = (
            row is not None
            and (
                "poor_data_quality" in risk
                or "severe_data_quality" in risk
                or "unstable_over_time" in risk
                or float(row["stability_score"]) < 0.35
            )
        )
        add(
            "outlier_robustness_signal",
            robust_signal,
            "outlier-driven association does not retain a data-quality signal",
        )
        add(
            "outlier_grade_cap",
            row is not None and row["candidate_grade"] not in {"A", "B"},
            "outlier-driven association receives A/B candidate_grade despite data_quality risk",
        )
    elif scenario == "lag_boundary_artifact":
        row = _row(ranked, "x_boundary")
        add("lag_boundary_flag", row is not None and bool(row["lag_boundary_flag"]) and "lag_boundary" in str(row["risk_flags"]), "lag_boundary evidence missing")
        add("lag_boundary_grade_cap", row is not None and row["candidate_grade"] not in {"A", "B"}, "lag_boundary does not cap candidate_grade")
    elif scenario == "model_incremental_validation":
        true = _row(ranked, "x_incremental")
        proxy = _row(ranked, "x_proxy")
        noise = _row(ranked, "noise")
        add("model_followup_not_executed", "model_lift_status" not in ranked.columns and "prediction_score" not in ranked.columns, "initial screening exposes model follow-up fields")
        proxy_details = metrics(case, ranked)["proxy_separation_details"]
        resolution = proxy_details[0]["resolution_status"] if proxy_details else "not_detected"
        add("incremental_proxy_resolution_recorded", resolution in {"identified_proxy", "unresolved_group"}, "layer_3/layer_4 proxy relationship is neither identified nor recorded as unresolved")
    elif scenario == "noise_only":
        high = ranked["candidate_grade"].isin(["A", "B"])
        add("noise_grade_control", not high.any(), "layer_1 FDR/noise control produces A/B candidate")
        add("noise_fdr_control", metrics(case, ranked)["significant_q_count"] <= 3, "layer_1 FDR corrected significant count exceeds contract")
    elif scenario == "mixed_evidence":
        indexed = ranked.set_index("variable")
        for variable in ["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]:
            add(f"{variable}_present", variable in indexed.index, f"mixed_evidence missing {variable}")
        if set(["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]) <= set(indexed.index):
            add("true_drivers_present", all(name in indexed.index for name in ["x_driver", "z_driver"]), "mixed true drivers are missing")
            add("downstream_direction", "target_leads_variable" in str(indexed.loc["x_downstream", "risk_flags"]), "mixed downstream flag missing")
            add("downstream_grade_cap", indexed.loc["x_downstream", "candidate_grade"] not in {"A", "B"}, "mixed downstream candidate_grade not capped")
            add("common_driver_candidate", "x_common" in indexed.index, "mixed x_common candidate is missing")
            proxy_metrics = metrics(case, ranked)
            proxy_detail = proxy_metrics["proxy_separation_details"][0] if proxy_metrics["proxy_separation_details"] else {}
            add("proxy_redundancy_unresolved", all(
                "redundant_proxy" in str(indexed.loc[name, "risk_flags"])
                for name in ["x_driver", "x_proxy"]
            ) and indexed.loc["x_driver", "candidate_class"] == indexed.loc["x_proxy", "candidate_class"] and proxy_detail.get("resolution_status") == "unresolved_group", "mixed unresolved proxy group is not conservatively classified")
            add("proxy_identification_rate", proxy_metrics["redundancy_group_detection_rate"] == 1.0 and proxy_metrics["proxy_identification_rate"] == 0.0 and proxy_metrics["proxy_separation_rate"] == 0.0, "mixed unresolved proxy group contributes to proxy identification rate")
            add("noise_low", indexed.loc["noise", "candidate_grade"] not in {"A", "B"} and int(indexed.loc["noise", "final_rank"]) > int(indexed.loc["x_driver", "final_rank"]), "mixed noise receives high grade or precedes true driver")
            add("model_followup_not_executed", "prediction_score" not in ranked.columns, "initial screening exposes model follow-up fields")
    for name, passed, failure in stability_contract_checks(scenario, stability):
        add(name, passed, failure)
    passed_names = [name for name, passed, _ in checks if passed]
    failures = [failure for _, passed, failure in checks if not passed]
    return {
        "passed": not failures,
        "failure_reason": "; ".join(failures),
        "failed_expectations": failures,
        "passed_expectations": passed_names,
    }
