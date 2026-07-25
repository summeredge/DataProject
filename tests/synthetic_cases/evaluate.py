from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.service import analyze_numeric_frame

from .four_layer_cases import SyntheticCase


KEY_FIELDS = [
    "variable",
    "driver_rank",
    "final_score",
    "driver_priority_factor",
    "driver_priority_score",
    "candidate_grade",
    "candidate_class",
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
    "evidence_coverage_status",
    "model_lift_status",
    "prediction_score",
    "independent_signal_score",
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
        top_k=int(metadata.get("top_k", 50)),
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
    if not regime.empty and set(extra) <= set(regime):
        flags = regime[extra].drop_duplicates("variable")
        ranked = ranked.merge(flags, on="variable", how="left")
    return ranked.sort_values("driver_rank", kind="stable").reset_index(drop=True)


def _typed(case: SyntheticCase, kind: str) -> frozenset[str]:
    return case.variable_types.get(kind, frozenset())


def _average_rank(index: pd.DataFrame, variables: frozenset[str]) -> float | None:
    found = [float(index.loc[name, "driver_rank"]) for name in variables if name in index.index]
    return float(np.mean(found)) if found else None


def metrics(case: SyntheticCase, ranked: pd.DataFrame) -> dict[str, object]:
    index = ranked.set_index("variable")
    true_ranks = {
        name: int(index.loc[name, "driver_rank"])
        for name in case.true_drivers
        if name in index.index
    }
    spurious = case.spurious_variables.intersection(index.index)
    noise = _typed(case, "noise").intersection(index.index)
    high = index["candidate_grade"].isin(["A", "B"])
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
            sum(int(index.loc[name, "driver_rank"]) <= 5 for name in noise)
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
    top_k: int = 5,
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
        ranks = ranked.set_index("variable")["driver_rank"].astype(float).to_dict()
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
    missing_rank = max(
        (max(run["ranks"].values(), default=0) for run in runs), default=0
    ) + 1
    correlations = []
    candidate_missing_counts = []
    for run in runs[1:]:
        run_variables = set(run["ranks"])
        common = sorted(base_variables.intersection(run_variables))
        base_ranks = pd.Series(
            [runs[0]["ranks"][variable] for variable in common], dtype=float
        )
        variant_ranks = pd.Series(
            [run["ranks"][variable] for variable in common], dtype=float
        )
        correlations.append(float(base_ranks.corr(variant_ranks, method="spearman")))
        candidate_missing_counts.append(
            {
                "label": run["label"],
                "missing_from_variant": sorted(base_variables - run_variables),
                "new_in_variant": sorted(run_variables - base_variables),
            }
        )
    true_ranks = [
        run["ranks"].get(driver, missing_rank)
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
        add("x_driver_top_3", row is not None and int(row["driver_rank"]) <= 3, "layer_1 x_driver missing from Top-3")
        add("x_driver_lag", row is not None and abs(int(row["lag"]) - case.lags["x_driver"]) <= 1, "layer_2 x_driver lag exceeds tolerance")
        add("x_driver_direction", row is not None and "target_leads_variable" not in str(row["risk_flags"]), "layer_2 x_driver incorrectly flagged target_leads_variable")
    elif scenario == "downstream_response":
        row = _row(ranked, "x_downstream")
        add("downstream_flag", row is not None and "target_leads_variable" in str(row["risk_flags"]), "layer_2 target_leads_variable missing")
        add("downstream_class", row is not None and row["candidate_class"] == "downstream_response", "layer_2 candidate_class is not downstream_response")
        add("downstream_grade_cap", row is not None and row["candidate_grade"] not in {"A", "B"}, "layer_2 target_leads_variable does not cap candidate_grade")
    elif scenario == "common_driver":
        z = _row(ranked, "z_driver")
        common = _row(ranked, "x_common")
        add("both_candidates_visible", z is not None and common is not None, "layer_3 comparison candidates are not both visible")
        add("common_driver_flag", common is not None and "common_capacity_driver" in str(common["risk_flags"]), "layer_3 common_capacity_driver missing for x_common")
        add("true_driver_precedes_proxy", z is not None and common is not None and int(z["driver_rank"]) < int(common["driver_rank"]), "layer_3 x_common outranks z_driver")
    elif scenario == "collinear_proxy":
        true = _row(ranked, "x1_driver")
        proxy = _row(ranked, "x2_proxy")
        risk = "" if proxy is None else str(proxy["risk_flags"])
        separated = (
            true is not None
            and proxy is not None
            and (
                proxy["candidate_grade"] != true["candidate_grade"]
                or any(token in risk for token in ["redundancy", "proxy", "collinearity"])
                or float(true["driver_priority_score"]) - float(proxy["driver_priority_score"]) >= 0.05
            )
        )
        add("true_precedes_proxy", true is not None and proxy is not None and int(true["driver_rank"]) < int(proxy["driver_rank"]), "layer_3 x1_driver does not precede x2_proxy")
        add("proxy_redundancy_separated", separated, "layer_3 true driver and collinear proxy receive near-identical A-grade evidence without redundancy signal")
    elif scenario == "nonlinear_stable_driver":
        row = _row(ranked, "x_nonlinear")
        add("nonlinear_candidate_visible", row is not None and int(row["driver_rank"]) <= 3, "layer_1 Pearson/Spearman lag screening misses U-shaped driver")
        add("nonlinear_model_evidence", row is not None and pd.notna(row["prediction_score"]), "layer_4 prediction_score missing for nonlinear driver")
    elif scenario == "regime_sign_reversal":
        row = _row(ranked, "x_reversal")
        add("regime_reversal_flag", row is not None and bool(row["regime_sign_reversal_flag"]), "stability regime_sign_reversal_flag missing")
        add("regime_stability_low", row is not None and float(row["regime_stability_final"]) < 0.2 and float(row["stability_score"]) < 0.35, "stability regime_stability_final/stability_score does not reflect sign reversal")
        add("regime_grade_cap", row is not None and row["candidate_grade"] not in {"A", "B"}, "stability sign reversal does not cap candidate_grade")
    elif scenario == "outlier_driven_correlation":
        row = _row(ranked, "x_outlier")
        risk = "" if row is None else str(row["risk_flags"])
        robust_signal = (
            row is not None
            and (
                "poor_data_quality" in risk
                or "unstable_over_time" in risk
                or float(row["stability_score"]) < 0.35
            )
        )
        add(
            "outlier_robustness_signal",
            robust_signal,
            "layer_1 association is outlier-driven but data_quality/stability fields do not identify it",
        )
        add(
            "outlier_grade_cap",
            row is not None and row["candidate_grade"] not in {"A", "B"},
            "layer_1 outlier-driven association receives A/B candidate_grade despite data_quality risk",
        )
    elif scenario == "lag_boundary_artifact":
        row = _row(ranked, "x_boundary")
        add("lag_boundary_flag", row is not None and bool(row["lag_boundary_flag"]) and "lag_boundary" in str(row["risk_flags"]), "layer_2 lag_boundary evidence missing")
        add("lag_boundary_grade_cap", row is not None and row["candidate_grade"] not in {"A", "B"}, "layer_2 lag_boundary does not cap candidate_grade")
    elif scenario == "model_incremental_validation":
        true = _row(ranked, "x_incremental")
        proxy = _row(ranked, "x_proxy")
        noise = _row(ranked, "noise")
        add("model_lift_computed", true is not None and str(true["model_lift_status"]).startswith("ok") and pd.notna(true["prediction_score"]), "layer_4 model_lift_status/prediction_score missing")
        noise_has_no_support = noise is not None and pd.isna(noise["prediction_score"])
        add("incremental_above_noise", true is not None and noise is not None and pd.notna(true["prediction_score"]) and (noise_has_no_support or float(true["prediction_score"]) > float(noise["prediction_score"])), "layer_4 true incremental evidence does not exceed noise")
        proxy_risk = "" if proxy is None else str(proxy["risk_flags"])
        add("incremental_proxy_separated", true is not None and proxy is not None and (proxy["candidate_grade"] != true["candidate_grade"] or any(token in proxy_risk for token in ["redundancy", "proxy", "collinearity"]) or float(true["prediction_score"]) - float(proxy["prediction_score"]) >= 0.05), "layer_3/layer_4 collinear proxy receives indistinguishable incremental evidence")
    elif scenario == "noise_only":
        high = ranked["candidate_grade"].isin(["A", "B"])
        add("noise_grade_control", not high.any(), "layer_1 FDR/noise control produces A/B candidate")
        add("noise_fdr_control", metrics(case, ranked)["significant_q_count"] <= 3, "layer_1 FDR corrected significant count exceeds contract")
    elif scenario == "mixed_evidence":
        indexed = ranked.set_index("variable")
        for variable in ["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]:
            add(f"{variable}_present", variable in indexed.index, f"mixed_evidence missing {variable}")
        if set(["x_driver", "z_driver", "x_proxy", "x_common", "x_downstream", "noise"]) <= set(indexed.index):
            add("true_drivers_top_3", all(int(indexed.loc[name, "driver_rank"]) <= 3 for name in ["x_driver", "z_driver"]), "mixed layer_1 true drivers are not both Top-3")
            add("downstream_direction", "target_leads_variable" in str(indexed.loc["x_downstream", "risk_flags"]), "mixed layer_2 downstream flag missing")
            add("downstream_grade_cap", indexed.loc["x_downstream", "candidate_grade"] not in {"A", "B"}, "mixed layer_2 downstream candidate_grade not capped")
            add("common_driver_flag", "common_capacity_driver" in str(indexed.loc["x_common", "risk_flags"]), "mixed layer_3 common_capacity_driver missing")
            add("proxy_separation", indexed.loc["x_proxy", "candidate_grade"] != indexed.loc["x_driver", "candidate_grade"] or float(indexed.loc["x_driver", "driver_priority_score"]) - float(indexed.loc["x_proxy", "driver_priority_score"]) >= 0.05, "mixed layer_3 proxy is not separated from true driver")
            add("noise_low", indexed.loc["noise", "candidate_grade"] not in {"A", "B"} and int(indexed.loc["noise", "driver_rank"]) > int(indexed.loc["x_driver", "driver_rank"]), "mixed noise receives high grade or precedes true driver")
            add("model_evidence_present", pd.notna(indexed.loc["x_driver", "prediction_score"]), "mixed layer_4 prediction_score missing")
    if stability is not None:
        add(
            "stability_reproducible",
            float(stability["spearman_rank_stability"]) >= -1.0,
            "stability metric unavailable",
        )
        if scenario == "true_lagged_driver":
            add("stable_true_driver_top_3", int(stability["true_driver_rank_max"]) <= 3, "stability true driver leaves Top-3 under fixed perturbations")
        elif scenario == "mixed_evidence":
            add("stable_mixed_recall", float(stability["top_3_recall_mean"]) >= 0.5, "stability mixed true-driver Top-3 recall falls below 0.5")
    passed_names = [name for name, passed, _ in checks if passed]
    failures = [failure for _, passed, failure in checks if not passed]
    return {
        "passed": not failures,
        "failure_reason": "; ".join(failures),
        "failed_expectations": failures,
        "passed_expectations": passed_names,
    }
