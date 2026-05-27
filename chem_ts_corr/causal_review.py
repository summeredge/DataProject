from __future__ import annotations

import pandas as pd


def build_causal_review_candidates(ranked_features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variable", "final_score", "candidate_grade", "lag", "direction", "raw_corr", "residual_corr",
        "rolling_stability", "regime_stability_final", "lag_boundary_flag", "model_lift_score",
        "risk_level", "risk_flags", "recommended_use", "recommended_action", "force_included",
        "review_priority", "review_reason", "review_tier",
    ]
    if ranked_features.empty:
        return pd.DataFrame(columns=cols)

    frame = ranked_features.copy()
    for c in ["final_score", "candidate_grade", "recommended_use", "risk_level", "risk_flags", "model_lift_score", "lag_quality", "rolling_stability"]:
        if c not in frame.columns:
            frame[c] = pd.NA

    def priority(row: pd.Series) -> tuple[int, str, str]:
        g = row.get("candidate_grade", "")
        u = row.get("recommended_use", "")
        r = row.get("risk_level", "")
        fs = row.get("final_score", 0)
        grade = "" if pd.isna(g) else str(g)
        use = "" if pd.isna(u) else str(u)
        risk = "" if pd.isna(r) else str(r)
        score = 0.0 if pd.isna(fs) else float(fs)

        if grade == "A" and use in {"strong_screening_candidate", "prediction_candidate"} and risk in {"none", "weak", ""}:
            return 1, "优先三层复核：高分且低风险", "tier_1"
        if grade in {"A", "B"} and risk in {"none", "weak", "medium", ""}:
            return 2, "建议三层复核：较高分候选", "tier_2"
        if use in {"capacity_driven", "formula_coupled_reference", "closed_loop_suspect", "unstable_candidate", "control_variable_reference"}:
            return 4, "低优先级：更偏参考/风险提示变量", "tier_4"
        if score >= 0.45:
            return 3, "可排队复核：中等相关线索", "tier_3"
        return 5, "暂不建议三层复核", "tier_5"

    priorities = frame.apply(priority, axis=1, result_type="expand")
    priorities.columns = ["review_priority", "review_reason", "review_tier"]
    out = pd.concat([frame, priorities], axis=1)
    out = out.sort_values(["review_priority", "final_score"], ascending=[True, False]).reset_index(drop=True)
    return out[[c for c in cols if c in out.columns]]
