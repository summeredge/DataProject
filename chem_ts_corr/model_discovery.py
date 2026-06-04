from __future__ import annotations

import re

import pandas as pd


INTERPRETATION = "model explanation only; not a causal conclusion"

OUT_COLS = [
    "variable",
    "best_model_feature",
    "best_model_lag",
    "max_importance",
    "importance_rank",
    "model_feature_count",
    "nearby_lag_count",
    "ranked_feature_rank",
    "ranked_final_score",
    "in_screening_top_n",
    "missing_from_screening_top_n",
    "risk_flags",
    "recommended_use",
    "recommended_action",
    "discovery_reason",
    "interpretation",
]


def build_model_discovered_candidates(
    importance: pd.DataFrame,
    ranked_features: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    screening_top_n: int = 30,
    model_top_n: int = 50,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Summarize model-importance-only supplemental candidates.

    This output is a conservative review aid. It does not alter screening scores
    and does not claim causality.
    """
    if importance.empty:
        return pd.DataFrame(columns=OUT_COLS)

    top_features = _normalized_importance(importance).sort_values("importance", ascending=False).head(model_top_n)
    if top_features.empty:
        return pd.DataFrame(columns=OUT_COLS)
    top_features = top_features.reset_index(drop=True)
    top_features["importance_rank"] = range(1, len(top_features) + 1)

    ranked_lookup = _ranked_lookup(ranked_features)
    risk_lookup = _risk_lookup(risk_flags)

    rows: list[dict[str, object]] = []
    for variable, group in top_features.groupby("variable", sort=False):
        if pd.isna(variable) or str(variable).strip() == "":
            continue
        ordered = group.sort_values(["importance", "importance_rank"], ascending=[False, True])
        best = ordered.iloc[0]
        variable_name = str(variable)
        ranked_meta = ranked_lookup.get(variable_name, {})
        risk_meta = risk_lookup.get(variable_name, {})
        ranked_rank = ranked_meta.get("ranked_feature_rank", pd.NA)
        in_top = _is_in_top_n(ranked_rank, screening_top_n)
        risk_text = _first_non_empty(ranked_meta.get("risk_flags"), risk_meta.get("risk_flags"))
        row = {
            "variable": variable_name,
            "best_model_feature": best["feature"],
            "best_model_lag": best["lag"],
            "max_importance": best["importance"],
            "importance_rank": int(best["importance_rank"]),
            "model_feature_count": int(len(group)),
            "nearby_lag_count": int(pd.to_numeric(group["lag"], errors="coerce").dropna().nunique()),
            "ranked_feature_rank": ranked_rank,
            "ranked_final_score": ranked_meta.get("ranked_final_score", pd.NA),
            "in_screening_top_n": in_top,
            "missing_from_screening_top_n": not in_top,
            "risk_flags": risk_text,
            "recommended_use": _first_non_empty(ranked_meta.get("recommended_use"), risk_meta.get("recommended_use")),
            "recommended_action": _first_non_empty(ranked_meta.get("recommended_action"), risk_meta.get("recommended_action")),
            "interpretation": INTERPRETATION,
        }
        row["discovery_reason"] = _discovery_reason(row, max_lag=max_lag)
        rows.append(row)

    return pd.DataFrame(rows, columns=OUT_COLS)


def _normalized_importance(importance: pd.DataFrame) -> pd.DataFrame:
    frame = importance.copy(deep=True)
    if "feature" not in frame.columns:
        frame["feature"] = pd.NA
    if "importance" not in frame.columns:
        frame["importance"] = pd.NA
    if "variable" not in frame.columns:
        frame["variable"] = frame["feature"].apply(_variable_from_feature)
    if "lag" not in frame.columns:
        frame["lag"] = frame["feature"].apply(_lag_from_feature)

    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    frame["lag"] = pd.to_numeric(frame["lag"], errors="coerce")
    frame["variable"] = frame["variable"].where(frame["variable"].notna(), frame["feature"].apply(_variable_from_feature))
    return frame.dropna(subset=["importance", "variable"])


def _variable_from_feature(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return re.sub(r"__lag_\d+$", "", str(value))


def _lag_from_feature(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    match = re.search(r"__lag_(\d+)$", str(value))
    return int(match.group(1)) if match else pd.NA


def _ranked_lookup(ranked_features: pd.DataFrame) -> dict[str, dict[str, object]]:
    if ranked_features.empty or "variable" not in ranked_features.columns:
        return {}
    frame = ranked_features.copy(deep=True).reset_index(drop=True)
    frame["ranked_feature_rank"] = frame.index + 1
    if "final_score" in frame.columns:
        frame["ranked_final_score"] = frame["final_score"]
    else:
        frame["ranked_final_score"] = pd.NA
    lookup: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        lookup[str(row["variable"])] = row.to_dict()
    return lookup


def _risk_lookup(risk_flags: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if risk_flags is None or risk_flags.empty or "variable" not in risk_flags.columns:
        return {}
    return {str(row["variable"]): row.to_dict() for _, row in risk_flags.copy(deep=True).iterrows()}


def _is_in_top_n(rank: object, screening_top_n: int) -> bool:
    numeric = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return False
    return int(numeric) <= screening_top_n


def _first_non_empty(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            if value:
                return ";".join(str(item) for item in value)
            continue
        if pd.isna(value):
            continue
        text = str(value)
        if text:
            return value
    return pd.NA


def _discovery_reason(row: dict[str, object], max_lag: int | None) -> str:
    reasons: list[str] = []
    if bool(row["missing_from_screening_top_n"]):
        reasons.append("model_only_signal")
    if int(row["model_feature_count"]) >= 3:
        reasons.append("multi_lag_model_signal")

    best_lag = pd.to_numeric(pd.Series([row["best_model_lag"]]), errors="coerce").iloc[0]
    if max_lag is not None and not pd.isna(best_lag) and best_lag >= max_lag - 2:
        reasons.append("model_lag_boundary_risk")
    if not pd.isna(best_lag) and int(best_lag) == 0:
        reasons.append("synchronous_or_leakage_risk")

    risk_value = row.get("risk_flags")
    risk_text = "" if risk_value is None or pd.isna(risk_value) else str(risk_value)
    if "lag_boundary" in risk_text:
        reasons.append("screening_lag_boundary_risk")
    if "target_leads_variable" in risk_text:
        reasons.append("target_lead_risk")
    if "unstable_over_time" in risk_text or "unstable_across_regimes" in risk_text:
        reasons.append("stability_risk")

    if not reasons:
        reasons.append("model_supported_screening_candidate")
    return ";".join(dict.fromkeys(reasons))
