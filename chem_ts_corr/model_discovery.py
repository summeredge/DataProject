from __future__ import annotations

import re

import pandas as pd


INTERPRETATION = "model explanation only; not a causal conclusion"
DISCOVERY_INTERPRETATION = (
    "model discovery exploration only; not a validation conclusion or causal conclusion"
)
MAX_DISCOVERY_CANDIDATE_WINDOW = 10
MAX_DISCOVERY_CANDIDATES = 5
DISCOVERY_CANDIDATE_COLUMNS = ["variable", "source_rank", "discovery_reason"]

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

VARIABLE_IMPORTANCE_COLS = [
    "variable",
    "best_model_feature",
    "best_model_lag",
    "max_importance",
    "total_importance",
    "feature_count",
    "importance_rank",
    "method",
    "ranked_feature_rank",
    "ranked_final_score",
    "risk_flags",
    "recommended_use",
    "recommended_action",
    "interpretation",
]


def build_model_variable_importance(
    importance: pd.DataFrame,
    ranked_features: pd.DataFrame | None = None,
    risk_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate lag-level model importance into one conservative row per variable.

    This summary is intended for model-explanation review only. It does not
    alter screening scores and does not claim causality.
    """
    if importance.empty:
        return pd.DataFrame(columns=VARIABLE_IMPORTANCE_COLS)

    frame = _normalized_importance(importance)
    if frame.empty:
        return pd.DataFrame(columns=VARIABLE_IMPORTANCE_COLS)
    if "method" not in frame.columns:
        frame["method"] = pd.NA

    ranked_lookup = _ranked_lookup(ranked_features if ranked_features is not None else pd.DataFrame())
    risk_lookup = _risk_lookup(risk_flags)

    rows: list[dict[str, object]] = []
    for variable, group in frame.groupby("variable", sort=False):
        if pd.isna(variable) or str(variable).strip() == "":
            continue
        variable_name = str(variable)
        ordered = group.sort_values(["importance"], ascending=[False], kind="mergesort")
        best = ordered.iloc[0]
        ranked_meta = ranked_lookup.get(variable_name, {})
        risk_meta = risk_lookup.get(variable_name, {})
        rows.append(
            {
                "variable": variable_name,
                "best_model_feature": best["feature"],
                "best_model_lag": best["lag"],
                "max_importance": float(best["importance"]),
                "total_importance": float(group["importance"].sum()),
                "feature_count": int(len(group)),
                "method": best.get("method", pd.NA),
                "ranked_feature_rank": ranked_meta.get("ranked_feature_rank", pd.NA),
                "ranked_final_score": ranked_meta.get("ranked_final_score", pd.NA),
                "risk_flags": _first_non_empty(ranked_meta.get("risk_flags"), risk_meta.get("risk_flags")),
                "recommended_use": _first_non_empty(
                    ranked_meta.get("recommended_use"), risk_meta.get("recommended_use")
                ),
                "recommended_action": _first_non_empty(
                    ranked_meta.get("recommended_action"), risk_meta.get("recommended_action")
                ),
                "interpretation": INTERPRETATION,
            }
        )

    if not rows:
        return pd.DataFrame(columns=VARIABLE_IMPORTANCE_COLS)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["total_importance", "max_importance", "variable"], ascending=[False, False, True]
    ).reset_index(drop=True)
    result["importance_rank"] = range(1, len(result) + 1)
    return result[VARIABLE_IMPORTANCE_COLS]


def build_model_discovered_candidates(
    importance: pd.DataFrame,
    ranked_features: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    screening_top_n: int = 30,
    model_top_n: int = 50,
    max_lag: int | None = None,
) -> pd.DataFrame:
    """Summarize model-importance-only omission exploration signals.

    This output is a conservative exploration aid, not a validation conclusion.
    It does not alter screening scores, rankings, recommendations, or causality.
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
            "interpretation": DISCOVERY_INTERPRETATION,
        }
        row["discovery_reason"] = _discovery_reason(row, max_lag=max_lag)
        rows.append(row)

    return pd.DataFrame(rows, columns=OUT_COLS)


def build_exploration_candidate_pool(
    ranked_features: pd.DataFrame,
    *,
    top_k: int,
    discovery_candidate_window: int = MAX_DISCOVERY_CANDIDATE_WINDOW,
) -> pd.DataFrame:
    """Return the bounded initial-screening omission window.

    The pool is derived only from the existing initial-screening order. It is
    not a new ranking and never changes ``ranked_features``.
    """
    columns = ["variable", "source_rank"]
    if ranked_features is None or ranked_features.empty or "variable" not in ranked_features.columns:
        return pd.DataFrame(columns=columns)

    window_size = min(MAX_DISCOVERY_CANDIDATE_WINDOW, max(0, int(discovery_candidate_window)))
    if window_size == 0:
        return pd.DataFrame(columns=columns)

    frame = ranked_features[["variable"]].copy(deep=True)
    fallback_ranks = pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="Int64")
    if "driver_rank" in ranked_features.columns:
        source_rank = pd.to_numeric(ranked_features["driver_rank"], errors="coerce")
        source_rank = source_rank.where(source_rank.notna(), fallback_ranks)
    else:
        source_rank = fallback_ranks
    frame["source_rank"] = source_rank
    frame["variable"] = frame["variable"].astype("string").str.strip()
    frame = frame[frame["variable"].notna() & frame["variable"].ne("")]
    frame = frame[
        (frame["source_rank"] > max(0, int(top_k)))
        & (frame["source_rank"] <= max(0, int(top_k)) + window_size)
    ]
    frame = frame.sort_values("source_rank", kind="mergesort").drop_duplicates(
        subset=["variable"], keep="first"
    )
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce").astype("Int64")
    return frame[columns].reset_index(drop=True)


def build_discovery_candidates(
    model_discovered: pd.DataFrame,
    ranked_features: pd.DataFrame,
    *,
    top_k: int,
    discovery_candidate_window: int = MAX_DISCOVERY_CANDIDATE_WINDOW,
    max_discovery_candidates: int = MAX_DISCOVERY_CANDIDATES,
) -> pd.DataFrame:
    """Build the small manual-review view from model-discovery results.

    Candidates are filtered back to the fixed omission window and emitted in
    initial-screening order. Model importance is not used to create a second
    ranking.
    """
    columns = DISCOVERY_CANDIDATE_COLUMNS
    if model_discovered is None or model_discovered.empty or "variable" not in model_discovered.columns:
        return pd.DataFrame(columns=columns)

    limit = min(MAX_DISCOVERY_CANDIDATES, max(0, int(max_discovery_candidates)))
    if limit == 0:
        return pd.DataFrame(columns=columns)

    reasons: dict[str, object] = {}
    for row in model_discovered.to_dict(orient="records"):
        variable = row.get("variable")
        if pd.isna(variable):
            continue
        name = str(variable).strip()
        if not name or name in reasons:
            continue
        reason = row.get("discovery_reason", pd.NA)
        if reason is None or (not isinstance(reason, (list, tuple, set)) and pd.isna(reason)):
            reason = "模型发现候选"
        reasons[name] = reason

    pool = build_exploration_candidate_pool(
        ranked_features,
        top_k=top_k,
        discovery_candidate_window=discovery_candidate_window,
    )
    rows = [
        {
            "variable": row["variable"],
            "source_rank": row["source_rank"],
            "discovery_reason": reasons[str(row["variable"])],
        }
        for row in pool.to_dict(orient="records")
        if str(row["variable"]) in reasons
    ][:limit]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["source_rank"] = pd.to_numeric(result["source_rank"], errors="coerce").astype("Int64")
    return result


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
