from __future__ import annotations

import pandas as pd

from chem_ts_corr.common import as_text, to_float


INTERPRETATION = "screening near-miss only; not a causal conclusion"

OUT_COLS = [
    "variable",
    "near_miss_score",
    "lag",
    "direction",
    "raw_score",
    "residual_corr",
    "independent_signal_score",
    "lag_quality",
    "ranked_feature_rank",
    "ranked_final_score",
    "missing_from_screening_top_n",
    "risk_flags",
    "recommended_use",
    "recommended_action",
    "near_miss_reason",
    "interpretation",
]


def build_near_miss_candidates(
    lag_scores: pd.DataFrame,
    ranked_features: pd.DataFrame,
    residual_corr_scores: pd.DataFrame | None = None,
    lag_peak_quality: pd.DataFrame | None = None,
    risk_flags: pd.DataFrame | None = None,
    screening_top_n: int = 50,
    output_top_n: int = 50,
) -> pd.DataFrame:
    """Build conservative near-miss candidates without changing screening scores."""
    if lag_scores.empty or "variable" not in lag_scores.columns:
        return pd.DataFrame(columns=OUT_COLS)

    base = _best_lag_rows(lag_scores)
    if base.empty:
        return pd.DataFrame(columns=OUT_COLS)

    ranked = _ranked_metadata(ranked_features)
    base = _merge_optional(base, ranked)
    base = _merge_optional(base, _residual_metadata(residual_corr_scores))
    base = _merge_optional(base, _lag_quality_metadata(lag_peak_quality))
    base = _merge_optional(base, _risk_metadata(risk_flags), fill_existing=True)

    for col in ["ranked_feature_rank", "ranked_final_score", "residual_corr", "independent_signal_score", "lag_quality", "risk_flags", "recommended_use", "recommended_action"]:
        if col not in base.columns:
            base[col] = pd.NA

    base = base[base["variable"].notna()].copy()
    base["variable"] = base["variable"].astype(str)
    base = base[base["variable"].str.strip().ne("")]
    base["missing_from_screening_top_n"] = base["ranked_feature_rank"].apply(
        lambda rank: _is_missing_from_top_n(rank, screening_top_n)
    )
    base = base[base["missing_from_screening_top_n"]].copy()
    if base.empty:
        return pd.DataFrame(columns=OUT_COLS)

    base = base[~base.apply(_is_obvious_control_reference, axis=1)].copy()
    if base.empty:
        return pd.DataFrame(columns=OUT_COLS)

    base["near_miss_score"] = base.apply(_near_miss_score, axis=1)
    base["near_miss_reason"] = base.apply(_near_miss_reason, axis=1)
    base["interpretation"] = INTERPRETATION

    out = base.sort_values("near_miss_score", ascending=False).head(output_top_n).reset_index(drop=True)
    return out[OUT_COLS]


def _best_lag_rows(lag_scores: pd.DataFrame) -> pd.DataFrame:
    frame = lag_scores.copy(deep=True)
    if "score" not in frame.columns:
        score_cols = [col for col in ["abs_pearson", "abs_spearman"] if col in frame.columns]
        if not score_cols:
            return pd.DataFrame(columns=["variable", "lag", "direction", "raw_score"])
        frame["score"] = frame[score_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score", "variable"])
    if frame.empty:
        return pd.DataFrame(columns=["variable", "lag", "direction", "raw_score"])
    idx = frame.groupby("variable")["score"].idxmax()
    best = frame.loc[idx].copy()
    if "direction" not in best.columns:
        best["direction"] = best.get("lag", pd.Series(index=best.index, dtype=float)).apply(_direction_from_lag)
    return best.rename(columns={"score": "raw_score"})[["variable", "lag", "direction", "raw_score"]]


def _ranked_metadata(ranked_features: pd.DataFrame) -> pd.DataFrame:
    if ranked_features.empty or "variable" not in ranked_features.columns:
        return pd.DataFrame(columns=["variable"])
    frame = ranked_features.copy(deep=True).reset_index(drop=True)
    frame["ranked_feature_rank"] = frame.index + 1
    if "final_score" in frame.columns:
        frame["ranked_final_score"] = frame["final_score"]
    cols = [
        "variable",
        "ranked_feature_rank",
        "ranked_final_score",
        "risk_flags",
        "recommended_use",
        "recommended_action",
    ]
    return frame[[col for col in cols if col in frame.columns]]


def _residual_metadata(residual_corr_scores: pd.DataFrame | None) -> pd.DataFrame:
    if residual_corr_scores is None or residual_corr_scores.empty or "variable" not in residual_corr_scores.columns:
        return pd.DataFrame(columns=["variable"])
    frame = residual_corr_scores.copy(deep=True)
    if "residual_corr" not in frame.columns:
        return pd.DataFrame(columns=["variable"])
    frame["residual_corr"] = pd.to_numeric(frame["residual_corr"], errors="coerce").abs()
    frame["independent_signal_score"] = frame["residual_corr"].clip(0, 1)
    frame = frame.dropna(subset=["residual_corr"])
    if frame.empty:
        return pd.DataFrame(columns=["variable", "residual_corr", "independent_signal_score"])
    idx = frame.groupby("variable")["residual_corr"].idxmax()
    return frame.loc[idx, ["variable", "residual_corr", "independent_signal_score"]]


def _lag_quality_metadata(lag_peak_quality: pd.DataFrame | None) -> pd.DataFrame:
    if lag_peak_quality is None or lag_peak_quality.empty or "variable" not in lag_peak_quality.columns:
        return pd.DataFrame(columns=["variable"])
    cols = [col for col in ["variable", "lag_quality"] if col in lag_peak_quality.columns]
    return lag_peak_quality.copy(deep=True)[cols]


def _risk_metadata(risk_flags: pd.DataFrame | None) -> pd.DataFrame:
    if risk_flags is None or risk_flags.empty or "variable" not in risk_flags.columns:
        return pd.DataFrame(columns=["variable"])
    cols = [col for col in ["variable", "risk_flags", "recommended_use", "recommended_action"] if col in risk_flags.columns]
    return risk_flags.copy(deep=True)[cols]


def _merge_optional(left: pd.DataFrame, right: pd.DataFrame, fill_existing: bool = False) -> pd.DataFrame:
    if right.empty or "variable" not in right.columns:
        return left
    merged = left.merge(right, on="variable", how="left", suffixes=("", "__joined"))
    if fill_existing:
        for col in [c.removesuffix("__joined") for c in merged.columns if c.endswith("__joined")]:
            joined = f"{col}__joined"
            merged[col] = merged[col].combine_first(merged[joined]) if col in merged.columns else merged[joined]
            merged = merged.drop(columns=[joined])
    return merged


def _is_missing_from_top_n(rank: object, screening_top_n: int) -> bool:
    numeric = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    return bool(pd.isna(numeric) or int(numeric) > screening_top_n)


def _is_obvious_control_reference(row: pd.Series) -> bool:
    text = ";".join(_text(row.get(col, "")) for col in ["risk_flags", "recommended_use", "recommended_action"])
    return "control_variable_reference" in text


def _near_miss_score(row: pd.Series) -> float:
    raw = _number(row.get("raw_score"), 0.0)
    independent = row.get("independent_signal_score")
    residual = _number(independent, 0.0) if pd.notna(independent) else 0.0
    lag_quality = _number(row.get("lag_quality"), 0.0)
    score = raw + 0.3 * abs(residual) + 0.2 * lag_quality
    risks = _text(row.get("risk_flags", ""))
    if "strong_formula_leakage" in risks or "severe_data_quality" in risks:
        score *= 0.35
    return float(score)


def _near_miss_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    raw = _number(row.get("raw_score"), 0.0)
    independent = row.get("independent_signal_score")
    residual = _number(independent, 0.0) if pd.notna(independent) else None
    lag_quality = _number(row.get("lag_quality"), 0.0)
    risks = _text(row.get("risk_flags", ""))
    if raw >= 0.3:
        reasons.append("raw_lag_signal")
    if residual is not None and residual >= 0.2:
        reasons.append("residual_signal")
    if lag_quality >= 0.5:
        reasons.append("clear_lag_peak")
    if "lag_boundary" in risks:
        reasons.append("lag_boundary_risk")
    if "target_leads_variable" in risks:
        reasons.append("target_lead_risk")
    if "unstable_over_time" in risks or "unstable_across_regimes" in risks:
        reasons.append("stability_risk")
    if "poor_data_quality" in risks:
        reasons.append("poor_data_quality_warning")
    if "severe_data_quality" in risks:
        reasons.append("severe_data_quality_risk")
    if "strong_formula_leakage" in risks:
        reasons.append("data_or_formula_risk")
    if not reasons:
        reasons.append("near_miss_candidate")
    return ";".join(dict.fromkeys(reasons))


def _text(value: object) -> str:
    return as_text(value)


def _number(value: object, default: float) -> float:
    return to_float(value, default)


def _direction_from_lag(value: object) -> str:
    lag = _number(value, 0.0)
    if lag > 0:
        return "变量领先目标"
    if lag < 0:
        return "变量滞后目标"
    return "同步变化"
