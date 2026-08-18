from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from chem_ts_corr.common import to_float

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.feature_alignment import fit_linear_model, predict_linear_model
from chem_ts_corr.lag import build_lag_peak_quality, compute_lag_scores, summarize_best_lags
from chem_ts_corr.time_axis import (
    lagged_series,
    physical_gap_starts,
    physical_segment_ids,
    preserve_time_axis_metadata,
    sample_period_ns,
)

ROLES = {"TIME", "Y", "CAPACITY", "MV", "PV", "DV", "IGNORE"}
CONTROL_REFERENCE_COLUMNS = (
    "is_auto_control_reference",
    "is_control_reference",
    "control_reference_type",
    "control_reference_source",
)
_AUTO_CONTROL_REFERENCE_RE = re.compile(r"[._:\-](SV|SP|MV)$", re.IGNORECASE)
RISK_RELATIVE_PENALTY_WEIGHTS = {
    "formula_like": 0.00,
    "strong_formula_leakage": 0.50,
    "common_capacity_driver": 0.00,
    "target_leads_variable": 0.00,
    "unstable_across_regimes": 0.00,
    "unstable_over_time": 0.00,
    "lag_boundary": 0.00,
    "low_model_lift": 0.00,
    "poor_data_quality": 0.00,
    "residual_collinearity": 0.00,
    "redundant_proxy": 0.00,
}
EVIDENCE_SCORE_CAPS = {
    "strong_formula_leakage": 0.25,
    "severe_data_quality": 0.44,
}


def detect_auto_control_reference(variable: object) -> tuple[bool, str, str]:
    """Detect explicit PID setpoint/output suffixes without inferring from tag prefixes."""
    if not isinstance(variable, str):
        return False, "", ""
    match = _AUTO_CONTROL_REFERENCE_RE.search(variable.strip())
    if match is None:
        return False, "", ""
    suffix = match.group(1).upper()
    if suffix == "MV":
        return True, "pid_output", "tag_suffix_mv"
    return True, "pid_setpoint", f"tag_suffix_{suffix.lower()}"


CLASS_PRIORITY_FACTORS = {
    "upstream_driver_candidate": 1.00,
    "synchronous_association": 0.90,
    "downstream_response": 0.45,
    "capacity_driven": 0.75,
    "formula_or_derived": 0.25,
    "poor_quality": 0.35,
    "uncertain_candidate": 0.80,
}
# Fixed initial-screening temporal constraints. These are not user parameters.
TARGET_LEADS_PENALTY_RATE = 0.50
TARGET_LEADS_SCORE_CAP = 0.25
RAW_CANDIDATE_MIN_SCORE = 0.30
RESIDUAL_CANDIDATE_MIN_CORR = 0.30
RESIDUAL_CANDIDATE_MIN_N = 30
CANDIDATE_PRIORITY_COLUMNS = [
    "residual_signal_score",
    "residual_evidence_status",
    "load_adjusted_relation_status",
    "candidate_priority_tier",
    "candidate_priority_score",
    "candidate_priority_rank",
]


REGIME_NAMES = ("low", "mid", "high")
MIN_REGIMES_FOR_STABILITY = 2
REGIME_UNSTABLE_THRESHOLD = 0.50
EVIDENCE_SEPARATION_MARGIN = 0.05
REGIME_CONSISTENCY_WEIGHTS = {
    "strength": 0.60,
    "lag": 0.40,
}
REGIME_STABILITY_COLUMNS = [
    "variable",
    "regime_stability_final",
    "regime_consistency_score",
    "regime_coverage",
    "regime_strength_consistency",
    "regime_sign_consistency",
    "regime_lag_consistency",
    "regime_score_cv",
    "regime_count",
    "regime_evidence_status",
    "regime_sign_reversal_flag",
]
PRIMARY_RANK_COLUMN = "final_score"
PRIMARY_SCORE_COLUMN = "final_score"

V5_SHADOW_SCORE_METHOD = "initial_association_temporal_v5"
V5_SHADOW_COMPARISON_FILENAME = "screening_v5_shadow_comparison.csv"
V5_SHADOW_SUMMARY_FILENAME = "screening_v5_shadow_summary.csv"
V5_SHADOW_COMPARISON_COLUMNS = [
    "variable",
    "final_score_v4",
    "rank_v4",
    "association_score",
    "data_quality_score",
    "base_score_v5",
    "residual_corr",
    "residual_status",
    "residual_support",
    "residual_bonus_rate",
    "rolling_support",
    "rolling_support_status",
    "regime_support",
    "regime_support_status",
    "stability_support",
    "stability_bonus_rate",
    "support_bonus_rate",
    "evidence_score_v5",
    "shadow_final_score_v5",
    "rank_v5",
    "rank_delta",
    "score_delta",
    "temporal_direction_status",
    "risk_flags",
]
V5_SHADOW_SUMMARY_COLUMNS = [
    "k",
    "effective_k",
    "top_k_overlap_count",
    "top_k_overlap_ratio",
    "top_k_overlap",
    "top_k_entrants",
    "top_k_dropouts",
    "max_rank_rise_variable",
    "max_rank_rise_delta",
    "max_rank_drop_variable",
    "max_rank_drop_delta",
]


def compute_v5_shadow_components(
    *,
    association_score: object,
    data_quality_score: object,
    residual_corr: object = np.nan,
    residual_status: object = "not_computed",
    rolling_sign_consistency: object = np.nan,
    rolling_corr_iqr: object = np.nan,
    regime_coverage: object = np.nan,
    regime_strength_consistency: object = np.nan,
    regime_sign_consistency: object = np.nan,
    regime_lag_consistency: object = np.nan,
    regime_evidence_status: object = None,
    rolling_support_status: object = None,
    regime_support_status: object = None,
    temporal_direction_status: object = None,
) -> dict[str, object]:
    """Compute the isolated V5 Shadow score decomposition for one variable.

    This function intentionally accepts only the V5 evidence inputs.  It does
    not read or mutate the formal V4 score, risk flags, or any later-stage
    validation result.  Missing support evidence is represented by ``NaN``;
    its bonus is zero so that missing and a measured zero remain distinct.
    """
    association = _v5_bounded_number(association_score)
    quality = _v5_bounded_number(data_quality_score)
    base = association * quality if pd.notna(association) and pd.notna(quality) else np.nan

    residual_value = _v5_number(residual_corr)
    residual_support = (
        float(np.clip(residual_value, 0.0, 1.0))
        if _v5_status_is(residual_status, "ok") and pd.notna(residual_value)
        else np.nan
    )
    residual_bonus = (
        0.10 * residual_support if pd.notna(residual_support) else 0.0
    )

    rolling_support, rolling_status = _v5_rolling_support(
        rolling_sign_consistency,
        rolling_corr_iqr,
        rolling_support_status,
    )
    regime_support, regime_status = _v5_regime_support(
        regime_coverage,
        regime_strength_consistency,
        regime_sign_consistency,
        regime_lag_consistency,
        regime_evidence_status,
        regime_support_status,
    )
    if pd.notna(rolling_support) and pd.notna(regime_support):
        stability_support = float(np.sqrt(rolling_support * regime_support))
    elif pd.notna(rolling_support):
        stability_support = rolling_support
    elif pd.notna(regime_support):
        stability_support = regime_support
    else:
        stability_support = np.nan
    stability_bonus = (
        0.10 * stability_support if pd.notna(stability_support) else 0.0
    )

    support_bonus = float(np.clip(residual_bonus + stability_bonus, 0.0, 0.20))
    evidence = (
        float(min(1.0, base * (1.0 + support_bonus)))
        if pd.notna(base)
        else np.nan
    )
    if _v5_status_is(temporal_direction_status, "target_leads_supported"):
        shadow_final = (
            float(min(evidence * TARGET_LEADS_PENALTY_RATE, TARGET_LEADS_SCORE_CAP))
            if pd.notna(evidence)
            else np.nan
        )
    else:
        shadow_final = evidence

    return {
        "association_score": association,
        "data_quality_score": quality,
        "base_score_v5": base,
        "residual_corr": residual_value,
        "residual_support": residual_support,
        "residual_bonus_rate": float(residual_bonus),
        "rolling_support": rolling_support,
        "rolling_support_status": rolling_status,
        "regime_support": regime_support,
        "regime_support_status": regime_status,
        "stability_support": stability_support,
        "stability_bonus_rate": float(stability_bonus),
        "support_bonus_rate": support_bonus,
        "evidence_score_v5": evidence,
        "shadow_final_score_v5": shadow_final,
    }


def build_v5_shadow_comparison(
    ranked_features: pd.DataFrame,
    residual_corr_scores: pd.DataFrame | None = None,
    rolling_corr_scores: pd.DataFrame | None = None,
    regime_stability: pd.DataFrame | None = None,
    risk_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an auditable V4/V5 comparison without changing formal screening.

    ``ranked_features`` is the frozen formal V4 result.  The other frames are
    optional, independently generated evidence tables.  The returned frame is
    a new object and is suitable for ``write_v5_shadow_outputs``; no formal
    CSV/API/table is written or modified by this function.
    """
    if ranked_features is None or ranked_features.empty:
        return pd.DataFrame(columns=V5_SHADOW_COMPARISON_COLUMNS)
    if "variable" not in ranked_features.columns:
        raise ValueError("ranked_features must contain a variable column")

    frame = ranked_features.copy(deep=True)
    frame["variable"] = frame["variable"].astype(str)
    frame = frame.drop_duplicates(subset=["variable"], keep="first").reset_index(drop=True)
    frame = _v5_assign_formal_ranks(frame)

    residual_lookup = _v5_source_lookup(residual_corr_scores)
    rolling_lookup = _v5_source_lookup(rolling_corr_scores)
    regime_lookup = _v5_source_lookup(regime_stability)
    risk_lookup = _v5_source_lookup(risk_flags)

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        variable = str(row["variable"])
        residual = _v5_lookup_row(residual_lookup, variable)
        rolling = _v5_lookup_row(rolling_lookup, variable)
        regime = _v5_lookup_row(regime_lookup, variable)
        risk = _v5_lookup_row(risk_lookup, variable)

        association = _v5_association_value(row)
        quality = row.get("data_quality_score", np.nan)
        residual_corr = _v5_prefer_source_value(
            residual, row, "residual_corr", default=np.nan
        )
        residual_status = _v5_prefer_source_value(
            residual, row, "residual_status", default="not_computed"
        )
        regime_evidence_status = _v5_prefer_source_value(
            regime, row, "regime_evidence_status"
        )
        temporal_status = row.get("temporal_direction_status", pd.NA)
        components = compute_v5_shadow_components(
            association_score=association,
            data_quality_score=quality,
            residual_corr=residual_corr,
            residual_status=residual_status,
            rolling_sign_consistency=_v5_prefer_source_value(
                rolling, row, "rolling_sign_consistency"
            ),
            rolling_corr_iqr=_v5_prefer_source_value(
                rolling, row, "rolling_corr_iqr"
            ),
            rolling_support_status=_v5_prefer_source_value(
                rolling, row, "rolling_support_status", default=None
            ),
            regime_coverage=_v5_prefer_source_value(
                regime, row, "regime_coverage"
            ),
            regime_strength_consistency=_v5_prefer_source_value(
                regime, row, "regime_strength_consistency"
            ),
            regime_sign_consistency=_v5_prefer_source_value(
                regime, row, "regime_sign_consistency"
            ),
            regime_lag_consistency=_v5_prefer_source_value(
                regime, row, "regime_lag_consistency"
            ),
            regime_evidence_status=regime_evidence_status,
            regime_support_status=_v5_prefer_source_value(
                regime,
                row,
                "regime_support_status",
                default=regime_evidence_status,
            ),
            temporal_direction_status=temporal_status,
        )
        risk_value = _v5_prefer_source_value(risk, row, "risk_flags", default="")
        rows.append(
            {
                "variable": variable,
                "final_score_v4": row.get("final_score", np.nan),
                "rank_v4": row.get("rank_v4", np.nan),
                **components,
                "residual_status": residual_status,
                "rank_v5": np.nan,
                "rank_delta": np.nan,
                "score_delta": np.nan,
                "temporal_direction_status": temporal_status,
                "risk_flags": "" if pd.isna(risk_value) else risk_value,
            }
        )

    comparison = pd.DataFrame(rows, columns=V5_SHADOW_COMPARISON_COLUMNS)
    comparison["final_score_v4"] = pd.to_numeric(
        comparison["final_score_v4"], errors="coerce"
    )
    comparison["rank_v4"] = pd.to_numeric(comparison["rank_v4"], errors="coerce")
    comparison = _v5_assign_shadow_ranks(comparison)
    comparison["rank_delta"] = (
        pd.to_numeric(comparison["rank_v4"], errors="coerce")
        - pd.to_numeric(comparison["rank_v5"], errors="coerce")
    )
    comparison["score_delta"] = (
        pd.to_numeric(comparison["shadow_final_score_v5"], errors="coerce")
        - pd.to_numeric(comparison["final_score_v4"], errors="coerce")
    )
    return comparison.loc[:, V5_SHADOW_COMPARISON_COLUMNS]


def build_v5_shadow_summary(
    comparison: pd.DataFrame,
    top_ks: tuple[int, ...] = (10, 20, 30),
) -> pd.DataFrame:
    """Return objective Top-K overlap and rank-movement summaries."""
    if comparison is None or comparison.empty:
        return pd.DataFrame(columns=V5_SHADOW_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    frame = comparison.copy(deep=True)
    frame["variable"] = frame["variable"].astype(str)
    for requested_k in top_ks:
        k = int(requested_k)
        if k <= 0:
            continue
        v4_available = pd.to_numeric(frame["rank_v4"], errors="coerce").notna().sum()
        v5_available = pd.to_numeric(frame["rank_v5"], errors="coerce").notna().sum()
        effective_k = min(k, int(v4_available), int(v5_available))
        v4_names = _v5_top_variables(frame, "rank_v4", effective_k)
        v5_names = _v5_top_variables(frame, "rank_v5", effective_k)
        overlap = [name for name in v4_names if name in set(v5_names)]
        entrants = [name for name in v5_names if name not in set(v4_names)]
        dropouts = [name for name in v4_names if name not in set(v5_names)]
        rise = _v5_extreme_rank_delta(frame, rising=True)
        drop = _v5_extreme_rank_delta(frame, rising=False)
        rows.append(
            {
                "k": k,
                "effective_k": effective_k,
                "top_k_overlap_count": len(overlap),
                "top_k_overlap_ratio": len(overlap) / effective_k if effective_k else np.nan,
                "top_k_overlap": ";".join(overlap),
                "top_k_entrants": ";".join(entrants),
                "top_k_dropouts": ";".join(dropouts),
                "max_rank_rise_variable": rise[0],
                "max_rank_rise_delta": rise[1],
                "max_rank_drop_variable": drop[0],
                "max_rank_drop_delta": drop[1],
            }
        )
    return pd.DataFrame(rows, columns=V5_SHADOW_SUMMARY_COLUMNS)


def write_v5_shadow_outputs(
    output_dir: str | Path,
    comparison: pd.DataFrame,
    summary: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write only the independent V5 Shadow comparison artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / V5_SHADOW_COMPARISON_FILENAME
    summary_path = output_path / V5_SHADOW_SUMMARY_FILENAME
    comparison.loc[:, V5_SHADOW_COMPARISON_COLUMNS].to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
    (summary if summary is not None else build_v5_shadow_summary(comparison)).loc[
        :, V5_SHADOW_SUMMARY_COLUMNS
    ].to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"comparison": comparison_path, "summary": summary_path}


def _v5_number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _v5_bounded_number(value: object) -> float:
    numeric = _v5_number(value)
    return float(np.clip(numeric, 0.0, 1.0)) if pd.notna(numeric) else np.nan


def _v5_status_is(value: object, expected: str) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    return str(value) == expected


def _v5_rolling_support(
    sign_consistency: object,
    corr_iqr: object,
    status: object = None,
) -> tuple[float, str]:
    status_value = _v5_status_value(status)
    if status_value is not None and status_value != "ok":
        if status_value not in {
            "not_computed",
            "calculation_failed",
            "insufficient_data",
        }:
            status_value = "calculation_failed"
        return np.nan, status_value
    sign = _v5_number(sign_consistency)
    iqr = _v5_number(corr_iqr)
    if pd.isna(sign) or pd.isna(iqr):
        return np.nan, "insufficient_data" if status_value == "ok" else "not_computed"
    return float(np.clip(sign * (1.0 - min(1.0, iqr)), 0.0, 1.0)), "ok"


def _v5_regime_support(
    coverage: object,
    strength_consistency: object,
    sign_consistency: object,
    lag_consistency: object,
    evidence_status: object,
    support_status: object = None,
) -> tuple[float, str]:
    status_value = _v5_regime_status(support_status)
    if status_value is None:
        status_value = _v5_regime_status(evidence_status)
    if status_value is not None and status_value != "ok":
        return np.nan, status_value
    values = [_v5_number(value) for value in [
        coverage,
        strength_consistency,
        sign_consistency,
        lag_consistency,
    ]]
    if any(pd.isna(value) for value in values):
        return np.nan, "insufficient_metrics" if status_value == "ok" else "no_regime_basis"
    coverage_value, strength, sign, lag = values
    return (
        float(
            np.clip(
                coverage_value * sign * (0.60 * strength + 0.40 * lag),
                0.0,
                1.0,
            )
        ),
        "ok",
    )


def _v5_status_value(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


def _v5_regime_status(value: object) -> str | None:
    status = _v5_status_value(value)
    if status is None:
        return None
    if status in {
        "ok",
        "no_regime_basis",
        "insufficient_regimes",
        "insufficient_metrics",
        "calculation_failed",
    }:
        return status
    if status in {"full_coverage", "partial_coverage"}:
        return "ok"
    if status in {"not_computed", "unavailable"}:
        return "no_regime_basis"
    if status == "fit_failed":
        return "calculation_failed"
    return "calculation_failed"


def _v5_source_lookup(source: pd.DataFrame | None) -> pd.DataFrame:
    if source is None or source.empty or "variable" not in source.columns:
        return pd.DataFrame()
    prepared = source.copy(deep=True)
    prepared["variable"] = prepared["variable"].astype(str)
    return prepared.drop_duplicates(subset=["variable"], keep="first").set_index("variable")


def _v5_lookup_row(source: pd.DataFrame, variable: str) -> pd.Series:
    if source.empty or variable not in source.index:
        return pd.Series(dtype=object)
    value = source.loc[variable]
    return value if isinstance(value, pd.Series) else pd.Series(value)


def _v5_prefer_source_value(
    source_row: pd.Series,
    fallback_row: pd.Series,
    column: str,
    default: object = np.nan,
) -> object:
    if column in source_row.index:
        value = source_row.get(column)
        if not pd.isna(value):
            return value
    return fallback_row.get(column, default)


def _v5_association_value(row: pd.Series) -> object:
    association = _v5_number(row.get("association_score", np.nan))
    if pd.notna(association):
        return association
    for column in ["raw_corr", "score"]:
        value = _v5_number(row.get(column, np.nan))
        if pd.notna(value):
            return float(np.clip(value, 0.0, 1.0))
    best_lag_corr = _v5_number(row.get("best_lag_corr", np.nan))
    return float(np.clip(abs(best_lag_corr), 0.0, 1.0)) if pd.notna(best_lag_corr) else np.nan


def _v5_assign_formal_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy(deep=True)
    provided = pd.to_numeric(
        ranked["driver_rank"]
        if "driver_rank" in ranked.columns
        else pd.Series(np.nan, index=ranked.index),
        errors="coerce",
    )
    if provided.notna().all():
        ranked["rank_v4"] = provided.astype(int)
        return ranked
    score = pd.to_numeric(
        ranked["final_score"]
        if "final_score" in ranked.columns
        else pd.Series(np.nan, index=ranked.index),
        errors="coerce",
    )
    ranked["_v5_v4_score"] = score.fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_v5_v4_score", "variable"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank_v4"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns=["_v5_v4_score"])


def _v5_assign_shadow_ranks(comparison: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison.copy(deep=True)
    ranked["_v5_score"] = pd.to_numeric(
        ranked["shadow_final_score_v5"], errors="coerce"
    )
    ranked["_v5_base"] = pd.to_numeric(ranked["base_score_v5"], errors="coerce")
    ranked["_v5_association"] = pd.to_numeric(
        ranked["association_score"], errors="coerce"
    )
    ordered = ranked.loc[ranked["_v5_score"].notna()].sort_values(
        ["_v5_score", "_v5_base", "_v5_association", "rank_v4", "variable"],
        ascending=[False, False, False, True, True],
        kind="stable",
    )
    rank_map = dict(zip(ordered["variable"], np.arange(1, len(ordered) + 1)))
    ranked["rank_v5"] = ranked["variable"].map(rank_map)
    return ranked.drop(columns=["_v5_score", "_v5_base", "_v5_association"])


def _v5_top_variables(frame: pd.DataFrame, rank_column: str, count: int) -> list[str]:
    if count <= 0:
        return []
    ranked = frame.copy(deep=True)
    ranked["_v5_rank"] = pd.to_numeric(ranked[rank_column], errors="coerce")
    return (
        ranked.loc[ranked["_v5_rank"].notna()]
        .sort_values(["_v5_rank", "variable"], kind="stable")
        .head(count)["variable"]
        .astype(str)
        .tolist()
    )


def _v5_extreme_rank_delta(
    frame: pd.DataFrame,
    *,
    rising: bool,
) -> tuple[str, float]:
    deltas = pd.to_numeric(frame["rank_delta"], errors="coerce")
    valid = frame.loc[deltas.notna() & (deltas.gt(0) if rising else deltas.lt(0))].copy()
    if valid.empty:
        return "", np.nan
    valid["_v5_delta"] = deltas.loc[valid.index]
    valid = valid.sort_values(
        ["_v5_delta", "variable"],
        ascending=[not rising, True],
        kind="stable",
    )
    row = valid.iloc[0]
    return str(row["variable"]), float(row["_v5_delta"])


class BestLagEvidence(TypedDict):
    best_lag: int | None
    best_score: float | None
    max_lag: int
    pair_alignment_key: str
    source: str
    status: Literal["ok", "scanned_no_result"]


def load_roles(config: AnalysisConfig, columns: list[str]) -> dict[str, str]:
    roles = {column: "PV" for column in columns}
    roles[config.target] = "Y"
    if config.segment_column and config.segment_column in roles:
        roles[config.segment_column] = "CAPACITY"
    for column in config.capacity_columns or []:
        if column in roles:
            roles[column] = "CAPACITY"
    for column in config.residual_control_columns or []:
        if column in roles:
            roles[column] = "CAPACITY"

    if config.roles_path:
        role_frame = pd.read_csv(config.roles_path)
        if not {"variable", "role"}.issubset(role_frame.columns):
            raise ValueError("roles file must contain columns: variable, role")
        for _, row in role_frame.iterrows():
            variable = str(row["variable"])
            role = str(row["role"]).upper()
            if variable in roles and role in ROLES:
                roles[variable] = role
    return roles


def apply_ignore_roles(frame: pd.DataFrame, roles: dict[str, str], target: str) -> pd.DataFrame:
    ignored = [column for column, role in roles.items() if role == "IGNORE" and column != target]
    return frame.drop(columns=[column for column in ignored if column in frame.columns], errors="ignore")


def diagnostics(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    columns = [
        "variable", "role", "missing_rate", "longest_missing_run", "duplicate_timestamps",
        "sampling_period_seconds", "constant_run_max", "abnormal_jump_count", "abnormal_jump_ratio", "robust_outlier_ratio", "saturation_ratio",
    ]
    rows: list[dict[str, object]] = []
    duplicate_timestamps = int(frame.attrs.get("duplicate_timestamps", 0))
    sampling_period = _sampling_period_seconds(frame.index)
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        diffs = series.diff().abs()
        non_na = series.dropna()
        q1 = non_na.quantile(0.25) if len(non_na) else np.nan
        q3 = non_na.quantile(0.75) if len(non_na) else np.nan
        iqr = q3 - q1 if pd.notna(q1) and pd.notna(q3) else np.nan
        jump_threshold = 10 * iqr if pd.notna(iqr) and iqr > 0 else np.nan
        abnormal_jump_count = int((diffs > jump_threshold).sum()) if pd.notna(jump_threshold) else 0
        valid_diff_count = int(diffs.notna().sum())
        rows.append({
            "variable": column,
            "role": roles.get(column, "PV"),
            "missing_rate": float(series.isna().mean()),
            "longest_missing_run": int(_longest_run(series.isna())),
            "duplicate_timestamps": duplicate_timestamps,
            "sampling_period_seconds": sampling_period,
            "constant_run_max": int(_longest_constant_run(series)),
            "abnormal_jump_count": abnormal_jump_count,
            "abnormal_jump_ratio": abnormal_jump_count / valid_diff_count if valid_diff_count else 0.0,
            "robust_outlier_ratio": float(_robust_outlier_ratio(non_na)),
            "saturation_ratio": float(_saturation_ratio(series)),
        })
    return pd.DataFrame(rows, columns=columns)


RESIDUAL_SCORE_COLUMNS = [
    "variable", "residual_pearson", "residual_spearman", "residual_signed_corr",
    "residual_corr", "residual_method", "residualization_method", "residual_lag", "residual_direction",
    "residual_n", "residual_lag_quality", "residual_lag_boundary_flag",
    "residual_status", "requested_control_columns", "effective_control_columns",
    "control_count", "control_matrix_rank", "control_condition_number",
]


def residual_corr_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_columns: list[str] | None,
    max_lag: int,
    best_lags: Mapping[str, int] | None = None,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    requested_controls = list(dict.fromkeys(str(column) for column in (capacity_columns or [])))
    available_controls = [
        column for column in requested_controls
        if column in frame.columns and column != target
    ]
    if not requested_controls:
        return pd.DataFrame(columns=RESIDUAL_SCORE_COLUMNS)
    controls_frame = frame[available_controls].copy()
    resolved_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column == target:
            continue
        base = _residual_result_row(column, requested_controls)
        if column in available_controls:
            base["residual_status"] = "control_reference_not_residualized"
            rows.append(base)
            continue
        pair = pd.concat([frame[[target, column]], controls_frame], axis=1).loc[resolved_mask]
        pair = pair.replace([np.inf, -np.inf], np.nan)
        usable_controls = [
            control for control in available_controls
            if pair[control].notna().any() and pair[control].nunique(dropna=True) > 1
        ]
        if not usable_controls:
            base["residual_status"] = "no_valid_controls"
            rows.append(base)
            continue
        pair = pair[[target, column, *usable_controls]].replace([np.inf, -np.inf], np.nan).dropna()
        usable_controls = [
            control for control in usable_controls if pair[control].nunique(dropna=True) > 1
        ]
        if not usable_controls:
            base["residual_status"] = "no_valid_controls"
            rows.append(base)
            continue
        if len(pair) < max(10, max_lag + 5):
            base.update({
                "effective_control_columns": ";".join(usable_controls),
                "control_count": len(usable_controls),
                "residual_n": len(pair),
                "residual_status": "insufficient_joint_samples",
            })
            rows.append(base)
            continue
        base.update({
            "effective_control_columns": ";".join(usable_controls),
            "control_count": len(usable_controls),
        })
        try:
            control_matrix = np.column_stack([np.ones(len(pair)), pair[usable_controls].to_numpy(dtype=float)])
            matrix_rank = int(np.linalg.matrix_rank(control_matrix))
            base["control_matrix_rank"] = matrix_rank
            condition_number = float(np.linalg.cond(control_matrix))
            base["control_condition_number"] = condition_number if np.isfinite(condition_number) else np.nan
            target_residual, residualization_method, _, _ = _residualize(pair[target], pair[usable_controls])
            candidate_residual, _, _, _ = _residualize(pair[column], pair[usable_controls])
            residual_pair = preserve_time_axis_metadata(
                frame,
                pd.DataFrame(
                    {target: target_residual, column: candidate_residual},
                    index=pair.index,
                ),
            )
            residual_lag_scores = compute_lag_scores(residual_pair, target, max_lag)
            best = summarize_best_lags(residual_lag_scores)
            quality = build_lag_peak_quality(residual_lag_scores, max_lag)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
            base["residual_status"] = "fit_failed"
            rows.append(base)
            continue
        residualization_method = (
            "ols_rank_deficient" if matrix_rank < control_matrix.shape[1] else residualization_method
        )
        base["residualization_method"] = residualization_method
        base["residual_status"] = "rank_deficient" if matrix_rank < control_matrix.shape[1] else "ok"
        if best.empty:
            base["residual_status"] = "no_valid_residual_lag" if matrix_rank == control_matrix.shape[1] else "rank_deficient_no_valid_residual_lag"
            rows.append(base)
            continue
        best_row = best.iloc[0]
        quality_row = quality.loc[quality["variable"] == column]
        method_name = str(best_row["method"])
        signed = float(best_row[method_name])
        base.update({
            "residual_pearson": best_row["pearson"],
            "residual_spearman": best_row["spearman"],
            "residual_signed_corr": signed,
            "residual_corr": abs(signed),
            "residual_method": method_name,
            "residual_lag": int(best_row["lag"]),
            "residual_direction": best_row["direction"],
            "residual_n": int(best_row["n"]),
            "residual_lag_boundary_flag": bool(best_row["lag_boundary_flag"]),
            "residual_lag_quality": quality_row.iloc[0]["lag_quality"] if not quality_row.empty else np.nan,
        })
        rows.append(base)
    return pd.DataFrame(rows, columns=RESIDUAL_SCORE_COLUMNS)


def _residual_result_row(variable: str, requested_controls: list[str]) -> dict[str, object]:
    row: dict[str, object] = {column: np.nan for column in RESIDUAL_SCORE_COLUMNS}
    row.update({
        "variable": variable,
        "requested_control_columns": ";".join(requested_controls),
        "effective_control_columns": "",
        "control_count": 0,
        "control_matrix_rank": np.nan,
        "control_condition_number": np.nan,
    })
    return row


def regime_scores(
    frame: pd.DataFrame,
    target: str,
    capacity_column: str | None,
    max_lag: int,
    best_lags: Mapping[str, int] | None = None,
    target_mask: pd.Series | None = None,
    regime_basis: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = ["variable", "regime", "regime_row_count", "score", "signed_corr", "lag", "direction", "p_value", "r2"]
    basis_frame = frame if regime_basis is None else regime_basis
    if not capacity_column or capacity_column not in basis_frame.columns:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    capacity = pd.to_numeric(
        basis_frame[capacity_column].reindex(frame.index), errors="coerce"
    )
    resolved_mask = (
        target_mask.reindex(frame.index).fillna(False).astype(bool)
        if target_mask is not None
        else pd.Series(True, index=frame.index, dtype=bool)
    )
    selected_capacity = capacity.where(resolved_mask)
    q1 = selected_capacity.quantile(1 / 3)
    q2 = selected_capacity.quantile(2 / 3)
    regimes = dict(zip(REGIME_NAMES, [
        resolved_mask & (capacity <= q1),
        resolved_mask & (capacity > q1) & (capacity <= q2),
        resolved_mask & (capacity > q2),
    ]))

    all_rows: list[pd.DataFrame] = []
    for name, regime_mask in regimes.items():
        if int(regime_mask.sum()) < max(10, max_lag + 5):
            continue
        regime_rows: list[pd.DataFrame] = []
        for column in frame.columns:
            if column == target:
                continue
            best = _best_lag_review_scores(
                frame[[target, column]],
                target,
                max_lag,
                (best_lags or {}).get(column),
                target_mask=regime_mask,
            )
            if best.empty:
                continue
            best["signed_corr"] = np.where(best["method"].eq("pearson"), best["pearson"], best["spearman"])
            best = best.assign(regime=name, regime_row_count=best["n"].astype(int))
            regime_rows.append(best[score_cols])
        if regime_rows:
            all_rows.append(
                pd.concat(regime_rows, ignore_index=True)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

    if not all_rows:
        return pd.DataFrame(columns=score_cols), pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    scores = pd.concat(all_rows, ignore_index=True)
    return scores, _summarize_regime_robustness(scores, max_lag)


def _best_lag_review_scores(
    pair: pd.DataFrame,
    target: str,
    max_lag: int,
    primary_best_lag,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    def scan(lag_values=None) -> pd.DataFrame:
        if target_mask is None:
            return compute_lag_scores(pair, target, max_lag, lag_values=lag_values)
        return compute_lag_scores(
            pair,
            target,
            max_lag,
            lag_values=lag_values,
            target_mask=target_mask,
        )

    limit = max(0, int(max_lag))
    if limit == 0:
        return summarize_best_lags(scan([0]))

    primary_lag = _valid_primary_lag(primary_best_lag, limit)
    if primary_lag is None or abs(primary_lag) == limit:
        return summarize_best_lags(scan())

    radius = min(limit, max(3, int(np.ceil(limit * 0.05))))
    lower = max(-limit, primary_lag - radius)
    upper = min(limit, primary_lag + radius)
    best = summarize_best_lags(scan(range(lower, upper + 1)))
    if best.empty:
        return summarize_best_lags(scan())

    best_lag = int(best.iloc[0]["lag"])
    touches_local_boundary = (
        (best_lag == lower and lower != -limit)
        or (best_lag == upper and upper != limit)
    )
    if touches_local_boundary:
        return summarize_best_lags(scan())
    return best


def _valid_primary_lag(value, max_lag: int) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    lag = int(numeric)
    return lag if abs(lag) <= max_lag else None


def _summarize_regime_robustness(scores: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if scores.empty or not {"variable", "regime"}.issubset(scores.columns):
        return pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)

    cleaned = scores.copy(deep=True)
    cleaned = cleaned[cleaned["regime"].isin(REGIME_NAMES)]
    cleaned = cleaned.drop_duplicates(subset=["variable", "regime"], keep="first")
    if cleaned.empty:
        return pd.DataFrame(columns=REGIME_STABILITY_COLUMNS)
    for column in ["score", "signed_corr", "lag"]:
        cleaned[column] = pd.to_numeric(cleaned.get(column), errors="coerce")

    rows: list[dict[str, object]] = []
    for variable, group in cleaned.groupby("variable", sort=False):
        regime_count = int(group["regime"].nunique())
        coverage = float(np.clip(regime_count / len(REGIME_NAMES), 0.0, 1.0))
        valid_scores = group["score"].dropna().clip(0, 1)
        valid_signed = group["signed_corr"].dropna()
        valid_lags = group["lag"].dropna()
        score_mean = float(valid_scores.mean()) if not valid_scores.empty else np.nan
        score_cv = (
            float(valid_scores.std(ddof=1) / abs(score_mean))
            if len(valid_scores) >= 2 and abs(score_mean) > 1e-12
            else np.nan
        )
        enough_metrics = all(
            len(values) >= MIN_REGIMES_FOR_STABILITY
            for values in [valid_scores, valid_signed, valid_lags]
        )

        reversal = bool((valid_signed > 0).any() and (valid_signed < 0).any())
        strength = sign = lag_consistency = consistency = final_score = np.nan
        if regime_count < MIN_REGIMES_FOR_STABILITY:
            status = "insufficient_regimes"
        elif not enough_metrics:
            status = "insufficient_metrics"
        else:
            score_max = float(valid_scores.max())
            score_min = float(valid_scores.min())
            strength = float(np.clip(score_min / score_max if score_max > 1e-12 else 0.0, 0, 1))
            signed_strength = float(valid_signed.abs().sum())
            sign = float(
                np.clip(
                    abs(float(valid_signed.sum())) / signed_strength
                    if signed_strength > 1e-12
                    else 0.0,
                    0,
                    1,
                )
            )
            lag_std = float(np.std(valid_lags, ddof=0))
            lag_consistency = float(np.clip(1.0 - lag_std / max(1.0, float(max_lag)), 0, 1))
            shape = (
                REGIME_CONSISTENCY_WEIGHTS["strength"] * strength
                + REGIME_CONSISTENCY_WEIGHTS["lag"] * lag_consistency
            )
            consistency = float(np.clip(sign * shape, 0, 1))
            final_score = float(np.clip(coverage * consistency, 0, 1))
            status = "full_coverage" if regime_count == len(REGIME_NAMES) else "partial_coverage"

        rows.append({
            "variable": variable,
            "regime_stability_final": final_score,
            "regime_consistency_score": consistency,
            "regime_coverage": coverage,
            "regime_strength_consistency": strength,
            "regime_sign_consistency": sign,
            "regime_lag_consistency": lag_consistency,
            "regime_score_cv": score_cv,
            "regime_count": regime_count,
            "regime_evidence_status": status,
            "regime_sign_reversal_flag": reversal,
        })
    return pd.DataFrame(rows, columns=REGIME_STABILITY_COLUMNS)


def model_lift_scores(frame: pd.DataFrame, target: str, candidate_variables: list[str], max_lag: int, n_splits: int = 4, best_lags: dict[str, int] | None = None, target_mask: pd.Series | None = None) -> pd.DataFrame:
    cols = [
        "variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift",
        "median_fold_lift", "positive_fold_ratio", "model_lift_score",
    ]
    rows: list[dict[str, object]] = []
    ar_lags = list(range(1, min(max_lag, 6) + 1))
    period_ns = sample_period_ns(frame)
    forced_starts = physical_gap_starts(frame)
    for variable in candidate_variables:
        if variable == target or variable not in frame.columns:
            continue
        best_lag = best_lags.get(variable) if best_lags else None
        if best_lag is not None and pd.notna(best_lag) and int(best_lag) <= 0:
            rows.append({"variable": variable, "status": "non_predictive_lag", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        candidate_lags = [lag for lag in _nearby_lags(best_lag, max_lag) if lag >= 1]
        if not candidate_lags:
            rows.append({"variable": variable, "status": "non_predictive_lag", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        dataset = pd.DataFrame(index=frame.index)
        dataset[target] = frame[target]
        for lag in ar_lags:
            dataset[f"{target}__lag_{lag}"] = lagged_series(
                frame[target], frame.index, lag, period_ns=period_ns, forced_starts=forced_starts
            )
        for lag in candidate_lags:
            lagged_candidate = lagged_series(
                frame[variable], frame.index, lag, period_ns=period_ns, forced_starts=forced_starts
            )
            dataset[f"{variable}__lag_{lag}"] = lagged_candidate
            # PR-8C nonlinear_stable_driver: expose a quadratic incremental basis.
            dataset[f"{variable}__lag_{lag}__squared"] = lagged_candidate.pow(2)
        if target_mask is not None:
            dataset = dataset.loc[target_mask.reindex(dataset.index).fillna(False).astype(bool)]
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
        if len(dataset) < 60:
            rows.append({"variable": variable, "status": "skipped: insufficient rows", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        base_cols = [f"{target}__lag_{lag}" for lag in ar_lags]
        full_cols = base_cols + [
            feature
            for lag in candidate_lags
            for feature in [f"{variable}__lag_{lag}", f"{variable}__lag_{lag}__squared"]
        ]
        base_errors: list[float] = []
        full_errors: list[float] = []
        splits = _time_series_splits(len(dataset), n_splits)
        if not splits:
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        for train_idx, test_idx in splits:
            y_train = dataset.iloc[train_idx][target].to_numpy()
            y_test = dataset.iloc[test_idx][target].to_numpy()
            base_pred = _linear_predict(dataset.iloc[train_idx][base_cols], y_train, dataset.iloc[test_idx][base_cols])
            full_pred = _linear_predict(dataset.iloc[train_idx][full_cols], y_train, dataset.iloc[test_idx][full_cols])
            base_errors.append(_rmse(y_test, base_pred))
            full_errors.append(_rmse(y_test, full_pred))
        base_rmse = float(np.mean(base_errors))
        full_rmse = float(np.mean(full_errors))
        if np.isnan(base_rmse) or np.isnan(full_rmse):
            rows.append({"variable": variable, "status": "skipped: no valid time series split", "ar_baseline_rmse": np.nan, "candidate_rmse": np.nan, "model_lift": np.nan, "median_fold_lift": np.nan, "positive_fold_ratio": np.nan, "model_lift_score": np.nan})
            continue
        lift = max(0.0, (base_rmse - full_rmse) / base_rmse) if base_rmse > 0 else 0.0
        fold_lifts = np.array([
            (base - full) / base if base > 0 else 0.0
            for base, full in zip(base_errors, full_errors)
        ])
        median_fold_lift = float(np.median(fold_lifts))
        positive_fold_ratio = float(np.mean(fold_lifts > 0))
        lift_strength = float(np.clip(max(0.0, median_fold_lift) / 0.05, 0.0, 1.0))
        model_lift_score = lift_strength * positive_fold_ratio
        rows.append({"variable": variable, "status": "ok", "ar_baseline_rmse": base_rmse, "candidate_rmse": full_rmse, "model_lift": lift, "median_fold_lift": median_fold_lift, "positive_fold_ratio": positive_fold_ratio, "model_lift_score": model_lift_score})
    return pd.DataFrame(rows, columns=cols)


def pair_alignment_key(pair: pd.DataFrame) -> str:
    index_hashes = pd.util.hash_pandas_object(pair.index, index=False).to_numpy(
        dtype=np.uint64,
        copy=False,
    )
    digest = hashlib.sha256()
    digest.update(str(len(pair)).encode("utf-8"))
    digest.update(str(pair.index.dtype).encode("utf-8"))
    digest.update(index_hashes.tobytes())
    return digest.hexdigest()[:24]


def prepare_best_lag_evidence(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    ranked: pd.DataFrame | None = None,
    allow_ranked_reuse: bool = True,
    ranked_source_frame: pd.DataFrame | None = None,
    target_mask: pd.Series | None = None,
) -> tuple[dict[str, BestLagEvidence], dict[str, int]]:
    evidence: dict[str, BestLagEvidence] = {}
    diagnostics = {
        "reused_evidence_count": 0,
        "recomputed_evidence_count": 0,
        "invalid_evidence_count": 0,
    }
    if target not in frame.columns or max_lag < 0:
        return evidence, diagnostics

    for variable in dict.fromkeys(candidate_variables):
        if variable == target or variable not in frame.columns:
            continue
        current_pair = frame[[target, variable]].dropna()
        current_alignment_key = pair_alignment_key(current_pair)
        ranked_row = _ranked_row(ranked, variable)
        if allow_ranked_reuse and ranked_row is not None:
            source_columns_available = (
                ranked_source_frame is not None
                and target in ranked_source_frame.columns
                and variable in ranked_source_frame.columns
            )
            if source_columns_available:
                source_pair = ranked_source_frame[[target, variable]].dropna()
                source_alignment_key = pair_alignment_key(source_pair)
                source_matches_current = (
                    source_alignment_key == current_alignment_key
                    and source_pair.index.equals(current_pair.index)
                )
                if source_matches_current:
                    candidate = _evidence_from_ranked_row(
                        ranked_row,
                        max_lag,
                        source_alignment_key,
                    )
                    if _validated_best_lag_evidence(candidate, source_pair, max_lag) is not None:
                        evidence[variable] = candidate
                        diagnostics["reused_evidence_count"] += 1
                        continue
            diagnostics["invalid_evidence_count"] += 1
        if len(current_pair) < max(10, max_lag + 5):
            continue
        scores = (
            compute_lag_scores(current_pair, target, max_lag)
            if target_mask is None
            else compute_lag_scores(current_pair, target, max_lag, target_mask=target_mask)
        )
        best = summarize_best_lags(scores)
        diagnostics["recomputed_evidence_count"] += 1
        if best.empty:
            evidence[variable] = {
                "best_lag": None,
                "best_score": None,
                "max_lag": int(max_lag),
                "pair_alignment_key": current_alignment_key,
                "source": "recomputed",
                "status": "scanned_no_result",
            }
            continue
        best_row = best.iloc[0]
        evidence[variable] = {
            "best_lag": int(best_row["lag"]),
            "best_score": float(best_row["score"]),
            "max_lag": int(max_lag),
            "pair_alignment_key": current_alignment_key,
            "source": "recomputed",
            "status": "ok",
        }
    return evidence, diagnostics


def _ranked_row(ranked: pd.DataFrame | None, variable: str) -> pd.Series | None:
    if ranked is None or ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return None
    matches = ranked.loc[ranked["variable"].astype(str).eq(variable)]
    return None if matches.empty else matches.iloc[0]


def _evidence_from_ranked_row(
    row: pd.Series,
    max_lag: int,
    alignment_key: str,
) -> BestLagEvidence:
    score = row.get("score", row.get("raw_corr", np.nan))
    return {
        "best_lag": row.get("lag"),
        "best_score": score,
        "max_lag": int(max_lag),
        "pair_alignment_key": alignment_key,
        "source": "ranked",
        "status": "ok",
    }


def _is_scanned_no_result_evidence(
    evidence: Mapping[str, object] | None,
    pair: pd.DataFrame,
    max_lag: int,
) -> bool:
    if not isinstance(evidence, Mapping) or evidence.get("status") != "scanned_no_result":
        return False
    try:
        evidence_max_lag = float(evidence.get("max_lag"))
    except (TypeError, ValueError):
        return False
    return (
        np.isfinite(evidence_max_lag)
        and evidence_max_lag.is_integer()
        and int(evidence_max_lag) == max_lag
        and evidence.get("best_lag") is None
        and evidence.get("best_score") is None
        and evidence.get("source") == "recomputed"
        and evidence.get("pair_alignment_key") == pair_alignment_key(pair)
    )


def _validated_best_lag_evidence(
    evidence: Mapping[str, object] | None,
    pair: pd.DataFrame,
    max_lag: int,
) -> tuple[int, float] | None:
    if not isinstance(evidence, Mapping):
        return None
    try:
        lag_value = float(evidence.get("best_lag"))
        score = float(evidence.get("best_score"))
        evidence_max_lag = float(evidence.get("max_lag"))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in [lag_value, score, evidence_max_lag]):
        return None
    if not lag_value.is_integer() or not evidence_max_lag.is_integer():
        return None
    lag = int(lag_value)
    if int(evidence_max_lag) != max_lag or not -max_lag <= lag <= max_lag:
        return None
    if not 0.0 <= score <= 1.0 or not str(evidence.get("source", "")).strip():
        return None
    if evidence.get("pair_alignment_key") != pair_alignment_key(pair):
        return None
    return lag, score


def rolling_corr_scores(
    frame: pd.DataFrame,
    target: str,
    candidate_variables: list[str],
    max_lag: int,
    window: int | None = None,
    min_periods: int | None = None,
    best_lag_evidence: Mapping[str, Mapping[str, object]] | None = None,
    target_mask: pd.Series | None = None,
) -> pd.DataFrame:
    cols = ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_abs_corr_median", "rolling_corr_iqr", "rolling_sign_consistency", "valid_window_count", "rolling_stability"]
    if target not in frame.columns or not candidate_variables:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    window_size = max(12, int(window or min(len(frame), max(24, max_lag * 4))))
    min_points = max(6, int(min_periods or window_size // 2))
    for variable in candidate_variables:
        if variable == target or variable not in frame.columns:
            continue
        pair = frame[[target, variable]].dropna()
        if len(pair) < max(window_size, max_lag + 10):
            continue
        candidate_evidence = best_lag_evidence.get(variable) if best_lag_evidence else None
        if _is_scanned_no_result_evidence(candidate_evidence, pair, max_lag):
            continue
        prepared = _validated_best_lag_evidence(
            candidate_evidence,
            pair,
            max_lag,
        )
        if prepared is None:
            scores = (
                compute_lag_scores(pair, target, max_lag)
                if target_mask is None
                else compute_lag_scores(pair, target, max_lag, target_mask=target_mask)
            )
            best = summarize_best_lags(scores)
            if best.empty:
                continue
            best_row = best.iloc[0]
            best_lag = int(best_row["lag"])
            best_score = float(best_row.get("score", 0.0) or 0.0)
        else:
            best_lag, best_score = prepared
        shifted = lagged_series(
            pair[variable],
            pair.index,
            best_lag,
            period_ns=sample_period_ns(frame),
            forced_starts=physical_gap_starts(frame),
        )
        segment_ids = physical_segment_ids(
            pair.index,
            sample_period_ns(frame),
            physical_gap_starts(frame),
        )
        rolling = pd.concat(
            [
                shifted.loc[segment.index]
                .rolling(window=window_size, min_periods=min_points)
                .corr(pair.loc[segment.index, target])
                for _, segment in pair.groupby(segment_ids, sort=False)
            ]
        )
        if target_mask is not None:
            rolling = rolling.where(target_mask.reindex(rolling.index).fillna(False).astype(bool))
        rolling = rolling.replace([np.inf, -np.inf], np.nan).dropna()
        if rolling.empty:
            continue
        sign_consistency = rolling.apply(lambda value: 1 if value >= 0 else -1).value_counts(normalize=True).max()
        iqr = float(rolling.quantile(0.75) - rolling.quantile(0.25))
        abs_median = float(rolling.abs().median())
        stability = max(0.0, min(1.0, abs_median * float(sign_consistency) * (1.0 - min(1.0, iqr))))
        rows.append({"variable": variable, "best_lag": best_lag, "best_score": best_score, "rolling_corr_median": float(rolling.median()), "rolling_abs_corr_median": abs_median, "rolling_corr_iqr": iqr, "rolling_sign_consistency": float(sign_consistency), "valid_window_count": int(len(rolling)), "rolling_stability": stability})
    return pd.DataFrame(rows, columns=cols)


def _safe_float(value: object, default: float = 0.0) -> float:
    return to_float(value, default)


def _risk_token_set(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def _risk_adjustment(value: object) -> tuple[float, float, str]:
    tokens = _risk_token_set(value)
    penalty_rate = min(0.80, sum(RISK_RELATIVE_PENALTY_WEIGHTS.get(token, 0.0) for token in tokens))
    cap = 1.0
    reason = ""
    for token, token_cap in EVIDENCE_SCORE_CAPS.items():
        if token in tokens and token_cap < cap:
            cap = token_cap
            reason = token
    return float(penalty_rate), float(cap), reason


def classify_candidate(row: pd.Series) -> str:
    flags = _risk_token_set(row.get("risk_flags", ""))
    for token, candidate_class in [
        ("severe_data_quality", "poor_quality"),
        ("strong_formula_leakage", "formula_or_derived"),
    ]:
        if token in flags:
            return candidate_class

    temporal_status = str(row.get("temporal_direction_status", ""))
    if temporal_status == "target_leads_supported":
        return "downstream_response"
    if temporal_status == "variable_leads_supported":
        return "upstream_driver_candidate"
    if temporal_status == "synchronous":
        return "synchronous_association"
    if temporal_status in {"direction_unresolved", "not_computed"}:
        return "uncertain_candidate"
    for token, candidate_class in [
        ("common_capacity_driver", "capacity_driven"),
        ("redundant_proxy", "uncertain_candidate"),
    ]:
        if token in flags:
            return candidate_class

    return "uncertain_candidate"


def _combine_correlation_evidence(
    association_score: pd.Series,
    independent_signal_score: pd.Series | None = None,
    innovation_score: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    association = pd.to_numeric(association_score, errors="coerce").astype(float)
    status = pd.Series("association_only", index=association_score.index, dtype=object)
    return association.clip(0, 1), status


def _temporal_adjustment(row: pd.Series) -> tuple[str, float, float]:
    value = row.get("temporal_direction_status", pd.NA)
    status = "not_computed" if pd.isna(value) else str(value).strip()
    if status not in {
        "variable_leads_supported",
        "target_leads_supported",
        "synchronous",
        "direction_unresolved",
        "not_computed",
    }:
        status = "not_computed"
    if status == "target_leads_supported":
        return status, TARGET_LEADS_PENALTY_RATE, TARGET_LEADS_SCORE_CAP
    return status, 0.0, 1.0


def _temporal_status_available(value: object) -> bool:
    return value in {
        "variable_leads_supported",
        "target_leads_supported",
        "synchronous",
        "direction_unresolved",
    }


def _data_quality_score(diag: Mapping[str, object]) -> float:
    rates = np.array(
        [
            max(0.0, _safe_float(diag.get("missing_rate", 0.0))),
            max(0.0, _safe_float(diag.get("saturation_ratio", 0.0))),
            max(0.0, _safe_float(diag.get("abnormal_jump_ratio", 0.0))),
            max(0.0, _safe_float(diag.get("robust_outlier_ratio", 0.0))),
        ]
    )
    reference_rates = np.array([0.20, 0.20, 0.01, 0.01])
    quality_components = np.exp(-np.log(2) * rates / reference_rates)
    return float(
        np.clip(
            np.prod(quality_components) ** (1 / len(quality_components)), 0.0, 1.0
        )
    )


def _redundant_proxy_variables(
    frame: pd.DataFrame | None,
    ranked: pd.DataFrame,
    target: str | None,
    *,
    residual_map: Mapping[str, object] | None = None,
    stability_map: Mapping[str, Mapping[str, object]] | None = None,
    diag_map: Mapping[str, Mapping[str, object]] | None = None,
    lag_map: Mapping[str, Mapping[str, object]] | None = None,
    rolling_map: Mapping[str, Mapping[str, object]] | None = None,
    lift_map: Mapping[str, Mapping[str, object]] | None = None,
) -> set[str]:
    """Resolve redundant positive-lag candidate groups from computed evidence."""
    if frame is None or not target or target not in frame.columns:
        return set()
    period_ns = sample_period_ns(frame)
    forced_starts = physical_gap_starts(frame)
    residual_map = residual_map or {}
    stability_map = stability_map or {}
    diag_map = diag_map or {}
    lag_map = lag_map or {}
    rolling_map = rolling_map or {}
    lift_map = lift_map or {}
    candidates = [
        str(row["variable"])
        for _, row in ranked.iterrows()
        if str(row.get("variable", "")) in frame.columns
        and str(row.get("variable", "")) != target
        and _safe_float(row.get("lag", 0), default=0.0) > 0
    ]
    candidate_lags = {
        str(row["variable"]): int(_safe_float(row.get("lag", 0), default=0.0))
        for _, row in ranked.iterrows()
    }
    adjacency = {variable: set() for variable in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            pair = pd.DataFrame(
                {
                    left: lagged_series(frame[left], frame.index, candidate_lags[left], period_ns=period_ns, forced_starts=forced_starts),
                    right: lagged_series(frame[right], frame.index, candidate_lags[right], period_ns=period_ns, forced_starts=forced_starts),
                }
            ).dropna()
            if len(pair) < 30 or abs(float(pair[left].corr(pair[right]))) < 0.995:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)

    profiles = {
        str(row["variable"]): _redundancy_evidence_profile(
            row,
            residual_map=residual_map,
            stability_map=stability_map,
            diag_map=diag_map,
            lag_map=lag_map,
            rolling_map=rolling_map,
            lift_map=lift_map,
        )
        for _, row in ranked.iterrows()
        if str(row.get("variable", "")) in adjacency
    }
    redundant: set[str] = set()
    visited: set[str] = set()
    for variable in candidates:
        if variable in visited:
            continue
        stack = [variable]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(adjacency[current] - group)
        visited.update(group)
        if len(group) < 2:
            continue
        representatives = [
            candidate
            for candidate in group
            if all(
                candidate == other
                or _evidence_clearly_exceeds(
                    profiles[candidate], profiles[other]
                )
                for other in group
            )
        ]
        if len(representatives) == 1:
            redundant.update(group - set(representatives))
        else:
            redundant.update(group)
    return redundant


def _redundancy_evidence_profile(
    row: pd.Series,
    *,
    residual_map: Mapping[str, object],
    stability_map: Mapping[str, Mapping[str, object]],
    diag_map: Mapping[str, Mapping[str, object]],
    lag_map: Mapping[str, Mapping[str, object]],
    rolling_map: Mapping[str, Mapping[str, object]],
    lift_map: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    variable = str(row["variable"])
    stability = stability_map.get(variable, {})
    rolling = rolling_map.get(variable, {})
    lift = lift_map.get(variable, {})
    return {
        "independence": _safe_float(residual_map.get(variable, np.nan), default=np.nan),
        "prediction": _safe_float(
            lift.get("model_lift_score", lift.get("model_lift", np.nan)), default=np.nan
        ),
        "data_quality": _data_quality_score(diag_map.get(variable, {})),
        "stability": _safe_float(
            stability.get("regime_stability_final", rolling.get("rolling_stability", np.nan)),
            default=np.nan,
        ),
        "lag_quality": _safe_float(lag_map.get(variable, {}).get("lag_quality", np.nan), default=np.nan),
        "association": _safe_float(
            row.get("association_score", row.get("score", np.nan)), default=np.nan
        ),
    }


def _evidence_clearly_exceeds(
    candidate: Mapping[str, float], other: Mapping[str, float]
) -> bool:
    differences = [
        candidate[field] - other[field]
        for field in candidate
        if np.isfinite(candidate[field]) and np.isfinite(other.get(field, np.nan))
    ]
    return bool(
        any(difference >= EVIDENCE_SEPARATION_MARGIN for difference in differences)
        and not any(difference <= -EVIDENCE_SEPARATION_MARGIN for difference in differences)
    )


def _residual_condition_number_map(residual: pd.DataFrame) -> dict[str, float]:
    if residual.empty or "variable" not in residual.columns:
        return {}
    field = (
        "control_condition_number"
        if "control_condition_number" in residual.columns
        else "condition_number" if "condition_number" in residual.columns else None
    )
    if field is None:
        return {}
    values = residual.loc[:, ["variable", field]].copy()
    values[field] = pd.to_numeric(values[field], errors="coerce")
    values.loc[~np.isfinite(values[field]), field] = np.nan
    values = values.drop_duplicates(subset="variable", keep="first").dropna(subset=[field])
    return values.set_index("variable")[field].to_dict()


def risk_flags(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, diag: pd.DataFrame, roles: dict[str, str], control_columns: list[str] | None, lag_peak_quality: pd.DataFrame | None = None, rolling_corr_scores: pd.DataFrame | None = None, model_lift_scores: pd.DataFrame | None = None, *, frame: pd.DataFrame | None = None, target: str | None = None) -> pd.DataFrame:
    cols = ["variable", "formula_like_flag", "strong_formula_leakage_flag", "common_capacity_driver_flag", "redundant_proxy_flag", "target_leads_variable_flag", "unstable_across_regimes_flag", "unstable_over_time_flag", "lag_boundary_flag", "low_model_lift_flag", "poor_data_quality_flag", "residual_collinearity_flag", "data_quality_score", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)

    residual_map = residual.set_index("variable")["residual_corr"].to_dict() if not residual.empty and "residual_corr" in residual.columns else {}
    residual_cond_map = _residual_condition_number_map(residual)
    stability_map = stability.set_index("variable").to_dict("index") if not stability.empty else {}
    diag_map = diag.set_index("variable").to_dict("index") if not diag.empty else {}
    lag_map = lag_peak_quality.set_index("variable").to_dict("index") if lag_peak_quality is not None and not lag_peak_quality.empty else {}
    roll_map = rolling_corr_scores.set_index("variable").to_dict("index") if rolling_corr_scores is not None and not rolling_corr_scores.empty else {}
    lift_map = model_lift_scores.set_index("variable").to_dict("index") if model_lift_scores is not None and not model_lift_scores.empty else {}
    redundant_variables = _redundant_proxy_variables(
        frame,
        ranked,
        target,
        residual_map=residual_map,
        stability_map=stability_map,
        diag_map=diag_map,
        lag_map=lag_map,
        rolling_map=roll_map,
        lift_map=lift_map,
    )

    rows = []
    for _, row in ranked.iterrows():
        variable = str(row.get("variable", ""))
        raw_corr = _safe_float(row.get("score", 0), default=0.0)
        residual_corr = _safe_float(residual_map.get(variable, raw_corr), default=raw_corr)
        regime_info = stability_map.get(variable, {})
        regime_stability = regime_info.get("regime_stability_final", np.nan)
        regime_status = str(regime_info.get("regime_evidence_status", ""))
        d = diag_map.get(variable, {})
        data_quality_score = _data_quality_score(d)
        poor_quality = (
            _safe_float(d.get("missing_rate", 0), default=0.0) > 0.2
            or _safe_float(d.get("saturation_ratio", 0), default=0.0) > 0.2
            or _safe_float(d.get("abnormal_jump_ratio", 0), default=0.0) > 0.01
            or _safe_float(d.get("robust_outlier_ratio", 0), default=0.0) > 0.01
        )
        severe_quality = (
            _safe_float(d.get("missing_rate", 0), default=0.0) > 0.50
            or _safe_float(d.get("saturation_ratio", 0), default=0.0) > 0.80
            or _safe_float(d.get("abnormal_jump_ratio", 0), default=0.0) > 0.05
            or _safe_float(d.get("robust_outlier_ratio", 0), default=0.0) > 0.05
        )
        lag_value = int(_safe_float(row.get("lag", 0), default=0.0))
        formula_like = _looks_like_formula_variable(variable)
        strong_formula = formula_like and raw_corr > 0.98 and lag_value == 0
        common_capacity = bool(control_columns) and raw_corr >= 0.5 and residual_corr < raw_corr * 0.65
        temporal_status = lag_map.get(variable, {}).get("temporal_direction_status")
        target_leads = (
            isinstance(temporal_status, str)
            and temporal_status == "target_leads_supported"
        )
        regime_evaluated = regime_status in {"partial_coverage", "full_coverage"}
        unstable_reg = (
            regime_evaluated
            and pd.notna(regime_stability)
            and float(regime_stability) < REGIME_UNSTABLE_THRESHOLD
        )
        lift_info = lift_map.get(variable, {})
        model_supported = str(lift_info.get("status", "")).startswith("ok") and _safe_float(
            lift_info.get("model_lift", 0.0), default=0.0
        ) >= 0.05
        unstable_time = _safe_float(
            roll_map.get(variable, {}).get("rolling_stability", 1.0), default=1.0
        ) < 0.35 and not (model_supported and raw_corr < 0.2)
        lag_boundary = bool(lag_map.get(variable, {}).get("lag_boundary_flag", False))
        low_lift = str(lift_info.get("status", "")).startswith("ok") and _safe_float(
            lift_info.get("model_lift", 0.0), default=0.0
        ) < 0.01
        residual_collinearity = _safe_float(residual_cond_map.get(variable, 0), default=0.0) > 1e8

        flags = [name for name, active in [
            ("formula_like", formula_like),
            ("strong_formula_leakage", strong_formula),
            ("common_capacity_driver", common_capacity),
            ("redundant_proxy", variable in redundant_variables),
            ("target_leads_variable", target_leads),
            ("unstable_across_regimes", unstable_reg),
            ("unstable_over_time", unstable_time),
            ("lag_boundary", lag_boundary),
            ("low_model_lift", low_lift),
            ("poor_data_quality", poor_quality and not severe_quality),
            ("severe_data_quality", severe_quality),
            ("residual_collinearity", residual_collinearity),
        ] if active]

        strong_risks = [f for f in flags if f in {"strong_formula_leakage", "common_capacity_driver", "severe_data_quality"}]
        weak_risks = [f for f in flags if f not in set(strong_risks)]
        level = "none" if not flags else ("strong" if len(strong_risks) >= 2 else ("medium" if strong_risks else "weak"))
        reason_map = {
            "formula_like": "疑似公式类变量",
            "strong_formula_leakage": "强公式泄漏风险",
            "common_capacity_driver": "疑似共同负荷驱动",
            "target_leads_variable": "目标领先变量",
            "unstable_across_regimes": "跨工况不稳定",
            "unstable_over_time": "随时间不稳定",
            "lag_boundary": "滞后触边界",
            "low_model_lift": "模型增益偏低",
            "poor_data_quality": "数据质量需关注，建议核查缺失、单值集中和异常点",
            "severe_data_quality": "数据质量严重不足",
            "residual_collinearity": "残差控制共线性高",
            "redundant_proxy": "与其他候选变量高度冗余，独立信息不足",
        }
        reason = "；".join(reason_map.get(flag, flag) for flag in flags)
        rows.append({"variable": variable, "formula_like_flag": formula_like, "strong_formula_leakage_flag": strong_formula, "common_capacity_driver_flag": common_capacity, "redundant_proxy_flag": variable in redundant_variables, "target_leads_variable_flag": target_leads, "unstable_across_regimes_flag": unstable_reg, "unstable_over_time_flag": unstable_time, "lag_boundary_flag": lag_boundary, "low_model_lift_flag": low_lift, "poor_data_quality_flag": poor_quality and not severe_quality, "residual_collinearity_flag": residual_collinearity, "data_quality_score": data_quality_score, "risk_flags": ";".join(flags), "risk_count": len(flags), "strong_risk_count": len(strong_risks), "weak_risk_count": len(weak_risks), "risk_level": level, "human_reason": reason})
    return pd.DataFrame(rows, columns=cols)


def final_ranked_features(ranked: pd.DataFrame, residual: pd.DataFrame, stability: pd.DataFrame, model_lift: pd.DataFrame, risks: pd.DataFrame, lag_peak_quality: pd.DataFrame, rolling_corr_scores: pd.DataFrame, force_include_variables: list[str] | None = None, top_k: int | None = None, control_columns: list[str] | None = None, capacity_columns: list[str] | None = None, segment_column: str | None = None) -> pd.DataFrame:
    cols = [
        "variable", "lag", "direction", "pearson", "spearman", "method",
        "pearson_p", "spearman_p", "pearson_q", "spearman_q", "corr_q_value",
        "pearson_r2", "spearman_r2", "n", "raw_corr", "association_score",
        "innovation_score", "innovation_lag", "innovation_direction", "innovation_sign",
        "innovation_status", "correlation_evidence_score", "correlation_evidence_status",
        "lag_quality", "lag_quality_status", "lag_boundary_flag", "data_quality_score",
        "evidence_strength", "evidence_available_count", "evidence_completeness",
        "evidence_confidence", "evidence_score_low", "evidence_score_high", "score_method",
        "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason",
        "risk_flags", "evidence_score", "risk_penalty_rate", "risk_penalty", "risk_score_cap",
        "risk_cap_reason", "final_score", "association_rank", "candidate_class",
        "driver_priority_factor", "driver_priority_score", "driver_rank", "candidate_grade",
        "recommended_use", "recommended_action", "force_included", "engineering_context",
        "is_residual_control", "is_capacity_reference", "is_segment_reference", "variable_role",
        "stability_score", "evidence_missing_items", "evidence_coverage_status",
        "near_peak_lag_min", "near_peak_lag_max", "near_peak_lag_count",
        "temporal_direction_status", "temporal_penalty_rate", "temporal_score_cap",
        *CONTROL_REFERENCE_COLUMNS,
    ]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    final = ranked.rename(columns={"score": "raw_corr"}).copy()
    residual_source = residual[[c for c in ["variable", "residual_corr"] if c in residual.columns]].copy()
    if "variable" not in residual_source.columns:
        residual_source = pd.DataFrame(columns=["variable"])
    final = final.merge(residual_source, on="variable", how="left")
    stability_columns = list(REGIME_STABILITY_COLUMNS)
    stability_source = stability[[c for c in stability_columns if c in stability.columns]].copy()
    if "variable" not in stability_source.columns:
        stability_source = pd.DataFrame(columns=["variable"])
    # Upstream callers may already carry selected regime fields.  Keep the
    # freshly computed stability evidence as the single source for this stage.
    final = final.drop(columns=[c for c in stability_source.columns if c != "variable" and c in final.columns])
    final = final.merge(stability_source, on="variable", how="left")
    model_columns = [c for c in ["variable", "model_lift", "model_lift_score", "status"] if c in model_lift.columns]
    model_source = model_lift[model_columns].copy()
    if "status" in model_source.columns:
        model_source = model_source.rename(columns={"status": "_model_lift_source_status"})
    final = final.merge(model_source, on="variable", how="left")
    risk_columns = ["variable", "risk_flags", "risk_count", "strong_risk_count", "weak_risk_count", "risk_level", "human_reason", "data_quality_score"]
    risk_source = risks[[c for c in risk_columns if c in risks.columns]].copy()
    if "variable" not in risk_source.columns:
        risk_source = pd.DataFrame(columns=["variable"])
    final = final.merge(risk_source, on="variable", how="left")
    lag_peak_columns = [
        "variable", "lag_quality", "near_peak_lag_min", "near_peak_lag_max",
        "near_peak_lag_count", "temporal_direction_status",
    ]
    if "lag_boundary_flag" not in final.columns:
        lag_peak_columns.append("lag_boundary_flag")
    final = final.merge(
        lag_peak_quality[[c for c in lag_peak_columns if c in lag_peak_quality.columns]],
        on="variable",
        how="left",
    )
    if "temporal_direction_status" not in final.columns:
        final["temporal_direction_status"] = pd.NA
    final = final.merge(rolling_corr_scores[[c for c in ["variable", "rolling_stability"] if c in rolling_corr_scores.columns]], on="variable", how="left")
    # Some legacy callers pre-merge regime evidence.  Pandas then suffixes the
    # duplicate flag; normalize it before the layer-status contract is built.
    if "regime_sign_reversal_flag" not in final.columns:
        for suffix in ("_y", "_x"):
            legacy_flag = f"regime_sign_reversal_flag{suffix}"
            if legacy_flag in final.columns:
                final["regime_sign_reversal_flag"] = final[legacy_flag]
                break

    residual_raw = pd.to_numeric(final["residual_corr"], errors="coerce") if "residual_corr" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    regime_raw = pd.to_numeric(final["regime_stability_final"], errors="coerce") if "regime_stability_final" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    rolling_raw = pd.to_numeric(final["rolling_stability"], errors="coerce") if "rolling_stability" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    lagq_raw = pd.to_numeric(final["lag_quality"], errors="coerce") if "lag_quality" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    innovation_raw = pd.to_numeric(final["innovation_score"], errors="coerce") if "innovation_score" in final.columns else pd.Series(np.nan, index=final.index, dtype=float)
    if "model_lift_score" in final.columns:
        lift_raw = final["model_lift_score"]
    elif "model_lift" in final.columns:
        lift_raw = final["model_lift"]
    else:
        lift_raw = pd.Series(np.nan, index=final.index, dtype=float)
    lift_raw = pd.to_numeric(lift_raw, errors="coerce")
    final["residual_status"] = np.where(residual_raw.notna(), "ok", "not_computed")
    regime_evidence_status = final.get("regime_evidence_status", pd.Series(np.nan, index=final.index))
    final["regime_status"] = regime_evidence_status.where(
        regime_evidence_status.notna(), np.where(regime_raw.notna(), "ok", "not_computed")
    )
    final["rolling_status"] = np.where(rolling_raw.notna(), "ok", "not_computed")
    model_source_status = final.get("_model_lift_source_status", pd.Series(np.nan, index=final.index))
    final["model_lift_status"] = model_source_status.where(
        model_source_status.notna(), np.where(lift_raw.notna(), "ok", "not_computed")
    )
    final["lag_quality_status"] = np.where(lagq_raw.notna(), "ok", "not_computed")
    final["association_score"] = pd.to_numeric(final["raw_corr"], errors="coerce").clip(0, 1)
    final["innovation_score"] = pd.to_numeric(innovation_raw, errors="coerce").clip(0, 1)
    final["regime_stability_final"] = regime_raw.clip(0,1)
    final["rolling_stability"] = rolling_raw.clip(0,1)
    final["lag_quality"] = lagq_raw.clip(0,1)
    final["model_lift_score"] = lift_raw.clip(0,1)
    model_lift_computed = final["model_lift_status"].astype(str).str.startswith("ok")
    final["prediction_score"] = final["model_lift_score"].where(model_lift_computed)
    final["independent_signal_score"] = pd.to_numeric(residual_raw, errors="coerce").clip(0, 1)
    (
        final["correlation_evidence_score"],
        final["correlation_evidence_status"],
    ) = _combine_correlation_evidence(
        final["association_score"],
        final["independent_signal_score"],
        final["innovation_score"],
    )
    both_stability = final["regime_stability_final"].notna() & final["rolling_stability"].notna()
    final["stability_score"] = final["rolling_stability"].where(
        final["rolling_stability"].notna(), final["regime_stability_final"]
    ).astype(float)
    final.loc[both_stability, "stability_score"] = np.sqrt(
        final.loc[both_stability, "rolling_stability"]
        * final.loc[both_stability, "regime_stability_final"]
    )

    data_quality_raw = final.get("data_quality_score", pd.Series(1.0, index=final.index))
    final["data_quality_score"] = pd.to_numeric(data_quality_raw, errors="coerce").clip(0, 1)
    final["evidence_confidence"] = final["data_quality_score"]
    temporal_values = final.apply(_temporal_adjustment, axis=1)
    final[["temporal_direction_status", "temporal_penalty_rate", "temporal_score_cap"]] = pd.DataFrame(
        temporal_values.tolist(), index=final.index
    )
    association_available = final["association_score"].notna()
    quality_available = final["data_quality_score"].notna()
    temporal_available = final["temporal_direction_status"].map(_temporal_status_available)
    final["evidence_available_count"] = (
        association_available.astype(int)
        + quality_available.astype(int)
        + temporal_available.astype(int)
    )
    final["evidence_completeness"] = final["evidence_available_count"] / 3.0
    final["evidence_missing_items"] = pd.DataFrame(
        {
            "基础关联": ~association_available,
            "数据质量": ~quality_available,
            "时间方向": ~temporal_available,
        }
    ).apply(lambda row: "；".join(row.index[row].tolist()), axis=1)
    missing_count = 3 - final["evidence_available_count"]
    final["evidence_coverage_status"] = np.select(
        [missing_count.eq(0), missing_count.eq(1)],
        ["完整", "部分完整"],
        default="证据不足",
    )

    final["evidence_strength"] = final["association_score"]
    final["evidence_score"] = (
        final["evidence_strength"] * final["evidence_confidence"]
    ).clip(0, 1)
    final["evidence_score_low"] = final["evidence_score"]
    final["evidence_score_high"] = final["evidence_score"]
    final["score_method"] = "initial_association_temporal_v4"
    risk_values = final.get("risk_flags", pd.Series("", index=final.index)).map(_risk_adjustment)
    final[["risk_penalty_rate", "risk_score_cap", "risk_cap_reason"]] = pd.DataFrame(
        risk_values.tolist(), index=final.index
    )
    final["risk_penalty"] = final["evidence_score"] * final["risk_penalty_rate"]
    risk_adjusted_score = (
        final["evidence_score"] * (1.0 - final["risk_penalty_rate"])
    ).clip(0, 1)
    temporal_adjusted_score = (
        risk_adjusted_score * (1.0 - final["temporal_penalty_rate"])
    ).clip(0, 1)
    final["final_score"] = pd.concat(
        [temporal_adjusted_score, final["risk_score_cap"], final["temporal_score_cap"]],
        axis=1,
    ).min(axis=1, skipna=False).clip(0, 1)
    final["association_rank"] = final["evidence_score"].rank(
        method="first", ascending=False
    ).astype("Int64")
    final["candidate_class"] = final.apply(classify_candidate, axis=1)
    # These columns are retained only for readers of historical CSV files. They
    # are aliases of the statistical score and never affect screening behavior.
    final["driver_priority_factor"] = 1.0
    final["driver_priority_score"] = final["final_score"]
    final = _finalize_driver_ranking(
        final,
        force_include_variables=force_include_variables,
        control_columns=control_columns,
        capacity_columns=capacity_columns,
        segment_column=segment_column,
        primary_rank_column=PRIMARY_RANK_COLUMN,
    )
    final = order_initial_candidates(final)
    final["driver_rank"] = np.arange(1, len(final) + 1)
    for c in cols:
        if c not in final.columns:
            final[c] = np.nan
    return final.reset_index(drop=True)[cols]


def _finalize_driver_ranking(
    final: pd.DataFrame,
    force_include_variables: list[str] | None = None,
    top_k: int | None = None,
    control_columns: list[str] | None = None,
    capacity_columns: list[str] | None = None,
    segment_column: str | None = None,
    primary_rank_column: str = PRIMARY_RANK_COLUMN,
) -> pd.DataFrame:
    final = final.copy()
    final["driver_priority_score"] = final["final_score"]
    forced = set(force_include_variables or [])
    final["force_included"] = final["variable"].astype(str).isin(forced)
    final["candidate_grade"] = final.apply(_grade_candidate, axis=1)
    final["recommended_use"] = final.apply(_recommend_use, axis=1)
    variables = final["variable"].astype(str)
    residual_set = {str(value) for value in (control_columns or [])}
    capacity_set = {str(value) for value in (capacity_columns or [])}
    final["is_residual_control"] = variables.isin(residual_set)
    final["is_capacity_reference"] = variables.isin(capacity_set)
    final["is_segment_reference"] = bool(segment_column) & variables.eq(str(segment_column))
    final["variable_role"] = np.select(
        [final["is_residual_control"], final["is_capacity_reference"], final["is_segment_reference"]],
        ["residual_control", "capacity_reference", "segment_reference"],
        default="candidate",
    )
    detected = variables.map(detect_auto_control_reference)
    final["is_auto_control_reference"] = detected.map(lambda value: value[0]).astype(bool)
    final["is_control_reference"] = (
        final[["is_residual_control", "is_capacity_reference", "is_segment_reference"]]
        .any(axis=1)
        | final["is_auto_control_reference"]
    )
    reference_type = detected.map(lambda value: value[1]).replace("", pd.NA)
    reference_source = detected.map(lambda value: value[2]).replace("", pd.NA)
    reference_type = reference_type.mask(final["is_segment_reference"], "segment_reference")
    reference_source = reference_source.mask(final["is_segment_reference"], "configured_segment")
    reference_type = reference_type.mask(final["is_capacity_reference"], "capacity_reference")
    reference_source = reference_source.mask(final["is_capacity_reference"], "configured_capacity")
    reference_type = reference_type.mask(final["is_residual_control"], "residual_control")
    reference_source = reference_source.mask(
        final["is_residual_control"], "configured_residual_control"
    )
    final["control_reference_type"] = reference_type
    final["control_reference_source"] = reference_source
    final["recommended_action"] = final.apply(_recommended_action, axis=1)
    final = order_initial_candidates(final)
    final["driver_rank"] = np.arange(1, len(final) + 1)
    return final


def build_recommended_candidates(
    ranked_features: pd.DataFrame,
    top_k: int | None,
    force_include_variables: list[str] | None = None,
    exclude_control_columns: bool = True,
    *,
    residual_corr_scores: pd.DataFrame | None = None,
    residual_top_k: int | None = None,
) -> pd.DataFrame:
    """Build the downstream candidate pool from raw, residual, and forced channels."""
    if ranked_features.empty:
        empty = ranked_features.drop(
            columns=list(CONTROL_REFERENCE_COLUMNS), errors="ignore"
        ).copy(deep=True)
        for column in ["selected_by_raw", "selected_by_residual", "candidate_source", "raw_candidate_rank", "residual_candidate_rank", "candidate_pool_rank", "common_capacity_candidate_flag"]:
            if column not in empty.columns:
                empty[column] = pd.Series(dtype=bool if column.startswith("selected_") or column.endswith("_flag") else float if column.endswith("_rank") else str)
        return empty
    frame = ranked_features.copy(deep=True)
    frame["variable"] = frame["variable"].astype(str)
    forced_order = list(dict.fromkeys(str(value) for value in (force_include_variables or [])))
    forced = set(forced_order)
    reference_columns = [
        "is_residual_control",
        "is_capacity_reference",
        "is_segment_reference",
        "is_auto_control_reference",
    ]
    references = frame.reindex(columns=reference_columns, fill_value=False).fillna(False).astype(bool).any(axis=1)
    eligible = ~references if exclude_control_columns else pd.Series(True, index=frame.index)
    raw_score = pd.to_numeric(frame.get("final_score", frame.get("score", pd.Series(np.nan, index=frame.index))), errors="coerce")
    raw = order_initial_candidates(frame.loc[eligible & np.isfinite(raw_score) & raw_score.ge(RAW_CANDIDATE_MIN_SCORE)])
    raw = raw.head(top_k) if top_k is not None else raw
    raw_rank = {name: index + 1 for index, name in enumerate(raw["variable"])}

    residual_top_k = top_k if residual_top_k is None else residual_top_k
    residual_rank: dict[str, int] = {}
    valid_residual_rows = pd.DataFrame(columns=["variable"])
    eligible_residual_rows = pd.DataFrame(columns=["variable"])
    selected_residual_rows = pd.DataFrame(columns=["variable"])
    required_residual_columns = {"variable", "residual_corr", "residual_lag", "residual_n", "residual_status"}
    if (
        residual_corr_scores is not None
        and not residual_corr_scores.empty
        and required_residual_columns.issubset(residual_corr_scores.columns)
    ):
        residual = residual_corr_scores.copy(deep=True)
        residual["variable"] = residual["variable"].astype(str)
        residual = residual[residual["variable"].isin(frame["variable"])]
        if exclude_control_columns:
            residual = residual[residual["variable"].isin(frame.loc[eligible, "variable"])]
        residual["_residual_corr"] = pd.to_numeric(residual.get("residual_corr"), errors="coerce")
        residual["_residual_lag"] = pd.to_numeric(residual.get("residual_lag"), errors="coerce")
        residual["_residual_n"] = pd.to_numeric(residual.get("residual_n"), errors="coerce")
        residual["_residual_quality"] = pd.to_numeric(residual.get("residual_lag_quality"), errors="coerce")
        residual = residual[
            residual.get("residual_status", pd.Series("", index=residual.index)).isin(["ok", "rank_deficient"])
            & np.isfinite(residual["_residual_corr"])
            & np.isfinite(residual["_residual_lag"])
            & np.isfinite(residual["_residual_n"])
            & residual["_residual_n"].gt(0)
        ].copy()
        residual["_status_priority"] = residual["residual_status"].map({"ok": 0, "rank_deficient": 1})
        residual["_quality_sort"] = residual["_residual_quality"].fillna(-np.inf)
        valid_residual_rows = residual.sort_values(
            ["_residual_corr", "_quality_sort", "_residual_n", "_status_priority", "variable"],
            ascending=[False, False, False, True, True], kind="stable",
        ).drop_duplicates(subset="variable", keep="first")
        eligible_residual_rows = valid_residual_rows[
            valid_residual_rows["_residual_corr"].ge(RESIDUAL_CANDIDATE_MIN_CORR)
            & valid_residual_rows["_residual_n"].ge(RESIDUAL_CANDIDATE_MIN_N)
            & np.isfinite(valid_residual_rows["_residual_quality"])
        ]
        selected_residual_rows = eligible_residual_rows.head(residual_top_k) if residual_top_k is not None else eligible_residual_rows
        residual_rank = {name: index + 1 for index, name in enumerate(selected_residual_rows["variable"])}

    selected_variables = set(raw_rank) | set(residual_rank) | (forced & set(frame["variable"]))
    pool = frame[frame["variable"].isin(selected_variables)].copy()
    pool["selected_by_raw"] = pool["variable"].isin(raw_rank)
    pool["selected_by_residual"] = pool["variable"].isin(residual_rank)
    pool["raw_candidate_rank"] = pool["variable"].map(raw_rank)
    pool["residual_candidate_rank"] = pool["variable"].map(residual_rank)
    pool["force_included"] = pool["variable"].isin(forced)
    pool["_reference"] = references.reindex(pool.index).fillna(False)
    pool["candidate_source"] = np.select(
        [
            pool["_reference"],
            pool["selected_by_raw"] & pool["selected_by_residual"],
            pool["selected_by_raw"],
            pool["selected_by_residual"],
        ],
        ["control_reference", "raw_and_residual", "raw_only", "residual_only"],
        default="force_included",
    )
    raw_corr = pd.to_numeric(pool.get("raw_corr", pool.get("score", pd.Series(np.nan, index=pool.index))), errors="coerce")
    residual_lookup = valid_residual_rows.set_index("variable")["_residual_corr"] if not valid_residual_rows.empty else pd.Series(dtype=float)
    pool["common_capacity_candidate_flag"] = (
        pool["selected_by_raw"]
        & ~pool["selected_by_residual"]
        & pool["variable"].isin(residual_lookup.index)
        & raw_corr.ge(0.5)
        & pool["variable"].map(residual_lookup).lt(raw_corr * 0.65)
    )
    source_priority = pool["candidate_source"].map({"raw_and_residual": 0, "raw_only": 1, "residual_only": 1, "force_included": 2, "control_reference": 3})
    force_rank = {name: index + 1 for index, name in enumerate(forced_order)}
    pool["_best_rank"] = pool[["raw_candidate_rank", "residual_candidate_rank"]].min(axis=1).fillna(np.inf)
    pool["_force_rank"] = pool["variable"].map(force_rank).fillna(np.inf)
    pool = pool.assign(_source_priority=source_priority).sort_values(
        ["_source_priority", "_best_rank", "raw_candidate_rank", "residual_candidate_rank", "_force_rank", "variable"],
        ascending=[True, True, True, True, True, True], kind="stable",
    )
    pool["candidate_pool_rank"] = np.arange(1, len(pool) + 1)
    return pool.drop(
        columns=[
            "_reference", "_best_rank", "_force_rank", "_source_priority",
            *CONTROL_REFERENCE_COLUMNS,
        ],
        errors="ignore",
    ).reset_index(drop=True)


def prioritize_recommended_candidates(
    recommended_candidates: pd.DataFrame,
    residual_corr_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach load-adjusted evidence and deterministically prioritize the PR-3 pool."""
    base_columns = [
        column for column in recommended_candidates.columns
        if column not in CANDIDATE_PRIORITY_COLUMNS
    ]
    if recommended_candidates.empty:
        empty = recommended_candidates.loc[:, base_columns].copy(deep=True)
        for column in CANDIDATE_PRIORITY_COLUMNS:
            empty[column] = pd.Series(dtype="Int64" if column.endswith("_rank") or column.endswith("_tier") else "object")
        return empty

    frame = recommended_candidates.loc[:, base_columns].copy(deep=True)
    frame["variable"] = frame["variable"].astype(str)
    frame["_candidate_pool_sort"] = pd.to_numeric(frame.get("candidate_pool_rank"), errors="coerce")
    frame["_candidate_raw_sort"] = _candidate_bool_series(frame, "selected_by_raw")
    frame["_candidate_residual_sort"] = _candidate_bool_series(frame, "selected_by_residual")
    frame["_candidate_dual_sort"] = frame["_candidate_raw_sort"] & frame["_candidate_residual_sort"]
    frame["_candidate_common_load_sort"] = _candidate_bool_series(
        frame, "common_capacity_candidate_flag"
    )
    frame["_candidate_force_sort"] = _candidate_bool_series(frame, "force_included")
    frame["_candidate_source_priority"] = frame.get(
        "candidate_source", pd.Series("", index=frame.index)
    ).map(
        {
            "raw_and_residual": 0,
            "raw_only": 1,
            "residual_only": 2,
            "force_included": 3,
            "control_reference": 4,
        }
    ).fillna(5)
    frame["_candidate_raw_rank_sort"] = pd.to_numeric(
        frame.get("raw_candidate_rank"), errors="coerce"
    )
    frame["_candidate_residual_rank_sort"] = pd.to_numeric(
        frame.get("residual_candidate_rank"), errors="coerce"
    )
    frame["_candidate_final_sort"] = pd.to_numeric(frame.get("final_score"), errors="coerce")
    frame = frame.sort_values(
        [
            "_candidate_pool_sort", "_candidate_dual_sort", "_candidate_raw_sort",
            "_candidate_residual_sort", "_candidate_common_load_sort",
            "_candidate_source_priority", "_candidate_raw_rank_sort",
            "_candidate_residual_rank_sort", "_candidate_final_sort",
            "_candidate_force_sort", "variable",
        ],
        ascending=[True, False, False, False, True, True, True, True, False, False, True],
        na_position="last",
        kind="stable",
    ).drop_duplicates(subset="variable", keep="first")

    residual_columns = [
        "variable", "_priority_residual_corr", "_priority_residual_lag",
        "_priority_residual_n", "_priority_residual_quality",
        "_priority_residual_complete", "_priority_residual_present",
    ]
    residual_best = pd.DataFrame(columns=residual_columns)
    if (
        residual_corr_scores is not None
        and not residual_corr_scores.empty
        and "variable" in residual_corr_scores.columns
    ):
        residual = residual_corr_scores.copy(deep=True)
        residual["variable"] = residual["variable"].astype(str)
        residual = residual[residual["variable"].isin(frame["variable"])]
        residual["_priority_residual_corr"] = pd.to_numeric(residual.get("residual_corr"), errors="coerce")
        residual["_priority_residual_lag"] = pd.to_numeric(residual.get("residual_lag"), errors="coerce")
        residual["_priority_residual_n"] = pd.to_numeric(residual.get("residual_n"), errors="coerce")
        residual["_priority_residual_quality"] = pd.to_numeric(residual.get("residual_lag_quality"), errors="coerce")
        valid_status = residual.get("residual_status", pd.Series("", index=residual.index)).isin(
            ["ok", "rank_deficient"]
        )
        residual["_priority_residual_complete"] = (
            valid_status
            & np.isfinite(residual["_priority_residual_corr"])
            & np.isfinite(residual["_priority_residual_lag"])
            & np.isfinite(residual["_priority_residual_n"])
            & np.isfinite(residual["_priority_residual_quality"])
        )
        residual["_priority_residual_present"] = True
        residual["_priority_status_sort"] = residual.get(
            "residual_status", pd.Series("", index=residual.index)
        ).map({"ok": 0, "rank_deficient": 1}).fillna(2)
        residual["_priority_corr_sort"] = residual["_priority_residual_corr"].where(
            np.isfinite(residual["_priority_residual_corr"]), -np.inf
        )
        residual["_priority_quality_sort"] = residual["_priority_residual_quality"].where(
            np.isfinite(residual["_priority_residual_quality"]), -np.inf
        )
        residual["_priority_n_sort"] = residual["_priority_residual_n"].where(
            np.isfinite(residual["_priority_residual_n"]), -np.inf
        )
        residual_best = residual.sort_values(
            [
                "_priority_residual_complete", "_priority_corr_sort",
                "_priority_quality_sort", "_priority_n_sort",
                "_priority_status_sort", "variable",
            ],
            ascending=[False, False, False, False, True, True],
            kind="stable",
        ).drop_duplicates(subset="variable", keep="first")[residual_columns]

    frame = frame.merge(residual_best, on="variable", how="left", sort=False)
    selected_by_raw = frame["_candidate_raw_sort"]
    selected_by_residual = frame["_candidate_residual_sort"]
    common_load_risk = frame["_candidate_common_load_sort"]
    force_included = frame["_candidate_force_sort"]
    control_reference = frame.get(
        "candidate_source", pd.Series("", index=frame.index)
    ).astype(str).eq("control_reference")
    residual_present = frame["_priority_residual_present"].eq(True)
    residual_complete = frame["_priority_residual_complete"].eq(True)
    residual_strong = (
        residual_complete
        & frame["_priority_residual_corr"].ge(RESIDUAL_CANDIDATE_MIN_CORR)
        & frame["_priority_residual_n"].ge(RESIDUAL_CANDIDATE_MIN_N)
    )

    frame["residual_signal_score"] = (
        frame["_priority_residual_corr"].clip(0.0, 1.0)
        + frame["_priority_residual_quality"].clip(0.0, 1.0)
    ).div(2.0).where(residual_complete)
    frame["residual_evidence_status"] = np.select(
        [control_reference, ~residual_present, ~residual_complete, residual_strong],
        ["control_reference", "missing", "insufficient", "strong"],
        default="weak",
    )

    raw_only = selected_by_raw & ~selected_by_residual
    frame["load_adjusted_relation_status"] = np.select(
        [
            control_reference,
            selected_by_raw & selected_by_residual,
            ~selected_by_raw & selected_by_residual,
            raw_only & common_load_risk,
            raw_only & frame["residual_evidence_status"].eq("weak"),
            raw_only & frame["residual_evidence_status"].isin(["missing", "insufficient"]),
            raw_only,
            ~selected_by_raw & ~selected_by_residual & force_included,
        ],
        [
            "control_reference",
            "dual_channel_supported",
            "residual_only_supported",
            "raw_only_common_load_risk",
            "raw_only_residual_weak",
            "raw_only_residual_missing",
            "raw_only_supported",
            "force_included_only",
        ],
        default="force_included_only",
    )

    final_score = pd.to_numeric(
        frame.get("final_score", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    priority_score = pd.Series(np.nan, index=frame.index, dtype=float)
    dual_channel = selected_by_raw & selected_by_residual
    residual_only = ~selected_by_raw & selected_by_residual
    dual_score = 0.60 * final_score + 0.40 * frame["residual_signal_score"]
    dual_available = dual_channel & dual_score.notna()
    raw_available = raw_only & final_score.notna()
    residual_available = residual_only & frame["residual_signal_score"].notna()
    if dual_available.any():
        priority_score.loc[dual_available] = dual_score.loc[dual_available]
    if raw_available.any():
        priority_score.loc[raw_available] = final_score.loc[raw_available]
    if residual_available.any():
        priority_score.loc[residual_available] = frame.loc[residual_available, "residual_signal_score"]
    frame["candidate_priority_score"] = priority_score
    frame["candidate_priority_tier"] = frame["load_adjusted_relation_status"].map(
        {
            "dual_channel_supported": 0,
            "raw_only_supported": 1,
            "residual_only_supported": 1,
            "raw_only_residual_weak": 1,
            "raw_only_residual_missing": 1,
            "raw_only_common_load_risk": 2,
            "force_included_only": 3,
            "control_reference": 4,
        }
    ).astype("Int64")
    frame = frame.sort_values(
        [
            "candidate_priority_tier", "candidate_priority_score",
            "_priority_residual_n", "_candidate_pool_sort", "variable",
        ],
        ascending=[True, False, False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    frame["candidate_priority_rank"] = np.arange(1, len(frame) + 1)
    return frame.loc[:, [*base_columns, *CANDIDATE_PRIORITY_COLUMNS]]


def _candidate_bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.get(column, pd.Series(False, index=frame.index))

    def normalize(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes"}

    return values.map(normalize).astype(bool)


def order_initial_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Order initial candidates by score, evidence, lag quality, then variable."""
    if frame.empty:
        return frame.copy()
    ordered = frame.copy()
    ordered["_initial_final_score"] = pd.to_numeric(
        ordered.get("final_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_association_score"] = pd.to_numeric(
        ordered.get("association_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_lag_quality"] = pd.to_numeric(
        ordered.get("lag_quality", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    ).fillna(-np.inf)
    ordered["_initial_variable"] = ordered.get("variable", pd.Series("", index=ordered.index)).astype(str)
    return ordered.sort_values(
        ["_initial_final_score", "_initial_association_score", "_initial_lag_quality", "_initial_variable"],
        ascending=[False, False, False, True],
        kind="stable",
    ).drop(columns=["_initial_final_score", "_initial_association_score", "_initial_lag_quality", "_initial_variable"])


def _grade_candidate(row: pd.Series) -> str:
    score = _safe_float(row.get("final_score", 0), default=0.0)
    return "A" if score >= 0.75 else "B" if score >= 0.6 else "C" if score >= 0.45 else "D" if score >= 0.3 else "E"


def _recommend_use(row: pd.Series) -> str:
    flags = _risk_token_set(row.get("risk_flags", ""))
    grade = str(row.get("candidate_grade", "E"))
    if "severe_data_quality" in flags:
        return "poor_quality_variable"
    raw_corr = _safe_float(row.get("raw_corr", 0), default=0.0)
    lag = int(_safe_float(row.get("lag", 0), default=0.0))
    has_formula = "formula_like" in flags
    has_strong_formula = "strong_formula_leakage" in flags
    has_common = "common_capacity_driver" in flags
    if has_strong_formula or (has_formula and has_common) or (has_formula and lag == 0 and raw_corr >= 0.95):
        return "formula_coupled_reference"
    if str(row.get("temporal_direction_status", "")) == "target_leads_supported":
        return "state_indicator"
    if "common_capacity_driver" in flags:
        return "capacity_driven"
    if "unstable_across_regimes" in flags or "unstable_over_time" in flags:
        return "unstable_candidate"
    if grade == "A":
        return "strong_screening_candidate"
    return "manual_review_required"


def _recommended_action(row: pd.Series) -> str:
    flags = _risk_token_set(row.get("risk_flags", ""))
    if "severe_data_quality" in flags:
        return "数据质量严重不足，建议清洗数据或剔除该变量后重新分析"
    if str(row.get("temporal_direction_status", "")) == "target_leads_supported":
        return (
            "目标明显领先该变量，不适合作为上游原因候选；"
            "可能是下游响应、反馈动作或其他滞后结果，具体机制需工艺确认"
        )
    use_value = row.get("recommended_use", "manual_review_required")
    use = "manual_review_required" if pd.isna(use_value) else str(use_value)
    mapping = {
        "strong_screening_candidate": "优先进入机理复核",
        "prediction_candidate": "可作为预测候选",
        "capacity_driven": "疑似共同负荷驱动",
        "formula_coupled_reference": "疑似公式耦合，仅参考",
        "unstable_candidate": "跨工况/时间不稳定，建议复核",
        "poor_quality_variable": "数据质量严重不足，建议清洗数据或剔除该变量后重新分析",
        "state_indicator": "更可能是状态指示量",
        "control_variable_reference": "残差/负荷控制变量，仅作控制基准参考。",
    }
    return mapping.get(use, "建议人工工艺复核")


def _sampling_period_seconds(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return np.nan
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.median()) if len(diffs) else np.nan


def _longest_run(mask: pd.Series) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _longest_constant_run(series: pd.Series) -> int:
    values = series.dropna()
    if values.empty:
        return 0
    best = current = 1
    previous = values.iloc[0]
    for value in values.iloc[1:]:
        current = current + 1 if value == previous else 1
        previous = value
        best = max(best, current)
    return best


def _saturation_ratio(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return 0.0
    counts = values.value_counts(normalize=True)
    return float(counts.iloc[0]) if len(counts) else 0.0


def _robust_outlier_ratio(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 10:
        return 0.0
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad <= 1e-12:
        return 0.0
    robust_z = 0.6745 * (values - median).abs() / mad
    return float((robust_z > 6.0).mean())


def _residualize(
    y: pd.Series,
    x: pd.DataFrame,
    fit_mask: pd.Series | None = None,
) -> tuple[pd.Series, str, float, list[str]]:
    application_data = pd.concat([y, x], axis=1).dropna()
    fit_data = application_data
    if fit_mask is not None:
        resolved_mask = fit_mask.reindex(application_data.index).fillna(False).astype(bool)
        fit_data = application_data.loc[resolved_mask]
    fit_x = fit_data.iloc[:, 1:]
    usable_columns = [column for column in fit_x.columns if fit_x[column].nunique() > 1]
    if len(fit_data) < 5 or not usable_columns:
        return y - fit_data.iloc[:, 0].mean(), "demean", np.nan, []
    fit_features = fit_x[usable_columns]
    fit_matrix = np.column_stack([np.ones(len(fit_data)), fit_features.to_numpy(dtype=float)])
    cond = float(np.linalg.cond(fit_matrix))
    coefficients, _, _, _ = np.linalg.lstsq(
        fit_matrix, fit_data.iloc[:, 0].to_numpy(dtype=float), rcond=None
    )
    application_matrix = np.column_stack(
        [np.ones(len(application_data)), application_data[usable_columns].to_numpy(dtype=float)]
    )
    fitted = np.dot(application_matrix, coefficients)
    residual = pd.Series(
        index=application_data.index,
        data=application_data.iloc[:, 0].to_numpy() - fitted,
    )
    return residual.reindex(y.index), "ols", cond, usable_columns


def _nearby_lags(best_lag: int | None, max_lag: int, radius: int = 2) -> list[int]:
    if best_lag is None or pd.isna(best_lag):
        return list(range(0, min(max_lag, 6) + 1))
    center = int(best_lag)
    if center <= 0:
        return [0]
    lower = max(0, center - radius)
    upper = min(max_lag, center + radius)
    return list(range(lower, upper + 1))


def _time_series_splits(n_rows: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size = max(5, n_rows // (n_splits + 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for split in range(n_splits):
        train_end = n_rows - test_size * (n_splits - split)
        test_start = train_end
        test_end = test_start + test_size
        if train_end <= test_size or test_end > n_rows:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def _linear_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    model = fit_linear_model(x_train, y_train)
    return predict_linear_model(model, x_test)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _looks_like_formula_variable(name: str) -> bool:
    lower = name.lower()
    tokens = ["单耗", "消耗", "比值", "ratio", "rate", "%", "百分比", "折算", "累计", "平均", "total", "consumption", "specific"]
    return any(token in lower for token in tokens)
