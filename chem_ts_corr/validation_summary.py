from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


VALIDATION_SUMMARY_FILENAME = "validation_summary.csv"
VALIDATION_SUMMARY_COLUMNS = [
    "variable",
    "validation_status",
    "evidence_consistency",
    "supporting_methods",
    "limiting_factors",
]

# V1 deliberately keeps ``validation_summary.csv`` as the five-column
# conclusion contract.  These fields are the V3 semantic sidecar exposed by
# the API/UI; keeping them separate prevents a stage-specific value from being
# mistaken for a unified conclusion or entering the screening workflow.
VALIDATION_FIELDS_COLUMNS = [
    "variable",
    "initial_screening_lag",
    "validation_lag",
    "conditional_validation_lag",
    "screening_model_lift",
    "validation_model_lift",
]

_METHOD_LABELS = {
    "enhanced": "enhanced_screening",
    "granger": "granger",
    "model": "model_explanation",
}


@dataclass(frozen=True)
class _Observation:
    state: str


_SUPPORT = "support"
_ZERO_EVIDENCE = "zero_evidence"
_COMPUTED_NO_SUPPORT = "computed_no_support"
_NOT_COMPUTED = "not_computed"
_MISSING_EVIDENCE = "missing"
_SKIPPED = "skipped"
_FAILED = "failed"
_VARIABLE_MISSING = "variable_missing"


def build_validation_summary(
    ranked_features: pd.DataFrame | None = None,
    *,
    enhanced_validation_summary: pd.DataFrame | None = None,
    model_lift_scores: pd.DataFrame | None = None,
    rolling_corr_scores: pd.DataFrame | None = None,
    granger_tests: pd.DataFrame | None = None,
    model_variable_importance: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the read-only, per-variable secondary validation summary.

    ``None`` means that a method did not run and an empty DataFrame means that
    the method ran but did not produce a row.  This distinction is kept in the
    limiting-factor text; missing numeric evidence is never converted to zero
    or a positive/negative conclusion.

    The summary is deliberately independent of initial-screening scores and
    ranks.  It only combines the existing secondary outputs and returns the
    frozen five-column contract.
    """
    ranked = _normalize_frame(ranked_features)
    method_frames = {
        "enhanced": _normalize_frame(enhanced_validation_summary),
        "model_lift": _normalize_frame(model_lift_scores),
        "rolling": _normalize_frame(rolling_corr_scores),
        "granger": _normalize_frame(granger_tests),
        "model": _normalize_frame(model_variable_importance),
    }
    executed = {
        "enhanced": any(
            frame is not None
            for frame in (
                enhanced_validation_summary,
                model_lift_scores,
                rolling_corr_scores,
            )
        ),
        "granger": granger_tests is not None,
        "model": model_variable_importance is not None,
    }

    evidence_frames = (
        method_frames["enhanced"],
        method_frames["model_lift"],
        method_frames["rolling"],
        method_frames["granger"],
        method_frames["model"],
    )
    variables = _ordered_variables(ranked, *evidence_frames)
    if not variables:
        return _empty_summary()

    rows: list[dict[str, object]] = []
    for variable in variables:
        supporting: list[str] = []
        limiting: list[str] = []

        enhanced_observations = _enhanced_observations(
            variable,
            method_frames["enhanced"],
            method_frames["model_lift"],
            method_frames["rolling"],
        )
        _append_method_result(
            "enhanced",
            enhanced_observations,
            executed["enhanced"],
            supporting,
            limiting,
        )

        granger_observations = _granger_observations(variable, method_frames["granger"])
        _append_method_result(
            "granger",
            granger_observations,
            executed["granger"],
            supporting,
            limiting,
        )

        model_observations = _model_observations(variable, method_frames["model"])
        _append_method_result(
            "model",
            model_observations,
            executed["model"],
            supporting,
            limiting,
        )

        executed_count = sum(executed.values())
        support_count = len(supporting)
        limiting_states = {
            factor.rsplit(":", 1)[-1]
            for factor in limiting
        }
        if executed_count == 0:
            validation_status = "not_run"
            consistency = "not_run"
        elif support_count == 0:
            has_computed_zero_or_weak = bool(
                limiting_states
                & {_ZERO_EVIDENCE, _COMPUTED_NO_SUPPORT}
            )
            validation_status = "limited" if has_computed_zero_or_weak else "not_computed"
            consistency = "partial" if has_computed_zero_or_weak else "not_computed"
        elif limiting:
            validation_status = "limited"
            consistency = "partial"
        elif support_count >= 2:
            validation_status = "supported"
            consistency = "consistent"
        else:
            validation_status = "limited"
            consistency = "partial"

        rows.append(
            {
                "variable": variable,
                "validation_status": validation_status,
                "evidence_consistency": consistency,
                "supporting_methods": "; ".join(supporting),
                "limiting_factors": "; ".join(limiting),
            }
        )

    return pd.DataFrame(rows, columns=VALIDATION_SUMMARY_COLUMNS)


def build_validation_summary_from_output_dir(
    output_dir: str | Path,
    *,
    write: bool = False,
) -> pd.DataFrame:
    """Build a summary from the result files already present in ``output_dir``.

    The file-existence check is intentional: absent optional results become
    explicit ``not_run`` / ``not_computed`` limitations, never fabricated
    validation evidence.
    """
    output_path = Path(output_dir)
    ranked = _read_csv(output_path / "ranked_features.csv")
    enhanced = _read_optional_csv(output_path / "enhanced_validation_summary.csv")
    lift = _read_optional_csv(output_path / "model_lift_scores.csv")
    rolling = _read_optional_csv(output_path / "rolling_corr_scores.csv")
    granger = _read_optional_csv(output_path / "granger_tests.csv")
    model = _read_optional_csv(output_path / "model_variable_importance.csv")
    summary = build_validation_summary(
        ranked,
        enhanced_validation_summary=enhanced,
        model_lift_scores=lift,
        rolling_corr_scores=rolling,
        granger_tests=granger,
        model_variable_importance=model,
    )
    if write:
        summary.to_csv(
            output_path / VALIDATION_SUMMARY_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
    return summary


def build_validation_fields(
    ranked_features: pd.DataFrame | None = None,
    *,
    enhanced_validation_summary: pd.DataFrame | None = None,
    model_lift_scores: pd.DataFrame | None = None,
    rolling_corr_scores: pd.DataFrame | None = None,
    granger_tests: pd.DataFrame | None = None,
    conditional_granger_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expose stage-specific lag and model-lift values without combining them.

    ``initial_screening_lag`` and ``screening_model_lift`` are read only from
    the initial ``ranked_features`` table.  ``validation_lag`` is read from
    validation outputs (rolling validation has precedence over ordinary
    Granger when both are available), while ``validation_model_lift`` is read
    from the validation model-lift table or its enhanced-validation summary.
    Conditional Granger has its own ``conditional_validation_lag`` field.

    Values retain their sign and missingness.  In particular, a missing value
    is never replaced with ``0.0`` and lag direction is never normalized to an
    absolute value.  The result is read-only metadata and is not used for scoring,
    ranking, or candidate selection.
    """
    ranked = _normalize_frame(ranked_features)
    enhanced = _normalize_frame(enhanced_validation_summary)
    lift = _normalize_frame(model_lift_scores)
    rolling = _normalize_frame(rolling_corr_scores)
    granger = _normalize_frame(granger_tests)
    conditional = _normalize_frame(conditional_granger_scores)

    variables = _ordered_variables(ranked, enhanced, lift, rolling, granger, conditional)
    if not variables:
        return _empty_validation_fields()

    rows: list[dict[str, object]] = []
    for variable in variables:
        ranked_row = _first_row(ranked, variable)
        enhanced_row = _first_row(enhanced, variable)
        lift_row = _first_row(lift, variable)
        rolling_row = _first_row(rolling, variable)
        granger_row = _first_row(granger, variable)
        conditional_row = _first_row(conditional, variable)

        validation_row = rolling_row or enhanced_row or granger_row
        rows.append(
            {
                "variable": variable,
                "initial_screening_lag": _stage_value(
                    ranked_row,
                    "initial_screening_lag",
                    "lag",
                    signed=True,
                ),
                "validation_lag": _validation_lag(
                    validation_row,
                    rolling_row=rolling_row,
                    enhanced_row=enhanced_row,
                    granger_row=granger_row,
                ),
                "conditional_validation_lag": _stage_value(
                    conditional_row,
                    "conditional_validation_lag",
                    "best_lag",
                    signed=True,
                ),
                "screening_model_lift": _stage_value(
                    ranked_row,
                    "screening_model_lift",
                    "model_lift",
                ),
                "validation_model_lift": _validation_model_lift(
                    lift_row=lift_row,
                    enhanced_row=enhanced_row,
                ),
            }
        )

    return pd.DataFrame(rows, columns=VALIDATION_FIELDS_COLUMNS)


def build_validation_fields_from_output_dir(output_dir: str | Path) -> pd.DataFrame:
    """Derive V3 fields from existing result files without writing them."""
    output_path = Path(output_dir)
    return build_validation_fields(
        _read_csv(output_path / "ranked_features.csv"),
        enhanced_validation_summary=_read_optional_csv(
            output_path / "enhanced_validation_summary.csv"
        ),
        model_lift_scores=_read_optional_csv(output_path / "model_lift_scores.csv"),
        rolling_corr_scores=_read_optional_csv(
            output_path / "rolling_corr_scores.csv"
        ),
        granger_tests=_read_optional_csv(output_path / "granger_tests.csv"),
        conditional_granger_scores=_read_optional_csv(
            output_path / "conditional_granger_scores.csv"
        ),
    )


def _empty_validation_fields() -> pd.DataFrame:
    return pd.DataFrame(columns=VALIDATION_FIELDS_COLUMNS)


def _first_row(frame: pd.DataFrame, variable: str) -> dict[str, object]:
    rows = _rows_for_variable(frame, variable)
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _stage_value(
    row: dict[str, object],
    explicit_column: str,
    legacy_column: str,
    *,
    signed: bool = False,
) -> object:
    if not row:
        return np.nan
    value = row.get(explicit_column)
    if _missing(value):
        value = row.get(legacy_column)
    if _missing(value):
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(numeric):
        return np.nan
    # Keep integer lags convenient for CSV/API consumers, but never alter the
    # sign.  ``signed`` documents that negative values are meaningful; it does
    # not trigger any normalization.
    if signed and numeric.is_integer():
        return int(numeric)
    return numeric


def _validation_lag(
    validation_row: dict[str, object],
    *,
    rolling_row: dict[str, object],
    enhanced_row: dict[str, object],
    granger_row: dict[str, object],
) -> object:
    # Prefer the rolling validation lag because it is the lag used by the
    # enhanced validation stability result.  Explicitly named fields always
    # win within their own source.  Ordinary Granger is a fallback only when
    # no rolling/enhanced lag is available; it is never merged with them.
    for row, legacy in (
        (rolling_row, "best_lag"),
        (enhanced_row, "best_lag"),
    ):
        value = _stage_value(row, "validation_lag", legacy, signed=True)
        if not _missing(value):
            return value
    value = _stage_value(
        granger_row,
        "validation_lag",
        "best_granger_lag",
        signed=True,
    )
    if _missing(value):
        value = _stage_value(granger_row, "validation_lag", "best_lag", signed=True)
    if not _missing(value):
        return value
    return _stage_value(validation_row, "validation_lag", "best_lag", signed=True)


def _validation_model_lift(
    *,
    lift_row: dict[str, object],
    enhanced_row: dict[str, object],
) -> object:
    value = _stage_value(lift_row, "validation_model_lift", "model_lift")
    if not _missing(value):
        return value
    return _stage_value(
        enhanced_row,
        "validation_model_lift",
        "model_lift",
    )


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=VALIDATION_SUMMARY_COLUMNS)


def _normalize_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.copy(deep=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return _read_csv(path) if path.exists() else None


def _ordered_variables(*frames: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for frame in frames:
        if "variable" not in frame.columns:
            continue
        for value in frame["variable"].tolist():
            if _missing(value):
                continue
            variable = str(value).strip()
            if variable and variable not in values:
                values.append(variable)
    return values


def _rows_for_variable(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    if frame.empty or "variable" not in frame.columns:
        return frame.iloc[0:0]
    values = frame["variable"].astype("string")
    return frame.loc[values.eq(variable)]


def _enhanced_observations(
    variable: str,
    enhanced: pd.DataFrame,
    lift: pd.DataFrame,
    rolling: pd.DataFrame,
) -> list[_Observation]:
    rows = [
        *_rows_for_variable(enhanced, variable).to_dict("records"),
        *_rows_for_variable(lift, variable).to_dict("records"),
        *_rows_for_variable(rolling, variable).to_dict("records"),
    ]
    return [_observation_from_row(row, _ENHANCED_STRENGTH_COLUMNS) for row in rows]


def _granger_observations(variable: str, frame: pd.DataFrame) -> list[_Observation]:
    rows = _rows_for_variable(frame, variable)
    observations: list[_Observation] = []
    for row in rows.to_dict("records"):
        status = str(row.get("status", "")).strip().lower()
        if status == "ok":
            contribution = row.get("predictive_contribution")
            if _missing(contribution):
                observations.append(_Observation(_MISSING_EVIDENCE))
            elif not _finite(contribution):
                observations.append(_Observation(_NOT_COMPUTED))
            elif float(contribution) == 0.0:
                observations.append(_Observation(_ZERO_EVIDENCE))
            elif float(contribution) > 0.0:
                observations.append(_Observation(_SUPPORT))
            else:
                observations.append(_Observation(_COMPUTED_NO_SUPPORT))
        else:
            observations.append(_status_observation(status))
    return observations


def _model_observations(variable: str, frame: pd.DataFrame) -> list[_Observation]:
    rows = _rows_for_variable(frame, variable)
    return [
        _observation_from_row(row, _MODEL_STRENGTH_COLUMNS)
        for row in rows.to_dict("records")
    ]


_ENHANCED_STRENGTH_COLUMNS = (
    "model_lift",
    "median_fold_lift",
    "rolling_stability",
)
_MODEL_STRENGTH_COLUMNS = ("max_importance", "total_importance")


def _observation_from_row(
    row: dict[str, object], strength_columns: tuple[str, ...]
) -> _Observation:
    status = str(row.get("status", "")).strip().lower()
    if status.startswith("failed"):
        return _Observation(_FAILED)
    if status.startswith("skipped"):
        return _Observation(_SKIPPED)
    if status.startswith("not_computed") or status.startswith("unavailable"):
        return _Observation(_NOT_COMPUTED)
    if status and status != "ok":
        return _status_observation(status)

    values = [row.get(column) for column in strength_columns]
    finite_values = [float(value) for value in values if _finite(value)]
    if any(value > 0.0 for value in finite_values):
        return _Observation(_SUPPORT)
    if finite_values and all(value == 0.0 for value in finite_values):
        return _Observation(_ZERO_EVIDENCE)
    if status == "ok" or finite_values:
        return _Observation(_COMPUTED_NO_SUPPORT)
    return _Observation(_NOT_COMPUTED)


def _status_observation(status: str) -> _Observation:
    if status.startswith("failed"):
        return _Observation(_FAILED)
    if status.startswith("skipped"):
        return _Observation(_SKIPPED)
    if status.startswith("not_computed") or status.startswith("unavailable"):
        return _Observation(_NOT_COMPUTED)
    return _Observation(_NOT_COMPUTED)


def _append_method_result(
    method: str,
    observations: list[_Observation],
    executed: bool,
    supporting: list[str],
    limiting: list[str],
) -> None:
    label = _METHOD_LABELS[method]
    if not executed:
        limiting.append(f"{label}:not_run")
        return
    if not observations:
        limiting.append(f"{label}:{_VARIABLE_MISSING}")
        return
    if any(observation.state == _SUPPORT for observation in observations):
        supporting.append(label)
    for observation in observations:
        if observation.state != _SUPPORT:
            limiting.append(f"{label}:{observation.state}")


def _finite(value: object) -> bool:
    try:
        return bool(pd.notna(value) and np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def write_validation_summary(output_dir: str | Path) -> pd.DataFrame:
    """Persist the summary without touching any initial-screening artifact."""
    return build_validation_summary_from_output_dir(output_dir, write=True)
