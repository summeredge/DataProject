from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.causal_review_evidence import EVIDENCE_COLUMNS, build_causal_review_evidence
from chem_ts_corr.causal_review_service import REPORT_COLUMNS, build_causal_review_report
from chem_ts_corr.conditional_granger import OUT_COLS, run_conditional_granger_tests


def run_causal_review_stage(
    frame: pd.DataFrame,
    target: str,
    ranked_features: pd.DataFrame,
    causal_review_candidates: pd.DataFrame,
    risk_flags: pd.DataFrame | None = None,
    enhanced_validation_summary: pd.DataFrame | None = None,
    granger_tests: pd.DataFrame | None = None,
    model_variable_importance: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    control_columns: list[str] | None = None,
    maxlag: int = 12,
    min_rows: int = 60,
    top_n: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the standalone v0.4 causal-review stage.

    This runner intentionally stays independent from the v0.3 pipeline and
    Web/UI. It produces predictive-validation evidence for manual review only;
    it does not claim final causality.
    """
    selected_candidates = _select_candidates(causal_review_candidates, top_n=top_n)
    variables = _candidate_variables(selected_candidates)

    if not variables:
        return {
            "conditional_granger_scores": pd.DataFrame(columns=OUT_COLS),
            "causal_review_report": pd.DataFrame(columns=REPORT_COLUMNS),
            "causal_review_evidence": pd.DataFrame(columns=EVIDENCE_COLUMNS),
        }

    optional_tables = _load_optional_evidence_tables(
        output_dir,
        enhanced_validation_summary=enhanced_validation_summary,
        granger_tests=granger_tests,
        model_variable_importance=model_variable_importance,
    )

    conditional_granger_scores = run_conditional_granger_tests(
        frame=frame,
        target=target,
        variables=variables,
        control_columns=control_columns,
        maxlag=maxlag,
        min_rows=min_rows,
    )
    causal_review_report = build_causal_review_report(
        ranked_features=ranked_features,
        causal_review_candidates=selected_candidates,
        conditional_granger_scores=conditional_granger_scores,
        risk_flags=risk_flags,
    )
    causal_review_evidence = build_causal_review_evidence(
        ranked_features=ranked_features,
        conditional_granger_scores=conditional_granger_scores,
        risk_flags=risk_flags,
        enhanced_validation_summary=optional_tables["enhanced_validation_summary"],
        granger_tests=optional_tables["granger_tests"],
        model_variable_importance=optional_tables["model_variable_importance"],
    )
    return {
        "conditional_granger_scores": conditional_granger_scores,
        "causal_review_report": causal_review_report,
        "causal_review_evidence": causal_review_evidence,
    }


def _select_candidates(causal_review_candidates: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    selected = causal_review_candidates.copy(deep=True)
    if top_n is not None:
        selected = selected.head(top_n).copy(deep=True)
    return selected


def _candidate_variables(causal_review_candidates: pd.DataFrame) -> list[str]:
    if causal_review_candidates.empty or "variable" not in causal_review_candidates.columns:
        return []
    values = causal_review_candidates["variable"].dropna()
    return [str(value) for value in values]


def _load_optional_evidence_tables(
    output_dir: str | Path | None,
    *,
    enhanced_validation_summary: pd.DataFrame | None,
    granger_tests: pd.DataFrame | None,
    model_variable_importance: pd.DataFrame | None,
) -> dict[str, pd.DataFrame | None]:
    tables = {
        "enhanced_validation_summary": enhanced_validation_summary,
        "granger_tests": granger_tests,
        "model_variable_importance": model_variable_importance,
    }
    if output_dir is None:
        return tables

    directory = Path(output_dir)
    files = {
        "enhanced_validation_summary": "enhanced_validation_summary.csv",
        "granger_tests": "granger_tests.csv",
        "model_variable_importance": "model_variable_importance.csv",
    }
    for key, file_name in files.items():
        if tables[key] is None:
            tables[key] = _safe_read_csv(directory / file_name)
    return tables


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
