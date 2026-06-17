from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.causal_review_evidence import EVIDENCE_COLUMNS, build_causal_review_evidence
from chem_ts_corr.causal_review_service import REPORT_COLUMNS, build_causal_review_report
from chem_ts_corr.final_review_summary import SUMMARY_COLUMNS, build_final_review_summary
from chem_ts_corr.conditional_granger import (
    OUT_COLS,
    build_candidate_lag_windows,
    run_conditional_granger_tests,
)


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
    conditional_lag_mode: str = "ranked_window",
    conditional_lag_window: int = 5,
    conditional_fallback_maxlag: int = 24,
    conditional_baseline_maxlag: int | None = 24,
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
            "final_review_summary": pd.DataFrame(columns=SUMMARY_COLUMNS),
        }

    optional_tables = _load_optional_evidence_tables(
        output_dir,
        enhanced_validation_summary=enhanced_validation_summary,
        granger_tests=granger_tests,
        model_variable_importance=model_variable_importance,
    )

    candidate_lags = _conditional_candidate_lags(
        ranked_features=ranked_features,
        variables=variables,
        maxlag=maxlag,
        mode=conditional_lag_mode,
        window=conditional_lag_window,
        fallback_maxlag=conditional_fallback_maxlag,
    )
    conditional_granger_scores = run_conditional_granger_tests(
        frame=frame,
        target=target,
        variables=variables,
        control_columns=control_columns,
        maxlag=maxlag,
        min_rows=min_rows,
        candidate_lags=candidate_lags,
        baseline_maxlag=conditional_baseline_maxlag,
        lag_mode=conditional_lag_mode,
        lag_window=conditional_lag_window,
        fallback_maxlag=conditional_fallback_maxlag,
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
    final_review_summary = build_final_review_summary(
        causal_review_evidence=causal_review_evidence,
        conditional_granger_scores=conditional_granger_scores,
        ranked_features=ranked_features,
    )
    return {
        "conditional_granger_scores": conditional_granger_scores,
        "causal_review_report": causal_review_report,
        "causal_review_evidence": causal_review_evidence,
        "final_review_summary": final_review_summary,
    }


def _conditional_candidate_lags(
    *,
    ranked_features: pd.DataFrame,
    variables: list[str],
    maxlag: int,
    mode: str,
    window: int,
    fallback_maxlag: int,
) -> dict[str, list[int]] | None:
    if mode == "full_scan":
        return None
    if mode == "ranked_window":
        effective_window = window
    elif mode == "best_only":
        effective_window = 0
    else:
        raise ValueError(f"unsupported conditional_lag_mode: {mode}")
    return build_candidate_lag_windows(
        ranked_features=ranked_features,
        variables=variables,
        maxlag=maxlag,
        window=effective_window,
        fallback_maxlag=fallback_maxlag,
    )


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
