from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_prompt_has_apc_term_guardrails(tmp_path: Path):
    run_dir = tmp_path / "run_terms"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _csv(run_dir / "ranked_features.csv", [{"variable": "FIC421002.PV", "final_score": 0.9, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target"}])

    prompt = build_llm_prompt(build_llm_analysis_package(run_dir, top_n=10), report_type="apc_advice")

    assert "DV / FF" in prompt
    assert "CV" in prompt
    assert "loop_mv_candidate" in prompt


def test_aic_pv_is_not_direct_mv(tmp_path: Path):
    run_dir = tmp_path / "run_aic"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _csv(run_dir / "ranked_features.csv", [{"variable": "AIC450005.PV", "final_score": 0.9, "candidate_grade": "A", "lag": 2, "direction": "variable_leads_target"}])

    row = build_llm_analysis_package(run_dir, top_n=10)["control_candidate_variables"][0]

    assert row["variable"] == "AIC450005.PV"
    assert row["suggested_control_role"] != "mv_candidate"


def test_prompt_mentions_skipped_model_evidence_guardrail(tmp_path: Path):
    run_dir = tmp_path / "run_skip"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n- model_status: skipped\n- skip_model_lift: True\n- skip_rolling_corr: True\n", encoding="utf-8")
    _csv(run_dir / "ranked_features.csv", [{"variable": "FIC421002.PV", "final_score": 0.9, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target"}])

    package = build_llm_analysis_package(run_dir, top_n=10)
    prompt = build_llm_prompt(package, report_type="apc_advice")

    assert package["meta"].get("skip_model_lift") is True
    assert package["meta"].get("skip_rolling_corr") is True
    assert "skip_model_lift" in prompt
    assert "skip_rolling_corr" in prompt
