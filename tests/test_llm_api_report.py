from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_llm_report_api_contract_functions_exist(tmp_path: Path):
    from chem_ts_corr.llm_api import LLMCallConfig, call_openai_compatible_chat, redact_secret

    cfg = LLMCallConfig(provider="deepseek", base_url="https://api.example.com", model="deepseek-chat", api_key="sk-test")
    assert cfg.provider == "deepseek"
    assert cfg.model == "deepseek-chat"
    assert redact_secret("sk-1234567890") != "sk-1234567890"
    assert callable(call_openai_compatible_chat)


def test_build_prompt_can_feed_llm_report_call(tmp_path: Path, monkeypatch):
    from chem_ts_corr.llm_api import LLMCallConfig, generate_llm_report

    run_dir = tmp_path / "run_api"
    run_dir.mkdir()
    (run_dir / "summary.md").write_text("- target: Y.PV\n", encoding="utf-8")
    _csv(run_dir / "ranked_features.csv", [{"variable": "FIC421002.PV", "final_score": 0.9, "candidate_grade": "A", "lag": 5, "direction": "variable_leads_target"}])

    def fake_call(config, prompt):
        assert config.api_key == "sk-test"
        assert "FIC421002.PV" in prompt
        return {"report": "# report\n", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "raw": {}}

    monkeypatch.setattr("chem_ts_corr.llm_api.call_openai_compatible_chat", fake_call)
    result = generate_llm_report(run_dir, LLMCallConfig(provider="deepseek", base_url="https://api.example.com", model="deepseek-chat", api_key="sk-test"), top_n=10)

    assert result["report"].startswith("# report")
    assert "prompt" in result
    assert result["usage"]["prompt_tokens"] == 10
    assert (run_dir / "llm_report.md").exists()


def test_web_exposes_llm_report_ui_and_download():
    from chem_ts_corr.web import DOWNLOAD_FILES, INDEX_HTML

    assert "llm_report.md" in DOWNLOAD_FILES
    assert "/api/llm_report" in INDEX_HTML
    assert "生成 DeepSeek 报告" in INDEX_HTML or "生成 LLM 报告" in INDEX_HTML
    assert "llmReportRendered" in INDEX_HTML
