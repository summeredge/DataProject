from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib import error, request

from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


@dataclass
class LLMCallConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 15000
    timeout: float = 120.0


def redact_secret(value: str | None) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def _chat_completions_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ValueError("base_url is required")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_openai_compatible_chat(config: LLMCallConfig, prompt: str) -> dict[str, Any]:
    if not config.api_key:
        raise ValueError("API Key is required")
    if not config.model:
        raise ValueError("model is required")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你是工业过程控制和APC/DCS分析报告助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(config.temperature),
        "max_tokens": int(config.max_tokens),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        _chat_completions_url(config.base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(config.timeout)) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail.replace(config.api_key, redact_secret(config.api_key))}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc).replace(config.api_key, redact_secret(config.api_key))) from exc

    try:
        report = raw["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError("LLM API response missing choices[0].message.content") from exc
    return {"report": report, "usage": raw.get("usage", {}), "raw": raw}


def generate_llm_report(
    run_dir: str | Path,
    config: LLMCallConfig,
    top_n: int = 20,
    report_type: str = "apc_advice",
    anonymize: bool = False,
) -> dict[str, Any]:
    path = Path(run_dir)
    package = build_llm_analysis_package(path, top_n=top_n)
    prompt = build_llm_prompt(package, report_type=report_type)
    result = call_openai_compatible_chat(config, prompt)
    report = str(result.get("report", ""))
    (path / "llm_report.md").write_text(report, encoding="utf-8")
    return {
        "report": report,
        "prompt": prompt,
        "package": package,
        "usage": result.get("usage", {}),
    }
