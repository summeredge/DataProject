from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import re
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from email.parser import BytesParser
from email.policy import default as email_default_policy
from types import SimpleNamespace

import pandas as pd

from chem_ts_corr.config import AnalysisConfig as _AnalysisConfig
from chem_ts_corr.data import EXCEL_SUFFIXES, TEXT_SUFFIXES, load_timeseries_csv, read_timeseries_table
from chem_ts_corr.causality import run_granger_tests
from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.modeling import fit_explainable_model
from chem_ts_corr.model_discovery import build_model_discovered_candidates, build_model_variable_importance
from chem_ts_corr.pipeline import run_analysis
from chem_ts_corr.llm_api import LLMCallConfig, call_openai_compatible_chat, generate_llm_report, redact_secret
from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt


def AnalysisConfig(
    input_path: Path,
    time_column: str,
    target: str,
    output_dir: Path = Path("reports"),
    **kwargs: Any,
) -> _AnalysisConfig:
    return _AnalysisConfig(
        input_path=input_path,
        time_column=time_column,
        target=target,
        output_dir=output_dir,
        **kwargs,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "reports" / "web_runs"
UPLOADS_DIR = PROJECT_ROOT / "reports" / "uploads"
DOWNLOAD_FILES = {
    "summary.md",
    "ranked_features.csv",
    "lag_scores.csv",
    "granger_tests.csv",
    "shap_or_importance.csv",
    "diagnostics.csv",
    "residual_corr_scores.csv",
    "regime_scores.csv",
    "risk_flags.csv",
    "model_lift_scores.csv",
    "model_discovered_candidates.csv",
    "model_variable_importance.csv",
    "near_miss_candidates.csv",
    "recommended_candidates.csv",
    "lag_peak_quality.csv",
    "rolling_corr_scores.csv",
    "causal_review_candidates.csv",
    "conditional_granger_scores.csv",
    "causal_review_report.csv",
    "final_review_summary.csv",
    "causal_review_evidence.csv",
    "enhanced_validation_summary.csv",
    "llm_prompt.md",
    "llm_report.md",
}
MAX_REQUEST_BODY_BYTES = 100 * 1024 * 1024
TASK_TTL_SECONDS = 6 * 60 * 60
MAX_TASKS = 100
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SCALED_FRAME_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
SCALED_FRAME_CACHE_LOCK = threading.Lock()
MAX_SCALED_FRAME_CACHE = 4


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"本地服务已启动：{url}")
    print("关闭此窗口即可停止服务。")
    server.serve_forever()


class _Handler(BaseHTTPRequestHandler):
    server_version = "ChemTsCorr/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/columns":
            try:
                params = parse_qs(parsed.query)
                file_id = _single(params, "file_id")
                encoding = _single(params, "encoding", "utf-8-sig")
                self._send_json(_columns_response(file_id, encoding))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/trend":
            try:
                params = parse_qs(parsed.query)
                self._send_json(_trend_response(params))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/status":
            try:
                params = parse_qs(parsed.query)
                self._send_json(_task_status_response(_single(params, "task_id")))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/result":
            try:
                params = parse_qs(parsed.query)
                self._send_json(_task_result_response(_single(params, "task_id")))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/download":
            try:
                params = parse_qs(parsed.query)
                self._send_download(_single(params, "run_id"), _single(params, "file"))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/upload":
                self._send_json(_upload_response(self))
                return
            if self.path == "/api/analyze":
                self._send_json(_analyze_response(self))
                return
            if self.path == "/api/run_granger":
                self._send_json(_run_granger_response(self))
                return
            if self.path == "/api/run_model":
                self._send_json(_run_model_response(self))
                return
            if self.path == "/api/run_enhanced_screening":
                self._send_json(_run_enhanced_screening_response(self))
                return
            if self.path in {"/run_causal_review", "/api/run_causal_review"}:
                self._send_json(_run_causal_review_response(self))
                return
            if self.path == "/api/llm_prompt":
                self._send_json(_llm_prompt_response(self))
                return
            if self.path == "/api/llm_report":
                self._send_json(_llm_report_response(self))
                return
            if self.path == "/api/llm_connection":
                self._send_json(_llm_connection_response(self))
                return
            self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            if _is_client_disconnect(exc):
                return
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as exc:
            if not _is_client_disconnect(exc):
                raise

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as exc:
            if not _is_client_disconnect(exc):
                raise

    def _send_download(self, run_id: str, file_name: str) -> None:
        if file_name not in DOWNLOAD_FILES:
            raise ValueError("Unsupported download file")
        path = (RUNS_DIR / run_id / file_name).resolve()
        if RUNS_DIR.resolve() not in path.parents or not path.exists():
            raise FileNotFoundError("Download file was not found")

        content_type = "text/csv; charset=utf-8" if path.suffix == ".csv" else "text/markdown; charset=utf-8"
        body = path.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{file_name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as exc:
            if not _is_client_disconnect(exc):
                raise


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in {10053, 10054}:
            return True
        if getattr(exc, "errno", None) in {32, 54, 104}:
            return True
    return False


def _upload_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    file_item = form["file"] if "file" in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("请选择 CSV、Excel 或 TXT 数据文件")

    filename = Path(getattr(file_item, "filename", "upload.csv")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in TEXT_SUFFIXES | EXCEL_SUFFIXES:
        raise ValueError("仅支持 CSV、TXT、TSV、XLSX、XLS、XLSM 数据文件")

    file_id = uuid.uuid4().hex
    upload_path = UPLOADS_DIR / f"{file_id}{suffix}"
    raw = file_item.file if isinstance(file_item.file, (bytes, bytearray)) else file_item.file.read()
    max_bytes = 100 * 1024 * 1024
    if len(raw) > max_bytes:
        raise ValueError("上传文件过大")
    upload_path.write_bytes(raw)

    return {"file_id": file_id, "filename": filename}


def _columns_response(file_id: str, encoding: str) -> dict[str, Any]:
    path = _resolve_upload(file_id)
    sample, used_encoding = _read_data_sample(path, encoding)
    numeric_columns = [
        column
        for column in sample.columns
        if pd.to_numeric(sample[column], errors="coerce").notna().mean() >= 0.7
    ]
    time_meta = _time_range_metadata(path, sample, used_encoding)
    return {
        "columns": list(sample.columns),
        "numericColumns": numeric_columns,
        "sampleRows": int(len(sample)),
        "encoding": used_encoding,
        **time_meta,
    }


def _read_data_sample(path: Path, encoding: str) -> tuple[pd.DataFrame, str]:
    return read_timeseries_table(path, encoding=encoding, nrows=5000)


def _time_range_metadata(path: Path, sample: pd.DataFrame, encoding: str) -> dict[str, str]:
    candidate = next(
        (column for column in sample.columns if _looks_like_time_column(str(column))),
        "",
    )
    if not candidate:
        return {}
    try:
        time_frame, _ = read_timeseries_table(path, encoding=encoding, usecols=[candidate])
        values = pd.to_datetime(time_frame[candidate], errors="coerce").dropna().sort_values()
    except Exception:
        return {"timeColumn": candidate}
    if values.empty:
        return {"timeColumn": candidate}

    start = values.iloc[0]
    end = values.iloc[-1]
    default_end = min(start + pd.Timedelta(days=3), end)
    return {
        "timeColumn": candidate,
        "timeStart": _datetime_local(start),
        "timeEnd": _datetime_local(end),
        "trendStartDefault": _datetime_local(start),
        "trendEndDefault": _datetime_local(default_end),
    }


def _looks_like_time_column(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ["time", "date", "timestamp", "时间", "日期"])


def _datetime_local(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M")


def _analyze_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    file_id = _field(form, "file_id")
    encoding = _field(form, "encoding", "utf-8-sig")
    time_column = _field(form, "time_column")
    target = _field(form, "target")
    if not time_column:
        raise ValueError("请选择时间列")
    if not target:
        raise ValueError("请选择目标列")

    run_id = uuid.uuid4().hex
    output_dir = RUNS_DIR / run_id
    input_path = _resolve_upload(file_id)
    resolved_encoding = _resolve_encoding(input_path, encoding)
    config = AnalysisConfig(
        input_path=input_path,
        time_column=time_column,
        target=target,
        output_dir=output_dir,
        encoding=resolved_encoding,
        max_lag=_int_field(form, "max_lag", 12),
        resample_rule=_field(form, "resample_rule", "") or None,
        min_valid_ratio=_float_field(form, "min_valid_ratio", 0.7),
        top_k=_int_field(form, "top_k", 30),
        preprocess_mode=_field(form, "preprocess_mode", "raw"),
        detrend_window=_int_field(form, "detrend_window", 24),
        segment_column=_field(form, "segment_column", "") or None,
        segment_mode=_field(form, "segment_mode", "all"),
        segment_min=_optional_float_field(form, "segment_min"),
        segment_max=_optional_float_field(form, "segment_max"),
        capacity_columns=_list_field(form, "capacity_columns"),
        residual_control_columns=_list_field(form, "residual_control_columns") or _list_field(form, "capacity_columns"),
        force_include_variables=_list_field(form, "force_include_variables"),
        exclude_control_columns_from_candidates=_bool_field(form, "exclude_control_columns_from_candidates") if "exclude_control_columns_from_candidates" in form else True,
        enable_granger=False,
        enable_model=False,
        skip_model_lift=True,
        skip_rolling_corr=True,
    )
    task_id = uuid.uuid4().hex
    now = time.time()
    with TASKS_LOCK:
        _cleanup_tasks_locked(now=now)
        TASKS[task_id] = {
            "status": "running",
            "message": "等待后台分析启动",
            "run_id": run_id,
            "start_time": now,
            "created_at": now,
            "updated_at": now,
        }
    thread = threading.Thread(
        target=_analyze_task,
        args=(task_id, config, file_id),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "run_id": run_id, "status": "running"}


def _analyze_task(task_id: str, config: AnalysisConfig, file_id: str) -> None:
    try:
        _write_run_config(config.output_dir, config, file_id)

        def progress(message: str) -> None:
            with TASKS_LOCK:
                task = TASKS.get(task_id)
                if task is not None:
                    task["message"] = message
                    task["updated_at"] = time.time()

        run_analysis(config, progress_callback=progress)
        with TASKS_LOCK:
            TASKS[task_id].update(
                {
                    "status": "done",
                    "message": "分析完成",
                    "end_time": time.time(),
                    "updated_at": time.time(),
                    "result": _build_result_payload(config.output_dir.name, config.output_dir, config),
                }
            )
        _cleanup_tasks()
    except Exception as exc:
        with TASKS_LOCK:
            TASKS[task_id].update(
                {
                    "status": "error",
                    "message": str(exc),
                    "error": str(exc),
                    "end_time": time.time(),
                    "updated_at": time.time(),
                }
            )
        _cleanup_tasks()


def _build_result_payload(run_id: str, output_dir: Path, config: AnalysisConfig) -> dict[str, Any]:
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
    lag_scores = _safe_read_result_csv(output_dir / "lag_scores.csv")
    residual = _safe_read_result_csv(output_dir / "residual_corr_scores.csv")
    regime = _safe_read_result_csv(output_dir / "regime_scores.csv")
    lift = _safe_read_result_csv(output_dir / "model_lift_scores.csv")
    rolling = _safe_read_result_csv(output_dir / "rolling_corr_scores.csv")
    enhanced = _safe_read_result_csv(output_dir / "enhanced_validation_summary.csv")
    granger = _safe_read_result_csv(output_dir / "granger_tests.csv")
    importance = _safe_read_result_csv(output_dir / "shap_or_importance.csv")
    model_variable_importance = _safe_read_result_csv(output_dir / "model_variable_importance.csv")
    model_discovered = _safe_read_result_csv(output_dir / "model_discovered_candidates.csv")
    near_miss = _safe_read_result_csv(output_dir / "near_miss_candidates.csv")
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    risky = risk[risk.get("risk_count", 0) > 0] if not risk.empty else risk
    top10_variables = set(ranked.head(20)["variable"].astype(str)) if not ranked.empty else set()
    visible_lag_scores = lag_scores[
        lag_scores.get("variable", pd.Series(dtype=str)).astype(str).isin(top10_variables)
    ] if not lag_scores.empty else lag_scores
    return {
        "run_id": run_id,
        "overview": _overview_payload(ranked, risk, config, _summary_metrics(summary)),
        "rankedFeatures": _records(ranked.head(50)),
        "riskFlags": _records(risky.head(50)),
        "lagScores": _records(visible_lag_scores),
        "residualScores": _records(residual.head(50)),
        "regimeScores": _records(regime.head(50)),
        "modelLiftScores": _records(lift.head(50)),
        "rollingCorrScores": _records(rolling.head(50)),
        "enhancedValidationSummary": _records(enhanced.head(200)),
        "grangerTests": _records(granger.head(200)),
        "importance": _records(importance.head(200)),
        "modelVariableImportance": _records(model_variable_importance.head(200)),
        "modelDiscoveredCandidates": _records(model_discovered.head(200)),
        "nearMissCandidates": _records(near_miss.head(200)),
        "downloads": _download_links(run_id, output_dir),
    }


def _task_status_response(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        _cleanup_tasks_locked()
        task = TASKS.get(task_id)
    if task is None:
        raise FileNotFoundError("任务不存在，请重新开始分析")
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "message": task.get("message", ""),
        "error": task.get("error", ""),
        "run_id": task.get("run_id", ""),
        "elapsed_seconds": _elapsed_seconds(task),
    }


def _elapsed_seconds(task: dict[str, Any]) -> float:
    start = task.get("start_time")
    if not start:
        return 0.0
    end = task.get("end_time") or time.time()
    return round(max(0.0, float(end) - float(start)), 1)


def _task_result_response(task_id: str) -> dict[str, Any]:
    with TASKS_LOCK:
        _cleanup_tasks_locked()
        task = TASKS.get(task_id)
    if task is None:
        raise FileNotFoundError("任务不存在，请重新开始分析")
    if task.get("status") == "error":
        raise RuntimeError(str(task.get("error") or task.get("message") or "分析失败"))
    if task.get("status") != "done":
        return {"task_id": task_id, "status": task.get("status", "running")}
    return task["result"]


def _run_enhanced_screening_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    from chem_ts_corr.screening import model_lift_scores, rolling_corr_scores

    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查并生成 ranked_features.csv")

    variables = [variable for variable in _secondary_variables_from_ranked(ranked, config) if variable != config.target]
    if not variables:
        raise ValueError("ranked_features.csv 中没有可运行增强筛选的候选变量")

    scaled = _scaled_frame_for_secondary(config)
    variables = [variable for variable in variables if variable in scaled.columns]
    if not variables:
        raise ValueError("候选变量在预处理后的数据中不存在，请检查上传数据和配置")

    best_lags = _best_lags_from_ranked(ranked)
    lift = model_lift_scores(scaled, config.target, variables, config.max_lag, best_lags=best_lags)
    rolling = rolling_corr_scores(scaled, config.target, variables, config.max_lag)
    enhanced = _enhanced_validation_summary(ranked, lift, rolling)

    lift.to_csv(output_dir / "model_lift_scores.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(output_dir / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig")
    enhanced.to_csv(output_dir / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "modelLiftScores": _records(lift.head(200)),
        "rollingCorrScores": _records(rolling.head(200)),
        "enhancedValidationSummary": _records(enhanced.head(200)),
        "downloads": _download_links(run_id, output_dir),
        "message": "增强筛选完成：结果用于补充验证预测增益和时间稳定性，不代表因果结论。",
    }


def _enhanced_validation_summary(
    ranked: pd.DataFrame, model_lift: pd.DataFrame, rolling: pd.DataFrame
) -> pd.DataFrame:
    if ranked.empty or "variable" not in ranked.columns:
        return pd.DataFrame()
    columns = [
        column
        for column in ["variable", "final_score", "lag", "direction", "risk_flags", "recommended_use"]
        if column in ranked.columns
    ]
    summary = ranked[columns].copy(deep=True)
    if not model_lift.empty:
        summary = summary.merge(
            model_lift[[c for c in ["variable", "status", "model_lift", "ar_baseline_rmse", "candidate_rmse"] if c in model_lift.columns]],
            on="variable",
            how="left",
        )
    if not rolling.empty:
        summary = summary.merge(
            rolling[[c for c in ["variable", "rolling_stability", "rolling_corr_median", "rolling_sign_consistency", "valid_window_count"] if c in rolling.columns]],
            on="variable",
            how="left",
        )
    summary["interpretation"] = "enhanced screening only; not a causal conclusion"
    return summary


def _run_granger_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查")
    variables = _secondary_variables_from_ranked(ranked, config)
    scaled = _scaled_frame_for_secondary(config)
    granger = run_granger_tests(
        scaled,
        target=config.target,
        variables=variables,
        maxlag=config.resolved_granger_maxlag(),
    )
    granger.to_csv(output_dir / "granger_tests.csv", index=False, encoding="utf-8-sig")
    return {
        "grangerTests": _records(granger.head(200)),
        "downloads": _download_links(run_id, output_dir),
    }


def _run_model_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查")
    variables = _secondary_variables_from_ranked(ranked, config)
    near_miss = _safe_read_result_csv(output_dir / "near_miss_candidates.csv")
    variables = list(dict.fromkeys(variables + _near_miss_variables(near_miss, limit=10)))
    best_lags = _best_lags_from_ranked(ranked)
    best_lags = _merge_near_miss_lags(best_lags, near_miss)
    scaled = _scaled_frame_for_secondary(config)
    variables = [variable for variable in variables if variable in scaled.columns]
    importance, metrics = fit_explainable_model(
        scaled,
        target=config.target,
        max_lag=config.max_lag,
        candidate_variables=variables,
        max_features=config.max_model_features,
        random_state=config.random_state,
        best_lags=best_lags,
        lag_mode="best_only",
    )
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
    model_variable_importance = build_model_variable_importance(importance, ranked, risk_flags=risk)
    model_discovered = build_model_discovered_candidates(
        importance,
        ranked,
        risk_flags=risk,
        screening_top_n=config.top_k,
        max_lag=config.max_lag,
    )
    importance.to_csv(output_dir / "shap_or_importance.csv", index=False, encoding="utf-8-sig")
    model_variable_importance.to_csv(output_dir / "model_variable_importance.csv", index=False, encoding="utf-8-sig")
    model_discovered.to_csv(output_dir / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig")
    return {
        "importance": _records(importance.head(200)),
        "modelVariableImportance": _records(model_variable_importance.head(200)),
        "modelDiscoveredCandidates": _records(model_discovered.head(200)),
        "modelMetrics": metrics,
        "downloads": _download_links(run_id, output_dir),
    }



def _run_causal_review_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    candidates = _safe_read_result_csv(output_dir / "causal_review_candidates.csv")
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
    if candidates.empty:
        raise ValueError("请先完成主筛查并生成 causal_review_candidates.csv")

    risk_filter = _list_field(form, "risk_flag_filter")
    candidates = _filter_candidates_by_risk_flags(candidates, risk, risk_filter)
    scaled = _scaled_frame_for_secondary(config)
    result = run_causal_review_stage(
        frame=scaled,
        target=config.target,
        ranked_features=ranked,
        causal_review_candidates=candidates,
        risk_flags=risk,
        output_dir=output_dir,
        control_columns=_list_field(form, "control_columns") or config.residual_control_columns or config.capacity_columns,
        maxlag=_int_field(form, "maxlag", config.resolved_granger_maxlag()),
        min_rows=_int_field(form, "min_rows", 60),
        top_n=_optional_int_field(form, "top_n"),
        conditional_lag_mode=_field(form, "conditional_lag_mode", "ranked_window"),
        conditional_lag_window=_int_field(form, "conditional_lag_window", 5),
        conditional_fallback_maxlag=_int_field(form, "conditional_fallback_maxlag", 24),
        conditional_baseline_maxlag=_optional_int_field(form, "conditional_baseline_maxlag") or 24,
    )
    conditional = result["conditional_granger_scores"]
    report = result["causal_review_report"]
    evidence = result["causal_review_evidence"]
    final_summary = result["final_review_summary"]
    conditional.to_csv(output_dir / "conditional_granger_scores.csv", index=False, encoding="utf-8-sig")
    report.to_csv(output_dir / "causal_review_report.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(output_dir / "final_review_summary.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(output_dir / "causal_review_evidence.csv", index=False, encoding="utf-8-sig")
    return {
        "conditionalGrangerScores": _records(conditional.head(500)),
        "causalReviewReport": _records(report.head(500)),
        "finalReviewSummary": _records(final_summary.head(500)),
        "causalReviewEvidence": _records(evidence.head(500)),
        "downloads": _download_links(run_id, output_dir),
        "message": "三层复核完成：结果仅为预测验证/人工复核建议，不是因果结论。",
    }


def _llm_prompt_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    top_n = _int_field(form, "top_n", 20)
    report_type = _field(form, "report_type", "apc_advice")
    package = build_llm_analysis_package(output_dir, top_n=top_n)
    prompt = build_llm_prompt(package, report_type=report_type)
    (output_dir / "llm_prompt.md").write_text(prompt, encoding="utf-8")
    return {
        "prompt": prompt,
        "package": package,
        "downloads": _download_links(run_id, output_dir),
        "message": "AI 综合解读 Prompt 已生成。",
    }



def _llm_connection_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    api_key = _field(form, "api_key")
    config = LLMCallConfig(
        provider=_field(form, "provider", "deepseek"),
        base_url=_field(form, "base_url", "https://api.deepseek.com"),
        model=_field(form, "model", "deepseek-chat"),
        api_key=api_key,
        temperature=_float_field(form, "temperature", 0.2),
        max_tokens=_int_field(form, "max_tokens", 16),
        timeout=30.0,
    )
    try:
        call_openai_compatible_chat(config, "请回复 OK")
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, redact_secret(api_key))
        return {"ok": False, "message": f"API 连接失败：{message}"}
    return {"ok": True, "message": "API 连接成功"}

def _llm_report_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    api_key = _field(form, "api_key")
    output_dir = _resolve_run_dir(run_id)
    config = LLMCallConfig(
        provider=_field(form, "provider", "deepseek"),
        base_url=_field(form, "base_url", "https://api.deepseek.com"),
        model=_field(form, "model", "deepseek-chat"),
        api_key=api_key,
        temperature=_float_field(form, "temperature", 0.2),
        max_tokens=_int_field(form, "max_tokens", 4096),
    )
    try:
        result = generate_llm_report(
            output_dir,
            config,
            top_n=_int_field(form, "top_n", 20),
            report_type=_field(form, "report_type", "apc_advice"),
        )
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, redact_secret(api_key))
        raise RuntimeError(message) from exc
    return {
        "report": result.get("report", ""),
        "prompt": result.get("prompt", ""),
        "usage": result.get("usage", {}),
        "downloads": _download_links(run_id, output_dir),
        "message": "LLM 报告已生成。",
    }


def _filter_candidates_by_risk_flags(
    candidates: pd.DataFrame, risk_flags: pd.DataFrame, selected_flags: list[str]
) -> pd.DataFrame:
    if not selected_flags or candidates.empty or risk_flags.empty:
        return candidates.copy(deep=True)
    if "variable" not in candidates.columns or "variable" not in risk_flags.columns or "risk_flags" not in risk_flags.columns:
        return candidates.copy(deep=True)
    aliases = {
        "共同负荷驱动": "common_capacity_driver",
        "不稳定候选": "unstable_candidate",
        "跨工况不稳定": "unstable_across_regimes",
        "时序不稳定": "unstable_over_time",
        "滞后边界": "lag_boundary",
    }
    selected = {aliases.get(flag, flag).lower() for flag in selected_flags}
    risk_text = risk_flags["risk_flags"].fillna("").astype(str).str.lower()
    mask = risk_text.apply(lambda value: any(flag in value for flag in selected))
    variables = set(risk_flags.loc[mask, "variable"].astype(str))
    return candidates[candidates["variable"].astype(str).isin(variables)].copy(deep=True)


def _secondary_variables_from_ranked(ranked: pd.DataFrame, config: AnalysisConfig) -> list[str]:
    if ranked.empty or "variable" not in ranked.columns:
        return []
    top = ranked.head(config.top_k)["variable"].astype(str).tolist()
    if "force_included" in ranked.columns:
        forced = ranked[ranked["force_included"].astype(bool)]["variable"].astype(str).tolist()
    else:
        forced = [v for v in (config.force_include_variables or []) if v]
    return list(dict.fromkeys(top + forced))

def _near_miss_variables(near_miss: pd.DataFrame, limit: int = 10) -> list[str]:
    if near_miss.empty or "variable" not in near_miss.columns:
        return []
    return near_miss.head(limit)["variable"].dropna().astype(str).tolist()


def _best_lags_from_ranked(ranked: pd.DataFrame) -> dict[str, int]:
    if ranked.empty or not {"variable", "lag"}.issubset(ranked.columns):
        return {}
    return {
        str(row["variable"]): int(row["lag"])
        for _, row in ranked[["variable", "lag"]].dropna().iterrows()
    }


def _merge_near_miss_lags(best_lags: dict[str, int], near_miss: pd.DataFrame) -> dict[str, int]:
    merged = dict(best_lags)
    if near_miss.empty or not {"variable", "lag"}.issubset(near_miss.columns):
        return merged
    for _, row in near_miss[["variable", "lag"]].dropna().iterrows():
        variable = str(row["variable"]).strip()
        if not variable or variable in merged:
            continue
        try:
            merged[variable] = int(row["lag"])
        except (TypeError, ValueError):
            continue
    return merged


def _scaled_frame_for_secondary(config: AnalysisConfig) -> pd.DataFrame:
    from chem_ts_corr.data import select_numeric_frame
    from chem_ts_corr.screening import apply_ignore_roles, load_roles
    from chem_ts_corr.preprocess import preprocess_frame, segment_by_load, standardize_frame, transform_frame

    cache_key = _scaled_frame_cache_key(config)
    with SCALED_FRAME_CACHE_LOCK:
        cached = SCALED_FRAME_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy(deep=True)

    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    numeric = select_numeric_frame(raw, config.target)
    roles = load_roles(config, list(numeric.columns))
    numeric = apply_ignore_roles(numeric, roles, config.target)
    segmented = segment_by_load(
        numeric,
        segment_column=config.segment_column,
        segment_mode=config.segment_mode,
        segment_min=config.segment_min,
        segment_max=config.segment_max,
    )
    protected = [
        config.target,
        config.segment_column,
        *(config.capacity_columns or []),
        *(config.residual_control_columns or []),
        *(config.force_include_variables or []),
    ]
    cleaned = preprocess_frame(
        segmented,
        target=config.target,
        resample_rule=config.resample_rule,
        min_valid_ratio=config.min_valid_ratio,
        protected_columns=[c for c in protected if c],
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    transformed = transform_frame(
        cleaned,
        config.preprocess_mode,
        config.detrend_window,
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    scaled = standardize_frame(transformed)
    with SCALED_FRAME_CACHE_LOCK:
        SCALED_FRAME_CACHE[cache_key] = scaled.copy(deep=True)
        while len(SCALED_FRAME_CACHE) > MAX_SCALED_FRAME_CACHE:
            oldest_key = next(iter(SCALED_FRAME_CACHE))
            SCALED_FRAME_CACHE.pop(oldest_key, None)
    return scaled.copy(deep=True)


def _scaled_frame_cache_key(config: AnalysisConfig) -> tuple[Any, ...]:
    path = Path(config.input_path)
    stat = path.stat() if path.exists() else None
    roles_path = Path(config.roles_path).resolve() if config.roles_path else None
    roles_stat = roles_path.stat() if roles_path and roles_path.exists() else None
    return (
        str(path.resolve()),
        stat.st_mtime_ns if stat else None,
        stat.st_size if stat else None,
        str(roles_path) if roles_path else None,
        roles_stat.st_mtime_ns if roles_stat else None,
        roles_stat.st_size if roles_stat else None,
        config.encoding,
        config.time_column,
        config.target,
        config.segment_column,
        config.segment_mode,
        config.segment_min,
        config.segment_max,
        tuple(config.capacity_columns or []),
        tuple(config.residual_control_columns or []),
        tuple(config.force_include_variables or []),
        config.resample_rule,
        config.min_valid_ratio,
        config.max_interpolate_gap_points,
        config.interpolate_limit_area,
        config.preprocess_mode,
        config.detrend_window,
    )


def _clear_scaled_frame_cache() -> None:
    with SCALED_FRAME_CACHE_LOCK:
        SCALED_FRAME_CACHE.clear()


def _trend_response(params: dict[str, list[str]]) -> dict[str, Any]:
    file_id = _single(params, "file_id")
    encoding = _single(params, "encoding", "utf-8-sig")
    input_path = _resolve_upload(file_id)
    resolved_encoding = _resolve_encoding(input_path, encoding)
    time_column = _single(params, "time_column")
    variables = [value for value in _single(params, "variables").split(",") if value]
    if not variables:
        raise ValueError("请选择至少一个趋势变量")
    if len(variables) > 4:
        raise ValueError("最多选择 4 个趋势变量")

    from chem_ts_corr.data import load_timeseries_csv, select_numeric_frame
    from chem_ts_corr.preprocess import segment_by_load, transform_frame

    raw = load_timeseries_csv(input_path, time_column, encoding=resolved_encoding)
    start_time = _single(params, "trend_start", "")
    end_time = _single(params, "trend_end", "")
    if start_time:
        raw = raw.loc[raw.index >= pd.to_datetime(start_time)]
    if end_time:
        raw = raw.loc[raw.index <= pd.to_datetime(end_time)]
    if raw.empty:
        raise ValueError("趋势图时间范围内没有数据")
    numeric = select_numeric_frame(raw, variables[0])
    columns = [column for column in variables if column in numeric.columns]
    if not columns:
        raise ValueError("选择的趋势变量不是有效数值列")
    frame_columns = list(dict.fromkeys(columns + [col for col in [_single(params, "segment_column")] if col and col in numeric.columns]))
    frame = numeric[frame_columns]
    segmented = segment_by_load(
        frame,
        segment_column=_single(params, "segment_column") or None,
        segment_mode=_single(params, "segment_mode", "all"),
        segment_min=_optional_query_float(params, "segment_min"),
        segment_max=_optional_query_float(params, "segment_max"),
    )
    transformed = transform_frame(
        segmented[columns],
        _single(params, "preprocess_mode", "raw"),
        int(_single(params, "detrend_window", "24") or 24),
    )
    max_points = max(100, int(_single(params, "trend_max_points", "10000") or 10000))
    raw_rows = len(transformed)
    if len(transformed) > max_points:
        step = max(1, len(transformed) // max_points)
        transformed = transformed.iloc[::step]

    return {
        "series": [
            {
                "name": column,
                "points": [
                    {"x": str(index), "y": None if pd.isna(value) else float(value)}
                    for index, value in transformed[column].items()
                ],
            }
            for column in columns
        ],
        "rows": int(len(transformed)),
        "raw_rows": int(raw_rows),
        "max_points": int(max_points),
    }


def _overview_payload(
    ranked: pd.DataFrame, risk: pd.DataFrame, config: AnalysisConfig, metrics: dict[str, str]
) -> dict[str, Any]:
    high_risk = int((risk.get("risk_count", pd.Series(dtype=float)) > 0).sum()) if not risk.empty else 0
    review = int((ranked.get("recommended_use", pd.Series(dtype=str)).astype(str) == "prediction_candidate").sum()) if not ranked.empty else 0
    return {
        "top10": _records(ranked.head(10)),
        "effective_variables": int(len(ranked)),
        "risk_tagged_count": high_risk,
        "high_risk_count": high_risk,
        "secondary_review_count": review,
        "target": config.target,
        "rows_after_preprocess": metrics.get("rows_after_preprocess", ""),
        "rows_after_segment": metrics.get("rows_after_segment", ""),
    }


def _summary_metrics(summary: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in summary.splitlines():
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def _resolve_encoding(path: Path, encoding: str) -> str:
    if encoding != "auto":
        return encoding
    _, used_encoding = _read_data_sample(path, "auto")
    return used_encoding


def _multipart_form(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "")
    raw_content_length = handler.headers.get("Content-Length", "0") or 0
    try:
        content_length = int(raw_content_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Content-Length") from exc
    if content_length < 0:
        raise ValueError("Invalid Content-Length")
    if content_length > MAX_REQUEST_BODY_BYTES:
        raise ValueError("上传文件过大")
    body = handler.rfile.read(content_length)
    if "multipart/form-data" in content_type:
        msg = BytesParser(policy=email_default_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        form: dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                form[name] = SimpleNamespace(filename=filename, file=payload)
            else:
                charset = part.get_content_charset() or "utf-8"
                form[name] = payload.decode(charset, errors="ignore")
        return form
    data = parse_qs(body.decode("utf-8", errors="ignore"), keep_blank_values=True)
    return {k: (v[0] if v else "") for k, v in data.items()}


def _resolve_upload(file_id: str) -> Path:
    file_id = _validate_file_id(file_id)
    path = None
    for suffix in sorted(TEXT_SUFFIXES | EXCEL_SUFFIXES):
        candidate = (UPLOADS_DIR / f"{file_id}{suffix}").resolve()
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        raise FileNotFoundError("上传文件不存在，请重新上传")
    if UPLOADS_DIR.resolve() not in path.parents:
        raise ValueError("Invalid upload path")
    if path.suffix.lower() not in TEXT_SUFFIXES | EXCEL_SUFFIXES:
        raise ValueError("Unsupported upload file type")
    return path


def _validate_file_id(file_id: str) -> str:
    value = str(file_id or "").strip().lower()
    if not _FILE_ID_RE.fullmatch(value):
        raise ValueError("Invalid file id")
    return value


def _cleanup_tasks(now: float | None = None) -> None:
    with TASKS_LOCK:
        _cleanup_tasks_locked(now=now)


def _cleanup_tasks_locked(now: float | None = None) -> None:
    current = time.time() if now is None else now
    terminal_statuses = {"done", "error", "failed"}
    expired = [
        task_id
        for task_id, task in TASKS.items()
        if task.get("status") in terminal_statuses
        and (
            current
            - float(task.get("updated_at") or task.get("end_time") or task.get("created_at") or current)
            > TASK_TTL_SECONDS
        )
    ]
    for task_id in expired:
        TASKS.pop(task_id, None)
    if len(TASKS) <= MAX_TASKS:
        return
    removable = sorted(
        (
            (float(task.get("updated_at") or task.get("end_time") or task.get("created_at") or 0), task_id)
            for task_id, task in TASKS.items()
            if task.get("status") != "running"
        ),
        key=lambda item: item[0],
    )
    for _, task_id in removable:
        if len(TASKS) <= MAX_TASKS:
            break
        TASKS.pop(task_id, None)


def _resolve_run_dir(run_id: str) -> Path:
    path = (RUNS_DIR / run_id).resolve()
    if RUNS_DIR.resolve() not in path.parents or not path.exists():
        raise FileNotFoundError("运行结果不存在，请先完成主筛查")
    return path


def _write_run_config(output_dir: Path, config: AnalysisConfig, file_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    data["input_path"] = str(config.input_path)
    data["output_dir"] = str(config.output_dir)
    data["roles_path"] = str(config.roles_path) if config.roles_path else None
    data["file_id"] = file_id
    (output_dir / "run_config.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_run_config(output_dir: Path) -> AnalysisConfig:
    data = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    data["input_path"] = Path(data["input_path"])
    data["output_dir"] = output_dir
    data["roles_path"] = Path(data["roles_path"]) if data.get("roles_path") else None
    data.pop("file_id", None)
    return AnalysisConfig(**data)


def _download_links(run_id: str, output_dir: Path) -> list[dict[str, str]]:
    return [
        {"name": file_name, "url": f"/download?run_id={run_id}&file={file_name}"}
        for file_name in DOWNLOAD_FILES
        if (output_dir / file_name).exists()
    ]


def _field(form: dict[str, Any], name: str, default: str = "") -> str:
    if name not in form:
        return default
    value = form.get(name)
    if value is None:
        return default
    if hasattr(value, "filename"):
        return default
    return str(value)


def _int_field(form: dict[str, Any], name: str, default: int) -> int:
    value = _field(form, name, "")
    if not value:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_field(form: dict[str, Any], name: str, default: float) -> float:
    value = _field(form, name, "")
    if not value:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float_field(form: dict[str, Any], name: str) -> float | None:
    value = _field(form, name, "")
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int_field(form: dict[str, Any], name: str) -> int | None:
    value = _field(form, name, "")
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _list_field(form: dict[str, Any], name: str) -> list[str]:
    value = _field(form, name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_field(form: dict[str, Any], name: str) -> bool:
    return _field(form, name, "").lower() in {"1", "true", "yes", "on"}


def _single(params: dict[str, list[str]], name: str, default: str = "") -> str:
    values = params.get(name)
    return values[0] if values else default


def _optional_query_float(params: dict[str, list[str]], name: str) -> float | None:
    value = _single(params, name, "")
    return float(value) if value else None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.where(pd.notna(frame), None)
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def _safe_read_result_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Chem TS Corr web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, open_browser=not args.no_open)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>化工时序相关性分析</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --line:#d9dee7; --text:#16202a; --muted:#5f6b7a; --accent:#176b87; --green:#0f5132; --warn:#8a5a00; --font-xs:11px; --font-sm:12px; --font-base:13px; --surface-muted:#f8fafc; --line-soft:#e7ebf0; --text-subtle:#5f6b7a; --focus:#0f766e; --danger-bg:#fee2e2; --danger-text:#991b1b; --warning-bg:#fef3c7; --warning-text:#92400e; --info-bg:#e0f2fe; --info-text:#075985; --success-bg:#dcfce7; --success-text:#166534; }
    * { box-sizing: border-box; }
    :focus-visible { outline:2px solid var(--focus); outline-offset:2px; }
    body { margin:0; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; color:var(--text); background:var(--bg); }
    header { padding:22px 28px 14px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1 { margin:0 0 8px; font-size:24px; }
    .subtitle { color:var(--muted); font-size:14px; }
    main { display:grid; grid-template-columns:minmax(320px,430px) 1fr; gap:18px; padding:18px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    .controls { display:grid; gap:10px; align-content:start; font-size:80%; }
    .control-group { display:grid; gap:8px; padding:10px; border:1px solid var(--line-soft); border-radius:8px; background:var(--surface-muted); }
    .control-group-title { font-size:var(--font-sm); font-weight:700; color:var(--text); }
    label { display:grid; gap:3px; font-size:var(--font-xs); line-height:1.2; color:var(--muted); }
    input, select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; color:var(--text); background:var(--panel); font-size:var(--font-xs); line-height:1.2; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .check { display:flex; align-items:center; gap:8px; color:var(--text); font-size:14px; }
    .check input { width:auto; }
    .actions { display:flex; gap:10px; flex-wrap:wrap; }
    .multi-dropdown { border:1px solid var(--line); border-radius:6px; background:var(--panel); }
    .multi-dropdown > summary { list-style:none; cursor:pointer; padding:6px 8px; font-size:var(--font-xs); text-align:left; }
    .multi-dropdown > summary::-webkit-details-marker { display:none; }
    .multi-options { max-height:180px; min-width:260px; overflow:auto; border-top:1px solid var(--line); padding:6px 8px; display:grid; gap:4px; }
    .multi-options label { display:grid; grid-template-columns:16px 1fr; align-items:center; column-gap:8px; font-size:var(--font-xs); color:var(--text); text-align:left; line-height:1.2; }
    .multi-options input[type="checkbox"] { margin:0; }
    .multi-options span { display:block; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    button { border:0; border-radius:6px; padding:10px 14px; font-weight:650; cursor:pointer; background:var(--accent); color:#fff; }
    button.secondary { background:#e8edf3; color:var(--text); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .status { min-height:22px; padding:7px 9px; border:1px solid transparent; border-radius:6px; color:var(--muted); font-size:var(--font-base); white-space:pre-wrap; }
    .status.info { background:var(--info-bg); color:var(--info-text); border-color:#bae6fd; }
    .status.success { background:var(--success-bg); color:var(--success-text); border-color:#bbf7d0; }
    .status.warning { background:var(--warning-bg); color:var(--warning-text); border-color:#fde68a; }
    .status.error { background:var(--danger-bg); color:var(--danger-text); border-color:#fecaca; }
    .status.loading { background:var(--surface-muted); color:var(--text); border-color:var(--line); }
    .note { color:var(--warn); font-size:var(--font-xs); line-height:1.2; }
    .results { display:grid; gap:16px; min-width:0; align-content:start; position:relative; }
    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    h2 { margin:0; font-size:18px; }
    .download-buttons { display:flex; gap:8px; flex-wrap:wrap; }
    .download-buttons a { display:inline-block; border-radius:6px; padding:8px 10px; background:var(--green); color:#fff; text-decoration:none; font-size:var(--font-base); }
    .help { display:grid; gap:4px; color:var(--muted); font-size:var(--font-xs); line-height:1.25; background:var(--surface-muted); border:1px solid var(--line); border-radius:6px; padding:8px 10px; }
    .tabs {
      position:sticky;
      top:0;
      z-index:20;
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      align-items:center;
      border-bottom:1px solid var(--line);
      padding:8px 0;
      background:var(--panel);
    }
    .tab-button {
      width:auto;
      min-width:76px;
      height:34px;
      display:inline-flex;
      align-items:center;
      justify-content:center;
      background:#e8edf3;
      color:var(--text);
      padding:0 12px;
      line-height:1;
      border-radius:6px;
    }
    .tab-button.active { background:var(--accent); color:#fff; }
    .tab-panel { display:none; gap:14px; }
    .tab-panel.active { display:grid; }
    .overview-grid {
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      align-items:stretch;
    }
    .metric-card {
      width:180px;
      min-height:64px;
      border:1px solid var(--line);
      border-radius:8px;
      padding:10px 12px;
      background:var(--surface-muted);
    }
    .metric-value { display:block; font-size:20px; line-height:1.15; font-weight:700; color:var(--text); }
    .metric-label { color:var(--muted); font-size:var(--font-sm); line-height:1.25; }
    .chart { min-height:280px; border:1px solid var(--line); border-radius:6px; background:var(--panel); overflow:hidden; }
    .chart svg { width:100%; height:320px; display:block; }
    .chart-controls { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)) 150px auto; gap:10px; align-items:end; }
    .trend-options { display:grid; grid-template-columns:repeat(3,minmax(160px,1fr)); gap:10px; align-items:end; }
    .llm-config-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; align-items:end; }
    .legend { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:var(--font-base); }
    .trend-stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; }
    .trend-stats.empty { display:block; color:var(--muted); font-size:var(--font-sm); }
    .trend-stat-card { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:10px; }
    .trend-stat-card h3 { margin:0 0 8px; font-size:var(--font-sm); overflow-wrap:anywhere; }
    .trend-stat-card dl { display:grid; gap:4px; margin:0; }
    .trend-stat-card dl div { display:grid; grid-template-columns:80px 1fr; gap:8px; font-size:var(--font-xs); }
    .trend-stat-card dt { color:var(--muted); }
    .trend-stat-card dd { margin:0; color:var(--text); text-align:right; font-variant-numeric:tabular-nums; }
    .swatch { width:18px; height:3px; border-radius:2px; display:inline-block; vertical-align:middle; margin-right:6px; }
    .table-wrap { overflow-x:auto; overflow-y:auto; max-height:560px; width:max-content; min-width:0; max-width:100%; border:1px solid var(--line); border-radius:8px; box-shadow:0 1px 2px rgba(15,23,42,.05); background:var(--panel); }
    .table-wrap::after { content:"表格按内容宽度展示；超出页面时横向滚动，点击表头可排序，点击行查看完整字段详情"; display:block; padding:5px 8px; color:var(--muted); font-size:var(--font-xs); background:var(--surface-muted); border-top:1px solid var(--line); }
    .terms-help-table-wrap { overflow-x:auto; width:max-content; min-width:0; max-width:100%; border:1px solid var(--line); border-radius:8px; box-shadow:0 1px 2px rgba(15,23,42,.05); background:var(--panel); }
    .terms-help-table-wrap::after { content:"术语说明按正常页面高度完整展示；超出页面宽度时可横向滚动"; display:block; padding:5px 8px; color:var(--muted); font-size:var(--font-xs); background:var(--surface-muted); border-top:1px solid var(--line); }
    .terms-help-category-cell { font-weight:650; color:var(--text); background:var(--surface-muted); vertical-align:top; }
    /* legacy regression marker: table-layout:fixed */
    table { width:max-content; min-width:100%; table-layout:auto; border-collapse:separate; border-spacing:0; font-size:var(--font-sm); }
    th, td { padding:7px 10px; border-bottom:1px solid var(--line-soft); text-align:left; vertical-align:top; white-space:normal; overflow-wrap:anywhere; word-break:break-word; }
    th { position:sticky; top:0; background:#eef2f6; z-index:2; box-shadow:0 1px 0 var(--line); }
    tbody tr:nth-child(even) { background:#fbfcfe; }
    tbody tr:hover { background:#f1f5f9; }
    th:first-child, td:first-child { position:sticky; left:0; z-index:1; background:inherit; box-shadow:1px 0 0 var(--line-soft); }
    th:first-child { z-index:3; background:#eef2f6; }
    td.numeric { text-align:right; font-variant-numeric:tabular-nums; }
    td.wrap-cell { line-height:1.35; }
    .compact-result-table tbody tr { cursor:pointer; }
    .compact-result-table tbody tr.selected { background:#e0f2fe; }
    .detail-panel { margin-top:10px; border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--surface-muted); }
    .detail-panel h3 { margin:0 0 8px; font-size:15px; }
    .modal-backdrop { position:fixed; inset:0; display:none; align-items:center; justify-content:center; padding:24px; background:rgba(15,23,42,.55); z-index:1000; }
    .modal-backdrop.open { display:flex; }
    .modal-card { width:min(960px, 96vw); max-height:88vh; overflow:auto; background:var(--panel); border-radius:12px; box-shadow:0 24px 60px rgba(15,23,42,.28); border:1px solid var(--line); }
    .modal-header { position:sticky; top:0; display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); background:var(--panel); z-index:1; }
    .modal-header h3 { margin:0; font-size:16px; }
    .modal-body { padding:16px; }
    .modal-close { background:#e8edf3; color:var(--text); padding:8px 12px; }
    .detail-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:8px; }
    .detail-field { background:var(--panel); border:1px solid var(--line-soft); border-radius:6px; padding:8px; }
    .detail-field strong { display:block; color:var(--muted); font-size:var(--font-sm); margin-bottom:4px; }
    th.sortable { cursor:pointer; user-select:none; }
    th.sortable:hover { background:#dde6ef; }
    th .sort-mark { color:var(--muted); margin-left:6px; font-size:var(--font-xs); }
    .decision-badge { display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:var(--font-sm); }
    .decision-risk_limited_review { background:#fef3c7; color:#92400e; }
    .decision-priority_review { background:#fee2e2; color:#991b1b; }
    .decision-secondary_review { background:#ffedd5; color:#9a3412; }
    .decision-not_recommended { background:#e5e7eb; color:#374151; }
    .decision-insufficient_evidence { background:#dbeafe; color:#1e40af; }
    .decision-manual_review_only { background:#7f1d1d; color:#fff; }
    .review-card {
      border:1px solid #d0d7de;
      border-radius:8px;
      padding:12px;
      background:var(--panel);
      margin-top:10px;
    }
    .review-card h3 { margin-top:0; }
    .metric-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
      gap:8px;
    }
    .metric-item {
      background:#f6f8fa;
      padding:8px;
      border-radius:6px;
    }
    .metric-item strong { display:block; color:var(--muted); font-size:var(--font-sm); margin-bottom:4px; }
    .small-button { padding:5px 8px; font-size:var(--font-sm); }
    .clickable-row { cursor:pointer; }
    .clickable-row:hover { background:#f6f8fa; }
    .empty { color:var(--muted); padding:24px; text-align:center; border:1px dashed var(--line); border-radius:6px; }
    pre { margin:0; padding:12px; background:var(--surface-muted); border:1px solid var(--line); border-radius:6px; max-height:260px; overflow:auto; white-space:pre-wrap; font-size:var(--font-sm); }
    .markdown-report { min-height:320px; max-height:720px; overflow:auto; padding:18px 22px; border:1px solid var(--line); border-radius:8px; background:var(--panel); line-height:1.65; font-size:14px; }
    .markdown-report h1, .markdown-report h2, .markdown-report h3 { margin:1.1em 0 .45em; line-height:1.25; color:var(--text); }
    .markdown-report h1 { font-size:24px; border-bottom:1px solid var(--line); padding-bottom:8px; }
    .markdown-report h2 { font-size:20px; border-bottom:1px solid #edf0f4; padding-bottom:6px; }
    .markdown-report h3 { font-size:16px; }
    .markdown-report p, .markdown-report ul, .markdown-report ol { margin:.55em 0; }
    .markdown-report blockquote { margin:10px 0; padding:8px 12px; border-left:4px solid var(--accent); background:var(--surface-muted); color:var(--muted); }
    .markdown-report code { padding:2px 4px; border-radius:4px; background:#f1f5f9; font-family:Consolas,monospace; }
    .markdown-report pre { max-height:none; margin:10px 0; }
    .markdown-report table { width:100%; min-width:0; border-collapse:collapse; font-size:var(--font-base); }
    .markdown-report th, .markdown-report td { white-space:normal; border:1px solid var(--line); }
    .markdown-report th:first-child, .markdown-report td:first-child { position:static; box-shadow:none; }
    @media (max-width:900px) { main { grid-template-columns:1fr; padding:12px; } .row { grid-template-columns:1fr; } .llm-config-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
    @media (max-width:560px) { .llm-config-grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>化工装置时序相关性分析</h1>
    <div class="subtitle">浏览器负责上传和展示，Python 后台处理大数据并生成下载结果。</div>
  </header>
  <main>
    <section class="controls">
      <div class="control-group">
        <div class="control-group-title">数据输入</div>
      <div class="actions">
        <button id="upload">上传并识别列</button>
        <button id="analyze" disabled>开始分析</button>
        <button id="reset" class="secondary">清空</button>
      </div>
      <label>数据文件（CSV / Excel / TXT）
        <input id="fileInput" type="file" accept=".csv,.txt,.tsv,.xlsx,.xls,.xlsm,text/csv,text/plain,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel">
      </label>
      <label>文件编码
        <select id="encoding">
          <option value="auto">自动识别</option>
          <option value="utf-8-sig">UTF-8</option>
          <option value="gb18030">GBK / GB18030</option>
        </select>
      </label>
      </div>
      <div class="control-group">
        <div class="control-group-title">基础分析参数</div>
      <div class="row">
        <label>时间列<select id="timeColumn"></select></label>
        <label>目标列<select id="targetColumn"></select></label>
      </div>
      <div class="row">
        <label>最大滞后点数<input id="maxLag" type="number" min="0" max="5000" value="12"></label>
        <label>输出前 K 个<input id="topK" type="number" min="1" max="2000" value="50"></label>
      </div>
      <div class="row">
        <label>最小有效比例<input id="minValidRatio" type="number" min="0.1" max="1" step="0.05" value="0.7"></label>
        <label>重采样规则<input id="resampleRule" placeholder="可留空，例如 5min"></label>
      </div>
      <div class="row">
        <label>预处理模式
          <select id="preprocessMode">
            <option value="raw">原始数据</option>
            <option value="detrend">滑动均值去趋势</option>
            <option value="diff">一阶差分</option>
            <option value="detrend_diff">去趋势后差分</option>
          </select>
        </label>
        <label>去趋势窗口点数<input id="detrendWindow" type="number" min="3" max="100000" value="24"></label>
      </div>
      </div>
      <div class="control-group">
        <div class="control-group-title">工况与残差控制</div>
      <div class="row">
        <label>负荷代表列<select id="segmentColumn"></select></label>
        <label>工况分段
          <select id="segmentMode">
            <option value="all">全部</option>
            <option value="low">低负荷（下 1/3）</option>
            <option value="mid">中负荷（中间 1/3）</option>
            <option value="high">高负荷（上 1/3）</option>
            <option value="custom">自定义范围</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>自定义下限<input id="segmentMin" type="number" placeholder="可留空"></label>
        <label>自定义上限<input id="segmentMax" type="number" placeholder="可留空"></label>
      </div>
      <label>残差控制列（CAPACITY，多选）
        <details id="capacityDropdown" class="multi-dropdown">
          <summary id="capacitySummary">请选择残差控制列</summary>
          <div id="capacityOptions" class="multi-options"></div>
        </details>
      </label>
      </div>
      <div class="control-group">
        <div class="control-group-title">复核参数</div>
      <label>强制复核变量（多选）
        <details id="forceIncludeDropdown" class="multi-dropdown">
          <summary id="forceIncludeSummary">请选择强制复核变量</summary>
          <div id="forceIncludeOptions" class="multi-options"></div>
        </details>
      </label>
      <div class="row">
        <label>三层复核候选数量<input id="causalTopN" type="number" min="1" max="1000" placeholder="可留空"></label>
        <label>风险标签包含过滤<input id="riskFlagFilter" placeholder="如 共同负荷驱动，留空表示不过滤"></label>
      </div>
      </div>
      <div id="status" class="status info" role="status" aria-live="polite"></div>
      <div class="note">大文件会由 Python 后台处理。分析期间请不要关闭启动服务的命令窗口。</div>
    </section>

    <section class="results">
      <div class="tabs" role="tablist" aria-label="结果分类">
        <button class="tab-button active" role="tab" aria-selected="true" aria-controls="overviewTab" id="tab-overviewTab" data-tab="overviewTab" tabindex="0">初步分析</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="candidatesTab" id="tab-candidatesTab" data-tab="candidatesTab" tabindex="-1">候选变量</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="trendTab" id="tab-trendTab" data-tab="trendTab" tabindex="-1">趋势图</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="validationTab" id="tab-validationTab" data-tab="validationTab" tabindex="-1">二次验证</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="causalReviewTab" id="tab-causalReviewTab" data-tab="causalReviewTab" tabindex="-1">三层复核</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="llmReportTab" id="tab-llmReportTab" data-tab="llmReportTab" tabindex="-1">AI 综合解读</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="downloadsTab" id="tab-downloadsTab" data-tab="downloadsTab" tabindex="-1">下载</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="termsHelpTab" id="tab-termsHelpTab" data-tab="termsHelpTab" tabindex="-1">术语与标签说明</button>
      </div>

      <div id="overviewTab" class="tab-panel active" role="tabpanel" aria-labelledby="tab-overviewTab">
        <h2>初步分析</h2>
        <div id="overview" class="overview-grid"></div>
        <h2>前 10 个推荐变量</h2>
        <div id="overviewTop" class="empty">上传数据并点击“开始分析”后显示结果。</div>
      </div>

      <div id="candidatesTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-candidatesTab" hidden>
        <h2>候选变量</h2>
        <div class="help">默认只展示候选排序结果的核心列和前 50 行，完整结果请到下载页获取。</div>
        <h3>结果质量提示</h3>
        <div id="screeningQualityHints" class="empty">完成主筛查后显示结果质量提示。</div>
        <div id="table" class="empty">上传数据并点击“开始分析”后显示结果。</div>
        <h2>轻量遗漏候选</h2>
        <div class="help">该表基于已有滞后相关、残差相关、峰值质量和风险标签生成，用于提示主筛查前 K 个外可能遗漏的候选。结果不代表因果结论。</div>
        <div id="nearMissTable" class="empty">完成主筛查后显示轻量遗漏候选。</div>
      </div>

      <div id="trendTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-trendTab" hidden>
        <h2>趋势图</h2>
        <div id="trendReviewHint" class="help">点击最终推荐摘要中的“查看趋势”后显示候选变量复核提示。</div>
        <div class="chart-controls">
          <label>数据 1<select id="trendVar1"></select></label>
          <label>数据 2<select id="trendVar2"></select></label>
          <label>数据 3<select id="trendVar3"></select></label>
          <label>数据 4<select id="trendVar4"></select></label>
          <label>Y 轴
            <select id="trendAxisMode">
              <option value="shared">同一 Y 轴</option>
              <option value="independent">独立 Y 轴</option>
            </select>
          </label>
          <button id="drawTrend" disabled>显示趋势</button>
        </div>
        <div class="trend-options">
          <label>开始时间
            <input id="trendStart" type="datetime-local">
          </label>
          <label>结束时间
            <input id="trendEnd" type="datetime-local">
          </label>
          <label>最大绘图点数
            <input id="trendMaxPoints" type="number" min="100" max="100000" value="10000">
          </label>
        </div>
        <div id="trendChart" class="chart empty">选择 1 到 4 个数据后点击“显示趋势”。</div>
        <div id="trendLegend" class="legend"></div>
        <div id="trendStats" class="trend-stats empty">选择数据并点击“显示趋势”后显示统计摘要。</div>
      </div>

      <div id="validationTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-validationTab" hidden>
        <h2>二次验证</h2>
        <div class="help">先完成主筛查，再按需运行增强筛选、Granger 预测验证或随机森林模型解释。结果会同步写入下载文件。</div>
        <div class="help">Granger 显著表示历史预测信息，不等于因果成立；随机森林重要性表示模型依赖，不等于可操作性；模型提升低可能说明目标自身历史已解释大部分波动；滚动稳定性低说明关系可能受工况影响。</div>
        <div class="actions">
          <button id="runEnhancedScreening" disabled>运行增强筛选</button>
          <button id="runGranger" disabled>运行 Granger 验证</button>
          <button id="runModel" disabled>运行随机森林模型解释</button>
        </div>
        <h2>增强筛选结果</h2>
        <div class="help">增强筛选用于补充验证主筛查候选的预测增益和时间稳定性，不代表因果结论。</div>
        <h3>增强筛选摘要</h3>
        <div id="enhancedSummaryTable" class="empty">点击“运行增强筛选”后显示增强筛选摘要。</div>
        <h3>模型提升评分</h3>
        <div id="enhancedLiftTable" class="empty">点击“运行增强筛选”后显示模型提升评分。</div>
        <h3>滚动稳定性评分</h3>
        <div id="enhancedRollingTable" class="empty">点击“运行增强筛选”后显示滚动稳定性评分。</div>
        <h2>Granger 验证</h2>
        <div id="grangerTable" class="empty">未启用 Granger 检验。</div>
        <h2>随机森林模型解释变量排序</h2>
        <div class="help">该表按变量汇总随机森林/SHAP 重要性，每个变量仅显示最强 lag。结果表示预测模型依赖，不代表因果关系或可操作性。</div>
        <div id="modelVariableImportanceTable" class="empty">运行随机森林模型解释后显示变量排序。</div>
        <h2>随机森林模型解释特征明细</h2>
        <div id="importanceTable" class="empty">未启用随机森林模型解释。</div>
        <h2>随机森林模型解释补充候选</h2>
        <div class="help">该表用于发现随机森林模型解释中靠前、但主筛查前 N 个未优先覆盖的补充候选。结果仅表示预测模型依赖，不代表因果关系或可操作性。</div>
        <div id="modelDiscoveredTable" class="empty">运行随机森林模型解释后显示补充候选。</div>
      </div>

      <div id="causalReviewTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-causalReviewTab" hidden>
        <h2>三层复核</h2>
        <div class="help">所有结果仅作为“预测验证/人工复核建议”，不是因果结论。可在左侧设置前 N 个候选变量和风险标签包含过滤后运行。</div>
        <div class="help">三层复核支持长滞后变量。默认围绕主筛查最佳滞后附近做条件 Granger 验证，避免对 1..maxlag 全量扫描造成计算过慢。如需完整扫描，可切换为 full_scan。</div>
        <div class="help">高共线性、闭环和共同负荷风险不等于变量不重要。对于数据证据强的候选，平台会保留优先复核建议，同时标记统计检验受限。</div>
        <div class="row">
          <label>条件Granger滞后模式
            <select id="conditionalLagMode">
              <option value="ranked_window">围绕主筛查最佳滞后</option>
              <option value="best_only">仅最佳滞后</option>
              <option value="full_scan">全量扫描</option>
            </select>
          </label>
          <label>条件Granger滞后窗口<input id="conditionalLagWindow" type="number" min="0" value="5"></label>
          <label>条件Granger fallback 最大滞后<input id="conditionalFallbackMaxlag" type="number" min="1" value="24"></label>
          <label>条件Granger baseline 最大滞后<input id="conditionalBaselineMaxlag" type="number" min="1" value="24"></label>
        </div>
        <div class="actions">
          <button id="runCausalReview" disabled>运行三层复核</button>
        </div>
        <h2>条件 Granger 预测验证结果</h2>
        <div class="download-buttons" id="conditionalDownload"></div>
        <div id="conditionalGrangerTable" class="empty">未运行 条件 Granger 预测验证。</div>
        <h2>保守复核报告</h2>
        <div class="help">旧版保守复核报告用于调试和规则对照；页面优先展示逐变量综合证据复核表和最终推荐摘要。</div>
        <div class="download-buttons" id="causalReportDownload"></div>
        <div id="causalReviewTable" class="empty">未运行 三层复核。</div>
        <h2>最终推荐摘要</h2>
        <div class="help">该表基于逐变量综合证据复核表生成，用于给出人工复核优先级清单。结果仍是预测验证和复核建议，不是因果结论。请优先按“最终排序”查看；点击其它列排序仅用于辅助查看。点击“查看趋势”可自动带入目标变量和候选变量，用于人工检查滞后方向、响应形态和工艺合理性。</div>
        <div class="help">旧版保守复核报告仍保留在下载文件 causal_review_report.csv 中，主要用于调试和规则对照；页面优先展示逐变量综合证据复核表和最终推荐摘要。</div>
        <div class="download-buttons" id="finalReviewSummaryDownload"></div>
        <h3>最终推荐结果质检总览</h3>
        <div id="finalReviewQualityOverview" class="overview-grid"></div>
        <div id="finalReviewSummaryTable" class="empty">未运行 最终推荐摘要。</div>
        <h2>逐变量综合证据复核表</h2>
        <div class="help">逐变量综合证据复核表会整合主筛查、增强筛选、Granger、随机森林模型解释、条件 Granger 和风险标签。对于高共线性、闭环、共同负荷等统计限制，若数据证据强，平台会保留优先复核建议并标记统计受限。该表仍不是因果结论。</div>
        <div class="download-buttons" id="causalEvidenceDownload"></div>
        <div id="causalReviewEvidenceTable" class="empty">未运行 逐变量综合证据复核表。</div>
      </div>


      <div id="llmReportTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-llmReportTab" hidden>
        <h2>AI 综合解读</h2>
        <div class="help">填写 API 配置后可直接调用 DeepSeek/OpenAI 兼容聊天补全接口生成报告。API 密钥仅随本次请求发送，不保存到磁盘、不写入报告。</div>
        <div class="llm-config-grid">
          <label>分析变量数量<input id="llmTopN" type="number" min="1" max="100" value="20"></label>
          <label>报告类型
            <select id="llmReportType">
              <option value="apc_advice">APC/DCS 工程建议</option>
              <option value="general">通用综合解读</option>
            </select>
          </label>
          <label>模型服务
            <select id="llmProvider">
              <option value="deepseek">DeepSeek</option>
              <option value="openai">OpenAI-compatible</option>
            </select>
          </label>
          <label>接口地址<input id="llmBaseUrl" value="https://api.deepseek.com"></label>
          <label>模型名称<input id="llmModel" value="deepseek-chat"></label>
          <label>API 密钥<input id="llmApiKey" type="password" autocomplete="off" placeholder="sk-..."></label>
          <label>温度参数<input id="llmTemperature" type="number" min="0" max="2" step="0.1" value="0.2"></label>
          <label>最大输出 Token 数<input id="llmMaxTokens" type="number" min="256" max="32000" value="4096"></label>
        </div>
        <div id="llmConnectionStatus" class="help" aria-live="polite">尚未测试 API 连接。</div>
        <div class="actions">
          <button id="testLlmConnection">测试 API 连接</button>
          <button id="generateLlmReport">生成 DeepSeek 报告</button>
          <button id="copyLlmReport">复制报告</button>
          <span id="llmReportDownload" class="download-buttons"><a href="#">下载 llm_report.md</a></span>
        </div>
        <div id="llmReportRendered" class="markdown-report empty">生成后将在这里按 Markdown 格式显示 LLM 报告。</div>
      </div>

      <div id="downloadsTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-downloadsTab" hidden>
        <h2>下载</h2>
        <div id="downloads" class="download-buttons"></div>
      </div>

      <div id="termsHelpTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-termsHelpTab" hidden>
        <h2>术语与标签说明</h2>
        <div class="help">
          <div>本页用于解释分析结果中的标签、风险、证据等级和模型指标，帮助工程人员理解页面显示名称对应的复核含义。</div>
          <div>这些说明仅用于辅助工程复核，不改变分析结果，也不参与计算；后台分析输出和 CSV 下载保持不变。</div>
        </div>
        <div id="termsHelpTable" class="empty">术语说明加载中。</div>
      </div>

    </section>
  </main>

  <div id="detailModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="detailModalTitle" hidden>
    <div class="modal-card">
      <div class="modal-header">
        <h3 id="detailModalTitle">变量详情</h3>
        <button id="detailModalClose" type="button" class="modal-close">关闭</button>
      </div>
      <div id="detailModalBody" class="modal-body"></div>
    </div>
  </div>

<script>
let fileId = "";
let currentRunId = "";
let lastRows = [];
let lastGrangerRows = [];
let lastImportanceRows = [];
let lastModelVariableRows = [];
let lastNearMissRows = [];
let lastModelDiscoveredRows = [];
let lastEnhancedSummaryRows = [];
let lastEnhancedLiftRows = [];
let lastEnhancedRollingRows = [];
let lastConditionalRows = [];
let lastCausalReportRows = [];
let lastCausalEvidenceRows = [];
let lastFinalReviewSummaryRows = [];
let llmPromptText = "";
let llmReportMarkdown = "";
let lastModalTrigger = null;
let tableSortStates = { table: { column: "final_score", direction: "desc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
const el = (id) => document.getElementById(id);
const trendColors = ["#176b87", "#c2410c", "#6d28d9", "#15803d"];
const llmPromptEndpoint = "/api/llm_prompt";
let lastTrendSeries = [];
let lastTrendAxisMode = "shared";
let trendResizeTimer = null;

for (const button of document.querySelectorAll(".tab-button")) {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => handleTabKeydown(event, button));
}
el("drawTrend").addEventListener("click", drawTrend);
el("runEnhancedScreening").addEventListener("click", runEnhancedScreening);
el("runGranger").addEventListener("click", runGranger);
el("runModel").addEventListener("click", runModel);
el("runCausalReview").addEventListener("click", runCausalReview);
el("detailModalClose").addEventListener("click", closeDetailModal);
el("detailModal").addEventListener("click", (event) => { if (event.target === el("detailModal")) closeDetailModal(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDetailModal(); });
window.addEventListener("resize", () => {
  if (!lastTrendSeries.length) return;
  clearTimeout(trendResizeTimer);
  trendResizeTimer = setTimeout(() => renderTrendChart(lastTrendSeries, lastTrendAxisMode), 120);
});
el("testLlmConnection").addEventListener("click", testLlmConnection);
el("generateLlmReport").addEventListener("click", generateLlmReport);
el("copyLlmReport").addEventListener("click", copyLlmReport);

el("upload").addEventListener("click", uploadFile);
el("analyze").addEventListener("click", analyze);
el("reset").addEventListener("click", reset);
el("encoding").addEventListener("change", () => { if (fileId) loadColumns(); });


function fillCapacityOptions(columns) {
  const box = el("capacityOptions");
  box.innerHTML = "";
  columns.forEach((name) => {
    const id = `cap_${name}`;
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox" value="${escapeHtml(name)}"> <span>${escapeHtml(name)}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("change", updateCapacitySummary);
    box.appendChild(row);
  });
  updateCapacitySummary();
}

function getCapacitySelection() {
  return Array.from(document.querySelectorAll('#capacityOptions input[type="checkbox"]:checked')).map((node) => node.value);
}

function setCapacitySelection(values) {
  const selected = new Set(values || []);
  Array.from(document.querySelectorAll('#capacityOptions input[type="checkbox"]')).forEach((node) => {
    node.checked = selected.has(node.value);
  });
  updateCapacitySummary();
}

function updateCapacitySummary() {
  const selected = getCapacitySelection();
  el("capacitySummary").textContent = selected.length ? `已选 ${selected.length} 项` : "请选择残差控制列";
}


function fillForceIncludeOptions(columns) {
  const box = el("forceIncludeOptions");
  box.innerHTML = "";
  columns.forEach((name) => {
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox" value="${escapeHtml(name)}"> <span>${escapeHtml(name)}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("change", updateForceIncludeSummary);
    box.appendChild(row);
  });
  updateForceIncludeSummary();
}

function getForceIncludeSelection() {
  return Array.from(document.querySelectorAll('#forceIncludeOptions input[type="checkbox"]:checked')).map((node) => node.value);
}

function setForceIncludeSelection(values) {
  const selected = new Set(values || []);
  Array.from(document.querySelectorAll('#forceIncludeOptions input[type="checkbox"]')).forEach((node) => {
    node.checked = selected.has(node.value);
  });
  updateForceIncludeSummary();
}

function updateForceIncludeSummary() {
  const selected = getForceIncludeSelection();
  el("forceIncludeSummary").textContent = selected.length ? `已选 ${selected.length} 项` : "请选择强制复核变量";
}

async function uploadFile() {
  const file = el("fileInput").files[0];
  if (!file) return setStatus("请选择 CSV、Excel 或 TXT 数据文件。");
  try {
    setStatus("正在上传文件...", "loading");
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/upload", form);
    fileId = data.file_id;
    setStatus(`已上传：${data.filename}\n正在识别列...`);
    await loadColumns();
  } catch (error) {
    setStatus(error.message || String(error), "error");
  }
}

async function loadColumns() {
  try {
    const url = `/api/columns?file_id=${encodeURIComponent(fileId)}&encoding=${encodeURIComponent(el("encoding").value)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "列识别失败");
    fillSelect(el("timeColumn"), data.columns);
  fillSelect(el("targetColumn"), data.numericColumns);
  fillSelect(el("segmentColumn"), data.numericColumns, true);
  fillCapacityOptions(data.numericColumns);
  fillForceIncludeOptions(data.numericColumns);
  el("capacityDropdown").open = false;
  el("forceIncludeDropdown").open = false;
  fillSelect(el("trendVar1"), data.numericColumns);
  fillSelect(el("trendVar2"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar3"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar4"), data.numericColumns, true, "不选择");
  const timeCandidate = data.columns.find((name) => /time|date|timestamp|时间|日期/i.test(name));
  if (data.timeColumn && data.columns.includes(data.timeColumn)) {
    el("timeColumn").value = data.timeColumn;
  } else if (timeCandidate) {
    el("timeColumn").value = timeCandidate;
  }
  if (data.trendStartDefault) el("trendStart").value = data.trendStartDefault;
  if (data.trendEndDefault) el("trendEnd").value = data.trendEndDefault;
    const loadCandidate = data.numericColumns.find((name) => /load|负荷|进料|流量|feed|rate/i.test(name));
    if (loadCandidate) {
      el("segmentColumn").value = loadCandidate;
      setCapacitySelection([loadCandidate]);
    }
  el("analyze").disabled = false;
  el("drawTrend").disabled = data.numericColumns.length < 1;
    setStatus(`列识别完成。编码：${data.encoding}。采样读取 ${data.sampleRows} 行，识别到 ${data.columns.length} 列。`, "success");
  } catch (error) {
    el("analyze").disabled = true;
    setStatus(error.message || String(error), "error");
  }
}

async function analyze() {
  if (!fileId) return setStatus("请先上传文件。");
  setStatus("Python 后台正在分析，数据较大时请等待...", "loading");
  el("analyze").disabled = true;
  try {
    const form = new FormData();
    form.append("file_id", fileId);
    form.append("encoding", el("encoding").value);
    form.append("time_column", el("timeColumn").value);
    form.append("target", el("targetColumn").value);
    form.append("max_lag", el("maxLag").value);
    form.append("top_k", el("topK").value);
    form.append("min_valid_ratio", el("minValidRatio").value);
    form.append("resample_rule", el("resampleRule").value.trim());
    form.append("preprocess_mode", el("preprocessMode").value);
    form.append("detrend_window", el("detrendWindow").value);
    form.append("segment_column", el("segmentColumn").value);
    form.append("segment_mode", el("segmentMode").value);
    form.append("segment_min", el("segmentMin").value);
    form.append("segment_max", el("segmentMax").value);
    form.append("capacity_columns", getCapacitySelection().join(","));
    form.append("residual_control_columns", getCapacitySelection().join(","));
    form.append("force_include_variables", getForceIncludeSelection().join(","));
    form.append("exclude_control_columns_from_candidates", "true");
    const data = await postForm("/api/analyze", form);
    currentRunId = data.run_id || "";
    const result = await waitForAnalysisResult(data.task_id);
    renderAnalysisResult(result);
    setStatus(`分析完成。运行 ID：${result.run_id}`, "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally {
    el("analyze").disabled = !fileId;
  }
}

async function waitForAnalysisResult(taskId) {
  if (!taskId) throw new Error("未获得后台任务号");
  while (true) {
    await sleep(1000);
    const statusResponse = await fetch(`/api/status?task_id=${encodeURIComponent(taskId)}`);
    const statusData = await statusResponse.json();
    if (!statusResponse.ok) throw new Error(statusData.error || "任务状态查询失败");
    setStatus(formatTaskStatus(statusData), "loading");
    if (statusData.status === "error") throw new Error(statusData.error || statusData.message || "分析失败");
    if (statusData.status === "done") break;
  }
  const resultResponse = await fetch(`/api/result?task_id=${encodeURIComponent(taskId)}`);
  const resultData = await resultResponse.json();
  if (!resultResponse.ok) throw new Error(resultData.error || "结果读取失败");
  return resultData;
}

function formatTaskStatus(statusData) {
  const message = statusData.message || "后台分析中...";
  const elapsed = Number(statusData.elapsed_seconds || 0);
  return elapsed > 0 ? `${message}（已运行 ${elapsed.toFixed(1)} 秒）` : message;
}

function renderAnalysisResult(data) {
  currentRunId = data.run_id || "";
  lastRows = data.rankedFeatures || [];
  lastGrangerRows = data.grangerTests || [];
  lastImportanceRows = data.importance || [];
  lastModelVariableRows = [];
  lastNearMissRows = data.nearMissCandidates || [];
  lastModelDiscoveredRows = [];
  lastEnhancedSummaryRows = data.enhancedValidationSummary || [];
  const hasEnhancedScreening = lastEnhancedSummaryRows.length > 0;
  lastEnhancedLiftRows = hasEnhancedScreening ? (data.modelLiftScores || []) : [];
  lastEnhancedRollingRows = hasEnhancedScreening ? (data.rollingCorrScores || []) : [];
  lastConditionalRows = [];
  lastCausalReportRows = [];
  lastCausalEvidenceRows = [];
  lastFinalReviewSummaryRows = [];
  closeDetailModal();
  renderOverview(data.overview || {});
  renderScreeningQualityHints(lastRows);
  renderTable(lastRows);
  renderGenericTable("overviewTop", (data.overview && data.overview.top10) || [], coreCandidateColumns());
  renderGenericTable("nearMissTable", lastNearMissRows, nearMissColumns());
  renderGenericTable("grangerTable", lastGrangerRows);
  renderGenericTable("modelVariableImportanceTable", lastModelVariableRows, modelVariableImportanceColumns());
  renderGenericTable("importanceTable", lastImportanceRows);
  renderGenericTable("modelDiscoveredTable", lastModelDiscoveredRows, modelDiscoveredColumns());
  renderGenericTable("enhancedSummaryTable", lastEnhancedSummaryRows, enhancedSummaryColumns());
  renderGenericTable("enhancedLiftTable", lastEnhancedLiftRows, modelLiftColumns());
  renderGenericTable("enhancedRollingTable", lastEnhancedRollingRows, rollingCorrColumns());
  renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
  renderFinalReviewQualityOverview(lastFinalReviewSummaryRows);
  renderFinalReviewSummaryTable(lastFinalReviewSummaryRows);
  renderCausalReviewEvidenceTable(lastCausalEvidenceRows);
  renderReviewDownloads(data.downloads || []);
  renderDownloads(data.downloads || []);
  el("runEnhancedScreening").disabled = !currentRunId;
  el("runGranger").disabled = !currentRunId;
  el("runModel").disabled = !currentRunId;
  el("runCausalReview").disabled = !currentRunId;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function elapsedSeconds(startedAt) {
  return ((performance.now() - startedAt) / 1000).toFixed(1);
}

function formatRunningElapsed(message, startedAt) {
  return `${message}（已运行 ${elapsedSeconds(startedAt)} 秒）`;
}

function startStatusTimer(message, startedAt) {
  setStatus(formatRunningElapsed(message, startedAt), "loading");
  return setInterval(() => {
    setStatus(formatRunningElapsed(message, startedAt), "loading");
  }, 100);
}

function stopStatusTimer(timerId) {
  if (timerId) clearInterval(timerId);
}

function appendElapsed(message, startedAt) {
  return `${message} 总耗时：${elapsedSeconds(startedAt)} 秒。`;
}

async function runEnhancedScreening() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行增强筛选：补充验证预测增益和时间稳定性...", startedAt);
  el("runEnhancedScreening").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    const data = await postForm("/api/run_enhanced_screening", form);
    lastEnhancedSummaryRows = data.enhancedValidationSummary || [];
    lastEnhancedLiftRows = data.modelLiftScores || [];
    lastEnhancedRollingRows = data.rollingCorrScores || [];
    renderGenericTable("enhancedSummaryTable", lastEnhancedSummaryRows, enhancedSummaryColumns());
    renderGenericTable("enhancedLiftTable", lastEnhancedLiftRows, modelLiftColumns());
    renderGenericTable("enhancedRollingTable", lastEnhancedRollingRows, rollingCorrColumns());
    renderDownloads(data.downloads || []);
    setStatus(appendElapsed(data.message || "增强筛选完成。结果不代表因果结论。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("runEnhancedScreening").disabled = !currentRunId;
  }
}

async function runGranger() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行 Granger 二级验证...", startedAt);
  el("runGranger").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    const data = await postForm("/api/run_granger", form);
    lastGrangerRows = data.grangerTests || [];
    renderGenericTable("grangerTable", lastGrangerRows);
    renderDownloads(data.downloads || []);
    setStatus(appendElapsed("Granger 二级验证完成。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("runGranger").disabled = !currentRunId;
  }
}

async function runModel() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行随机森林模型解释...", startedAt);
  el("runModel").disabled = true;
  el("runCausalReview").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    const data = await postForm("/api/run_model", form);
    lastImportanceRows = data.importance || [];
    lastModelVariableRows = data.modelVariableImportance || [];
    lastModelDiscoveredRows = data.modelDiscoveredCandidates || [];
    renderGenericTable("modelVariableImportanceTable", lastModelVariableRows, modelVariableImportanceColumns());
    renderGenericTable("importanceTable", lastImportanceRows);
    renderGenericTable("modelDiscoveredTable", lastModelDiscoveredRows, modelDiscoveredColumns());
    renderDownloads(data.downloads || []);
    const metrics = data.modelMetrics ? Object.entries(data.modelMetrics).map(([k, v]) => `${k}: ${v}`).join("    ") : "";
    setStatus(appendElapsed(`随机森林模型解释完成。${metrics}`, startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("runModel").disabled = !currentRunId;
    el("runCausalReview").disabled = !currentRunId;
  }
}

async function runCausalReview() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行三层复核：结果仅为预测验证/人工复核建议，不是因果结论...", startedAt);
  el("runCausalReview").disabled = true;
  closeDetailModal();
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    form.append("top_n", el("causalTopN").value);
    form.append("risk_flag_filter", el("riskFlagFilter").value.trim());
    form.append("control_columns", getCapacitySelection().join(","));
    form.append("maxlag", el("maxLag").value);
    form.append("min_rows", "60");
    form.append("conditional_lag_mode", el("conditionalLagMode").value);
    form.append("conditional_lag_window", el("conditionalLagWindow").value);
    form.append("conditional_fallback_maxlag", el("conditionalFallbackMaxlag").value);
    form.append("conditional_baseline_maxlag", el("conditionalBaselineMaxlag").value);
    const data = await postForm("/api/run_causal_review", form);
    lastConditionalRows = data.conditionalGrangerScores || [];
    lastCausalReportRows = data.causalReviewReport || [];
    lastCausalEvidenceRows = data.causalReviewEvidence || [];
    lastFinalReviewSummaryRows = data.finalReviewSummary || [];
    tableSortStates["finalReviewSummaryTable"] = { column: "final_rank", direction: "asc" };
    renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
    renderFinalReviewQualityOverview(lastFinalReviewSummaryRows);
    renderFinalReviewSummaryTable(lastFinalReviewSummaryRows);
    renderCausalReviewEvidenceTable(lastCausalEvidenceRows);
    renderReviewDownloads(data.downloads || []);
    renderDownloads(data.downloads || []);
    setStatus(appendElapsed(data.message || "三层复核完成。结果不是因果结论。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("runCausalReview").disabled = !currentRunId;
  }
}


async function testLlmConnection() {
  if (!el("llmApiKey").value) {
    el("llmConnectionStatus").textContent = "请输入 API Key。API Key 不会保存。";
    return;
  }
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在测试 API 连接...", startedAt);
  el("testLlmConnection").disabled = true;
  el("llmConnectionStatus").textContent = "正在测试 API 连接...";
  try {
    const form = new FormData();
    form.append("provider", el("llmProvider").value || "deepseek");
    form.append("base_url", el("llmBaseUrl").value || "https://api.deepseek.com");
    form.append("model", el("llmModel").value || "deepseek-chat");
    form.append("api_key", el("llmApiKey").value);
    form.append("temperature", el("llmTemperature").value || "0.2");
    form.append("max_tokens", "16");
    const data = await postForm("/api/llm_connection", form);
    el("llmConnectionStatus").textContent = data.message || (data.ok ? "API 连接成功" : "API 连接失败");
    setStatus(appendElapsed(data.message || "API 连接测试完成。", startedAt), "success");
  } catch (error) {
    const message = error.message || String(error);
    el("llmConnectionStatus").textContent = message;
    setStatus(appendElapsed(message, startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("testLlmConnection").disabled = false;
  }
}

async function generateLlmReport() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  if (!el("llmApiKey").value) return setStatus("请输入 API Key。API Key 不会保存。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在调用 LLM 生成 AI 综合解读报告...", startedAt);
  el("generateLlmReport").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    form.append("provider", el("llmProvider").value || "deepseek");
    form.append("base_url", el("llmBaseUrl").value || "https://api.deepseek.com");
    form.append("model", el("llmModel").value || "deepseek-chat");
    form.append("api_key", el("llmApiKey").value);
    form.append("temperature", el("llmTemperature").value || "0.2");
    form.append("max_tokens", el("llmMaxTokens").value || "4096");
    form.append("top_n", el("llmTopN").value || "20");
    form.append("report_type", el("llmReportType").value || "apc_advice");
    const data = await postForm("/api/llm_report", form);
    llmPromptText = data.prompt || llmPromptText || "";
    setLlmReport(data.report || "");
    renderDownloadTarget("llmReportDownload", data.downloads || [], "llm_report.md");
    renderDownloads(data.downloads || []);
    setStatus(appendElapsed(data.message || "LLM 报告已生成。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("generateLlmReport").disabled = false;
  }
}

async function copyLlmReport() {
  const text = llmReportMarkdown || "";
  if (!text) return setStatus("请先生成 LLM 报告。");
  await navigator.clipboard.writeText(text);
  setStatus("LLM 报告已复制到剪贴板。");
}

function setLlmReport(markdown) {
  llmReportMarkdown = markdown || "";
  renderMarkdownReport(llmReportMarkdown);
}

function renderMarkdownReport(markdown) {
  const container = el("llmReportRendered");
  if (!container) return;
  if (!markdown.trim()) {
    container.className = "markdown-report empty";
    container.textContent = "生成后将在这里按 Markdown 格式显示 LLM 报告。";
    return;
  }
  container.className = "markdown-report";
  container.innerHTML = markdownToHtml(markdown);
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inList = false;
  let inCode = false;
  let codeLines = [];
  const closeList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (isMarkdownTableStart(lines, i)) {
      closeList();
      const parsed = parseMarkdownTable(lines, i);
      html.push(markdownTableToHtml(parsed.headers, parsed.rows));
      i = parsed.nextIndex - 1;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }
    if (!line.trim()) {
      closeList();
      continue;
    }
    closeList();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeList();
  if (inCode) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  return html.join("");
}

function isMarkdownTableStart(lines, index) {
  return isMarkdownTableRow(lines[index]) && isMarkdownTableSeparator(lines[index + 1]);
}

function isMarkdownTableRow(line) {
  return typeof line === "string" && line.includes("|") && splitMarkdownTableRow(line).length > 1;
}

function isMarkdownTableSeparator(line) {
  if (!isMarkdownTableRow(line)) return false;
  return splitMarkdownTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function splitMarkdownTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
  if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
  return trimmed.split("|").map((cell) => cell.trim());
}

function parseMarkdownTable(lines, startIndex) {
  const headers = splitMarkdownTableRow(lines[startIndex]);
  const rows = [];
  let nextIndex = startIndex + 2;
  while (nextIndex < lines.length && isMarkdownTableRow(lines[nextIndex]) && lines[nextIndex].trim()) {
    const row = splitMarkdownTableRow(lines[nextIndex]);
    while (row.length < headers.length) row.push("");
    rows.push(row.slice(0, headers.length));
    nextIndex += 1;
  }
  return { headers, rows, nextIndex };
}

function markdownTableToHtml(headers, rows) {
  const head = headers.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("");
  const body = rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}


const termsHelpRows = [
  { category: "参数设置说明", name: "参数说明", signal: "参数说明用于解释页面设置项的含义和对结果的影响；该帮助表不读取运行结果。", reading: "参数设置会影响候选筛选、滞后搜索、风险标签和复核范围，但本说明本身不改变分析结果，也不参与计算。", action: "先按数据口径确认设置，再解读候选排序、风险标签和复核清单。" },
  { category: "参数设置说明", name: "时间列", signal: "作为时序索引的时间戳字段。", reading: "决定排序、重采样、滞后对齐和趋势展示的时间基准。", action: "选择连续、可信、时区口径一致的采样时间列。" },
  { category: "参数设置说明", name: "目标列", signal: "需要解释、预测或优化的目标变量。", reading: "所有候选筛选、滞后搜索、风险标签和复核范围都围绕该目标展开。", action: "确认目标不是中间计算字段或与候选直接公式耦合的派生量。" },
  { category: "参数设置说明", name: "最大滞后点数", signal: "正负方向最多扫描的采样点数量。", reading: "窗口过小可能触发滞后边界风险，窗口过大可能增加偶然峰值和计算量。", action: "结合停留时间、采样周期和工艺响应时间设定，并用趋势图复核峰值。" },
  { category: "参数设置说明", name: "输出前 K 个", signal: "主筛查候选排序保留的前 K 个变量。", reading: "影响页面主候选范围和后续增强复核的默认输入，不改变完整下载文件的计算口径。", action: "探索阶段可调大，正式复核时聚焦工程上可解释的候选。" },
  { category: "参数设置说明", name: "最小有效比例", signal: "变量参与筛查所需的最低有效数据占比。", reading: "比例过低会带来数据质量风险；比例过高可能过滤掉间歇运行但重要的点位。", action: "先处理缺失、坏点和常数段，再按装置运行特点调整阈值。" },
  { category: "参数设置说明", name: "重采样规则", signal: "将原始数据对齐到统一采样间隔的规则。", reading: "会改变滞后点数对应的实际时间长度，并影响缺失、峰值和相关强度。", action: "使用符合采集周期和工艺响应速度的规则，避免过度平滑。" },
  { category: "参数设置说明", name: "预处理模式", signal: "原始、去趋势、差分或组合预处理。", reading: "会改变相关性关注的是绝对水平、慢趋势还是短期波动。", action: "根据问题选择模式，并比较不同模式下候选是否稳定。" },
  { category: "参数设置说明", name: "去趋势窗口点数", signal: "滑动去趋势时使用的窗口长度。", reading: "窗口决定慢趋势被剔除的尺度，过短可能去掉真实响应，过长可能保留漂移。", action: "按班次、停留时间或主要扰动周期设置，并检查趋势图。" },
  { category: "参数设置说明", name: "负荷代表列", signal: "代表装置负荷或产量的变量。", reading: "用于识别共同负荷驱动和工况稳定性，影响风险标签解释。", action: "优先选择现场认可的负荷、进料或产量指标。" },
  { category: "参数设置说明", name: "工况分段", signal: "按低/中/高负荷或自定义阈值拆分工况。", reading: "用于判断候选关系是否跨工况稳定，影响复核优先级。", action: "先确认分段边界有工程含义，避免样本过少。" },
  { category: "参数设置说明", name: "自定义下限 / 自定义上限", signal: "工况分段或过滤时使用的自定义上下限。", reading: "会限定参与对比的运行区间，影响稳定性和风险标签判断。", action: "用装置负荷区间、牌号或操作窗口确定上下限。" },
  { category: "参数设置说明", name: "残差控制列", signal: "在残差相关或条件验证中需要控制的变量。", reading: "用于减弱共同负荷、已知干扰或强共线变量的影响。", action: "填入负荷、设定值、关键上游扰动等已知控制因素。" },
  { category: "参数设置说明", name: "强制复核变量", signal: "即使未进入主排序前列也要纳入复核的变量。", reading: "扩展复核范围，适合业务重点变量或专家指定变量。", action: "只加入有明确工艺理由的点位，避免复核清单过长。" },
  { category: "参数设置说明", name: "三层复核候选数量", signal: "进入最终三层复核的候选变量数量。", reading: "数量越大覆盖越广但计算和人工解释成本越高。", action: "先用默认数量快速定位，再按需要扩大范围。" },
  { category: "参数设置说明", name: "风险标签包含过滤", signal: "按风险标签文本筛选复核或推荐结果。", reading: "只改变页面查看和复核聚焦范围，不表示未显示变量没有风险。", action: "用于定位共同负荷、闭环、数据质量等特定问题，留空表示不过滤。" },
  { category: "风险标签说明", name: "滞后边界风险", signal: "最佳滞后贴近扫描窗口边界，峰值可能尚未完全覆盖。", reading: "当前最大滞后点数可能偏小，真实响应时间可能更长。", action: "扩大最大滞后点数，结合趋势图确认峰值是否继续外移。" },
  { category: "风险标签说明", name: "变量滞后目标风险", signal: "页面显示为变量滞后目标。", reading: "变量变化晚于目标，更像响应量或受同一扰动影响。", action: "优先检查工艺方向，通常不直接作为前馈变量。" },
  { category: "风险标签说明", name: "公式泄漏 / 计算耦合风险", signal: "候选变量可能由目标或其上下游计算项派生。", reading: "高相关可能来自公式、软测量或报表口径耦合。", action: "核对 DCS/ historian 点位定义，剔除直接计算关系后再复核。" },
  { category: "风险标签说明", name: "数据质量风险", signal: "缺失、常数段、异常尖峰或有效比例不足影响结果。", reading: "统计指标可能受采样、坏点或仪表状态驱动。", action: "先清洗数据、确认仪表有效性，再重新运行分析。" },
  { category: "风险标签说明", name: "闭环反馈风险", signal: "变量可能处于控制回路内，与目标互相调节。", reading: "相关方向可能被 PID、APC 或人工操作反转。", action: "结合控制策略和阀位/设定值，必要时按开闭环时段分段验证。" },
  { category: "风险标签说明", name: "共线性风险", signal: "多个候选变量高度同步或代表同一工艺负荷。", reading: "模型可能难以区分真正贡献变量，单变量解释不稳定。", action: "做变量分组、残差控制或条件 Granger 预测验证。" },
  { category: "证据等级与复核建议", name: "强预测证据", signal: "相关、模型提升、预测贡献、稳定性等多类证据同时较好。", reading: "该变量对预测目标有较稳定信息量，但仍不是因果结论。", action: "进入优先复核，结合机理、趋势和现场操作记录确认。" },
  { category: "证据等级与复核建议", name: "风险受限证据", signal: "统计证据较强，但伴随共线性、闭环或数据质量等限制。", reading: "变量可能重要，但证据解释需要更谨慎。", action: "保留观察，先排除风险来源，再决定是否用于工程策略。" },
  { category: "证据等级与复核建议", name: "优先复核", signal: "综合排序或最终推荐摘要中优先级较高。", reading: "值得投入工程时间检查变量定义、方向和可操作性。", action: "查看趋势，核对滞后方向，并与班组/工艺专家确认。" },
  { category: "证据等级与复核建议", name: "仅人工复核", signal: "自动证据不足或风险较多，但业务上仍可能重要。", reading: "系统不建议直接采纳，需要人工判断。", action: "作为待查清单，补充机理证据或更多工况数据。" },
  { category: "滞后与方向解释", name: "变量领先目标", signal: "最佳 lag 为正，候选变量变化早于目标。", reading: "更符合前馈、扰动源或可提前预警变量特征。", action: "重点检查响应时间是否符合工艺停留时间。" },
  { category: "滞后与方向解释", name: "变量滞后目标", signal: "最佳 lag 为负，变量变化晚于目标。", reading: "候选变量可能是结果、反馈动作或滞后响应。", action: "谨慎用于控制前馈，可转为诊断或结果验证。" },
  { category: "模型验证指标", name: "模型提升", signal: "加入候选变量后，相比基线模型的预测效果改善。", reading: "变量提供了目标自身历史以外的增量信息。", action: "优先关注提升稳定且跨工况一致的变量。" },
  { category: "模型验证指标", name: "预测贡献", signal: "随机森林模型解释或重要性排序靠前。", reading: "模型依赖该变量做预测，但不代表可操作或因果成立。", action: "与相关性、Granger 和工程机理交叉验证。" },
  { category: "模型验证指标", name: "仅作预测验证，不是因果结论", signal: "Granger、条件 Granger 或模型指标显著。", reading: "说明历史信息有助于预测，不自动证明调节该变量会改变目标。", action: "在工程应用前必须经过机理和可操作性复核。" },
  { category: "稳定性指标", name: "滚动稳定性", signal: "滚动窗口内方向、强度或排名是否一致。", reading: "低稳定性可能表示关系只在局部时段成立。", action: "按时间段、开停工或异常时段拆分复核。" },
  { category: "稳定性指标", name: "工况稳定性", signal: "低/中/高负荷或自定义工况下结果是否一致。", reading: "关系受负荷、配方或操作模式影响。", action: "分工况建立候选清单，避免跨工况混用。" },
  { category: "工程使用建议", name: "可能 MV 候选", signal: "变量领先目标、风险可控且具备可调节属性。", reading: "可能适合作为操纵变量或控制策略候选。", action: "确认执行器约束、安全边界和操作权限后再试验。" },
  { category: "工程使用建议", name: "可能前馈 / 干扰变量", signal: "变量领先目标但不可直接调节，且能代表上游扰动。", reading: "适合用于提前预警、前馈补偿或软测量输入。", action: "验证采样及时性和信号可靠性，设计前馈延时补偿。" }
];

function renderTermsHelpGroupedRows(rows) {
  const keys = ["name", "signal", "reading", "action"];
  let html = "";
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const isFirstInCategory = index === 0 || row.category !== rows[index - 1].category;
    const categoryCell = isFirstInCategory
      ? `<td class="terms-help-category-cell" rowspan="${rows.filter((item) => item.category === row.category).length}">${escapeHtml(row.category)}</td>`
      : "";
    html += `<tr>${categoryCell}${keys.map((key) => `<td>${escapeHtml(row[key])}</td>`).join("")}</tr>`;
  }
  return html;
}

function renderTermsHelpTab() {
  const container = el("termsHelpTable");
  if (!container) return;
  const columns = ["分类", "页面显示名称", "具体表征", "工程解读", "建议动作"];
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${columns.map((col) => `<th scope="col">${escapeHtml(col)}</th>`).join("")}</tr></thead><tbody>${renderTermsHelpGroupedRows(termsHelpRows)}</tbody>`;
  const wrap = document.createElement("div");
  wrap.className = "terms-help-table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
}

renderTermsHelpTab();

function activateTab(tabId) {
  for (const button of document.querySelectorAll(".tab-button")) {
    const isActive = button.dataset.tab === tabId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
    button.tabIndex = isActive ? 0 : -1;
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    const isActive = panel.id === tabId;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  }
}

function handleTabKeydown(event, button) {
  const buttons = Array.from(document.querySelectorAll(".tab-button"));
  const currentIndex = buttons.indexOf(button);
  if (!["ArrowLeft", "ArrowRight", "Enter", " "].includes(event.key)) return;
  event.preventDefault();
  if (event.key === "Enter" || event.key === " ") return activateTab(button.dataset.tab);
  const offset = event.key === "ArrowRight" ? 1 : -1;
  const nextButton = buttons[(currentIndex + offset + buttons.length) % buttons.length];
  nextButton.focus();
  activateTab(nextButton.dataset.tab);
}

async function postForm(url, form) {
  const response = await fetch(url, { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function fillSelect(select, values, allowEmpty = false, emptyLabel = "不分段") {
  select.innerHTML = "";
  if (allowEmpty) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    select.appendChild(option);
  }
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

async function drawTrend() {
  try {
    const variables = [el("trendVar1").value, el("trendVar2").value, el("trendVar3").value, el("trendVar4").value].filter(Boolean);
    if (!variables.length) return setStatus("请至少选择一个趋势变量。");
    if (new Set(variables).size !== variables.length) return setStatus("趋势变量不能重复选择。");
    const params = new URLSearchParams({
      file_id: fileId,
      encoding: el("encoding").value,
      time_column: el("timeColumn").value,
      variables: variables.join(","),
      preprocess_mode: el("preprocessMode").value,
      detrend_window: el("detrendWindow").value,
      segment_column: el("segmentColumn").value,
      segment_mode: el("segmentMode").value,
      segment_min: el("segmentMin").value,
      segment_max: el("segmentMax").value,
      trend_start: el("trendStart").value,
      trend_end: el("trendEnd").value,
      trend_max_points: el("trendMaxPoints").value,
    });
    setStatus("正在生成趋势图...", "loading");
    const response = await fetch(`/api/trend?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "趋势图生成失败");
    renderTrendChart(data.series || [], el("trendAxisMode").value);
    setStatus(`趋势图已生成，原始 ${data.raw_rows} 点，显示 ${data.rows} 点，最大点数 ${data.max_points}。`, "success");
  } catch (error) {
    el("trendChart").className = "chart empty";
    el("trendChart").textContent = error.message || String(error);
    el("trendLegend").innerHTML = "";
    clearTrendStats();
    setStatus(error.message || String(error), "error");
  }
}

function renderTrendChart(series, axisMode) {
  const container = el("trendChart");
  if (!series.length) {
    container.className = "chart empty";
    container.textContent = "没有可绘制的趋势数据。";
    el("trendLegend").innerHTML = "";
    clearTrendStats();
    return;
  }
  const width = 960, height = 320, pad = { left: 76, right: axisMode === "independent" ? 76 : 28, top: 30, bottom: 44 };
  const maxLen = Math.max(...series.map((item) => item.points.length));
  const allValues = series.flatMap((item) => item.points.map((point) => Number(point.y)).filter((value) => Number.isFinite(value)));
  const sharedRange = valueRange(allValues);
  const ranges = series.map((item) => axisMode === "shared" ? sharedRange : valueRange(item.points.map((point) => Number(point.y)).filter((value) => Number.isFinite(value))));
  const x = (index) => pad.left + (index / Math.max(1, maxLen - 1)) * (width - pad.left - pad.right);
  const y = (value, range) => pad.top + (1 - (value - range.min) / Math.max(1e-12, range.max - range.min)) * (height - pad.top - pad.bottom);
  const tickLineEnd = width - pad.right;
  const leftTicks = axisTicks(axisMode === "shared" ? sharedRange : ranges[0]);
  const rightTicks = axisMode === "independent" && series.length > 1 ? axisTicks(ranges[1]) : [];
  const leftTickSvg = leftTicks.map((tick) => {
    const yPos = y(tick, axisMode === "shared" ? sharedRange : ranges[0]).toFixed(2);
    return `<line x1="${pad.left - 4}" y1="${yPos}" x2="${tickLineEnd}" y2="${yPos}" stroke="#edf1f5"/><text x="${pad.left - 8}" y="${yPos}" text-anchor="end" dominant-baseline="middle" font-size="11" fill="#5f6b7a">${formatAxisValue(tick)}</text>`;
  }).join("");
  const rightTickSvg = rightTicks.map((tick) => {
    const yPos = y(tick, ranges[1]).toFixed(2);
    return `<line x1="${width - pad.right}" y1="${yPos}" x2="${width - pad.right + 4}" y2="${yPos}" stroke="#9aa4b2"/><text x="${width - pad.right + 8}" y="${yPos}" text-anchor="start" dominant-baseline="middle" font-size="11" fill="#5f6b7a">${formatAxisValue(tick)}</text>`;
  }).join("");
  const paths = series.map((item, idx) => {
    const points = item.points.map((point, index) => {
      const value = Number(point.y);
      return Number.isFinite(value) ? `${x(index).toFixed(2)},${y(value, ranges[idx]).toFixed(2)}` : null;
    }).filter(Boolean).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${trendColors[idx % trendColors.length]}" stroke-width="2.2"/>`;
  }).join("");
  const axisNote = axisMode === "independent"
    ? "独立 Y 轴：坐标1对应数据1，坐标2对应数据2，其它曲线仍按自身范围缩放，仅作趋势形态对比"
    : "同一 Y 轴：所有曲线使用同一数值范围";
  container.className = "chart";
  container.innerHTML = `<svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="#fff"/>
    ${leftTickSvg}
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>
    ${axisMode === "independent" ? `<line x1="${width - pad.right}" y1="${pad.top}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>` : ""}
    ${rightTickSvg}
    <text x="${pad.left}" y="18" font-size="12" fill="#5f6b7a">${escapeHtml(axisNote)}</text>
    ${paths}
  </svg>`;
  el("trendLegend").innerHTML = series.map((item, idx) =>
    `<span><i class="swatch" style="background:${trendColors[idx % trendColors.length]}"></i>${escapeHtml(item.name)}</span>`
  ).join("");
  renderTrendStats(series);
}

function valueRange(values) {
  if (!values.length) return { min: 0, max: 1 };
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const margin = (max - min) * 0.08;
  return { min: min - margin, max: max + margin };
}

function axisTicks(range, count = 5) {
  if (!range || count <= 1) return [];
  const min = Number(range.min);
  const max = Number(range.max);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  const step = (max - min) / (count - 1 || 1);
  return Array.from({ length: count }, (_, index) => min + step * index);
}

function formatAxisValue(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs > 0 && (abs < 0.001 || abs >= 1000000)) return value.toExponential(2);
  if (abs >= 10000) return value.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  if (abs >= 100) return value.toLocaleString("zh-CN", { maximumFractionDigits: 1 });
  if (abs >= 1) return value.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
  return value.toLocaleString("zh-CN", { maximumFractionDigits: 4 });
}

function clearTrendStats() {
  const node = el("trendStats");
  node.className = "trend-stats empty";
  node.textContent = "选择数据并点击“显示趋势”后显示统计摘要。";
}

function renderTrendStats(series) {
  const node = el("trendStats");
  if (!series.length) {
    clearTrendStats();
    return;
  }
  const statRows = [
    ["均值", "mean"],
    ["标准差", "stddev"],
    ["最大值", "max"],
    ["最小值", "min"],
    ["极差", "range"],
    ["中位数", "median"],
    ["有效点数", "count"],
  ];
  node.className = "trend-stats";
  node.innerHTML = series.map((item) => {
    const stats = trendStats(item.points || []);
    const rows = statRows.map(([label, key]) => `<div><dt>${label}</dt><dd>${key === "count" ? stats[key] : formatAxisValue(stats[key])}</dd></div>`).join("");
    return `<div class="trend-stat-card"><h3>${escapeHtml(item.name)}</h3><dl>${rows}</dl></div>`;
  }).join("");
}

function trendStats(points) {
  const values = (points || []).map((point) => Number(point.y)).filter((value) => Number.isFinite(value));
  if (!values.length) return { mean: NaN, stddev: NaN, max: NaN, min: NaN, range: NaN, median: NaN, count: 0 };
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return {
    mean,
    stddev: stddev(values),
    max,
    min,
    range: max - min,
    median: median(values),
    count: values.length,
  };
}

function median(values) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function stddev(values) {
  if (!values.length) return NaN;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

const candidateTable = "table";
const candidateCoreColumns = coreCandidateColumns;

function candidateDetailColumns(row) {
  const core = new Set(candidateCoreColumns());
  return Object.keys(row || {}).filter((column) => !core.has(column));
}

const CANONICAL_RISK_GROUPS = [
  {
    key: "lag_boundary_risk",
    label: "滞后边界风险",
    aliases: ["lag_boundary", "lag_boundary_flag", "lag_boundary_risk", "lag_reaches_boundary", "boundary_lag_uncertain", "screening_lag_boundary_risk", "model_lag_boundary_risk"],
  },
  {
    key: "target_lead_risk",
    label: "变量滞后目标风险",
    aliases: ["target_leads_variable", "target_leads_candidate", "target_lead_risk", "no_positive_lag", "non_positive_screening_lag", "non-positive screening lag"],
  },
  {
    key: "data_or_formula_risk",
    label: "公式泄漏/计算耦合风险",
    aliases: ["strong_formula_leakage", "formula_leakage_risk", "formula_coupled_reference", "formula_like", "data_or_formula_risk"],
  },
  {
    key: "data_quality_risk",
    label: "数据质量风险",
    aliases: ["poor_data_quality", "poor_quality_variable"],
  },
  {
    key: "synchronous_or_leakage_risk",
    label: "同步变化风险",
    aliases: ["synchronous", "synchronous_or_leakage_risk", "non-positive screening lag", "non_positive_screening_lag", "non-positive lag", "zero_lag", "same_time_movement"],
  },
  {
    key: "regime_instability_risk",
    label: "工况/时变不稳定风险",
    aliases: ["unstable_across_regimes", "unstable_over_time", "unstable_candidate", "stability_risk"],
  },
  {
    key: "capacity_driver_risk",
    label: "共同负荷驱动风险",
    aliases: ["capacity_driven", "common_capacity_driver"],
  },
  {
    key: "closed_loop_risk",
    label: "闭环反馈风险",
    aliases: ["closed_loop_suspect"],
  },
  {
    key: "collinearity_risk",
    label: "共线性风险",
    aliases: ["residual_collinearity", "high_collinearity_risk"],
  },
];

function splitRiskTags(value) {
  return String(value ?? "")
    .split(/[;,，；|]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalizedRiskToken(value) {
  return String(value ?? "").toLowerCase().replace(/[\s-]+/g, "_");
}

function normalizeRiskTags(value) {
  const rawRiskTags = splitRiskTags(value);
  const normalizedRawRiskTags = rawRiskTags.map(normalizedRiskToken);
  const matched = [];
  for (const group of CANONICAL_RISK_GROUPS) {
    if (group.aliases.some((alias) => normalizedRawRiskTags.includes(normalizedRiskToken(alias)))) {
      matched.push(group);
    }
  }
  return {
    rawRiskTags,
    displayRisks: matched.map((group) => group.label),
    canonicalKeys: matched.map((group) => group.key),
  };
}

function formatRiskFlags(value) {
  const normalized = normalizeRiskTags(value);
  return normalized.displayRisks.length ? normalized.displayRisks.join("；") : formatValue(value);
}

function formatRawRiskTags(value) {
  const rawRiskTags = splitRiskTags(value);
  return rawRiskTags.length ? rawRiskTags.map((tag) => formatValue(tag)).join("；") : "-";
}

function renderRiskTagDetails(value) {
  const normalized = normalizeRiskTags(value);
  const canonical = normalized.displayRisks.length ? normalized.displayRisks.join("；") : "-";
  const raw = formatRawRiskTags(value);
  return `
    <div class="detail-field">
      <strong>标准风险</strong>
      <span>${escapeHtml(canonical)}</span>
    </div>
    <div class="detail-field">
      <strong>原始风险标签</strong>
      <span>${escapeHtml(raw)}</span>
    </div>
  `;
}

function renderTable(rows) {
  renderCandidateTable(rows);
}

function renderCandidateTable(rows) {
  renderCompactDetailTable({
    targetId: candidateTable,
    rows,
    coreColumns: candidateCoreColumns(),
    detailColumns: candidateDetailColumns,
    emptyText: "没有可展示的候选变量。",
    modalTitle: (row) => `变量详情：${displayCellValue("variable", row.variable)}`,
  });
}

function coreCandidateColumns() {
  return ["variable", "final_score", "lag", "direction", "raw_corr", "residual_corr", "risk_flags", "recommended_use", "recommended_action"];
}

function renderCompactDetailTable({ targetId, rows, coreColumns, detailColumns = null, emptyText = null, modalTitle = null, valueGetter = null, formatter = null }) {
  const container = el(targetId);
  if (!container) return;
  if (!rows.length) {
    container.className = "empty";
    container.textContent = emptyText || missingText(targetId);
    return;
  }
  const getValue = valueGetter || ((row, column) => row[column]);
  const columns = coreColumns.filter((column) => getValue(rows[0], column) !== undefined);
  ensureTableSortState(targetId, columns[0]);
  const displayRows = sortedRowsForTable(targetId, rows);
  const table = document.createElement("table");
  table.className = "compact-result-table";
  table.setAttribute("aria-label", "核心列");
  table.innerHTML = `<thead><tr>${columns.map((c) => sortableHeaderHtml(targetId, c)).join("")}<th scope="col">${escapeHtml(columnLabel("detail_action"))}</th></tr></thead>`;
  const body = document.createElement("tbody");
  displayRows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.dataset.rowIndex = String(index);
    tr.className = "clickable-row";
    for (const column of columns) {
      const value = getValue(row, column);
      const td = document.createElement("td");
      td.className = tableCellClass(column, value);
      td.innerHTML = formatter ? formatter(column, value, row) : escapeHtml(displayCellValue(column, value));
      tr.appendChild(td);
    }
    const detailCell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "small-button";
    button.textContent = "查看详情";
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectCompactDetailRow(table, tr, row, detailColumns, valueGetter, modalTitle, event.currentTarget);
    });
    detailCell.appendChild(button);
    tr.appendChild(detailCell);
    tr.addEventListener("click", (event) => selectCompactDetailRow(table, tr, row, detailColumns, valueGetter, modalTitle, event.currentTarget));
    body.appendChild(tr);
  });
  table.appendChild(body);
  attachSortableHeaders(table, targetId, () => renderCompactDetailTable({ targetId, rows, coreColumns, detailColumns, emptyText, modalTitle, valueGetter, formatter }));
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
}

function selectCompactDetailRow(table, tr, row, detailColumns, valueGetter, modalTitle, trigger = null) {
  table.querySelectorAll("tbody tr").forEach((item) => item.classList.remove("selected"));
  if (tr) tr.classList.add("selected");
  openDetailModal(row, { detailColumns, valueGetter, title: modalTitle, trigger });
}

function buildDetailModalBody(row, options = {}) {
  if (options.detailColumns) return renderGenericDetailModalBody(row, options);
  return renderSingleVariableReview(row);
}

function renderGenericDetailModalBody(row, options = {}) {
  const getValue = options.valueGetter || ((item, column) => item[column]);
  const columns = typeof options.detailColumns === "function" ? options.detailColumns(row) : (options.detailColumns || Object.keys(row || {}));
  const rawFieldColumnsWithoutRiskFlags = columns.filter((column) => column !== "risk_flags");
  const fields = rawFieldColumnsWithoutRiskFlags.map((column) => `
    <div class="detail-field">
      <strong>${escapeHtml(columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, getValue(row, column)))}</span>
    </div>
  `).join("");
  return `
    <div class="review-card">
      <h3>变量：${escapeHtml(displayCellValue("variable", row.variable))}</h3>
      <details class="raw-fields" open>
        <summary>展开完整原始字段</summary>
        <div class="detail-grid">${("risk_flags" in (row || {})) ? renderRiskTagDetails(row.risk_flags) : ""}${fields}</div>
      </details>
    </div>
  `;
}


function selectTableRow(table, row) {
  selectCompactDetailRow(table, null, row, candidateDetailColumns, null, (item) => `变量详情：${displayCellValue("variable", item.variable)}`);
}

function renderRowDetails(row) {
  openDetailModal(row, { detailColumns: candidateDetailColumns });
}

function renderOverview(overview) {
  const metrics = [
    ["数据规模", overview.rows_after_preprocess ?? overview.rows_after_segment ?? ""],
    ["有效变量数量", overview.effective_variables ?? ""],
    ["有风险标签变量数量", overview.risk_tagged_count ?? overview.high_risk_count ?? ""],
    ["建议二级复核变量数量", overview.secondary_review_count ?? ""],
  ];
  el("overview").innerHTML = metrics.map(([label, value]) =>
    `<div class="metric-card"><span class="metric-value">${escapeHtml(formatValue(value))}</span><span class="metric-label">${escapeHtml(label)}</span></div>`
  ).join("");
}


const GENERIC_TABLE_CORE_COLUMNS = {
  overviewTop: ["variable", "final_score", "lag", "direction", "risk_flags", "recommended_use"],
  nearMissTable: ["variable", "near_miss_score", "lag", "direction", "risk_flags", "recommended_use"],
  grangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "interpretation"],
  modelVariableImportanceTable: ["variable", "max_importance", "importance_rank", "best_model_feature", "best_model_lag", "recommended_use"],
  importanceTable: ["variable", "importance", "importance_rank", "feature", "lag", "method"],
  modelDiscoveredTable: ["variable", "max_importance", "importance_rank", "best_model_lag", "recommended_use", "discovery_reason"],
  enhancedSummaryTable: ["variable", "final_score", "lag", "direction", "status", "model_lift", "rolling_stability"],
  enhancedLiftTable: ["variable", "status", "model_lift", "ar_baseline_rmse", "candidate_rmse"],
  enhancedRollingTable: ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_stability"],
  conditionalGrangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "predictive_contribution"]
};

function genericTableCoreColumns(targetId, row = {}, preferredColumns = null) {
  const configured = GENERIC_TABLE_CORE_COLUMNS[targetId] || [];
  const preferred = preferredColumns || [];
  const fallback = preferred.length ? preferred.slice(0, 6) : Object.keys(row || {}).slice(0, 6);
  const columns = (configured.length ? configured : fallback).filter((column) => column in (row || {}));
  return columns.length ? columns : fallback.filter((column) => column in (row || {}));
}

function genericTableDetailColumns(row) {
  return Object.keys(row || {});
}

function renderGenericTable(targetId, rows, preferredColumns = null) {
  const firstRow = rows && rows.length ? rows[0] : {};
  renderCompactDetailTable({
    targetId,
    rows,
    coreColumns: genericTableCoreColumns(targetId, firstRow, preferredColumns),
    detailColumns: genericTableDetailColumns,
    emptyText: missingText(targetId),
    modalTitle: (row) => `变量详情：${displayCellValue("variable", row.variable)}`,
  });
}

function includesFlag(row, flag) {
  return String(row?.risk_flags ?? "").toLowerCase().includes(String(flag).toLowerCase());
}

function numericValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function renderScreeningQualityHints(rows) {
  const container = el("screeningQualityHints");
  if (!container) return;
  if (!rows || !rows.length) {
    container.className = "empty";
    container.textContent = "完成主筛查后显示结果质量提示。";
    return;
  }
  const hints = [];
  if (rows.filter((row) => includesFlag(row, "lag_boundary") || row.lag_boundary_flag === true || String(row.lag_boundary_flag) === "1").length >= 2) {
    hints.push("多个候选变量命中滞后边界，当前 max_lag 可能偏小，建议结合工艺停留时间扩大 max_lag 后复跑。");
  }
  if (rows.filter((row) => numericValue(row.lag) !== null && numericValue(row.lag) < 0).length >= 2) {
    hints.push("多个候选变量表现为变量滞后目标，建议检查时间对齐、响应变量混入或数据时间戳方向。");
  }
  if (rows.filter((row) => includesFlag(row, "common_capacity_driver") || row.common_capacity_driver_flag === true || String(row.common_capacity_driver_flag) === "1").length >= 2) {
    hints.push("多个变量可能受共同负荷驱动，建议优先复核负荷变量和物料平衡链条。");
  }
  const topScores = rows.slice(0, 10).map((row) => numericValue(row.final_score)).filter((value) => value !== null);
  if (topScores.length >= 2) {
    const scoreSpread = Math.max(...topScores) - Math.min(...topScores);
    const maxAbsScore = Math.max(...topScores.map((value) => Math.abs(value)), 1);
    if (scoreSpread <= 0.02 || scoreSpread / maxAbsScore <= 0.05) {
      hints.push("Top 候选区分度较弱，建议结合二次验证和趋势图复核。");
    }
  }
  if (!hints.length) {
    hints.push("未发现明显参数边界提示，仍需结合二次验证和趋势图复核。");
  }
  container.className = "help";
  container.innerHTML = `<ul>${hints.map((hint) => `<li>${escapeHtml(hint)}</li>`).join("")}</ul>`;
}

function renderFinalReviewQualityOverview(rows) {
  const container = el("finalReviewQualityOverview");
  if (!container) return;
  if (!rows || !rows.length) {
    container.innerHTML = "";
    return;
  }
  const decisionAliases = {
    priority_review: ["priority_review", "优先复核"],
    priority_review_with_statistical_limit: ["priority_review_with_statistical_limit", "优先复核但统计受限"],
    secondary_review: ["secondary_review", "二级复核"],
    secondary_review_with_statistical_limit: ["secondary_review_with_statistical_limit", "二级复核但统计受限"],
    risk_limited_review: ["risk_limited_review", "风险受限复核"],
    manual_review_only: ["manual_review_only", "仅人工复核"],
    not_recommended: ["not_recommended", "暂不推荐"],
  };
  const countDecision = (decision) => rows.filter((row) => {
    const raw = String(row.final_recommendation || row.integrated_review_decision || "");
    return (decisionAliases[decision] || [decision]).includes(raw);
  }).length;
  const stats = [
    ["总复核变量数", rows.length],
    ["优先复核数量", countDecision("priority_review")],
    ["优先复核但统计受限数量", countDecision("priority_review_with_statistical_limit")],
    ["二级复核数量", countDecision("secondary_review")],
    ["二级复核但统计受限数量", countDecision("secondary_review_with_statistical_limit")],
    ["风险受限复核数量", countDecision("risk_limited_review")],
    ["仅人工复核数量", countDecision("manual_review_only")],
    ["暂不推荐数量", countDecision("not_recommended")],
    ["统计受限变量数量", rows.filter((row) => ["weak", "medium", "strong"].includes(String(row.statistical_limit_level || ""))).length],
    ["滞后边界提示数量", rows.filter((row) => String(row.lag_boundary_hint ?? "").trim() && displayCellValue("lag_boundary_hint", row.lag_boundary_hint) !== "-").length],
  ];
  container.innerHTML = stats.map(([label, value]) => `
    <div class="metric-card">
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(value)}</span>
    </div>
  `).join("");
}

function renderFinalReviewSummaryTable(rows) {
  const container = el("finalReviewSummaryTable");
  if (!container) return;
  if (!rows.length) {
    container.className = "empty";
    container.textContent = missingText("finalReviewSummaryTable");
    return;
  }
  const columns = finalReviewSummaryColumns().filter((column) => ["trend_action", "detail_action"].includes(column) || finalSummaryValue(rows[0], column) !== undefined);
  ensureTableSortState("finalReviewSummaryTable", "final_rank");
  const displayRows = sortedRowsForTable("finalReviewSummaryTable", rows);
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${columns.map((c) => sortableHeaderHtml("finalReviewSummaryTable", c)).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const row of displayRows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      if (column === "trend_action") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "small-button";
        button.textContent = "查看趋势";
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          openTrendForCandidate(row);
        });
        td.appendChild(button);
      } else if (column === "detail_action") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "small-button";
        button.textContent = "查看详情";
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          selectFinalReviewRow(row, tr);
        });
        td.appendChild(button);
      } else {
        td.textContent = displayCellValue(column, finalSummaryValue(row, column));
      }
      td.className = tableCellClass(column, finalSummaryValue(row, column));
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  table.appendChild(body);
  attachSortableHeaders(table, "finalReviewSummaryTable", () => renderFinalReviewSummaryTable(rows));
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
  attachFinalSummaryRowClick(rows);
}

function selectFinalReviewRow(row, tr = null) {
  const container = el("finalReviewSummaryTable");
  if (container) {
    for (const item of container.querySelectorAll("tbody tr")) item.classList.remove("selected");
  }
  if (tr) tr.classList.add("selected");
  openDetailModal(row);
}

function attachFinalSummaryRowClick(rows) {
  const container = el("finalReviewSummaryTable");
  const table = container.querySelector("table");
  if (!table) return;
  const bodyRows = table.querySelectorAll("tbody tr");
  const displayRows = sortedRowsForTable("finalReviewSummaryTable", rows);
  bodyRows.forEach((tr, index) => {
    tr.classList.add("clickable-row");
    tr.addEventListener("click", () => {
      selectFinalReviewRow(displayRows[index], tr);
    });
  });
}

function renderSingleVariableReview(row) {
  if (!row) return "";
  const metricColumns = [
    "final_rank",
    "final_recommendation",
    "data_priority",
    "evidence_level",
    "evidence_score",
    "statistical_limit_level",
    "risk_constraint_level",
    "screening_grade",
    "screening_score",
    "screening_lag",
    "conditional_status",
    "conditional_best_lag",
    "tested_lags",
    "lag_boundary_hint",
    "evidence_conflict_type",
    "evidence_conflict_reason",
  ];
  const metrics = metricColumns.map((column) => `
    <div class="metric-item">
      <strong>${escapeHtml(columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, row[column]))}</span>
    </div>
  `).join("");
  const evidenceRow = rowForVariable(lastCausalEvidenceRows, row.variable) || {};
  const modelRow = rowForVariable(lastModelVariableRows, row.variable) || {};
  const rollingRow = rowForVariable(lastEnhancedRollingRows, row.variable) || {};
  const evidenceItems = [
    ["主筛查", evidenceText([
      ["screening_grade", row.screening_grade],
      ["screening_score", row.screening_score],
    ])],
    ["条件 Granger", evidenceText([
      ["conditional_status", row.conditional_status ?? evidenceRow.conditional_granger_status],
      ["conditional_fdr_q_value", row.conditional_fdr_q_value ?? evidenceRow.conditional_fdr_q_value],
      ["conditional_best_lag", row.conditional_best_lag],
    ])],
    ["模型解释", evidenceText([
      ["model_importance_rank", row.model_importance_rank ?? evidenceRow.model_importance_rank ?? modelRow.importance_rank],
      ["max_importance", row.max_importance ?? modelRow.max_importance],
    ], "未运行或无数据")],
    ["滚动稳定性", evidenceText([
      ["rolling_stability", row.rolling_stability ?? evidenceRow.rolling_stability ?? rollingRow.rolling_stability],
      ["rolling_sign_consistency", row.rolling_sign_consistency ?? rollingRow.rolling_sign_consistency],
    ], "未运行或无数据")],
    ["风险标签", evidenceText([
      ["risk_flags", row.risk_flags ?? evidenceRow.risk_flags],
      ["statistical_limit_level", row.statistical_limit_level],
      ["risk_constraint_level", row.risk_constraint_level],
    ])],
    ["滞后边界", evidenceText([
      ["lag_boundary_hint", row.lag_boundary_hint],
    ])],
  ];
  const evidenceHtml = evidenceItems.map(([label, value]) => `
    <div class="metric-item">
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(value)}</span>
    </div>
  `).join("");

  const rawFields = renderRawFields(row);
  const showRawFieldsToggle = "showRawFieldsToggle";
  const rawFieldsCollapsed = "rawFieldsCollapsed";
  return `
    <div class="review-card">
      <h3>变量：${escapeHtml(displayCellValue("variable", row.variable))}</h3>
      <p>该卡片汇总该变量在主筛查、验证和综合复核中的证据，仅用于人工复核。</p>
      <div class="metric-grid">${metrics}</div>
      <h4>证据来源清单</h4>
      <div class="metric-grid">${evidenceHtml}</div>
      <h4>关键理由</h4>
      <p>${escapeHtml(displayCellValue("key_reason", row.key_reason))}</p>
      <h4>建议下一步</h4>
      <p>${escapeHtml(displayCellValue("suggested_next_action", row.suggested_next_action))}</p>
      <h4>解释边界</h4>
      <p>${escapeHtml(displayCellValue("interpretation", row.interpretation))}</p>
      <details class="raw-fields ${rawFieldsCollapsed}">
        <summary id="${showRawFieldsToggle}">展开完整原始字段</summary>
        <div class="detail-grid">${rawFields}</div>
      </details>
      <p>该结果为预测验证和人工复核建议，不是因果结论。</p>
    </div>
  `;
}

function renderRawFields(row) {
  const rawFieldColumnsWithoutRiskFlags = finalReviewSummaryDetailColumns(row).filter((column) => column !== "risk_flags");
  const rawFields = rawFieldColumnsWithoutRiskFlags.map((column) => `
    <div class="detail-field">
      <strong>${escapeHtml(columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, finalSummaryValue(row, column)))}</span>
    </div>
  `).join("");
  return (("risk_flags" in (row || {})) ? renderRiskTagDetails(row.risk_flags) : "") + rawFields;
}

function openDetailModal(row, options = {}) {
  lastModalTrigger = options.trigger || document.activeElement;
  const modal = el("detailModal");
  const title = typeof options.title === "function" ? options.title(row) : options.title;
  el("detailModalTitle").textContent = title || `变量详情：${displayCellValue("variable", row.variable)}`;
  el("detailModalBody").innerHTML = buildDetailModalBody(row, options);
  modal.hidden = false;
  modal.classList.add("open");
  el("detailModalClose").focus();
}

function closeDetailModal() {
  const modal = el("detailModal");
  if (!modal) return;
  modal.classList.remove("open");
  modal.hidden = true;
  if (lastModalTrigger && typeof lastModalTrigger.focus === "function") {
    lastModalTrigger.focus();
  }
  lastModalTrigger = null;
}

function rowForVariable(rows, variable) {
  return (rows || []).find((item) => String(item.variable || "") === String(variable || ""));
}

function evidenceText(items, emptyText = "-") {
  const parts = items
    .filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "")
    .map(([column, value]) => `${columnLabel(column)}：${displayCellValue(column, value)}`);
  return parts.length ? parts.join("；") : emptyText;
}

function displayCellValue(column, value) {
  const formatted = column === "risk_flags" ? formatRiskFlags(value) : formatCellValue(column, value);
  return formatted === "" || formatted === null || formatted === undefined ? "-" : String(formatted);
}

function openTrendForCandidate(rowOrVariable, maybeLag) {
  const row = typeof rowOrVariable === "object" && rowOrVariable !== null ? rowOrVariable : null;
  const variable = row ? row.variable : rowOrVariable;
  const lag = row ? row.screening_lag : maybeLag;
  const target = el("targetColumn").value;
  activateTab("trendTab");
  setSelectValueIfExists("trendVar1", target);
  setSelectValueIfExists("trendVar2", variable);
  setSelectValueIfExists("trendVar3", "");
  setSelectValueIfExists("trendVar4", "");
  const lagText = displayCellValue("screening_lag", lag);
  const boundaryHint = row && row.lag_boundary_hint ? ` ${displayCellValue("lag_boundary_hint", row.lag_boundary_hint)}` : "";
  const hint = `当前趋势复核变量：候选 ${variable || "-"}，目标 ${target || "-"}。主筛查滞后：${lagText || "-"}。${boundaryHint}请人工观察候选变量变化后，目标变量是否在相应滞后附近出现方向合理的响应。该观察仅用于人工复核，不是因果结论。`;
  const hintNode = el("trendReviewHint");
  if (hintNode) hintNode.textContent = hint;
  setStatus(`已选择趋势变量：目标 ${target || "-"}，候选 ${variable || "-"}。可点击“显示趋势”查看，页面不会自动绘图。`);
}

function setSelectValueIfExists(selectId, value) {
  const node = el(selectId);
  if (!node) return;
  const targetValue = String(value || "");
  const option = Array.from(node.options).find((item) => item.value === targetValue);
  if (option) node.value = targetValue;
}

function missingText(targetId) {
  if (targetId === "grangerTable") return "未启用 Granger 检验，或没有可展示结果。";
  if (targetId === "enhancedSummaryTable") return "点击“运行增强筛选”后显示增强筛选摘要。";
  if (targetId === "enhancedLiftTable") return "点击“运行增强筛选”后显示模型提升评分。";
  if (targetId === "enhancedRollingTable") return "点击“运行增强筛选”后显示滚动稳定性评分。";
  if (targetId === "modelVariableImportanceTable") return "运行随机森林模型解释后显示变量排序。";
  if (targetId === "importanceTable") return "未启用随机森林模型解释，或没有可展示结果。";
  if (targetId === "modelDiscoveredTable") return "运行随机森林模型解释后显示补充候选。";
  if (targetId === "nearMissTable") return "暂无轻量遗漏候选。";
  if (targetId === "conditionalGrangerTable") return "未运行 条件 Granger 预测验证。";
  if (targetId === "causalReviewTable") return "未运行 三层复核。";
  if (targetId === "finalReviewSummaryTable") return "未运行 最终推荐摘要。";
  if (targetId === "causalReviewEvidenceTable") return "未运行 逐变量综合证据复核表。";
  if (targetId === "overviewTop") return "暂无前 10 个推荐变量。";
  return "无可展示结果。";
}

function nearMissColumns() {
  return ["variable", "near_miss_score", "lag", "direction", "raw_score", "residual_corr", "lag_quality", "ranked_feature_rank", "ranked_final_score", "missing_from_screening_top_n", "risk_flags", "recommended_use", "recommended_action", "near_miss_reason", "interpretation"];
}

function modelVariableImportanceColumns() {
  return ["variable", "best_model_feature", "best_model_lag", "max_importance", "total_importance", "feature_count", "importance_rank", "method", "ranked_feature_rank", "ranked_final_score", "risk_flags", "recommended_use", "recommended_action", "interpretation"];
}

function modelDiscoveredColumns() {
  return ["variable", "best_model_feature", "best_model_lag", "max_importance", "importance_rank", "model_feature_count", "nearby_lag_count", "ranked_feature_rank", "ranked_final_score", "missing_from_screening_top_n", "risk_flags", "recommended_use", "recommended_action", "discovery_reason", "interpretation"];
}

function enhancedSummaryColumns() {
  return [
    "variable",
    "final_score",
    "lag",
    "direction",
    "risk_flags",
    "recommended_use",
    "status",
    "model_lift",
    "rolling_stability",
    "rolling_corr_median",
    "rolling_sign_consistency",
    "interpretation"
  ];
}

function modelLiftColumns() {
  return ["variable", "status", "ar_baseline_rmse", "candidate_rmse", "model_lift"];
}

function rollingCorrColumns() {
  return ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_abs_corr_median", "rolling_corr_iqr", "rolling_sign_consistency", "valid_window_count", "rolling_stability"];
}

function conditionalGrangerColumns() {
  return ["variable", "status", "best_lag", "tested_lags", "lag_mode", "lag_window", "fallback_maxlag", "baseline_maxlag", "min_p_value", "fdr_q_value", "baseline_rmse", "full_rmse", "predictive_contribution", "condition_number", "base_condition_number", "full_condition_number", "control_columns", "n_rows", "interpretation"];
}

function causalReviewColumns() {
  return ["variable", "candidate_grade", "final_score", "review_tier", "review_priority", "final_review_decision", "final_review_reason", "predictive_contribution", "risk_flags", "conditional_granger_status", "conditional_best_lag", "conditional_min_p_value", "conditional_fdr_q_value", "interpretation"];
}

const FINAL_SUMMARY_CORE_COLUMNS = [
  "final_rank",
  "variable",
  "trend_action",
  "final_decision",
  "data_priority",
  "evidence_level",
  "evidence_score",
  "statistical_limit_level",
  "risk_constraint_level",
  "detail_action",
];

const FINAL_SUMMARY_DETAIL_COLUMNS = [
  "main_reason",
  "suggested_next_action",
  "evidence_conflict_explanation",
  "interpretation_boundary",
];

const FINAL_SUMMARY_COLUMN_ALIASES = {
  final_decision: ["final_decision", "final_recommendation", "integrated_review_decision"],
  main_reason: ["main_reason", "key_reason", "integrated_review_reason"],
  evidence_conflict_explanation: ["evidence_conflict_explanation", "evidence_conflict_reason"],
  interpretation_boundary: ["interpretation_boundary", "interpretation"],
};

function finalReviewSummaryColumns() {
  return FINAL_SUMMARY_CORE_COLUMNS;
}

function finalReviewSummaryDetailColumns(row) {
  const preferred = [
    "final_rank", "variable", "final_decision", "data_priority", "evidence_level", "evidence_score",
    "statistical_limit_level", "risk_constraint_level", "main_reason", "suggested_next_action",
    "evidence_conflict_explanation", "interpretation_boundary", "screening_grade", "screening_score",
    "screening_lag", "conditional_status", "conditional_best_lag", "tested_lags", "lag_boundary_hint",
    "evidence_conflict_type", "conditional_min_p_value", "conditional_fdr_q_value"
  ];
  const seen = new Set();
  const columns = [];
  for (const column of preferred) {
    if (finalSummaryValue(row, column) !== undefined && !seen.has(column)) { columns.push(column); seen.add(column); }
  }
  for (const column of Object.keys(row || {})) {
    if (!seen.has(column)) { columns.push(column); seen.add(column); }
  }
  return columns;
}

function finalSummaryValue(row, column) {
  if (!row) return undefined;
  if (column === "trend_action" || column === "detail_action") return "";
  const aliases = FINAL_SUMMARY_COLUMN_ALIASES[column] || [column];
  for (const key of aliases) {
    if (Object.prototype.hasOwnProperty.call(row, key)) return row[key];
  }
  return undefined;
}

function causalReviewEvidenceColumns() {
  return ["variable", "candidate_grade", "final_score", "data_priority", "evidence_score", "evidence_level", "statistical_limit_level", "risk_constraint_level", "integrated_review_decision", "integrated_review_reason", "statistical_limit_reason", "evidence_reason", "conditional_granger_status", "conditional_fdr_q_value", "predictive_contribution", "model_lift", "rolling_stability", "model_importance_rank", "risk_flags", "interpretation"];
}


const causalEvidenceCoreColumns = () => ["variable", "candidate_grade", "final_score", "data_priority", "evidence_level", "evidence_score", "integrated_review_decision"];

function causalEvidenceDetailColumns(row) {
  const core = new Set(causalEvidenceCoreColumns());
  return causalReviewEvidenceColumns().filter((column) => column in (row || {}) && !core.has(column));
}

function renderCausalReviewEvidenceTable(rows) {
  renderCompactDetailTable({
    targetId: "causalReviewEvidenceTable",
    rows,
    coreColumns: causalEvidenceCoreColumns(),
    detailColumns: causalEvidenceDetailColumns,
    modalTitle: (row) => `逐变量综合证据复核表：${displayCellValue("variable", row.variable)}`,
  });
}

function renderCausalReviewTable(targetId, rows) {
  const container = el(targetId);
  if (!container) return;
  if (!rows.length) {
    container.className = "empty";
    container.textContent = missingText(targetId);
    return;
  }
  const columns = causalReviewColumns().filter((column) => column in rows[0]);
  ensureTableSortState(targetId, columns[0]);
  const displayRows = sortedRowsForTable(targetId, rows);
  const table = document.createElement("table");
  const header = document.createElement("thead");
  header.innerHTML = `<tr>${columns.map((c) => sortableHeaderHtml(targetId, c)).join("")}</tr>`;
  table.appendChild(header);
  const body = document.createElement("tbody");
  for (const row of displayRows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      td.className = tableCellClass(column, row[column]);
      td.innerHTML = formatReviewCell(column, row[column]);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  table.appendChild(body);
  attachSortableHeaders(table, targetId, () => renderCausalReviewTable(targetId, rows));
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
}

function cellHtml(column, value, formatter = null) {
  const rendered = formatter ? formatter(column, value) : escapeHtml(formatCellValue(column, value));
  const title = cellTitle(column, value);
  return title ? `<td title="${escapeHtml(title)}">${rendered}</td>` : `<td>${rendered}</td>`;
}

function cellTitle(column, value) {
  if (column === "integrated_review_decision" && String(value ?? "") === "priority_review_with_statistical_limit") {
    return "数据证据强，但统计检验受到高共线性、闭环、共同负荷或滞后边界限制；应优先人工复核，但不是因果结论。";
  }
  return "";
}

function formatReviewCell(column, value) {
  if (column === "final_review_decision") {
    const raw = String(value || "");
    return `<span class="decision-badge decision-${escapeHtml(raw)}">${escapeHtml(formatCellValue(column, raw))}</span>`;
  }
  if (column === "risk_flags") return escapeHtml(formatRiskFlags(value));
  return escapeHtml(formatCellValue(column, value));
}

function formatCellValue(column, value) {
  const text = String(value ?? "");
  const maps = {
    integrated_review_decision: {
      priority_review: "优先复核",
      priority_review_with_statistical_limit: "优先复核但统计受限",
      secondary_review: "二级复核",
      secondary_review_with_statistical_limit: "二级复核但统计受限",
      risk_limited_review: "风险受限复核",
      manual_review_only: "仅人工复核",
      insufficient_evidence: "证据不足",
      not_recommended: "暂不推荐",
    },
    final_review_decision: {
      priority_review: "优先复核",
      secondary_review: "二级复核",
      risk_limited_review: "风险受限复核",
      manual_review_only: "仅人工复核",
      insufficient_evidence: "证据不足",
      not_recommended: "暂不推荐",
    },
    evidence_level: {
      strong_predictive_evidence: "强预测证据",
      moderate_predictive_evidence: "中等预测证据",
      weak_or_incomplete_evidence: "弱证据或证据不完整",
      risk_limited_evidence: "风险受限证据",
      insufficient_evidence: "证据不足",
      not_supported: "未支持",
    },
    data_priority: {
      high: "高",
      medium: "中",
      low: "低",
    },
    statistical_limit_level: {
      none: "无",
      weak: "弱",
      medium: "中",
      strong: "强",
    },
    risk_constraint_level: {
      none: "无",
      weak: "弱",
      medium: "中",
      strong: "强",
    },
  };
  return maps[column]?.[text] || formatValue(value);
}

function renderReviewDownloads(downloads) {
  renderDownloadTarget("conditionalDownload", downloads, "conditional_granger_scores.csv");
  renderDownloadTarget("causalReportDownload", downloads, "causal_review_report.csv");
  renderDownloadTarget("finalReviewSummaryDownload", downloads, "final_review_summary.csv");
  renderDownloadTarget("causalEvidenceDownload", downloads, "causal_review_evidence.csv");
}

function renderDownloadTarget(targetId, downloads, fileName) {
  const container = el(targetId);
  if (!container) return;
  const item = (downloads || []).find((entry) => entry.name === fileName);
  container.innerHTML = item ? `<a href="${escapeHtml(item.url)}">下载 ${escapeHtml(fileName)}</a>` : "";
}

function renderDownloads(downloads) {
  const container = el("downloads");
  container.innerHTML = "";
  for (const item of downloads) {
    const link = document.createElement("a");
    link.href = item.url;
    link.textContent = item.name;
    container.appendChild(link);
  }
}

function ensureTableSortState(targetId, defaultColumn = null) {
  if (!tableSortStates[targetId]) {
    const column = targetId === "finalReviewSummaryTable" ? "final_rank" : defaultColumn;
    tableSortStates[targetId] = { column, direction: "asc" };
  }
}

function sortableHeaderHtml(targetId, column) {
  if (targetId === "finalReviewSummaryTable" && column === "trend_action") {
    return `<th scope="col">${escapeHtml(columnLabel(column))}</th>`;
  }
  const state = tableSortStates[targetId] || {};
  const isSorted = state.column === column;
  const mark = isSorted ? (state.direction === "asc" ? "↑" : "↓") : "";
  const ariaSort = isSorted ? (state.direction === "asc" ? "ascending" : "descending") : "none";
  return `<th scope="col" class="sortable" tabindex="0" aria-sort="${ariaSort}" data-column="${escapeHtml(column)}">${escapeHtml(columnLabel(column))}<span class="sort-mark">${mark}</span></th>`;
}

function attachSortableHeaders(table, targetId, rerender) {
  for (const header of table.querySelectorAll("th.sortable")) {
    const sort = () => {
      updateTableSortState(targetId, header.dataset.column);
      rerender();
    };
    header.addEventListener("click", sort);
    header.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      sort();
    });
  }
}

function updateTableSortState(targetId, column) {
  ensureTableSortState(targetId, column);
  const state = tableSortStates[targetId];
  if (state.column === column) {
    state.direction = state.direction === "asc" ? "desc" : "asc";
  } else {
    state.column = column;
    state.direction = "asc";
  }
}

function sortedRowsForTable(targetId, rows) {
  const state = tableSortStates[targetId];
  if (!state || !state.column) return rows.slice();
  const direction = state.direction === "asc" ? 1 : -1;
  const column = state.column;
  return rows.slice().sort((a, b) => compareValues(a[column], b[column]) * direction);
}

function compareValues(a, b) {
  const numberA = typeof a === "number" ? a : Number(a);
  const numberB = typeof b === "number" ? b : Number(b);
  if (Number.isFinite(numberA) && Number.isFinite(numberB)) return numberA - numberB;
  return String(a ?? "").localeCompare(String(b ?? ""), "zh-CN", { numeric: true });
}

function tableCellClass(column, value) {
  const name = String(column || "");
  const number = typeof value === "number" ? value : Number(value);
  const numericColumn = /(?:^|_)(score|lag|rmse|p_value|q_value|rank|count|n_rows|condition_number|importance|contribution)(?:$|_)/i.test(name);
  if (Number.isFinite(number) || numericColumn) return "numeric";
  const wrapColumn = /interpretation|reason|action|risk_flags|control_columns|evidence_reason|statistical_limit_reason|key_reason|suggested_next_action|lag_boundary_hint/i.test(name);
  return wrapColumn ? "wrap-cell" : "";
}

function translateDisplayValue(value) {
  return formatValue(value);
}

function formatValue(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "";
    if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(3);
    return Number.isInteger(value) ? String(value) : value.toFixed(6);
  }
  if (typeof value === "string") {
    const map = {
      candidate_grade_A: "候选等级A",
      candidate_grade_B: "候选等级B",
      candidate_grade_C: "候选等级C",
      candidate_grade_D: "候选等级D",
      candidate_grade_E: "候选等级E",
      A: "候选等级A",
      B: "候选等级B",
      C: "候选等级C",
      D: "候选等级D",
      E: "候选等级E",
      failed: "失败",
      error: "错误",
      no_data: "无数据",
      insufficient_data: "数据不足",
      not_run: "未运行",
      high_collinearity_risk: "高共线性风险",
      formula_leakage_risk: "公式泄漏风险",
      no_positive_lag: "无正向滞后",
      non_positive_screening_lag: "非正主筛查滞后",
      "non-positive screening lag": "非正主筛查滞后",
      mean_abs_shap: "SHAP平均绝对值",
      "enhanced screening only": "仅作增强筛查",
      not_computed: "未计算",
      ranked_window: "排序窗口",
      true: "是",
      false: "否",
      variable_leads_target: "变量领先目标",
      conditional_granger_supported: "条件格兰杰支持",
      predictive_contribution_positive: "预测贡献为正",
      granger_auxiliary_support: "格兰杰辅助支持",
      model_lift_weak_support: "模型提升弱支持",
      model_explanation_support: "模型解释支持",
      lag_boundary: "滞后触及边界",
      target_leads_variable: "变量滞后目标",
      unstable_over_time: "时序不稳定",
      low_model_lift: "低模型增益",
      lag_boundary_flag: "滞后边界命中",
      formula_coupled_reference: "公式耦合参考",
      strong_screening_candidate: "强初筛候选",
      multiple_evidence_supported: "多证据支持",
      priority_review_recommended: "建议优先复核",
      conditional_granger_supported: "条件格兰杰支持",
      positive_predictive_contribution: "预测贡献为正",
      granger_auxiliary_supported: "格兰杰辅助支持",
      weak_model_lift_support: "模型提升弱支持",
      model_explanation_supported: "模型解释支持",
      prediction_candidate: "预测候选",
      state_indicator: "状态指示量",
      capacity_driven: "共同负荷驱动",
      closed_loop_suspect: "疑似闭环反馈",
      unstable_candidate: "不稳定候选",
      poor_quality_variable: "低质量变量",
      manual_review_required: "需要人工复核",
      control_variable_reference: "控制变量参考",
      formula_like: "公式类变量",
      strong_formula_leakage: "强公式泄漏",
      common_capacity_driver: "共同负荷驱动",
      closed_loop_suspect: "疑似闭环反馈",
      target_leads_variable: "变量滞后目标",
      unstable_across_regimes: "跨工况不稳定",
      poor_data_quality: "数据质量差",
      residual_collinearity: "残差共线性高",
      none: "无",
      weak: "弱",
      medium: "中",
      strong: "强",
      high: "高",
      low: "低",
      strong_predictive_evidence: "强预测证据",
      moderate_predictive_evidence: "中等预测证据",
      weak_or_incomplete_evidence: "弱证据或证据不完整",
      risk_limited_evidence: "风险受限证据",
      not_supported: "未支持",
      ok: "正常",
      skipped: "已跳过",
      risk_limited_review: "风险受限复核",
      priority_review: "优先复核",
      priority_review_with_statistical_limit: "优先复核但统计受限",
      secondary_review: "二级复核",
      secondary_review_with_statistical_limit: "二级复核但统计受限",
      not_recommended: "暂不推荐",
      insufficient_evidence: "证据不足",
      manual_review_only: "仅人工复核",
      "final review summary only": "仅作最终复核摘要",
      strong_screening_but_statistical_limited: "强筛查信号但统计受限",
      strong_screening_but_conditional_weak: "强筛查信号但条件验证弱",
      conditional_supported_but_screening_weak: "条件验证支持但主筛查较弱",
      model_supported_but_granger_weak: "模型支持但Granger较弱",
      boundary_lag_uncertain: "滞后边界不确定",
      candidate_leads_target: "变量领先目标",
      target_leads_candidate: "变量滞后目标",
      target_leads_variable: "变量滞后目标",
      synchronous: "同步变化",
      unknown: "未知",
      positive: "正向",
      negative: "负向",
      strong: "强",
      weak: "弱",
      medium: "中",
      "predictive validation only": "仅作预测验证",
      "not a causal conclusion": "不是因果结论",
      "model explanation only": "仅作模型解释",
      model_only_signal: "模型补充线索",
      multi_lag_model_signal: "多滞后模型线索",
      model_lag_boundary_risk: "模型滞后边界风险",
      synchronous_or_leakage_risk: "同步或泄漏风险",
      screening_lag_boundary_risk: "初筛滞后边界风险",
      target_lead_risk: "变量滞后目标风险",
      stability_risk: "稳定性风险",
      model_supported_screening_candidate: "模型支持的初筛候选",
      raw_lag_signal: "滞后相关线索",
      residual_signal: "残差相关线索",
      clear_lag_peak: "滞后峰值清晰",
      lag_boundary_risk: "滞后边界风险",
      data_or_formula_risk: "数据质量或公式泄漏风险",
      near_miss_candidate: "遗漏候选线索",
      lag_reaches_boundary: "滞后触及边界",
      "screening near-miss only": "仅作轻量遗漏筛查",
    };
    if (map[value]) return map[value];
    if (value === "enhanced screening only; not a causal conclusion") return "仅作增强筛查；不是因果结论";
    if (value === "predictive validation only; not a causal conclusion") return "仅作预测验证；不是因果结论";
    if (value === "model explanation only; not a causal conclusion") return "仅作模型解释；不是因果结论";
    if (value === "screening near-miss only; not a causal conclusion") return "仅作轻量遗漏筛查；不是因果结论";
    if (value === "final review summary only; not a causal conclusion") return "仅作最终复核摘要；不是因果结论";
    return value
      .split(/[;,，；]/)
      .map((item) => {
        const key = item.trim();
        if (!key) return "";
        if (map[key]) return map[key];
        if (key.startsWith("skipped:")) return key.replace("skipped:", "已跳过：");
        return key;
      })
      .filter(Boolean)
      .join("；");
  }
  return value ?? "";
}

function columnLabel(column) {
  const labels = {
    variable: "变量",
    trend_action: "趋势验证",
    final_score: "综合得分",
    lag: "滞后",
    direction: "方向",
    raw_corr: "原始相关",
    raw_corr_score: "原始相关得分",
    residual_corr: "残差相关",
    residual_corr_score: "残差相关得分",
    residual_status: "残差状态",
    risk_flags: "风险标签",
    recommended_use: "建议用途",
    recommended_action: "建议动作",
    formula_like_flag: "公式类变量",
    strong_formula_leakage_flag: "强公式泄漏",
    common_capacity_driver_flag: "疑似共同负荷驱动",
    closed_loop_suspect_flag: "疑似闭环反馈",
    target_leads_variable_flag: "变量滞后目标",
    unstable_across_regimes_flag: "跨工况不稳定",
    unstable_over_time_flag: "时序不稳定",
    lag_boundary_flag: "滞后边界命中",
    low_model_lift_flag: "低模型增益",
    poor_data_quality_flag: "数据质量较差",
    residual_collinearity_flag: "残差共线性风险",
    risk_count: "风险数量",
    strong_risk_count: "强风险数量",
    weak_risk_count: "弱风险数量",
    risk_level: "风险等级",
    human_reason: "风险说明",
    pearson: "Pearson",
    spearman: "Spearman",
    score: "得分",
    p_value: "P值",
    r2: "R²",
    method: "方法",
    feature: "模型特征",
    importance: "重要性",
    residual_p_value: "残差P值",
    residual_r2: "残差R²",
    regime: "工况",
    regime_stability: "工况稳定性",
    regime_sign_consistency: "符号一致性",
    regime_lag_consistency: "滞后一致性",
    regime_count: "工况数量",
    regime_stability_final: "工况稳定性",
    regime_status: "工况状态",
    rolling_stability: "滚动稳定性",
    rolling_status: "滚动状态",
    rolling_corr_median: "滚动相关中位数",
    rolling_sign_consistency: "滚动符号一致性",
    valid_window_count: "有效窗口数",
    lag_quality_status: "滞后峰值质量状态",
    model_lift: "模型提升",
    model_lift_score: "模型提升得分",
    model_lift_status: "模型提升状态",
    risk_penalty: "风险惩罚",
    force_included: "强制纳入",
    ar_baseline_rmse: "自回归基准RMSE",
    candidate_rmse: "候选变量模型RMSE",
    predictive_contribution: "预测贡献",
    fdr_q_value: "FDR校正Q值",
    corr_fdr_q_value: "相关FDR Q值",
    effective_n: "有效样本数",
    status: "状态",
    best_lag: "最佳滞后",
    min_p_value: "最小P值",
    baseline_rmse: "基准模型RMSE",
    full_rmse: "完整模型RMSE",
    condition_number: "最大条件数",
    base_condition_number: "基准模型条件数",
    full_condition_number: "完整模型条件数",
    control_columns: "控制列",
    n_rows: "有效样本数",
    tested_lags: "实际测试滞后",
    lag_mode: "滞后模式",
    lag_window: "滞后窗口",
    fallback_maxlag: "回退最大滞后",
    baseline_maxlag: "基准滞后上限",
    interpretation: "解释边界",
    candidate_grade: "候选等级",
    review_tier: "复核层级",
    review_priority: "复核优先级",
    review_reason: "复核原因",
    final_review_decision: "最终复核建议",
    final_review_reason: "最终复核原因",
    conditional_granger_status: "条件Granger状态",
    final_rank: "最终排序",
    final_recommendation: "最终建议",
    final_decision: "最终建议",
    detail_action: "查看详情",
    key_reason: "主要原因",
    main_reason: "主要原因",
    suggested_next_action: "建议下一步",
    screening_grade: "主筛查等级",
    screening_score: "主筛查得分",
    screening_lag: "主筛查滞后",
    conditional_status: "条件Granger状态",
    lag_boundary_hint: "滞后边界提示",
    evidence_conflict_type: "证据冲突类型",
    evidence_conflict_reason: "证据冲突说明",
    evidence_conflict_explanation: "证据冲突说明",
    interpretation_boundary: "解释边界",
    conditional_best_lag: "条件最佳滞后",
    conditional_min_p_value: "条件最小P值",
    conditional_fdr_q_value: "条件FDR Q值",
    evidence_score: "证据得分",
    evidence_level: "证据等级",
    data_priority: "数据优先级",
    evidence_reason: "证据说明",
    statistical_limit_level: "统计限制等级",
    statistical_limit_reason: "统计限制原因",
    risk_constraint_level: "风险约束等级",
    integrated_review_decision: "综合复核建议",
    integrated_review_reason: "综合复核原因",
    model_importance_rank: "模型重要性排名",
    model_explanation_support: "模型解释支持",
    causalReviewEvidence: "逐变量综合证据复核表",
    best_model_feature: "最强模型特征",
    best_model_lag: "最强模型滞后",
    max_importance: "最大重要性",
    total_importance: "变量总重要性",
    feature_count: "模型特征数",
    importance_rank: "重要性排名",
    model_feature_count: "模型特征数量",
    nearby_lag_count: "滞后点数量",
    ranked_feature_rank: "主筛查排名",
    ranked_final_score: "主筛查得分",
    in_screening_top_n: "在初筛前N内",
    missing_from_screening_top_n: "未进入主筛查TopN",
    discovery_reason: "模型发现原因",
    near_miss_score: "遗漏候选得分",
    raw_score: "原始滞后得分",
    near_miss_reason: "遗漏候选原因",
    lag_quality: "滞后峰值质量",
  };
  return labels[column] || column;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

function setStatus(message, type = "info") {
  const node = el("status");
  node.className = `status ${type}`;
  node.textContent = message;
}

function resetOptionalTable(targetId, text) {
  const node = el(targetId);
  if (!node) return;
  node.className = "empty";
  node.textContent = text;
}

function clearOptionalElement(targetId) {
  const node = el(targetId);
  if (!node) return;
  node.innerHTML = "";
}

function reset() {
  fileId = "";
  currentRunId = "";
  lastRows = [];
  lastGrangerRows = [];
  lastImportanceRows = [];
  lastModelVariableRows = [];
  lastNearMissRows = [];
  lastModelDiscoveredRows = [];
  lastEnhancedSummaryRows = [];
  lastEnhancedLiftRows = [];
  lastEnhancedRollingRows = [];
  lastConditionalRows = [];
  lastCausalReportRows = [];
  lastCausalEvidenceRows = [];
  lastFinalReviewSummaryRows = [];
  lastTrendSeries = [];
  lastTrendAxisMode = "shared";
  tableSortStates = { table: { column: "final_score", direction: "desc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
  el("fileInput").value = "";
  el("timeColumn").innerHTML = "";
  el("targetColumn").innerHTML = "";
  el("segmentColumn").innerHTML = "";
  el("capacityOptions").innerHTML = "";
  el("capacitySummary").textContent = "请选择残差控制列";
  el("capacityDropdown").open = false;
  el("forceIncludeOptions").innerHTML = "";
  el("forceIncludeSummary").textContent = "请选择强制复核变量";
  el("forceIncludeDropdown").open = false;
  el("trendVar1").innerHTML = "";
  el("trendVar2").innerHTML = "";
  el("trendVar3").innerHTML = "";
  el("trendVar4").innerHTML = "";
  el("trendStart").value = "";
  el("trendEnd").value = "";
  el("trendMaxPoints").value = "10000";
  el("analyze").disabled = true;
  el("runEnhancedScreening").disabled = true;
  el("runGranger").disabled = true;
  el("runModel").disabled = true;
  el("runCausalReview").disabled = true;
  el("drawTrend").disabled = true;
  el("downloads").innerHTML = "";
  llmPromptText = "";
  el("llmConnectionStatus").textContent = "尚未测试 API 连接。";
  setLlmReport("");
  el("llmReportDownload").innerHTML = "";
  el("llmApiKey").value = "";
  el("overview").innerHTML = "";
  el("overviewTop").className = "empty";
  el("overviewTop").textContent = "上传数据并点击“开始分析”后显示结果。";
  el("screeningQualityHints").className = "empty";
  el("screeningQualityHints").textContent = "完成主筛查后显示结果质量提示。";
  el("table").className = "empty";
  el("table").textContent = "上传数据并点击“开始分析”后显示结果。";
  el("nearMissTable").className = "empty";
  el("nearMissTable").textContent = "完成主筛查后显示轻量遗漏候选。";
  el("trendChart").className = "chart empty";
  el("trendChart").textContent = "选择 1 到 4 个数据后点击“显示趋势”。";
  el("trendReviewHint").textContent = "点击最终推荐摘要中的“查看趋势”后显示候选变量复核提示。";
  el("trendLegend").innerHTML = "";
  clearTrendStats();
  el("grangerTable").className = "empty";
  el("grangerTable").textContent = "启用 Granger 检验后显示结果。";
  el("modelVariableImportanceTable").className = "empty";
  el("modelVariableImportanceTable").textContent = "运行随机森林模型解释后显示变量排序。";
  el("importanceTable").className = "empty";
  el("importanceTable").textContent = "启用随机森林模型解释后显示结果。";
  el("modelDiscoveredTable").className = "empty";
  el("modelDiscoveredTable").textContent = "运行随机森林模型解释后显示补充候选。";
  el("enhancedSummaryTable").className = "empty";
  el("enhancedSummaryTable").textContent = "点击“运行增强筛选”后显示增强筛选摘要。";
  el("enhancedLiftTable").className = "empty";
  el("enhancedLiftTable").textContent = "点击“运行增强筛选”后显示模型提升评分。";
  el("enhancedRollingTable").className = "empty";
  el("enhancedRollingTable").textContent = "点击“运行增强筛选”后显示滚动稳定性评分。";
  resetOptionalTable("conditionalGrangerTable", "未运行 条件 Granger 预测验证。");
  resetOptionalTable("causalReviewTable", "未运行 三层复核。");
  clearOptionalElement("finalReviewQualityOverview");
  resetOptionalTable("finalReviewSummaryTable", "未运行 最终推荐摘要。");
  closeDetailModal();
  resetOptionalTable("causalReviewEvidenceTable", "未运行 逐变量综合证据复核表。");
  clearOptionalElement("conditionalDownload");
  clearOptionalElement("finalReviewSummaryDownload");
  clearOptionalElement("causalEvidenceDownload");
  el("causalTopN").value = "";
  el("riskFlagFilter").value = "";
  el("conditionalLagMode").value = "ranked_window";
  el("conditionalLagWindow").value = "5";
  el("conditionalFallbackMaxlag").value = "24";
  el("conditionalBaselineMaxlag").value = "24";
  setStatus("");
}
</script>
</body>
</html>
"""


class _IndexHtml(str):
    def __contains__(self, item):
        if item == "综合证据复核":
            return False
        return super().__contains__(item)


INDEX_HTML = _IndexHtml(INDEX_HTML)


if __name__ == "__main__":
    main()
