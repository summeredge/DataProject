from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from numbers import Integral
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
from chem_ts_corr.data import (
    EXCEL_SUFFIXES,
    TEXT_SUFFIXES,
    drop_excluded_columns,
    load_timeseries_csv,
    normalize_excluded_columns,
    read_timeseries_table,
)
from chem_ts_corr.causality import run_granger_tests
from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.modeling import fit_explainable_model
from chem_ts_corr.model_discovery import build_model_discovered_candidates, build_model_variable_importance
from chem_ts_corr.pipeline import run_analysis
from chem_ts_corr.service import run_xgb_analysis
from chem_ts_corr.xgb_validation import validate_xgb_top_n
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
    "xgb_validation/xgb_model_summary.csv",
    "xgb_validation/xgb_candidate_uplift.csv",
    "xgb_validation/xgb_validation_summary.json",
}
MAX_REQUEST_BODY_BYTES = 100 * 1024 * 1024
TASK_TTL_SECONDS = 6 * 60 * 60
MAX_TASKS = 100
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MINUTE_RESAMPLE_RE = re.compile(r"([1-9]\d*)(?:min)?")
_RESAMPLE_MINUTES_ERROR = "重采样间隔必须是大于 0 的整数分钟"
SCALED_FRAME_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
SCALED_FRAME_CACHE_LOCK = threading.Lock()
MAX_SCALED_FRAME_CACHE = 4
TARGET_SEGMENT_MASK_ATTR = "target_operating_segment_mask"
CORRELATION_DIRECTION_EPSILON = 0.05


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
        if parsed.path == "/api/scatter_matrix":
            try:
                params = parse_qs(parsed.query)
                self._send_json(_scatter_matrix_response(params))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/lag_profile":
            try:
                params = parse_qs(parsed.query)
                self._send_json(_lag_profile_response(params))
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
            if self.path == "/api/run_xgb_validation":
                self._send_json(_run_xgb_validation_response(self))
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

        content_types = {
            ".csv": "text/csv; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }
        content_type = content_types.get(path.suffix, "text/markdown; charset=utf-8")
        body = path.read_bytes()
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
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


def _resolve_encoding(path: Path, encoding: str) -> str:
    if encoding and encoding != "auto":
        return encoding
    _, used_encoding = _read_data_sample(path, encoding or "utf-8-sig")
    return used_encoding


def _time_range_metadata(path: Path, sample: pd.DataFrame, encoding: str) -> dict[str, Any]:
    candidate = next(
        (column for column in sample.columns if _looks_like_time_column(str(column))),
        None,
    )
    automatically_detected = False
    if candidate is None and len(sample.columns):
        first_column = sample.columns[0]
        if _looks_like_index_column(str(first_column)):
            candidate = first_column
            automatically_detected = True
    if candidate is None:
        return {}
    try:
        time_frame, _ = read_timeseries_table(path, encoding=encoding, usecols=[candidate])
        source_values = time_frame[candidate]
        values = pd.to_datetime(source_values, errors="coerce")
    except Exception:
        return {} if automatically_detected else {"timeColumn": candidate}
    if automatically_detected and not _is_high_quality_time_series(values, source_values):
        return {}
    values = values.dropna().sort_values()
    if values.empty:
        return {"timeColumn": candidate}

    start = values.iloc[0]
    end = values.iloc[-1]
    default_end = min(start + pd.Timedelta(days=3), end)
    metadata = {
        "timeColumn": candidate,
        "timeStart": _datetime_local(start),
        "timeEnd": _datetime_local(end),
        "trendStartDefault": _datetime_local(start),
        "trendEndDefault": _datetime_local(default_end),
    }
    sampling_interval = _median_sampling_interval(pd.DatetimeIndex(values))
    if sampling_interval is not None:
        metadata["trendSamplingIntervalMs"] = int(
            sampling_interval.total_seconds() * 1000
        )
    if automatically_detected:
        metadata["autoTimeColumn"] = candidate
    return metadata


def _looks_like_time_column(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ["time", "date", "timestamp", "时间", "日期"])


def _looks_like_index_column(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        not normalized
        or bool(re.fullmatch(r"unnamed:\s*0", normalized))
        or normalized == "index"
        or normalized.startswith(("index_", "index-", "index.", "index "))
    )


def _is_high_quality_time_series(values: pd.Series, source_values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(source_values):
        return False
    if values.notna().mean() < 0.95:
        return False
    valid_values = values.dropna()
    differences = valid_values.diff().dropna()
    return not differences.empty and (differences >= pd.Timedelta(0)).mean() >= 0.95


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
    resample_rule = _normalize_minute_resample_rule(_field(form, "resample_rule", ""))

    run_id = uuid.uuid4().hex
    output_dir = RUNS_DIR / run_id
    input_path = _resolve_upload(file_id)
    resolved_encoding = _resolve_encoding(input_path, encoding)
    excluded_columns = normalize_excluded_columns(_list_field(form, "excluded_columns"))
    capacity_columns = _list_field(form, "capacity_columns")
    residual_control_columns = (
        _list_field(form, "residual_control_columns") or capacity_columns
    )
    force_include_variables = _list_field(form, "force_include_variables")
    segment_column = _field(form, "segment_column", "") or None
    _validate_analysis_excluded_columns(
        input_path,
        resolved_encoding,
        time_column=time_column,
        target=target,
        excluded_columns=excluded_columns,
        segment_column=segment_column,
        capacity_columns=capacity_columns,
        residual_control_columns=residual_control_columns,
        force_include_variables=force_include_variables,
    )
    config = AnalysisConfig(
        input_path=input_path,
        time_column=time_column,
        target=target,
        output_dir=output_dir,
        encoding=resolved_encoding,
        max_lag=_int_field(form, "max_lag", 12),
        resample_rule=resample_rule,
        min_valid_ratio=_float_field(form, "min_valid_ratio", 0.7),
        top_k=_int_field(form, "top_k", 30),
        preprocess_mode=_field(form, "preprocess_mode", "raw"),
        detrend_window=_int_field(form, "detrend_window", 24),
        segment_column=segment_column,
        segment_mode=_field(form, "segment_mode", "all"),
        segment_min=_optional_float_field(form, "segment_min"),
        segment_max=_optional_float_field(form, "segment_max"),
        capacity_columns=capacity_columns,
        residual_control_columns=residual_control_columns,
        force_include_variables=force_include_variables,
        excluded_columns=excluded_columns,
        exclude_control_columns_from_candidates=_bool_field(form, "exclude_control_columns_from_candidates") if "exclude_control_columns_from_candidates" in form else True,
        enable_granger=False,
        enable_model=False,
        skip_model_lift=False,
        skip_rolling_corr=False,
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


def _validate_analysis_excluded_columns(
    input_path: Path,
    encoding: str,
    *,
    time_column: str,
    target: str,
    excluded_columns: list[str],
    segment_column: str | None,
    capacity_columns: list[str],
    residual_control_columns: list[str],
    force_include_variables: list[str],
) -> None:
    if time_column in excluded_columns:
        raise ValueError(f"剔除列不能同时作为时间列：{time_column}")
    if target in excluded_columns:
        raise ValueError(f"剔除列不能同时作为目标列：{target}")
    protected = [
        column
        for column in [
            segment_column,
            *capacity_columns,
            *residual_control_columns,
            *force_include_variables,
        ]
        if column
    ]
    conflicts = [column for column in excluded_columns if column in set(protected)]
    if conflicts:
        raise ValueError(f"剔除列与工况/控制/白名单参数冲突：{'、'.join(conflicts)}")

    sample, _ = read_timeseries_table(input_path, encoding=encoding, nrows=5000)
    if time_column not in sample.columns:
        raise ValueError(f"时间列不存在：{time_column}")
    if target not in sample.columns:
        raise ValueError(f"目标列不存在：{target}")
    filtered = drop_excluded_columns(sample, excluded_columns)
    numeric_columns = [
        column
        for column in filtered.columns
        if pd.to_numeric(filtered[column], errors="coerce").notna().mean() >= 0.7
    ]
    candidates = [
        column for column in numeric_columns if column not in {time_column, target}
    ]
    if not candidates:
        raise ValueError("剔除后至少需要保留一个可分析数值候选列")


def _analyze_task(task_id: str, config: AnalysisConfig, file_id: str) -> None:
    try:
        _write_run_config(config.output_dir, config, file_id)

        def progress(message: str) -> None:
            with TASKS_LOCK:
                task = TASKS.get(task_id)
                if task is not None:
                    task["message"] = message
                    task["updated_at"] = time.time()

        pipeline_timings = run_analysis(config, progress_callback=progress)
        if not isinstance(pipeline_timings, dict):
            pipeline_timings = {}
        analysis_timings = {
            key: _non_negative_seconds(pipeline_timings.get(key))
            for key in [
                "read_data_seconds",
                "analysis_core_seconds",
                "write_outputs_seconds",
                "pipeline_total_seconds",
            ]
        }

        payload_started = time.perf_counter()
        result = _build_result_payload(config.output_dir.name, config.output_dir, config)
        result_payload_seconds = _non_negative_seconds(time.perf_counter() - payload_started)
        ended_at = time.time()
        with TASKS_LOCK:
            started_at = float(TASKS[task_id].get("start_time") or ended_at)
            task_total_seconds = round(max(0.0, ended_at - started_at), 6)
            task_total_seconds = max(
                task_total_seconds,
                analysis_timings["pipeline_total_seconds"],
            )
            analysis_timings.update(
                {
                    "result_payload_seconds": result_payload_seconds,
                    "task_total_seconds": task_total_seconds,
                }
            )
            result["elapsed_seconds"] = task_total_seconds
            result["analysis_timings"] = analysis_timings
            result.setdefault("overview", {})["analysis_elapsed_seconds"] = task_total_seconds
            TASKS[task_id].update(
                {
                    "status": "done",
                    "message": "分析完成",
                    "end_time": ended_at,
                    "updated_at": ended_at,
                    "result": result,
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


def _non_negative_seconds(value: object) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0
    return round(seconds, 6)


def _build_result_payload(run_id: str, output_dir: Path, config: AnalysisConfig) -> dict[str, Any]:
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    display_ranked = _with_correlation_display_fields(ranked)
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
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
    return {
        "run_id": run_id,
        "analysisContext": {"preprocess_mode": config.preprocess_mode},
        "overview": _overview_payload(display_ranked, risk, config, _summary_metrics(summary)),
        "rankedFeatures": _records(display_ranked.head(50)),
        "riskFlags": _records(risky.head(50)),
        "lagScores": [],
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
    from chem_ts_corr.screening import (
        model_lift_scores,
        prepare_best_lag_evidence,
        rolling_corr_scores,
    )

    total_started = time.perf_counter()
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    base_config = _read_run_config(output_dir)
    secondary_config = _secondary_config_from_form(base_config, form)
    extra_variables = _secondary_extra_variables_from_form(form)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查并生成 ranked_features.csv")

    variables = _secondary_variables_from_ranked(
        ranked,
        base_config,
        extra_variables=extra_variables,
    )
    if not variables:
        raise ValueError("ranked_features.csv 中没有可运行增强筛选的候选变量")

    scaled = _scaled_frame_for_secondary(secondary_config, protected_columns=extra_variables)
    target_mask = _target_segment_mask(scaled)
    variables = [variable for variable in variables if variable in scaled.columns and variable != secondary_config.target]
    if not variables:
        raise ValueError("二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。")

    lag_search_changed = _secondary_lag_search_changed(base_config, secondary_config)
    lag_evidence_started = time.perf_counter()
    ranked_source_scaled = None
    if not lag_search_changed:
        ranked_source_scaled = _scaled_frame_for_secondary(base_config)
    best_lag_evidence, _ = prepare_best_lag_evidence(
        scaled,
        secondary_config.target,
        variables,
        secondary_config.max_lag,
        ranked=ranked,
        ranked_source_frame=ranked_source_scaled,
        allow_ranked_reuse=not lag_search_changed,
        target_mask=target_mask,
    )
    lag_evidence_seconds = time.perf_counter() - lag_evidence_started
    best_lags = {
        variable: evidence["best_lag"]
        for variable, evidence in best_lag_evidence.items()
        if evidence["best_lag"] is not None
    }

    model_lift_started = time.perf_counter()
    lift = model_lift_scores(scaled, secondary_config.target, variables, secondary_config.max_lag, best_lags=best_lags, target_mask=target_mask)
    model_lift_seconds = time.perf_counter() - model_lift_started

    rolling_started = time.perf_counter()
    rolling = rolling_corr_scores(
        scaled,
        secondary_config.target,
        variables,
        secondary_config.max_lag,
        best_lag_evidence=best_lag_evidence,
        target_mask=target_mask,
    )
    rolling_seconds = time.perf_counter() - rolling_started

    output_started = time.perf_counter()
    enhanced = _enhanced_validation_summary(ranked, lift, rolling)

    lift.to_csv(output_dir / "model_lift_scores.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(output_dir / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig")
    enhanced.to_csv(output_dir / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig")

    result = {
        "modelLiftScores": _records(lift.head(200)),
        "rollingCorrScores": _records(rolling.head(200)),
        "enhancedValidationSummary": _records(enhanced.head(200)),
        "downloads": _download_links(run_id, output_dir),
        "message": "增强筛选完成：结果用于补充验证预测增益和时间稳定性，不代表因果结论。",
    }
    output_seconds = time.perf_counter() - output_started
    result["timings"] = {
        "lag_evidence_seconds": lag_evidence_seconds,
        "model_lift_seconds": model_lift_seconds,
        "rolling_seconds": rolling_seconds,
        "output_seconds": output_seconds,
        "total_seconds": time.perf_counter() - total_started,
    }
    return result


def _enhanced_validation_summary(
    ranked: pd.DataFrame, model_lift: pd.DataFrame, rolling: pd.DataFrame
) -> pd.DataFrame:
    base_columns = [
        column
        for column in ["variable", "final_score", "lag", "direction", "risk_flags", "recommended_use"]
        if column in ranked.columns
    ]

    variable_sources: list[str] = []
    for frame in [ranked, model_lift, rolling]:
        if not frame.empty and "variable" in frame.columns:
            variable_sources.extend(frame["variable"].dropna().astype(str).tolist())

    variables = list(dict.fromkeys([v for v in variable_sources if v]))
    if not variables:
        return pd.DataFrame()

    summary = pd.DataFrame({"variable": variables})

    if base_columns and "variable" in base_columns:
        ranked_meta = ranked[base_columns].copy(deep=True)
        ranked_meta["variable"] = ranked_meta["variable"].astype(str)
        ranked_meta = ranked_meta.drop_duplicates(subset=["variable"], keep="first")
        summary = summary.merge(ranked_meta, on="variable", how="left")
    else:
        for column in ["final_score", "lag", "direction", "risk_flags", "recommended_use"]:
            summary[column] = pd.NA

    if not model_lift.empty and "variable" in model_lift.columns:
        lift_columns = [
            c
            for c in ["variable", "status", "model_lift", "ar_baseline_rmse", "candidate_rmse"]
            if c in model_lift.columns
        ]
        lift_meta = model_lift[lift_columns].copy(deep=True)
        lift_meta["variable"] = lift_meta["variable"].astype(str)
        lift_meta = lift_meta.drop_duplicates(subset=["variable"], keep="first")
        summary = summary.merge(lift_meta, on="variable", how="left")

    if not rolling.empty and "variable" in rolling.columns:
        rolling_columns = [
            c
            for c in [
                "variable",
                "rolling_stability",
                "rolling_corr_median",
                "rolling_sign_consistency",
                "valid_window_count",
            ]
            if c in rolling.columns
        ]
        rolling_meta = rolling[rolling_columns].copy(deep=True)
        rolling_meta["variable"] = rolling_meta["variable"].astype(str)
        rolling_meta = rolling_meta.drop_duplicates(subset=["variable"], keep="first")
        summary = summary.merge(rolling_meta, on="variable", how="left")

    summary["interpretation"] = "enhanced screening only; not a causal conclusion"
    return summary


def _run_granger_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    base_config = _read_run_config(output_dir)
    secondary_config = _secondary_config_from_form(base_config, form)
    extra_variables = _secondary_extra_variables_from_form(form)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查")
    variables = _secondary_variables_from_ranked(
        ranked,
        base_config,
        extra_variables=extra_variables,
    )
    scaled = _scaled_frame_for_secondary(secondary_config, protected_columns=extra_variables)
    target_mask = _target_segment_mask(scaled)
    variables = [variable for variable in variables if variable in scaled.columns and variable != secondary_config.target]
    if not variables:
        raise ValueError("二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。")
    granger = run_granger_tests(
        scaled,
        target=secondary_config.target,
        variables=variables,
        maxlag=max(1, secondary_config.max_lag),
        target_mask=target_mask,
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
    base_config = _read_run_config(output_dir)
    secondary_config = _secondary_config_from_form(base_config, form)
    extra_variables = _secondary_extra_variables_from_form(form)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("请先完成主筛查")
    variables = _secondary_variables_from_ranked(
        ranked,
        base_config,
        extra_variables=extra_variables,
    )
    near_miss = _safe_read_result_csv(output_dir / "near_miss_candidates.csv")
    variables = list(dict.fromkeys(variables + _near_miss_variables(near_miss, limit=10)))
    scaled = _scaled_frame_for_secondary(secondary_config, protected_columns=extra_variables)
    target_mask = _target_segment_mask(scaled)
    variables = [variable for variable in variables if variable in scaled.columns and variable != secondary_config.target]
    lag_search_changed = _secondary_lag_search_changed(base_config, secondary_config)
    if lag_search_changed:
        best_lags = {}
    else:
        best_lags = _best_lags_from_ranked(ranked)
        best_lags = _merge_near_miss_lags(best_lags, near_miss)
    best_lags = _secondary_best_lags_for_missing_variables(
        scaled,
        secondary_config.target,
        variables,
        best_lags,
        secondary_config.max_lag,
        recompute_limit=None if lag_search_changed else 20,
        target_mask=target_mask,
    )
    if not variables:
        raise ValueError("二次验证候选变量在预处理后的数据中不存在，请检查 TopK、白名单和二次验证重采样设置。")
    importance, metrics = fit_explainable_model(
        scaled,
        target=secondary_config.target,
        max_lag=secondary_config.max_lag,
        candidate_variables=variables,
        max_features=secondary_config.max_model_features,
        random_state=secondary_config.random_state,
        best_lags=best_lags,
        lag_mode="best_only",
        target_mask=target_mask,
    )
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
    model_variable_importance = build_model_variable_importance(importance, ranked, risk_flags=risk)
    model_discovered = build_model_discovered_candidates(
        importance,
        ranked,
        risk_flags=risk,
        screening_top_n=base_config.top_k,
        max_lag=secondary_config.max_lag,
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
    control_columns = (
        _list_field(form, "control_columns")
        or config.residual_control_columns
        or config.capacity_columns
        or []
    )
    _ensure_columns_not_excluded(config, control_columns, "三层复核控制列")
    candidates = _filter_candidates_by_risk_flags(candidates, risk, risk_filter)
    scaled = _scaled_frame_for_secondary(config)
    target_mask = _target_segment_mask(scaled)
    result = run_causal_review_stage(
        frame=scaled,
        target=config.target,
        ranked_features=ranked,
        causal_review_candidates=candidates,
        risk_flags=risk,
        output_dir=output_dir,
        control_columns=control_columns,
        maxlag=_int_field(form, "maxlag", config.resolved_granger_maxlag()),
        min_rows=_int_field(form, "min_rows", 60),
        top_n=_optional_int_field(form, "top_n"),
        conditional_lag_mode=_field(form, "conditional_lag_mode", "ranked_window"),
        conditional_lag_window=_int_field(form, "conditional_lag_window", 5),
        conditional_fallback_maxlag=_int_field(form, "conditional_fallback_maxlag", 24),
        conditional_baseline_maxlag=_optional_int_field(form, "conditional_baseline_maxlag") or 24,
        target_mask=target_mask,
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


def _run_xgb_validation_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    if not _bool_field(form, "enable_xgb_validation"):
        return {
            "status": "skipped",
            "error_message": None,
            "xgbModelSummary": [],
            "xgbCandidateUplift": [],
            "xgbValidationSummary": {},
            "downloads": [],
            "message": "XGB 四级验证未启用。",
        }

    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    top_n_raw = _field(form, "top_n", str(config.xgb_top_n)).strip()
    try:
        top_n = validate_xgb_top_n(int(top_n_raw))
    except (TypeError, ValueError):
        return _xgb_response_payload(
            run_id,
            output_dir,
            status="invalid_input",
            error_message="top_n must be an integer between 1 and 10",
        )

    max_lag_raw = _field(form, "max_lag").strip()
    if max_lag_raw:
        try:
            max_lag = int(max_lag_raw)
        except ValueError:
            return _xgb_response_payload(
                run_id,
                output_dir,
                status="invalid_input",
                error_message="max_lag must be an integer between 1 and 5000",
            )
        if not 1 <= max_lag <= 5000:
            return _xgb_response_payload(
                run_id,
                output_dir,
                status="invalid_input",
                error_message="max_lag must be an integer between 1 and 5000",
            )
    else:
        max_lag = config.xgb_max_lag

    final_summary = _safe_read_result_csv(output_dir / "final_review_summary.csv")
    if final_summary.empty:
        return _xgb_response_payload(
            run_id,
            output_dir,
            status="invalid_input",
            error_message="missing final_review_summary; run the third-level review first",
        )

    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    control_columns = (
        _list_field(form, "control_columns")
        or config.residual_control_columns
        or config.capacity_columns
        or []
    )
    whitelist = _list_field(form, "whitelist")
    _ensure_columns_not_excluded(config, control_columns, "XGBoost 控制列")
    _ensure_columns_not_excluded(config, whitelist, "XGBoost 白名单")
    data = _prepared_frame_for_validation(config)
    target_mask = _target_segment_mask(data)
    result = run_xgb_analysis(
        run_dir=output_dir,
        data=data,
        target=config.target,
        final_review_summary=final_summary,
        ranked_features=ranked,
        control_columns=control_columns,
        whitelist=whitelist,
        top_n=top_n,
        max_lag=max_lag,
        target_mask=target_mask,
    )
    return _xgb_response_payload(
        run_id,
        output_dir,
        status=result.status,
        error_message=result.error_message,
    )


def _xgb_response_payload(
    run_id: str,
    output_dir: Path,
    *,
    status: str,
    error_message: str | None,
) -> dict[str, Any]:
    model_summary = pd.DataFrame()
    candidate_uplift = pd.DataFrame()
    validation_summary: dict[str, Any] = {}
    downloads = _download_links(run_id, output_dir)
    if status == "success":
        model_summary = _safe_read_result_csv(
            output_dir / "xgb_validation" / "xgb_model_summary.csv"
        )
        candidate_uplift = _safe_read_result_csv(
            output_dir / "xgb_validation" / "xgb_candidate_uplift.csv"
        )
        summary_path = output_dir / "xgb_validation" / "xgb_validation_summary.json"
        validation_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
    messages = {
        "success": "XGB 四级验证完成。",
        "missing_dependency": "XGB 四级验证缺少可选依赖。",
        "invalid_input": "XGB 四级验证输入无效。",
        "failed": "XGB 四级验证失败。",
    }
    return {
        "status": status,
        "error_message": error_message,
        "xgbModelSummary": _records(model_summary),
        "xgbCandidateUplift": _records(candidate_uplift),
        "xgbValidationSummary": validation_summary,
        "downloads": downloads,
        "message": messages.get(status, "XGB 四级验证未运行。"),
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
        max_tokens=_int_field(form, "max_tokens", 15000),
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


def _secondary_variables_from_ranked(
    ranked: pd.DataFrame,
    config: AnalysisConfig,
    extra_variables: list[str] | None = None,
) -> list[str]:
    _ensure_columns_not_excluded(
        config, extra_variables or [], "二次验证补充变量"
    )
    if ranked.empty or "variable" not in ranked.columns:
        return list(dict.fromkeys([v for v in (extra_variables or []) if v]))
    top = ranked.head(config.top_k)["variable"].astype(str).tolist()
    if "force_included" in ranked.columns:
        forced = ranked[ranked["force_included"].astype(bool)]["variable"].astype(str).tolist()
    else:
        forced = [v for v in (config.force_include_variables or []) if v]
    extra = [v for v in (extra_variables or []) if v]
    return list(dict.fromkeys(top + forced + extra))


def _secondary_config_from_form(config: AnalysisConfig, form: dict[str, Any]) -> AnalysisConfig:
    mode = _field(form, "secondary_resample_mode", "raw").strip().lower()
    secondary_max_lag = _int_field(form, "secondary_max_lag", config.max_lag)
    secondary_max_lag = min(5000, max(0, secondary_max_lag))

    if mode == "inherit":
        resample_rule = config.resample_rule
    elif mode == "custom":
        resample_rule = _normalize_minute_resample_rule(
            _field(form, "secondary_resample_rule", ""),
            allow_empty=False,
        )
    else:
        resample_rule = None

    extra_variables = _secondary_extra_variables_from_form(form)
    _ensure_columns_not_excluded(config, extra_variables, "二次验证补充变量")
    force_include_variables = list(
        dict.fromkeys(
            [v for v in (config.force_include_variables or []) if v]
            + [v for v in extra_variables if v]
        )
    )

    return replace(
        config,
        resample_rule=resample_rule,
        max_lag=secondary_max_lag,
        force_include_variables=force_include_variables,
    )


def _ensure_columns_not_excluded(
    config: AnalysisConfig, columns: list[str] | None, label: str
) -> None:
    excluded = set(normalize_excluded_columns(config.excluded_columns))
    conflicts = [
        column
        for column in normalize_excluded_columns(columns)
        if column in excluded
    ]
    if conflicts:
        raise ValueError(f"{label}不能包含已剔除列：{'、'.join(conflicts)}")


def _normalize_minute_resample_rule(
    value: object,
    *,
    allow_empty: bool = True,
) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_empty:
            return None
        raise ValueError(_RESAMPLE_MINUTES_ERROR)
    if isinstance(value, bool):
        raise ValueError(_RESAMPLE_MINUTES_ERROR)
    if isinstance(value, Integral):
        if value <= 0:
            raise ValueError(_RESAMPLE_MINUTES_ERROR)
        return f"{int(value)}min"
    if not isinstance(value, str):
        raise ValueError(_RESAMPLE_MINUTES_ERROR)

    match = _MINUTE_RESAMPLE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(_RESAMPLE_MINUTES_ERROR)
    return f"{int(match.group(1))}min"


def _normalized_resample_rule(rule: str | None) -> str:
    return "" if rule is None else str(rule).strip().lower()


def _secondary_lag_search_changed(
    base_config: AnalysisConfig,
    secondary_config: AnalysisConfig,
) -> bool:
    return (
        _normalized_resample_rule(base_config.resample_rule)
        != _normalized_resample_rule(secondary_config.resample_rule)
        or int(base_config.max_lag) != int(secondary_config.max_lag)
    )


def _secondary_extra_variables_from_form(form: dict[str, Any]) -> list[str]:
    return _list_field(form, "secondary_include_variables")


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


def _secondary_best_lags_for_missing_variables(
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    existing_best_lags: dict[str, int],
    max_lag: int,
    recompute_limit: int | None = 20,
    target_mask: pd.Series | None = None,
) -> dict[str, int]:
    from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags

    merged = dict(existing_best_lags or {})
    if target not in frame.columns or max_lag <= 0:
        return merged

    missing_lag_variables = [
        variable
        for variable in variables
        if variable not in merged and variable != target and variable in frame.columns
    ]

    if recompute_limit is not None:
        limit = max(0, int(recompute_limit))
        missing_lag_variables = missing_lag_variables[:limit]

    for variable in missing_lag_variables:
        pair = frame[[target, variable]].dropna()
        if len(pair) < max(10, max_lag + 5):
            continue

        scores = (
            compute_lag_scores(pair, target, max_lag)
            if target_mask is None
            else compute_lag_scores(pair, target, max_lag, target_mask=target_mask)
        )
        best = summarize_best_lags(scores)
        if best.empty or "lag" not in best.columns:
            continue

        try:
            merged[variable] = int(best.iloc[0]["lag"])
        except (TypeError, ValueError):
            continue

    return merged


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


def _scaled_frame_for_secondary(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> pd.DataFrame:
    from chem_ts_corr.preprocess import standardize_frame

    extra_protected = tuple(c for c in (protected_columns or []) if c)
    cache_key = _scaled_frame_cache_key(config, extra_protected)
    with SCALED_FRAME_CACHE_LOCK:
        cached = SCALED_FRAME_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy(deep=True)

    transformed = _prepared_frame_for_secondary(config, protected_columns)
    target_mask = _target_segment_mask(transformed)
    scaled = standardize_frame(transformed, fit_mask=target_mask)
    scaled.attrs[TARGET_SEGMENT_MASK_ATTR] = target_mask
    with SCALED_FRAME_CACHE_LOCK:
        SCALED_FRAME_CACHE[cache_key] = scaled.copy(deep=True)
        while len(SCALED_FRAME_CACHE) > MAX_SCALED_FRAME_CACHE:
            oldest_key = next(iter(SCALED_FRAME_CACHE))
            SCALED_FRAME_CACHE.pop(oldest_key, None)
    return scaled.copy(deep=True)


def _numeric_frame(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> pd.DataFrame:
    from chem_ts_corr.data import select_numeric_frame
    from chem_ts_corr.screening import apply_ignore_roles, load_roles

    raw = load_timeseries_csv(config.input_path, config.time_column, encoding=config.encoding)
    raw = drop_excluded_columns(
        raw,
        config.excluded_columns,
        protected_columns=[
            config.time_column,
            *_protected_validation_columns(config, protected_columns),
        ],
    )
    numeric = select_numeric_frame(raw, config.target)
    roles = load_roles(config, list(numeric.columns))
    return apply_ignore_roles(numeric, roles, config.target)


def _protected_validation_columns(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> list[str]:
    return [
        column
        for column in [
            config.target,
            config.segment_column,
            *(config.capacity_columns or []),
            *(config.residual_control_columns or []),
            *(config.force_include_variables or []),
            *(protected_columns or []),
        ]
        if column
    ]


def _prepared_frame_for_secondary(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> pd.DataFrame:
    from chem_ts_corr.preprocess import operating_segment_mask, preprocess_frame, transform_frame

    numeric = _numeric_frame(config, protected_columns)
    cleaned = preprocess_frame(
        numeric,
        target=config.target,
        resample_rule=config.resample_rule,
        min_valid_ratio=config.min_valid_ratio,
        protected_columns=_protected_validation_columns(config, protected_columns),
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    target_mask = operating_segment_mask(
        cleaned,
        config.segment_column,
        config.segment_mode,
        config.segment_min,
        config.segment_max,
    )
    transformed = transform_frame(
        cleaned,
        config.preprocess_mode,
        config.detrend_window,
        max_interpolate_gap_points=config.max_interpolate_gap_points,
        interpolate_limit_area=config.interpolate_limit_area,
    )
    transformed.attrs[TARGET_SEGMENT_MASK_ATTR] = target_mask.reindex(
        transformed.index
    ).fillna(False).astype(bool)
    return transformed


def _target_segment_mask(frame: pd.DataFrame) -> pd.Series | None:
    stored = frame.attrs.get(TARGET_SEGMENT_MASK_ATTR)
    if isinstance(stored, pd.Series):
        resolved = stored.reindex(frame.index).fillna(False).astype(bool)
        return None if bool(resolved.all()) else resolved
    return None


def _prepared_frame_for_validation(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> pd.DataFrame:
    from chem_ts_corr.preprocess import (
        operating_segment_mask,
        preprocess_frame_causal,
        transform_frame_causal,
    )

    numeric = _numeric_frame(config, protected_columns)
    protected = _protected_validation_columns(config, protected_columns)
    columns = [column for column in numeric.columns if column in protected or column == config.target]
    source = numeric.loc[:, columns] if protected_columns is not None else numeric
    cleaned = preprocess_frame_causal(
        source,
        target=config.target,
        resample_rule=config.resample_rule,
        max_forward_fill_gap_points=config.max_interpolate_gap_points,
    )
    target_mask = operating_segment_mask(
        cleaned,
        config.segment_column,
        config.segment_mode,
        config.segment_min,
        config.segment_max,
    )
    transformed = transform_frame_causal(
        cleaned, config.preprocess_mode, config.detrend_window
    )
    transformed.attrs[TARGET_SEGMENT_MASK_ATTR] = target_mask.reindex(
        transformed.index
    ).fillna(False).astype(bool)
    return transformed


def _scaled_frame_cache_key(
    config: AnalysisConfig, protected_columns: tuple[str, ...] = ()
) -> tuple[Any, ...]:
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
        tuple(config.excluded_columns or []),
        protected_columns,
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


def _chart_frame_from_params(
    params: dict[str, list[str]],
    variables: list[str],
) -> tuple[pd.DataFrame, int, int]:
    file_id = _single(params, "file_id")
    encoding = _single(params, "encoding", "utf-8-sig")
    input_path = _resolve_upload(file_id)
    resolved_encoding = _resolve_encoding(input_path, encoding)
    time_column = _single(params, "time_column")
    excluded_columns = normalize_excluded_columns(
        _single(params, "excluded_columns").split(",")
    )
    segment_column = _single(params, "segment_column") or None
    if time_column in excluded_columns:
        raise ValueError(f"剔除列不能同时作为时间列：{time_column}")
    excluded_variables = [
        variable for variable in variables if variable in set(excluded_columns)
    ]
    if excluded_variables:
        raise ValueError(f"已剔除列不能用于图表：{'、'.join(excluded_variables)}")
    if segment_column and segment_column in excluded_columns:
        raise ValueError(f"剔除列不能同时作为工况列：{segment_column}")

    from chem_ts_corr.data import load_timeseries_csv, select_numeric_frame
    from chem_ts_corr.preprocess import segment_by_load, transform_frame

    raw = load_timeseries_csv(input_path, time_column, encoding=resolved_encoding)
    raw = drop_excluded_columns(raw, excluded_columns)
    max_points = min(100000, max(100, int(_single(params, "trend_max_points", "10000") or 10000)))
    start_value = _single(params, "trend_start", "")
    end_value = _single(params, "trend_end", "")
    start_time = pd.to_datetime(start_value) if start_value else None
    end_time = pd.to_datetime(end_value) if end_value else None
    start_time, end_time = _trend_time_bounds(
        raw.index,
        start_time,
        end_time,
        max_points=max_points,
        mode=_single(params, "time_range_mode", "manual"),
    )
    if start_time is not None:
        raw = raw.loc[raw.index >= start_time]
    if end_time is not None:
        raw = raw.loc[raw.index <= end_time]
    if raw.empty:
        raise ValueError("图表时间范围内没有数据")
    numeric = select_numeric_frame(raw, variables[0])
    columns = [column for column in variables if column in numeric.columns]
    if not columns:
        raise ValueError("选择的变量不是有效数值列")
    frame_columns = list(
        dict.fromkeys(
            columns
            + [col for col in [segment_column] if col and col in numeric.columns]
        )
    )
    frame = numeric[frame_columns]
    segmented = segment_by_load(
        frame,
        segment_column=segment_column,
        segment_mode=_single(params, "segment_mode", "all"),
        segment_min=_optional_query_float(params, "segment_min"),
        segment_max=_optional_query_float(params, "segment_max"),
    )
    transformed = transform_frame(
        segmented[columns],
        _single(params, "preprocess_mode", "raw"),
        int(_single(params, "detrend_window", "24") or 24),
    )
    raw_rows = len(transformed)
    if len(transformed) > max_points:
        positions = [int(index * (len(transformed) - 1) / (max_points - 1)) for index in range(max_points)]
        positions = list(dict.fromkeys(positions))
        transformed = transformed.iloc[positions]
    return transformed, int(raw_rows), int(max_points)


def _median_sampling_interval(index: pd.Index) -> pd.Timedelta | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    values = pd.Series(index).dropna().drop_duplicates().sort_values()
    positive_deltas = values.diff().dropna()
    positive_deltas = positive_deltas.loc[positive_deltas > pd.Timedelta(0)]
    if positive_deltas.empty:
        return None
    interval = positive_deltas.median()
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        return None
    return pd.Timedelta(interval)


def _trend_time_bounds(
    index: pd.Index,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    *,
    max_points: int,
    mode: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if mode != "auto" or len(index) == 0:
        return start_time, end_time
    sampling_interval = _median_sampling_interval(index)
    if sampling_interval is None:
        return start_time, end_time
    first_time = pd.Timestamp(index.min())
    latest_time = pd.Timestamp(index.max())
    effective_start = start_time if start_time is not None else first_time
    effective_end = min(
        effective_start + (max_points - 1) * sampling_interval,
        latest_time,
    )
    return effective_start, effective_end


def _trend_response(params: dict[str, list[str]]) -> dict[str, Any]:
    variables = [value for value in _single(params, "variables").split(",") if value]
    if not variables:
        raise ValueError("请选择至少一个趋势变量")
    if len(variables) > 4:
        raise ValueError("最多选择 4 个趋势变量")

    transformed, raw_rows, max_points = _chart_frame_from_params(params, variables)
    columns = [column for column in variables if column in transformed.columns]
    if not columns:
        raise ValueError("选择的趋势变量不是有效数值列")

    return {
        "series": [
            {
                "name": column,
                "points": [
                    {"x": str(index), "y": _finite_json_number(value)}
                    for index, value in transformed[column].items()
                ],
            }
            for column in columns
        ],
        "rows": int(len(transformed)),
        "raw_rows": int(raw_rows),
        "max_points": int(max_points),
    }


def _finite_json_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _scatter_matrix_response(params: dict[str, list[str]]) -> dict[str, Any]:
    x_variables = list(
        dict.fromkeys(
            value.strip()
            for value in _single(params, "x_variables").split(",")
            if value.strip()
        )
    )
    y_variables = list(
        dict.fromkeys(
            value.strip()
            for value in _single(params, "y_variables").split(",")
            if value.strip()
        )
    )
    if not x_variables:
        raise ValueError("请选择至少一个 X 轴变量")
    if not y_variables:
        raise ValueError("请选择至少一个 Y 轴变量")
    if len(x_variables) > 3:
        raise ValueError("X 轴变量最多选择 3 个")
    if len(y_variables) > 3:
        raise ValueError("Y 轴变量最多选择 3 个")

    columns = list(dict.fromkeys(x_variables + y_variables))
    try:
        transformed, raw_rows, max_points = _chart_frame_from_params(params, columns)
    except ValueError as exc:
        if "Not enough rows in selected operating segment" in str(exc):
            raise ValueError("当前时间范围、工况和预处理条件下没有可绘制的散点数据") from exc
        raise
    if transformed.empty:
        raise ValueError("当前时间范围、工况和预处理条件下没有可绘制的散点数据")
    columns = [column for column in columns if column in transformed.columns]
    if not columns:
        raise ValueError("选择的散点矩阵变量不是有效数值列")

    values = [
        [_finite_json_number(value) for value in row]
        for row in transformed[columns].itertuples(
            index=False,
            name=None,
        )
    ]
    return {
        "x_variables": [column for column in x_variables if column in columns],
        "y_variables": [column for column in y_variables if column in columns],
        "columns": columns,
        "values": values,
        "rows": int(len(values)),
        "raw_rows": int(raw_rows),
        "max_points": int(max_points),
    }




def _overview_payload(
    ranked: pd.DataFrame, risk: pd.DataFrame, config: AnalysisConfig, metrics: dict[str, str]
) -> dict[str, Any]:
    high_risk = int((risk.get("risk_count", pd.Series(dtype=float)) > 0).sum()) if not risk.empty else 0
    review = int((ranked.get("recommended_use", pd.Series(dtype=str)).astype(str) == "prediction_candidate").sum()) if not ranked.empty else 0
    overview_ranked = (
        ranked.sort_values("driver_rank", ascending=True, kind="stable")
        if "driver_rank" in ranked.columns
        else ranked
    )
    return {
        "top10": _records(overview_ranked.head(10)),
        "effective_variables": int(len(ranked)),
        "risk_tagged_count": high_risk,
        "high_risk_count": high_risk,
        "secondary_review_count": review,
        "target": config.target,
        "rows_after_preprocess": metrics.get("rows_after_preprocess", ""),
        "rows_after_segment": metrics.get("rows_after_segment", ""),
    }


def _lag_profile_response(params: dict[str, list[str]]) -> dict[str, Any]:
    run_id = _single(params, "run_id")
    variable = _single(params, "variable")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("运行 ID 格式无效")
    if not variable:
        raise ValueError("变量名称不能为空")

    output_dir = _resolve_run_dir(run_id)
    lag_path = output_dir / "lag_scores.csv"
    if not lag_path.exists() or lag_path.stat().st_size == 0:
        raise FileNotFoundError("滞后相关结果不存在")
    lag_scores = _safe_read_result_csv(lag_path)
    if "variable" not in lag_scores.columns:
        raise ValueError("滞后相关结果缺少变量字段")
    profile = lag_scores[lag_scores["variable"].astype(str).eq(variable)].copy()
    if profile.empty:
        raise ValueError(f"变量 {variable} 没有滞后相关记录")
    if "lag" not in profile.columns:
        raise ValueError("滞后相关结果缺少 lag 字段")
    profile["lag"] = pd.to_numeric(profile["lag"], errors="coerce")
    profile = profile[profile["lag"].notna()].sort_values("lag", kind="stable")
    if profile.empty:
        raise ValueError(f"变量 {variable} 没有有效滞后点")

    ranked_path = output_dir / "ranked_features.csv"
    if not ranked_path.exists() or ranked_path.stat().st_size == 0:
        raise FileNotFoundError("候选结果不存在")
    ranked = _safe_read_result_csv(ranked_path)
    if "variable" not in ranked.columns:
        raise ValueError("候选结果缺少变量字段")
    candidate = ranked[ranked["variable"].astype(str).eq(variable)]
    if candidate.empty:
        raise ValueError(f"变量 {variable} 不在当前候选结果中")
    candidate_row = candidate.iloc[0]
    best_lag = _finite_json_number(candidate_row.get("lag"), integer=True)
    if best_lag is None:
        raise ValueError(f"变量 {variable} 的最佳滞后无效")
    method = str(candidate_row.get("method", "")).strip().lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError(f"变量 {variable} 的主导相关方法无效")

    points: list[dict[str, Any]] = []
    for _, row in profile.iterrows():
        point = {
            "variable": variable,
            "lag": _finite_json_number(row.get("lag"), integer=True),
            "pearson": _finite_json_number(row.get("pearson")),
            "spearman": _finite_json_number(row.get("spearman")),
            "pearson_q": _finite_json_number(row.get("pearson_q")),
            "spearman_q": _finite_json_number(row.get("spearman_q")),
            "n": _finite_json_number(row.get("n"), integer=True),
            "lag_boundary_flag": _bool_value(row.get("lag_boundary_flag", False)),
        }
        points.append(point)

    max_lag = max(abs(point["lag"]) for point in points if point["lag"] is not None)
    return {
        "variable": variable,
        "best_lag": best_lag,
        "method": method,
        "max_lag": max_lag,
        "sampling_interval_minutes": _lag_profile_sampling_minutes(output_dir),
        "points": points,
    }


def _finite_json_number(value: object, *, integer: bool = False) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if integer else number


def _bool_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    return bool(value)


def _lag_profile_sampling_minutes(output_dir: Path) -> int | None:
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        rule = str(data.get("resample_rule") or "").strip()
        match = _MINUTE_RESAMPLE_RE.fullmatch(rule)
    except (OSError, ValueError, TypeError):
        return None
    return int(match.group(1)) if match else None


def _with_correlation_display_fields(ranked: pd.DataFrame) -> pd.DataFrame:
    display = ranked.copy()
    method = display.get("method", pd.Series(index=display.index, dtype=str)).astype(str)
    pearson = pd.to_numeric(
        display.get("pearson", pd.Series(index=display.index, dtype=float)), errors="coerce"
    )
    spearman = pd.to_numeric(
        display.get("spearman", pd.Series(index=display.index, dtype=float)), errors="coerce"
    )
    display["dominant_corr"] = pearson.where(
        method.eq("pearson"), spearman.where(method.eq("spearman"))
    )
    display["correlation_direction"] = display["dominant_corr"].map(
        _correlation_direction
    )
    return display


def _correlation_direction(value: object) -> str:
    try:
        correlation = float(value)
    except (TypeError, ValueError):
        return "未计算"
    if not math.isfinite(correlation):
        return "未计算"
    if correlation > CORRELATION_DIRECTION_EPSILON:
        return "正向"
    if correlation < -CORRELATION_DIRECTION_EPSILON:
        return "负向"
    return "方向较弱"


def _summary_metrics(summary: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in summary.splitlines():
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        metrics[key.strip()] = value.strip()
    return metrics

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
    return list(dict.fromkeys(item.strip() for item in re.split(r"[,，]", value) if item.strip()))


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
    .grid { display:grid; grid-template-columns:repeat(2, minmax(160px, 1fr)); gap:10px; align-items:end; }
    .secondary-validation-params, .causal-review-params { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:6px; align-items:end; }
    .control-group-title { font-size:var(--font-sm); font-weight:700; color:var(--text); }
    label { display:grid; gap:3px; font-size:var(--font-xs); line-height:1.2; color:var(--muted); }
    input, select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; color:var(--text); background:var(--panel); font-size:var(--font-xs); line-height:1.2; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    label.checkbox-row { display:flex; align-items:center; align-self:end; gap:8px; min-height:31px; }
    label.checkbox-row input[type="checkbox"] { width:auto; margin:0; }
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
    .lag-profile-panel { min-height:300px; display:grid; gap:10px; margin-top:8px; padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--surface-muted); }
    .lag-profile-panel.loading, .lag-profile-panel.error { min-height:120px; place-items:center; color:var(--muted); }
    .lag-profile-panel.error { color:var(--danger-text); }
    .lag-profile-chart { min-width:0; overflow:hidden; border:1px solid var(--line-soft); border-radius:6px; background:var(--panel); }
    .lag-profile-chart svg { display:block; width:100%; height:300px; }
    .lag-profile-legend { display:flex; justify-content:center; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:var(--font-sm); }
    .lag-profile-legend span { display:inline-flex; align-items:center; gap:6px; }
    .lag-profile-line { width:24px; height:0; border-top:3px solid; }
    .lag-profile-line.spearman { border-top-style:dashed; }
    .lag-profile-directions { display:grid; grid-template-columns:1fr auto 1fr; gap:8px; color:var(--muted); font-size:var(--font-xs); }
    .lag-profile-directions span:last-child { text-align:right; }
    .lag-profile-summary { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:8px; }
    .lag-profile-summary div { padding:7px 8px; border:1px solid var(--line-soft); border-radius:6px; background:var(--panel); }
    .lag-profile-summary strong { display:block; margin-bottom:3px; color:var(--muted); font-size:var(--font-xs); }
    .lag-profile-message { margin:0; color:var(--text); font-size:var(--font-sm); }
    .lag-profile-warning { margin:0; color:var(--warn); font-size:var(--font-sm); font-weight:650; }
    .chart-controls { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)) 150px auto; gap:10px; align-items:end; }
    .trend-options { display:grid; grid-template-columns:repeat(3,minmax(160px,1fr)); gap:10px; align-items:end; }
    .scatter-matrix-section { display:grid; gap:12px; margin-top:10px; }
    .scatter-matrix-controls { display:grid; grid-template-columns:repeat(3, minmax(150px, 1fr)); gap:10px; align-items:end; }
    .scatter-matrix-chart { min-height:280px; border:1px solid var(--line); border-radius:6px; background:var(--panel); overflow:auto; }
    .scatter-matrix-chart.empty { display:grid; place-items:center; color:var(--muted); font-size:var(--font-sm); padding:16px; }
    .scatter-matrix-chart canvas { display:block; }
    .llm-config-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; align-items:end; }
    .legend { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:var(--font-base); }
    .trend-stats { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; align-items:start; }
    .trend-stats.empty { display:block; color:var(--muted); font-size:var(--font-sm); }
    .trend-stat-card { min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:10px; }
    .trend-stat-card h3 { margin:0 0 8px; font-size:var(--font-sm); overflow-wrap:anywhere; }
    .trend-stat-card dl { display:grid; gap:4px; margin:0; }
    .trend-stat-card dl div { display:grid; grid-template-columns:80px 1fr; gap:8px; font-size:var(--font-xs); }
    .trend-stat-card dt { color:var(--muted); }
    .trend-stat-card dd { margin:0; color:var(--text); text-align:right; font-variant-numeric:tabular-nums; }
    .trend-histogram { width:100%; min-width:0; margin-top:10px; }
    .trend-histogram-title { margin-bottom:4px; color:var(--muted); font-size:var(--font-xs); }
    .trend-histogram-bars { position:relative; display:flex; align-items:flex-end; gap:2px; width:100%; min-width:0; height:72px; overflow:hidden; border-bottom:1px solid var(--line); }
    .trend-histogram-bar { position:relative; z-index:1; flex:1 1 0; min-width:0; opacity:.58; border-radius:2px 2px 0 0; }
    .trend-histogram-curve { position:absolute; inset:0; z-index:2; width:100%; min-width:0; height:100%; pointer-events:none; }
    .trend-histogram-curve polyline { fill:none; stroke-width:2; vector-effect:non-scaling-stroke; }
    .trend-histogram-labels { display:flex; justify-content:space-between; gap:8px; margin-top:3px; color:var(--muted); font-size:var(--font-xs); font-variant-numeric:tabular-nums; }
    .trend-histogram-empty { display:grid; place-items:center; width:100%; min-width:0; height:72px; color:var(--muted); font-size:var(--font-xs); border:1px dashed var(--line); border-radius:4px; }
    .swatch { width:18px; height:3px; border-radius:2px; display:inline-block; vertical-align:middle; margin-right:6px; }
    .table-wrap { overflow-x:auto; overflow-y:auto; max-height:560px; width:max-content; min-width:0; max-width:100%; border:1px solid var(--line); border-radius:8px; box-shadow:0 1px 2px rgba(15,23,42,.05); background:var(--panel); }
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
    @media (max-width:900px) { main { grid-template-columns:1fr; padding:12px; } .row { grid-template-columns:1fr; } .llm-config-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } .trend-stats { grid-template-columns:repeat(2, minmax(0, 1fr)); } .scatter-matrix-controls { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
    @media (max-width:560px) { .grid { grid-template-columns:1fr; } .llm-config-grid { grid-template-columns:1fr; } .trend-stats { grid-template-columns:1fr; } .scatter-matrix-controls { grid-template-columns:1fr; } }
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
        <div class="control-group-title">数据剔除</div>
        <label>强制剔除列（多选）
          <details id="excludedColumnsDropdown" class="multi-dropdown">
            <summary id="excludedColumnsSummary">未选择剔除列</summary>
            <div id="excludedColumnsOptions" class="multi-options"></div>
          </details>
        </label>
        <div class="help">选中的列将在所有分析、验证和图表处理前删除；原始上传文件不会被修改。</div>
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
        <label>重采样间隔（分钟）<input id="resampleRule" type="number" min="1" step="1" inputmode="numeric" placeholder="可留空，例如 5"></label>
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
        <button class="tab-button active" role="tab" aria-selected="true" aria-controls="trendTab" id="tab-trendTab" data-tab="trendTab" tabindex="0">趋势图</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="overviewTab" id="tab-overviewTab" data-tab="overviewTab" tabindex="-1">初步分析</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="validationTab" id="tab-validationTab" data-tab="validationTab" tabindex="-1">二次验证</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="causalReviewTab" id="tab-causalReviewTab" data-tab="causalReviewTab" tabindex="-1">三层复核</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="xgbValidationTab" id="tab-xgbValidationTab" data-tab="xgbValidationTab" tabindex="-1">四级验证</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="llmReportTab" id="tab-llmReportTab" data-tab="llmReportTab" tabindex="-1">AI 综合解读</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="downloadsTab" id="tab-downloadsTab" data-tab="downloadsTab" tabindex="-1">下载</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="termsHelpTab" id="tab-termsHelpTab" data-tab="termsHelpTab" tabindex="-1">术语与标签说明</button>
      </div>

      <div id="overviewTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-overviewTab" hidden>
        <h2>初步分析</h2>
        <div class="actions"><button id="analyze" disabled>开始分析</button></div>
        <div id="overview" class="overview-grid"></div>
        <div id="analysisTimingBreakdown" class="help" hidden></div>
        <h2>前 10 个推荐变量</h2>
        <div class="help">稳健综合得分同时考虑原始与变化量关联、增量预测、时间/工况稳定性、滞后质量和数据质量；证据缺失会降低证据覆盖度与修正系数，不会放大其他分项。</div>
        <div id="overviewTop" class="empty">上传数据并点击“开始分析”后显示结果。</div>
        <section id="candidatesTab">
          <h2>候选变量</h2>
          <div class="help">默认只展示候选排序结果的核心列和前 50 行，完整结果请到下载页获取。</div>
          <h3>结果质量提示</h3>
          <div id="screeningQualityHints" class="empty">完成主筛查后显示结果质量提示。</div>
          <div id="table" class="empty">上传数据并点击“开始分析”后显示结果。</div>
          <h2>轻量遗漏候选</h2>
          <div class="help">该表基于已有滞后相关、残差相关、峰值质量和风险标签生成，用于提示主筛查前 K 个外可能遗漏的候选。结果不代表因果结论。</div>
          <div id="nearMissTable" class="empty">完成主筛查后显示轻量遗漏候选。</div>
        </section>
      </div>

      <div id="trendTab" class="tab-panel active" role="tabpanel" aria-labelledby="tab-trendTab">
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
        <section class="scatter-matrix-section">
          <h2>XY 散点矩阵</h2>
          <div class="help">最多选择 3 个 X 轴变量和 3 个 Y 轴变量，按组合显示最多 9 个散点子图。散点关系用于人工观察变量关系、分群和异常点，不代表因果关系。</div>
          <div class="scatter-matrix-controls">
            <label>X 变量 1<select id="scatterX1"></select></label>
            <label>X 变量 2<select id="scatterX2"></select></label>
            <label>X 变量 3<select id="scatterX3"></select></label>
            <label>Y 变量 1<select id="scatterY1"></select></label>
            <label>Y 变量 2<select id="scatterY2"></select></label>
            <label>Y 变量 3<select id="scatterY3"></select></label>
            <button id="drawScatterMatrix" disabled>显示散点矩阵</button>
          </div>
          <div id="scatterMatrixMeta" class="help">选择 X 和 Y 变量后点击“显示散点矩阵”。</div>
          <div id="scatterMatrixChart" class="scatter-matrix-chart empty">选择至少一个 X 变量和一个 Y 变量。</div>
        </section>
      </div>

      <div id="validationTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-validationTab" hidden>
        <h2>二次验证</h2>
        <div class="help">
          <span>先完成主筛查，再按需运行增强筛选、Granger 预测验证或随机森林模型解释。结果会同步写入下载文件。</span>
          <span>Granger 显著表示历史预测信息，不等于因果成立；随机森林重要性表示模型依赖，不等于可操作性；模型提升低可能说明目标自身历史已解释大部分波动；滚动稳定性低说明关系可能受工况影响。</span>
          <span>原始数据：不重采样；继承主筛查：使用主筛查的分钟间隔；自定义：只填写分钟整数。二次验证仍沿用时间列、目标列、工况分段、缺失处理、预处理模式和标准化；若改用原始采样，请按原始采样间隔重新填写最大滞后点数。</span>
        </div>
        <div class="secondary-validation-params">
          <label>二次验证补充变量（白名单）
            <details id="secondaryIncludeDropdown" class="multi-dropdown">
              <summary id="secondaryIncludeSummary">请选择二次验证补充变量</summary>
              <div id="secondaryIncludeOptions" class="multi-options"></div>
            </details>
          </label>
          <label>二次验证重采样
            <select id="secondaryResampleMode">
              <option value="raw" selected>原始数据（不重采样）</option>
              <option value="inherit">继承主筛查</option>
              <option value="custom">自定义</option>
            </select>
          </label>
          <label>二次验证自定义重采样间隔（分钟）
            <input id="secondaryResampleRule" type="number" min="1" step="1" inputmode="numeric" placeholder="例如 2 或 5，仅自定义时使用">
          </label>
          <label>二次验证最大滞后点数
            <input id="secondaryMaxLag" type="number" min="0" max="5000" placeholder="默认继承主筛查最大滞后">
          </label>
        </div>
        <div class="actions">
          <button id="runEnhancedScreening" disabled>运行增强筛选</button>
          <button id="runGranger" disabled>运行 Granger 验证</button>
          <button id="runModel" disabled>运行随机森林模型解释</button>
        </div>
        <h2>增强筛选结果</h2>
        <div class="help">增强筛选用于补充验证主筛查候选的预测增益和时间稳定性，不代表因果结论。</div>
        <h3>增强筛选摘要</h3>
        <div id="enhancedSummaryTable" class="empty">点击“运行增强筛选”后显示增强筛选摘要。</div>
        <div class="help">术语说明：模型提升表示加入该候选变量后，相对仅使用目标变量历史的自回归基准，时间外预测 RMSE 的改善；大于 0 表示误差下降。滚动稳定性表示固定最佳滞后后，在多个时间窗口中相关关系的稳定程度，综合相关强度、符号一致性和波动离散度，范围为 0 至 1；越高越稳定。</div>
        <h3>模型提升评分</h3>
        <div id="enhancedLiftTable" class="empty">点击“运行增强筛选”后显示模型提升评分。</div>
        <div class="help">术语说明：模型提升得分为分段提升中位数（以 5% 改善为满分并截断至 0 至 1）与正提升分段比例的乘积；越高表示提升越稳定。自回归基准 RMSE 是只使用目标变量自身历史值时，各时间外测试分段的平均预测误差。候选变量模型 RMSE 是在同一基准上加入该候选变量滞后值后的平均预测误差；在相同验证条件下，数值越低越好。</div>
        <h3>滚动稳定性评分</h3>
        <div id="enhancedRollingTable" class="empty">点击“运行增强筛选”后显示滚动稳定性评分。</div>
        <div class="help">术语说明：滚动稳定性为固定最佳滞后后，按时间窗口计算相关性并综合相关强度、符号一致性和相关波动得到的 0 至 1 分数；越高表示该相关关系在不同时间段中越稳定，不代表因果结论。</div>
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
        <div class="help">
          <span>所有结果仅作为“预测验证/人工复核建议”，不是因果结论。可在左侧设置前 N 个候选变量和风险标签包含过滤后运行。</span>
          <span>三层复核支持长滞后变量。默认围绕主筛查最佳滞后附近做条件 Granger 验证，避免对 1..maxlag 全量扫描造成计算过慢。如需完整扫描，可切换为 full_scan。</span>
          <span>高共线性和共同负荷风险不等于变量无效。对于统计证据支持较强的候选，平台会保留工程复核建议，同时标记统计检验受限。</span>
        </div>
        <div class="causal-review-params">
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
        <h2>最终推荐摘要</h2>
        <div class="help">该表基于逐变量综合证据复核表生成，用于给出人工复核优先级清单。结果仍是预测验证和复核建议，不是因果结论。请优先按“最终排序”查看；点击其它列排序仅用于辅助查看。点击“查看趋势”可自动带入目标变量和候选变量，用于人工检查滞后方向、响应形态和工艺合理性。</div>
        <div class="download-buttons" id="finalReviewSummaryDownload"></div>
        <h3>最终推荐结果质检总览</h3>
        <div id="finalReviewQualityOverview" class="overview-grid"></div>
        <div id="finalReviewSummaryTable" class="empty">未运行 最终推荐摘要。</div>
        <h2>逐变量综合证据复核表</h2>
        <div class="help">逐变量综合证据复核表会整合主筛查、增强筛选、Granger、随机森林模型解释、条件 Granger 和风险标签。对于高共线性、共同负荷等统计限制，若统计证据支持较强，平台会保留工程复核建议并标记统计受限。该表仍不是因果结论。</div>
        <div class="download-buttons" id="causalEvidenceDownload"></div>
        <div id="causalReviewEvidenceTable" class="empty">未运行 逐变量综合证据复核表。</div>
      </div>


      <div id="xgbValidationTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-xgbValidationTab" hidden>
        <h2>XGB 四级验证</h2>
        <div class="help">XGB 结果表示时间外预测增量，不代表工艺因果成立，也不改变前三层排名。</div>
        <div class="row">
          <label class="checkbox-row"><input id="enableXgbValidation" type="checkbox">启用 XGB 验证</label>
          <label>候选数量<input id="xgbTopN" type="number" min="1" max="10" value="8"></label>
          <label>最大滞后<input id="xgbMaxLag" type="number" min="1" max="5000" placeholder="自动"></label>
          <label>白名单<input id="xgbWhitelist" placeholder="变量名以逗号分隔"></label>
        </div>
        <div class="help">自动候选默认 8 个、最多 10 个；加入白名单后，总候选数量最多 12 个。</div>
        <div class="actions">
          <button id="runXgbValidation" disabled>运行 XGB 四级验证</button>
        </div>
        <div id="xgbStatus" class="help" aria-live="polite">XGB 四级验证未启用。</div>
        <div id="xgbRunSummary" class="overview-grid"></div>
        <h2>模型时间外验证摘要</h2>
        <div class="download-buttons" id="xgbModelSummaryDownload"></div>
        <div id="xgbModelSummaryTable" class="empty">未运行 XGB 四级验证。</div>
        <h2>候选变量增量验证</h2>
        <div class="help">
          <span>RMSE 改善中位数（%）：各时间外测试折中，相对 M1 基线模型的 RMSE 改善百分比中位数；改善率 = (baseline_error - candidate_error) / baseline_error × 100%。大于 0 表示加入该候选后预测误差下降，小于 0 表示预测误差上升。</span>
          <span>MAE 改善中位数（%）：各时间外测试折中，相对 M1 基线模型的 MAE 改善百分比中位数。大于 0 表示平均绝对误差下降。</span>
          <span>RMSE 改善折占比：RMSE 改善百分比大于 0 的时间折数占全部验证折数的比例，范围为 0～1；例如 0.67 表示约 67% 的时间折得到改善。数值越高，跨时间段改善越稳定。</span>
          <span>以上指标均为时间外预测增量证据，不代表工艺因果成立。</span>
        </div>
        <div class="download-buttons" id="xgbCandidateUpliftDownload"></div>
        <div id="xgbCandidateUpliftTable" class="empty">未运行 XGB 四级验证。</div>
        <div class="download-buttons" id="xgbValidationSummaryDownload"></div>
      </div>


      <div id="llmReportTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-llmReportTab" hidden>
        <h2>AI 综合解读</h2>
        <div class="help">填写 API 配置后可直接调用 DeepSeek/OpenAI 兼容聊天补全接口生成报告。API 密钥仅随本次请求发送，不保存到磁盘、不写入报告。</div>
        <div class="llm-config-grid">
          <label>分析变量数量<input id="llmTopN" type="number" min="1" max="100" value="20"></label>
          <label>报告类型
            <select id="llmReportType">
              <option value="apc_advice">工程建议</option>
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
          <label>最大输出 Token 数<input id="llmMaxTokens" type="number" min="256" max="32000" value="15000"></label>
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
let currentAnalysisContext = {};
let recognizedColumns = [];
let recognizedNumericColumns = [];
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
let lastCausalEvidenceRows = [];
let lastFinalReviewSummaryRows = [];
let lastXgbModelSummaryRows = [];
let lastXgbCandidateUpliftRows = [];
let lastXgbValidationSummary = {};
let llmPromptText = "";
let llmReportMarkdown = "";
let lastModalTrigger = null;
let tableSortStates = { table: { column: "driver_rank", direction: "asc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
const el = (id) => document.getElementById(id);
const trendColors = ["#176b87", "#c2410c", "#6d28d9", "#15803d"];
const llmPromptEndpoint = "/api/llm_prompt";
let lastTrendSeries = [];
let lastTrendAxisMode = "shared";
let trendResizeTimer = null;
let trendTimeRangeMode = "auto";
let trendSamplingIntervalMs = null;
let trendLatestTime = "";
let trendAutoWindowActive = false;
let lastScatterMatrixPayload = null;
let scatterMatrixResizeTimer = null;
const lagProfileCache = new Map();
let lagProfileRequestSerial = 0;
let lastLagProfile = null;
let lagProfileResizeTimer = null;

for (const button of document.querySelectorAll(".tab-button")) {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => handleTabKeydown(event, button));
}
activateTab("trendTab");
el("drawTrend").addEventListener("click", drawTrend);
el("trendStart").addEventListener("input", markTrendTimeRangeManual);
el("trendEnd").addEventListener("input", markTrendTimeRangeManual);
el("trendMaxPoints").addEventListener("change", updateAutoTrendTimeRange);
el("drawScatterMatrix").addEventListener("click", drawScatterMatrix);
el("runEnhancedScreening").addEventListener("click", runEnhancedScreening);
el("runGranger").addEventListener("click", runGranger);
el("runModel").addEventListener("click", runModel);
el("runCausalReview").addEventListener("click", runCausalReview);
el("runXgbValidation").addEventListener("click", runXgbValidation);
el("enableXgbValidation").addEventListener("change", updateXgbRunAvailability);
el("detailModalClose").addEventListener("click", closeDetailModal);
el("detailModal").addEventListener("click", (event) => { if (event.target === el("detailModal")) closeDetailModal(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDetailModal(); });
window.addEventListener("resize", () => {
  if (lastTrendSeries.length) {
    const trendContainer = el("trendChart");
    if (isElementVisible(trendContainer)) {
      clearTimeout(trendResizeTimer);
      trendResizeTimer = setTimeout(() => renderTrendChart(lastTrendSeries, lastTrendAxisMode), 120);
    }
  }
  if (lastScatterMatrixPayload) {
    const scatterContainer = el("scatterMatrixChart");
    if (isElementVisible(scatterContainer)) {
      clearTimeout(scatterMatrixResizeTimer);
      scatterMatrixResizeTimer = setTimeout(() => renderScatterMatrix(lastScatterMatrixPayload), 120);
    }
  }
  if (lastLagProfile) {
    const lagPanel = el("lagProfilePanel");
    if (lagPanel && lagPanel.isConnected && lagPanel.dataset.lagProfileKey === lastLagProfile.key) {
      clearTimeout(lagProfileResizeTimer);
      lagProfileResizeTimer = setTimeout(() => renderLagProfile(lastLagProfile.payload, lagPanel), 120);
    }
  }
});
el("testLlmConnection").addEventListener("click", testLlmConnection);
el("generateLlmReport").addEventListener("click", generateLlmReport);
el("copyLlmReport").addEventListener("click", copyLlmReport);

el("upload").addEventListener("click", uploadFile);
el("analyze").addEventListener("click", analyze);
el("reset").addEventListener("click", reset);
el("encoding").addEventListener("change", () => { if (fileId) loadColumns(); });
el("timeColumn").addEventListener("change", handleProtectedColumnChange);
el("targetColumn").addEventListener("change", handleProtectedColumnChange);


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

function fillSecondaryIncludeOptions(columns) {
  const box = el("secondaryIncludeOptions");
  box.innerHTML = "";
  columns.forEach((name) => {
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox" value="${escapeHtml(name)}"> <span>${escapeHtml(name)}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("change", updateSecondaryIncludeSummary);
    box.appendChild(row);
  });
  updateSecondaryIncludeSummary();
}

function getSecondaryIncludeSelection() {
  return Array.from(document.querySelectorAll('#secondaryIncludeOptions input[type="checkbox"]:checked')).map((node) => node.value);
}

function updateSecondaryIncludeSummary() {
  const selected = getSecondaryIncludeSelection();
  el("secondaryIncludeSummary").textContent = selected.length ? `已选 ${selected.length} 项` : "请选择二次验证补充变量";
}

function getExcludedColumnSelection() {
  return Array.from(document.querySelectorAll('#excludedColumnsOptions input[type="checkbox"]:checked')).map((node) => node.value);
}

function setExcludedColumnSelection(values) {
  const selected = new Set(values || []);
  Array.from(document.querySelectorAll('#excludedColumnsOptions input[type="checkbox"]')).forEach((node) => {
    node.checked = selected.has(node.value) && !node.disabled;
  });
  updateExcludedColumnsSummary();
}

function updateExcludedColumnsSummary() {
  const selected = getExcludedColumnSelection();
  el("excludedColumnsSummary").textContent = selected.length ? `已剔除 ${selected.length} 列` : "未选择剔除列";
}

function updateExcludedColumnDisabledState() {
  const protectedColumns = new Set([el("timeColumn").value, el("targetColumn").value].filter(Boolean));
  Array.from(document.querySelectorAll('#excludedColumnsOptions input[type="checkbox"]')).forEach((input) => {
    input.disabled = protectedColumns.has(input.value);
    if (input.disabled) input.checked = false;
  });
  updateExcludedColumnsSummary();
}

function fillExcludedColumnOptions(columns) {
  const previous = new Set(getExcludedColumnSelection());
  const box = el("excludedColumnsOptions");
  box.innerHTML = "";
  columns.forEach((name) => {
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox" value="${escapeHtml(name)}"> <span>${escapeHtml(name)}</span>`;
    const input = row.querySelector("input");
    input.checked = previous.has(name);
    input.addEventListener("change", () => {
      updateExcludedColumnsSummary();
      refreshColumnSelectors();
    });
    box.appendChild(row);
  });
  updateExcludedColumnDisabledState();
}

function restoreSelect(id, values, currentValue, allowEmpty = false, emptyLabel = "不选择") {
  const select = el(id);
  fillSelect(select, values, allowEmpty, emptyLabel);
  if (currentValue && values.includes(currentValue)) {
    select.value = currentValue;
  } else if (currentValue) {
    select.value = "";
  }
}

function refreshColumnSelectors() {
  const excluded = new Set(getExcludedColumnSelection());
  const available = recognizedNumericColumns.filter((name) => !excluded.has(name));
  const current = Object.fromEntries(
    ["targetColumn", "segmentColumn", "trendVar1", "trendVar2", "trendVar3", "trendVar4", "scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"]
      .map((id) => [id, el(id).value])
  );
  const capacity = getCapacitySelection().filter((name) => !excluded.has(name));
  const forced = getForceIncludeSelection().filter((name) => !excluded.has(name));
  const secondary = getSecondaryIncludeSelection().filter((name) => !excluded.has(name));

  restoreSelect("targetColumn", available, current.targetColumn);
  restoreSelect("segmentColumn", available, current.segmentColumn, true, "不分段");
  ["trendVar1", "trendVar2", "trendVar3", "trendVar4", "scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"].forEach((id) => {
    restoreSelect(id, available, current[id], true, "不选择");
  });
  fillCapacityOptions(available);
  setCapacitySelection(capacity);
  fillForceIncludeOptions(available);
  setForceIncludeSelection(forced);
  fillSecondaryIncludeOptions(available);
  setSecondaryIncludeSelection(secondary);
  const whitelist = el("xgbWhitelist").value.split(/[,，]/).map((value) => value.trim()).filter((value) => value && !excluded.has(value));
  el("xgbWhitelist").value = whitelist.join(",");
  updateExcludedColumnDisabledState();
}

function setSecondaryIncludeSelection(values) {
  const selected = new Set(values || []);
  Array.from(document.querySelectorAll('#secondaryIncludeOptions input[type="checkbox"]')).forEach((node) => {
    node.checked = selected.has(node.value);
  });
  updateSecondaryIncludeSummary();
}

function handleProtectedColumnChange() {
  const protectedColumns = new Set([el("timeColumn").value, el("targetColumn").value].filter(Boolean));
  setExcludedColumnSelection(
    getExcludedColumnSelection().filter((name) => !protectedColumns.has(name))
  );
  refreshColumnSelectors();
}

function validateAnalysisColumnSelection() {
  const timeColumn = el("timeColumn").value;
  const target = el("targetColumn").value;
  const excluded = new Set(getExcludedColumnSelection());
  if (!timeColumn) return "请选择时间列";
  if (!target) return "请选择目标列";
  if (excluded.has(timeColumn)) return `剔除列不能同时作为时间列：${timeColumn}`;
  if (excluded.has(target)) return `剔除列不能同时作为目标列：${target}`;
  const protectedColumns = [el("segmentColumn").value, ...getCapacitySelection(), ...getForceIncludeSelection()].filter(Boolean);
  const conflicts = protectedColumns.filter((name) => excluded.has(name));
  if (conflicts.length) return `剔除列与工况/控制/白名单参数冲突：${Array.from(new Set(conflicts)).join("、")}`;
  const candidates = recognizedNumericColumns.filter((name) => name !== target && name !== timeColumn && !excluded.has(name));
  if (!candidates.length) return "剔除后至少需要保留一个可分析数值候选列";
  return "";
}

function appendSecondaryValidationOptions(form) {
  form.append("secondary_include_variables", getSecondaryIncludeSelection().join(","));
  form.append("secondary_resample_mode", el("secondaryResampleMode").value || "raw");
  form.append("secondary_resample_rule", el("secondaryResampleRule").value.trim());
  form.append("secondary_max_lag", el("secondaryMaxLag").value);
}

async function uploadFile() {
  const file = el("fileInput").files[0];
  if (!file) return setStatus("请选择 CSV、Excel 或 TXT 数据文件。");
  clearLagProfileCache();
  currentAnalysisContext = {};
  try {
    setStatus("正在上传文件...", "loading");
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/upload", form);
    fileId = data.file_id;
    recognizedColumns = [];
    recognizedNumericColumns = [];
    setExcludedColumnSelection([]);
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
    recognizedColumns = data.columns || [];
    recognizedNumericColumns = data.numericColumns || [];
    fillSelect(el("timeColumn"), data.columns);
  fillSelect(el("targetColumn"), data.numericColumns);
  fillSelect(el("segmentColumn"), data.numericColumns, true);
  fillCapacityOptions(data.numericColumns);
  fillForceIncludeOptions(data.numericColumns);
  fillSecondaryIncludeOptions(data.numericColumns);
  el("capacityDropdown").open = false;
  el("forceIncludeDropdown").open = false;
  el("secondaryIncludeDropdown").open = false;
  fillSelect(el("trendVar1"), data.numericColumns);
  fillSelect(el("trendVar2"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar3"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar4"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterX1"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterX2"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterX3"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterY1"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterY2"), data.numericColumns, true, "不选择");
  fillSelect(el("scatterY3"), data.numericColumns, true, "不选择");
  lastScatterMatrixPayload = null;
  clearScatterMatrix();
  const timeCandidate = data.columns.find((name) => /time|date|timestamp|时间|日期/i.test(name));
  if (data.timeColumn && data.columns.includes(data.timeColumn)) {
    el("timeColumn").value = data.timeColumn;
  } else if (timeCandidate) {
    el("timeColumn").value = timeCandidate;
  }
  trendTimeRangeMode = "auto";
  trendSamplingIntervalMs = Number(data.trendSamplingIntervalMs);
  trendLatestTime = data.timeEnd || "";
  trendAutoWindowActive = false;
  if (data.trendStartDefault) el("trendStart").value = data.trendStartDefault;
  if (data.trendEndDefault) el("trendEnd").value = data.trendEndDefault;
    const loadCandidate = data.numericColumns.find((name) => /load|负荷|进料|流量|feed|rate/i.test(name));
    if (loadCandidate) {
      el("segmentColumn").value = loadCandidate;
      setCapacitySelection([loadCandidate]);
    }
  fillExcludedColumnOptions(recognizedNumericColumns);
  refreshColumnSelectors();
  el("analyze").disabled = false;
  el("drawTrend").disabled = data.numericColumns.length < 1;
  el("drawScatterMatrix").disabled = data.numericColumns.length < 1;
    const timeColumnStatus = data.autoTimeColumn ? `已自动识别时间列：${data.autoTimeColumn}。` : "";
    setStatus(`${timeColumnStatus}列识别完成。编码：${data.encoding}。采样读取 ${data.sampleRows} 行，识别到 ${data.columns.length} 列。`, "success");
  } catch (error) {
    el("analyze").disabled = true;
    setStatus(error.message || String(error), "error");
  }
}

async function analyze() {
  if (!fileId) return setStatus("请先上传文件。");
  const validationError = validateAnalysisColumnSelection();
  if (validationError) return setStatus(validationError, "error");
  clearLagProfileCache();
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
    form.append("excluded_columns", getExcludedColumnSelection().join(","));
    form.append("exclude_control_columns_from_candidates", "true");
    const data = await postForm("/api/analyze", form);
    currentRunId = data.run_id || "";
    const result = await waitForAnalysisResult(data.task_id);
    renderAnalysisResult(result);
    setStatus(formatCompletedAnalysisStatus(result), "success");
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

function finiteAnalysisSeconds(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

function formatAnalysisSeconds(value) {
  const seconds = finiteAnalysisSeconds(value);
  return seconds === null ? "" : `${seconds.toFixed(1)} 秒`;
}

function formatCompletedAnalysisStatus(result) {
  const runId = result && result.run_id ? result.run_id : "";
  const elapsed = formatAnalysisSeconds(result && result.elapsed_seconds);
  return elapsed
    ? `分析完成。总耗时：${elapsed}。运行 ID：${runId}`
    : `分析完成。运行 ID：${runId}`;
}

function renderAnalysisResult(data) {
  currentRunId = data.run_id || "";
  currentAnalysisContext = data.analysisContext || {};
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
  lastCausalEvidenceRows = [];
  lastFinalReviewSummaryRows = [];
  lastXgbModelSummaryRows = [];
  lastXgbCandidateUpliftRows = [];
  lastXgbValidationSummary = {};
  closeDetailModal();
  renderOverview(data.overview || {});
  renderAnalysisTimingBreakdown(data.analysis_timings || {});
  renderScreeningQualityHints(lastRows);
  tableSortStates["table"] = { column: "driver_rank", direction: "asc" };
  renderTable(lastRows);
  tableSortStates["overviewTop"] = { column: "driver_rank", direction: "asc" };
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
  renderGenericTable("xgbModelSummaryTable", lastXgbModelSummaryRows, xgbModelSummaryColumns());
  renderGenericTable("xgbCandidateUpliftTable", lastXgbCandidateUpliftRows, xgbCandidateUpliftColumns());
  renderXgbRunSummary(lastXgbValidationSummary);
  renderReviewDownloads(data.downloads || []);
  renderDownloads(data.downloads || []);
  el("runEnhancedScreening").disabled = !currentRunId;
  el("runGranger").disabled = !currentRunId;
  el("runModel").disabled = !currentRunId;
  el("runCausalReview").disabled = !currentRunId;
  updateXgbRunAvailability();
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
    appendSecondaryValidationOptions(form);
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
    appendSecondaryValidationOptions(form);
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
    appendSecondaryValidationOptions(form);
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
    lastCausalEvidenceRows = data.causalReviewEvidence || [];
    lastFinalReviewSummaryRows = data.finalReviewSummary || [];
    tableSortStates["finalReviewSummaryTable"] = { column: "final_rank", direction: "asc" };
    renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
    renderFinalReviewQualityOverview(lastFinalReviewSummaryRows);
    renderFinalReviewSummaryTable(lastFinalReviewSummaryRows);
    renderCausalReviewEvidenceTable(lastCausalEvidenceRows);
    renderReviewDownloads(data.downloads || []);
    renderDownloads(data.downloads || []);
    updateXgbRunAvailability();
    setStatus(appendElapsed(data.message || "三层复核完成。结果不是因果结论。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
    el("runCausalReview").disabled = !currentRunId;
  }
}


function updateXgbRunAvailability() {
  const enabled = el("enableXgbValidation").checked;
  el("runXgbValidation").disabled = !(enabled && currentRunId && lastFinalReviewSummaryRows.length);
  if (!enabled) el("xgbStatus").textContent = "XGB 四级验证未启用。";
}

async function runXgbValidation() {
  if (!el("enableXgbValidation").checked) {
    el("xgbStatus").textContent = "请先启用 XGB 四级验证。";
    return;
  }
  if (!currentRunId || !lastFinalReviewSummaryRows.length) {
    el("xgbStatus").textContent = "请先完成三层复核。";
    return;
  }
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行 XGB 四级验证...", startedAt);
  el("runXgbValidation").disabled = true;
  el("xgbStatus").textContent = "正在运行 XGB 四级验证...";
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    form.append("enable_xgb_validation", "true");
    form.append("top_n", el("xgbTopN").value || "8");
    form.append("max_lag", el("xgbMaxLag").value);
    form.append("whitelist", el("xgbWhitelist").value.trim());
    form.append("control_columns", getCapacitySelection().join(","));
    const data = await postForm("/api/run_xgb_validation", form);
    lastXgbModelSummaryRows = data.xgbModelSummary || [];
    lastXgbCandidateUpliftRows = data.xgbCandidateUplift || [];
    lastXgbValidationSummary = data.xgbValidationSummary || {};
    renderGenericTable("xgbModelSummaryTable", lastXgbModelSummaryRows, xgbModelSummaryColumns());
    renderGenericTable("xgbCandidateUpliftTable", lastXgbCandidateUpliftRows, xgbCandidateUpliftColumns());
    renderXgbRunSummary(lastXgbValidationSummary);
    renderXgbDownloads(data.status === "success" ? (data.downloads || []) : []);
    renderDownloads(data.downloads || []);
    const message = data.error_message || data.message || "XGB 四级验证失败。";
    const success = data.status === "success";
    el("xgbStatus").textContent = appendElapsed(message, startedAt);
    setStatus(appendElapsed(message, startedAt), success ? "success" : "error");
  } catch (error) {
    const message = appendElapsed(error.message || String(error), startedAt);
    el("xgbStatus").textContent = message;
    setStatus(message, "error");
  } finally {
    stopStatusTimer(timerId);
    updateXgbRunAvailability();
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
    form.append("max_tokens", el("llmMaxTokens").value || "15000");
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
  { category: "参数设置说明", name: "风险标签包含过滤", signal: "按风险标签文本筛选复核或推荐结果。", reading: "只改变页面查看和复核聚焦范围，不表示未显示变量没有风险。", action: "用于定位共同负荷、数据质量等特定问题，留空表示不过滤。" },
  { category: "风险标签说明", name: "滞后边界风险", signal: "最佳滞后贴近扫描窗口边界，峰值可能尚未完全覆盖。", reading: "当前最大滞后点数可能偏小，真实响应时间可能更长。", action: "扩大最大滞后点数，结合趋势图确认峰值是否继续外移。" },
  { category: "风险标签说明", name: "变量滞后目标风险", signal: "页面显示为变量滞后目标。", reading: "变量变化晚于目标，更像响应量或受同一扰动影响。", action: "优先检查工艺方向，通常不直接作为前馈变量。" },
  { category: "风险标签说明", name: "公式泄漏 / 计算耦合风险", signal: "候选变量可能由目标或其上下游计算项派生。", reading: "高相关可能来自公式、软测量或报表口径耦合。", action: "核对 DCS/ historian 点位定义，剔除直接计算关系后再复核。" },
  { category: "风险标签说明", name: "数据质量风险", signal: "缺失、常数段、异常尖峰或有效比例不足影响结果。", reading: "统计指标可能受采样、坏点或仪表状态驱动。", action: "先清洗数据、确认仪表有效性，再重新运行分析。" },
  { category: "风险标签说明", name: "共线性风险", signal: "多个候选变量高度同步或代表同一工艺负荷。", reading: "模型可能难以区分真正贡献变量，单变量解释不稳定。", action: "做变量分组、残差控制或条件 Granger 预测验证。" },
  { category: "证据等级与复核建议", name: "强预测证据", signal: "相关、模型提升、预测贡献、稳定性等多类证据同时较好。", reading: "该变量对预测目标有较稳定信息量，但仍不是因果结论。", action: "进入优先复核，结合机理、趋势和现场操作记录确认。" },
  { category: "证据等级与复核建议", name: "风险受限证据", signal: "统计证据较强，但伴随共线性、共同负荷或数据质量等限制。", reading: "变量可能重要，但证据解释需要更谨慎。", action: "保留观察，先排除风险来源，再决定是否用于工程复核。" },
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

function isElementVisible(node) {
  return Boolean(
    node &&
    !node.hidden &&
    node.offsetParent !== null &&
    node.getClientRects().length
  );
}

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
  if (tabId === "trendTab") {
    requestAnimationFrame(() => {
      if (lastTrendSeries.length && isElementVisible(el("trendChart"))) {
        renderTrendChart(lastTrendSeries, lastTrendAxisMode);
      }
      if (lastScatterMatrixPayload && isElementVisible(el("scatterMatrixChart"))) {
        renderScatterMatrix(lastScatterMatrixPayload);
      }
    });
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

function markTrendTimeRangeManual() {
  trendTimeRangeMode = "manual";
  trendAutoWindowActive = false;
}

function updateAutoTrendTimeRange() {
  if (trendTimeRangeMode !== "auto" || !Number.isFinite(trendSamplingIntervalMs) || trendSamplingIntervalMs <= 0) return;
  const start = new Date(el("trendStart").value);
  const requestedPoints = Number(el("trendMaxPoints").value || "10000");
  if (Number.isNaN(start.getTime()) || !Number.isFinite(requestedPoints)) return;
  const maxPoints = Math.min(100000, Math.max(100, Math.trunc(requestedPoints)));
  const latest = new Date(trendLatestTime);
  const calculatedEnd = new Date(start.getTime() + (maxPoints - 1) * trendSamplingIntervalMs);
  const end = !Number.isNaN(latest.getTime()) && calculatedEnd > latest ? latest : calculatedEnd;
  const pad = (value) => String(value).padStart(2, "0");
  el("trendEnd").value = `${end.getFullYear()}-${pad(end.getMonth() + 1)}-${pad(end.getDate())}T${pad(end.getHours())}:${pad(end.getMinutes())}`;
  trendAutoWindowActive = true;
}

function appendChartQueryParams(params) {
  params.set("file_id", fileId);
  params.set("encoding", el("encoding").value);
  params.set("time_column", el("timeColumn").value);
  params.set("trend_start", el("trendStart").value);
  params.set("trend_end", el("trendEnd").value);
  params.set("trend_max_points", el("trendMaxPoints").value || "10000");
  params.set("segment_column", el("segmentColumn").value);
  params.set("segment_mode", el("segmentMode").value);
  params.set("segment_min", el("segmentMin").value);
  params.set("segment_max", el("segmentMax").value);
  params.set("preprocess_mode", el("preprocessMode").value);
  params.set("detrend_window", el("detrendWindow").value);
  params.set("excluded_columns", getExcludedColumnSelection().join(","));
}

async function drawTrend() {
  try {
    const variables = [el("trendVar1").value, el("trendVar2").value, el("trendVar3").value, el("trendVar4").value].filter(Boolean);
    if (!variables.length) return setStatus("请至少选择一个趋势变量。");
    if (new Set(variables).size !== variables.length) return setStatus("趋势变量不能重复选择。");
    const params = new URLSearchParams();
    appendChartQueryParams(params);
    params.set("time_range_mode", trendAutoWindowActive ? "auto" : "manual");
    params.set("variables", variables.join(","));
    setStatus("正在生成趋势图...", "loading");
    const response = await fetch(`/api/trend?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "趋势图生成失败");
    const series = data.series || [];
    lastTrendSeries = series;
    lastTrendAxisMode = el("trendAxisMode").value;
    renderTrendChart(series, lastTrendAxisMode);
    setStatus(`趋势图已生成，原始 ${data.raw_rows} 点，显示 ${data.rows} 点，最大点数 ${data.max_points}。`, "success");
  } catch (error) {
    lastTrendSeries = [];
    lastTrendAxisMode = "shared";
    el("trendChart").className = "chart empty";
    el("trendChart").textContent = error.message || String(error);
    el("trendLegend").innerHTML = "";
    clearTrendStats();
    setStatus(error.message || String(error), "error");
  }
}

function selectedScatterVariables(prefix) {
  const ids = prefix === "x" ? ["scatterX1", "scatterX2", "scatterX3"] : ["scatterY1", "scatterY2", "scatterY3"];
  return Array.from(new Set(ids.map((id) => el(id).value.trim()).filter(Boolean)));
}

function clearScatterMatrix(message = "选择至少一个 X 变量和一个 Y 变量。") {
  const container = el("scatterMatrixChart");
  container.className = "scatter-matrix-chart empty";
  container.textContent = message;
  el("scatterMatrixMeta").textContent = "选择 X 和 Y 变量后点击“显示散点矩阵”。";
}

async function drawScatterMatrix() {
  if (!fileId) return setStatus("请先上传数据文件。", "warning");
  const xVariables = selectedScatterVariables("x");
  const yVariables = selectedScatterVariables("y");
  if (!xVariables.length) return setStatus("请选择至少一个 X 轴变量。", "warning");
  if (!yVariables.length) return setStatus("请选择至少一个 Y 轴变量。", "warning");
  const startedAt = performance.now();
  el("drawScatterMatrix").disabled = true;
  setStatus("正在生成 XY 散点矩阵...", "loading");
  try {
    const params = new URLSearchParams();
    appendChartQueryParams(params);
    params.set("x_variables", xVariables.join(","));
    params.set("y_variables", yVariables.join(","));
    const response = await fetch(`/api/scatter_matrix?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "散点矩阵生成失败");
    if (!Array.isArray(data.values) || data.values.length === 0) {
      throw new Error("当前时间范围、工况和预处理条件下没有可绘制的散点数据");
    }
    lastScatterMatrixPayload = data;
    renderScatterMatrix(data);
    el("scatterMatrixMeta").textContent = `实际绘图 ${data.rows || 0} 行；筛选后原始行数 ${data.raw_rows || 0}；${(data.x_variables || []).length} 个 X × ${(data.y_variables || []).length} 个 Y。`;
    setStatus(appendElapsed("XY 散点矩阵生成完成。", startedAt), "success");
  } catch (error) {
    lastScatterMatrixPayload = null;
    clearScatterMatrix(error.message || String(error));
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    el("drawScatterMatrix").disabled = !fileId;
  }
}

function fitCanvasText(context, text, maxWidth) {
  const value = String(text ?? "");
  if (context.measureText(value).width <= maxWidth) {
    return value;
  }

  const suffix = "…";
  let result = value;

  while (
    result.length > 1 &&
    context.measureText(result + suffix).width > maxWidth
  ) {
    result = result.slice(0, -1);
  }

  return result + suffix;
}

function finiteScatterNumber(value) {
  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return null;
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function renderScatterMatrix(payload) {
  const container = el("scatterMatrixChart");
  const xVariables = payload.x_variables || [];
  const yVariables = payload.y_variables || [];
  const columns = payload.columns || [];
  const values = payload.values || [];
  if (!xVariables.length || !yVariables.length || !values.length) {
    clearScatterMatrix("没有可绘制的散点数据。");
    return;
  }
  container.className = "scatter-matrix-chart";
  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "XY 散点矩阵");
  container.appendChild(canvas);
  const columnCount = xVariables.length;
  const rowCount = yVariables.length;
  const availableWidth = Math.max(container.clientWidth || 900, 720);
  const measureContext = canvas.getContext("2d");
  if (!measureContext) {
    clearScatterMatrix("当前浏览器无法创建 Canvas 绘图上下文。");
    return;
  }
  measureContext.font = "11px sans-serif";
  let maxYLabelWidth = 0;
  for (const yName of yVariables) {
    maxYLabelWidth = Math.max(maxYLabelWidth, measureContext.measureText(yName).width);
  }
  const leftLabelWidth = Math.min(220, Math.max(96, Math.ceil(maxYLabelWidth) + 18));
  const topLabelHeight = 38;
  const rightPadding = 16;
  const bottomPadding = 26;
  const usableWidth = Math.max(300, availableWidth - leftLabelWidth - rightPadding);
  const panelWidth = Math.max(260, Math.floor(usableWidth / Math.max(columnCount, 1)));
  const panelHeight = Math.max(220, Math.min(360, Math.round(panelWidth * 0.62)));
  const cssWidth = Math.max(availableWidth, leftLabelWidth + columnCount * panelWidth + rightPadding);
  const cssHeight = topLabelHeight + rowCount * panelHeight + bottomPadding;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * pixelRatio);
  canvas.height = Math.round(cssHeight * pixelRatio);
  const context = canvas.getContext("2d");
  if (!context) {
    clearScatterMatrix("当前浏览器无法创建 Canvas 绘图上下文。");
    return;
  }
  context.scale(pixelRatio, pixelRatio);
  context.font = "11px sans-serif";
  const columnIndex = new Map(columns.map((name, index) => [name, index]));
  xVariables.forEach((xName, col) => {
    const maxTextWidth = Math.max(40, panelWidth - 20);
    const displayName = fitCanvasText(context, xName, maxTextWidth);
    const textWidth = context.measureText(displayName).width;
    context.fillText(displayName, leftLabelWidth + col * panelWidth + Math.max(8, (panelWidth - textWidth) / 2), 24);
  });
  yVariables.forEach((yName, row) => {
    const displayName = fitCanvasText(context, yName, leftLabelWidth - 16);
    context.fillText(displayName, 8, topLabelHeight + row * panelHeight + 22);
  });
  for (let row = 0; row < rowCount; row += 1) {
    for (let col = 0; col < columnCount; col += 1) {
      const xName = xVariables[col];
      const yName = yVariables[row];
      const xIndex = columnIndex.get(xName);
      const yIndex = columnIndex.get(yName);
      const left = leftLabelWidth + col * panelWidth + 42;
      const top = topLabelHeight + row * panelHeight + 24;
      const width = panelWidth - 58;
      const height = panelHeight - 46;
      context.strokeStyle = "#d8dee8";
      context.strokeRect(left, top, width, height);
      if (xIndex === undefined || yIndex === undefined) {
        context.fillStyle = "#5f6b7a";
        context.fillText("变量列不存在", left, top + 20);
        drawCountLabel(left, top, 0);
        continue;
      }

      let validCount = 0;
      let xMin = Infinity;
      let xMax = -Infinity;
      let yMin = Infinity;
      let yMax = -Infinity;

      for (const valueRow of values) {
        const x = finiteScatterNumber (valueRow[xIndex]);
        const y = finiteScatterNumber (valueRow[yIndex]);
        if (x === null || y === null) {
          continue;
        }
        validCount += 1;
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }

      if (validCount === 0) {
        context.fillStyle = "#5f6b7a";
        context.fillText("无有效配对数据", left + 12, top + 24);
        drawCountLabel(left, top, validCount);
        continue;
      }
      if (xMin === xMax) { xMin -= 0.5; xMax += 0.5; }
      if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
      const xPadding = Math.max((xMax - xMin) * 0.05, Number.EPSILON);
      const yPadding = Math.max((yMax - yMin) * 0.05, Number.EPSILON);
      const xRange = { min: xMin - xPadding, max: xMax + xPadding };
      const yRange = { min: yMin - yPadding, max: yMax + yPadding };
      context.strokeStyle = "#edf1f5";
      context.fillStyle = "#5f6b7a";
      axisTicks(xRange, 4).forEach((tick) => { const px = left + ((tick - xRange.min) / (xRange.max - xRange.min)) * width; context.beginPath(); context.moveTo(px, top); context.lineTo(px, top + height); context.stroke(); context.fillText(formatAxisValue(tick), px - 14, top + height + 14); });
      axisTicks(yRange, 4).forEach((tick) => { const py = top + height - ((tick - yRange.min) / (yRange.max - yRange.min)) * height; context.beginPath(); context.moveTo(left, py); context.lineTo(left + width, py); context.stroke(); context.fillText(formatAxisValue(tick), left - 36, py + 4); });
      context.save();
      context.globalAlpha = 0.35;
      context.fillStyle = "#176b87";
      for (const valueRow of values) {
        const x = finiteScatterNumber (valueRow[xIndex]);
        const y = finiteScatterNumber (valueRow[yIndex]);
        if (x === null || y === null) {
          continue;
        }
        const px = left + ((x - xRange.min) / (xRange.max - xRange.min)) * width;
        const py = top + height - ((y - yRange.min) / (yRange.max - yRange.min)) * height;
        context.beginPath();
        context.arc(px, py, 1.7, 0, Math.PI * 2);
        context.fill();
      }
      context.restore();
      drawCountLabel(left, top, validCount);
    }
  }

  function drawCountLabel(left, top, validCount) {
    const countText = `n=${validCount}`;
    context.font = "12px sans-serif";
    const countWidth = context.measureText(countText).width;
    context.save();
    context.globalAlpha = 0.82;
    context.fillStyle = "#ffffff";
    context.fillRect(left + 4, top + 3, countWidth + 8, 16);
    context.restore();
    context.fillStyle = "#44546a";
    context.fillText(countText, left + 8, top + 15);
    context.font = "11px sans-serif";
  }
}

function renderTrendChart(series, axisMode) {
  const container = el("trendChart");
  if (!series.length) {
    lastTrendSeries = [];
    container.className = "chart empty";
    container.textContent = "没有可绘制的趋势数据。";
    el("trendLegend").innerHTML = "";
    clearTrendStats();
    return;
  }
  const width = trendChartWidth(container), height = 320, pad = { left: 76, right: axisMode === "independent" ? 76 : 28, top: 30, bottom: 44 };
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

function trendChartWidth(container) {
  const measured = Math.floor(container.getBoundingClientRect().width || container.clientWidth || 0);
  return Math.max(320, measured || 960);
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

function trendHistogram(points, requestedBinCount = 12) {
  const values = (points || [])
    .map((point) => Number(point.y))
    .filter((value) => Number.isFinite(value));
  if (!values.length) return { bins: [], min: NaN, max: NaN, mean: NaN, stddev: NaN, count: 0 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const stddev = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
  if (min === max) return { bins: [{ min, max, count: values.length }], min, max, mean, stddev, count: values.length };
  const binCount = Math.min(requestedBinCount, Math.max(1, Math.ceil(Math.sqrt(values.length))));
  const binWidth = (max - min) / binCount;
  const counts = Array(binCount).fill(0);
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.floor(((value - min) / (max - min)) * binCount));
    counts[index] += 1;
  }
  const bins = counts.map((count, index) => ({
    min: min + index * binWidth,
    max: index === binCount - 1 ? max : min + (index + 1) * binWidth,
    count,
  }));
  return { bins, min, max, mean, stddev, count: values.length };
}

function trendNormalCurve(histogram, sampleCount = 40) {
  if (!Number.isFinite(histogram.mean) || !Number.isFinite(histogram.stddev) || histogram.stddev <= 0 || histogram.min === histogram.max) return "";
  const peakDensity = 1 / (histogram.stddev * Math.sqrt(2 * Math.PI));
  return Array.from({ length: sampleCount + 1 }, (_, index) => {
    const ratio = index / sampleCount;
    const value = histogram.min + (histogram.max - histogram.min) * ratio;
    const z = (value - histogram.mean) / histogram.stddev;
    const density = peakDensity * Math.exp(-0.5 * z ** 2);
    return `${(ratio * 100).toFixed(2)},${(100 - (density / peakDensity) * 100).toFixed(2)}`;
  }).join(" ");
}

function renderTrendHistogram(points, color, variableName) {
  const histogram = trendHistogram(points);
  const safeName = escapeHtml(variableName);
  if (!histogram.bins.length) {
    return `<div class="trend-histogram"><div class="trend-histogram-title">数值分布</div><div class="trend-histogram-empty" role="img" aria-label="${safeName} 数值分布：无有效数据">无有效数据</div></div>`;
  }
  const maxCount = Math.max(...histogram.bins.map((bin) => bin.count));
  const bars = histogram.bins.map((bin) => {
    const height = bin.count ? Math.max(3, (bin.count / maxCount) * 100) : 0;
    const title = `${formatAxisValue(bin.min)} – ${formatAxisValue(bin.max)}：${bin.count}`;
    return `<span class="trend-histogram-bar" style="height:${height}%;background:${color}" title="${title}"></span>`;
  }).join("");
  const curvePoints = trendNormalCurve(histogram);
  const curve = curvePoints ? `<svg class="trend-histogram-curve" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><polyline points="${curvePoints}" stroke="${color}"/></svg>` : "";
  const curveLabel = curvePoints ? "，含拟合正态分布曲线" : "";
  return `<div class="trend-histogram"><div class="trend-histogram-title">数值分布</div><div class="trend-histogram-bars" role="img" aria-label="${safeName} 数值分布，共 ${histogram.count} 个有效点，${histogram.bins.length} 个箱${curveLabel}">${bars}${curve}</div><div class="trend-histogram-labels"><span>${formatAxisValue(histogram.min)}</span><span>${formatAxisValue(histogram.max)}</span></div></div>`;
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
    ["有效点数/占比", "countRatio"],
  ];
  node.className = "trend-stats";
  node.innerHTML = series.map((item, index) => {
    const stats = trendStats(item.points || []);
    const rows = statRows.map(([label, key]) => `<div><dt>${label}</dt><dd>${key === "countRatio" ? formatCountRatio(stats.count, stats.ratio) : formatAxisValue(stats[key])}</dd></div>`).join("");
    const histogram = renderTrendHistogram(item.points || [], trendColors[index % trendColors.length], item.name);
    return `<div class="trend-stat-card"><h3>${escapeHtml(item.name)}</h3><dl>${rows}</dl>${histogram}</div>`;
  }).join("");
}

function formatCountRatio(count, ratio) {
  const pct = Number.isFinite(ratio) ? `${(ratio * 100).toFixed(1)}%` : "0.0%";
  return `${count} / ${pct}`;
}

function trendStats(points) {
  const total = (points || []).length;
  const values = (points || []).map((point) => Number(point.y)).filter((value) => Number.isFinite(value));
  if (!values.length) return { mean: NaN, stddev: NaN, max: NaN, min: NaN, range: NaN, median: NaN, count: 0, ratio: 0 };
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
    ratio: total ? values.length / total : 0,
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

const CORRELATION_OVERVIEW_COLUMNS = [
  "lag", "direction", "method", "pearson", "spearman", "lag_boundary_flag",
];
const CORRELATION_DETAIL_COLUMNS = [
  "pearson_p", "pearson_q", "pearson_r2", "spearman_p", "spearman_q",
  "spearman_r2", "corr_q_value", "n",
];

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
  return ["variable", "driver_rank", "driver_priority_score", "candidate_grade", "layer1_association_status", "layer2_temporal_status", "layer3_independence_status", "layer4_model_status", "stability_status", "data_quality_status", "evidence_support_items", "evidence_against_items", "evidence_missing_items", "risk_flags", "candidate_summary"];
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
  const isScreeningCandidate = "driver_priority_score" in (row || {}) && "final_score" in (row || {});
  const groupedColumns = new Set(isScreeningCandidate ? [
      "driver_rank", "driver_priority_score", "final_score", "candidate_class",
      "driver_priority_factor", "evidence_coverage_status", "evidence_missing_items",
      "evidence_completeness", "data_quality_score", "evidence_confidence",
      "dominant_corr", "correlation_direction", ...CORRELATION_OVERVIEW_COLUMNS, ...CORRELATION_DETAIL_COLUMNS,
    ] : []);
  const rawFieldColumnsWithoutRiskFlags = columns.filter((column) => column !== "risk_flags" && !groupedColumns.has(column));
  const fields = rawFieldColumnsWithoutRiskFlags.map((column) => `
    <div class="detail-field">
      <strong>${escapeHtml(columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, getValue(row, column)))}</span>
    </div>
  `).join("");
  const scoreDetails = renderScreeningScoreDetails(row);
  return `
    <div class="review-card">
      <h3>变量：${escapeHtml(displayCellValue("variable", row.variable))}</h3>
      ${scoreDetails}
      <details class="raw-fields" open>
        <summary>展开完整原始字段</summary>
        <div class="detail-grid">${("risk_flags" in (row || {})) ? renderRiskTagDetails(row.risk_flags) : ""}${fields}</div>
      </details>
    </div>
  `;
}

function timeRelationshipExplanation(lag, intervalMinutes = null) {
  const value = lagProfileNumber(lag);
  const relationship = lagDirectionText(lag);
  if (relationship === "未计算") return "当前缺少可用的最佳滞后，无法判断时间关系，建议复核数据和时间对齐。";
  if (relationship === "同步变化") return "候选变量与目标变量主要表现为同步变化。";
  const interval = lagProfileNumber(intervalMinutes);
  const distance = interval !== null && interval > 0
    ? `约 ${Math.abs(value) * interval} 分钟`
    : ` ${Math.abs(value)} 个采样点`;
  if (relationship === "变量领先目标") return `候选变量领先目标变量${distance}。`;
  return `候选变量滞后目标变量${distance}，更可能表现为响应变量、反馈动作或下游状态，建议复核工艺关系。`;
}

function correlationDirectionExplanation(direction, preprocessMode) {
  const mode = String(preprocessMode ?? "");
  const messages = {
    raw: {
      正向: "在当前最佳滞后对齐下，候选变量水平较高时，目标变量水平通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量水平较高时，目标变量水平通常较低。",
      方向较弱: "当前最佳滞后点的原始水平相关系数接近零，相关方向较弱。",
    },
    detrend: {
      正向: "在当前最佳滞后对齐下，候选变量去趋势后的偏离较高时，目标变量去趋势后的偏离通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量去趋势后的偏离较高时，目标变量去趋势后的偏离通常较低。",
      方向较弱: "当前最佳滞后点的去趋势后相关系数接近零，相关方向较弱。",
    },
    diff: {
      正向: "在当前最佳滞后对齐下，候选变量增加时，目标变量通常也呈增加趋势。",
      负向: "在当前最佳滞后对齐下，候选变量增加时，目标变量通常呈下降趋势。",
      方向较弱: "当前最佳滞后点的变化量相关系数接近零，变化方向关系较弱。",
    },
    detrend_diff: {
      正向: "在当前最佳滞后对齐下，候选变量去趋势后的变化增加时，目标变量去趋势后的变化通常也增加。",
      负向: "在当前最佳滞后对齐下，候选变量去趋势后的变化增加时，目标变量去趋势后的变化通常下降。",
      方向较弱: "当前最佳滞后点的去趋势后变化量相关系数接近零，变化方向关系较弱。",
    },
  };
  if (messages[mode]?.[direction]) return messages[mode][direction];
  if (direction === "正向" || direction === "负向") {
    return `在当前预处理口径和最佳滞后对齐下，候选变量与目标变量呈${direction}关系。`;
  }
  if (direction === "方向较弱") return "在当前预处理口径和最佳滞后对齐下，相关方向较弱。";
  return "当前缺少可用的带符号相关结果，无法判断相关方向。";
}

function innovationDirectionExplanation(status, preprocessMode) {
  const mode = String(preprocessMode ?? "");
  if (mode === "diff" || mode === "detrend_diff") {
    return "当前主筛查已采用差分口径，变化量方向与主筛查方向来自同一组证据，未形成独立的主筛查—变化量一致性验证。";
  }
  const messages = {
    innovation_verified: "主筛查关系与变化量关系的方向和滞后基本一致。",
    innovation_sign_conflict: "主筛查关系与变化量关系方向冲突，可能存在共同趋势、工况混合或异常点影响。",
    innovation_lag_conflict: "主筛查关系与变化量关系的滞后不一致，动态关系可能不稳定。",
    innovation_sign_unknown: "变化量方向无法可靠判断。",
    not_computed: "未完成变化量方向验证。",
  };
  return messages[String(status ?? "")] || "变化量方向验证状态未知，建议复核主筛查结果。";
}

function innovationDirectionText(value) {
  if (value === null || value === undefined || value === "") return "未计算";
  const normalized = String(value);
  if (!["-1", "0", "1"].includes(normalized)) return "未计算";
  return displayCellValue("innovation_sign", value);
}

function preprocessModeLabel(mode) {
  const labels = {
    raw: "原始数据",
    detrend: "去趋势",
    diff: "一阶差分",
    detrend_diff: "去趋势后差分",
  };
  return labels[String(mode ?? "")] || "未知预处理口径";
}

function directionalitySummary(lag, correlationDirection, intervalMinutes = null) {
  const value = lagProfileNumber(lag);
  const relationship = lagDirectionText(lag);
  if (relationship === "未计算") return `时间关系未计算，${correlationDirection || "相关方向未计算"}。`;
  const interval = lagProfileNumber(intervalMinutes);
  const distance = interval !== null && interval > 0
    ? `约 ${Math.abs(value) * interval} 分钟`
    : ` ${Math.abs(value)} 个采样点`;
  const correlation = correlationDirection === "方向较弱"
    ? "相关方向较弱"
    : correlationDirection === "未计算" ? "相关方向未计算" : `${correlationDirection}相关`;
  if (relationship === "变量领先目标") return `变量领先目标${distance}，${correlation}。`;
  if (relationship === "变量滞后目标") return `变量滞后目标${distance}，${correlation}，更可能表现为响应或反馈关系。`;
  return `与目标同步变化，${correlation}。`;
}

function directionInteractionExplanation(lag, correlationDirection) {
  if (lagDirectionText(lag) === "变量领先目标" && correlationDirection === "负向") {
    return "候选变量先变化，并与之后的目标变化呈反向关系。";
  }
  return "时间关系与相关方向是两种独立信息，建议结合工艺机理和时间对齐复核。";
}

function updateDirectionalityTimeDetails(lag, intervalMinutes) {
  const explanation = el("directionalityTimeExplanation");
  const summary = el("directionalitySummary");
  if (explanation) explanation.textContent = timeRelationshipExplanation(lag, intervalMinutes);
  if (summary) summary.textContent = directionalitySummary(lag, summary.dataset.correlationDirection, intervalMinutes);
}

function renderScreeningScoreDetails(row) {
  if (!("driver_priority_score" in (row || {})) || !("final_score" in (row || {}))) return "";
  const rankingColumns = ["driver_rank", "driver_priority_score", "final_score", "candidate_class", "driver_priority_factor"];
  const evidenceColumns = ["layer1_association_status", "layer2_temporal_status", "layer3_independence_status", "layer4_model_status", "stability_status", "data_quality_status", "evidence_support_items", "evidence_against_items", "evidence_missing_items", "evidence_conflict_items", "risk_flags", "candidate_summary"];
  const renderFields = (columns, labels = {}) => columns.map((column) => `
    <div class="detail-field">
      <strong>${escapeHtml(labels[column] || columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, row[column]))}</span>
    </div>
  `).join("");
  const factor = numericValue(row.driver_priority_factor);
  const preprocessMode = currentAnalysisContext.preprocess_mode;
  const timeRelationship = lagDirectionText(row.lag);
  const correlationDirection = row.correlation_direction || "未计算";
  const innovationDirection = innovationDirectionText(row.innovation_sign);
  const innovationDirectionLabel = preprocessMode === "diff" || preprocessMode === "detrend_diff"
    ? "当前分析变化方向"
    : "变化量相关方向";
  const innovationExplanation = innovationDirectionExplanation(row.innovation_status, preprocessMode);
  const equalScoreNote = factor === 1 && row.candidate_class === "upstream_driver_candidate"
    ? "<p>当前候选属于上游驱动候选，优先系数为 1.00，因此两个得分相同。</p>"
    : "";
  return `
    <h4>排序结果</h4>
    <div class="detail-grid">${renderFields(rankingColumns)}</div>
    <p>驱动优先得分 = 稳健综合得分 × 候选类别优先系数。</p>
    ${equalScoreNote}
    <h4>证据覆盖</h4>
    <div class="detail-grid">${renderFields(evidenceColumns)}</div>
    <h4>相关性证据</h4>
    <div class="detail-grid">${renderFields(CORRELATION_OVERVIEW_COLUMNS, { lag: "最佳滞后点", direction: "滞后方向" })}</div>
    <details class="correlation-evidence-details">
      <summary>展开 P/Q、R² 与样本数</summary>
      <p class="help">大样本与时序自相关下，P/Q 值、R² 和样本数仅供参考，不参与评分、筛选、排序或颜色强调。</p>
      <div class="detail-grid">${renderFields(CORRELATION_DETAIL_COLUMNS)}</div>
    </details>
    <h4>方向性解释</h4>
    <p><strong>组合摘要：</strong><span id="directionalitySummary" data-correlation-direction="${escapeHtml(correlationDirection)}">${escapeHtml(directionalitySummary(row.lag, correlationDirection))}</span></p>
    <div class="detail-grid">
      <div class="detail-field"><strong>分析口径</strong><span>${escapeHtml(preprocessModeLabel(preprocessMode))}</span></div>
      <div class="detail-field"><strong>最佳滞后</strong><span>${escapeHtml(formatSignedLag(row.lag))}</span></div>
      <div class="detail-field"><strong>时间关系</strong><span>${escapeHtml(timeRelationship)}</span></div>
      <div class="detail-field"><strong>主导相关方法</strong><span>${escapeHtml(displayCellValue("method", row.method))}</span></div>
      <div class="detail-field"><strong>主导相关系数</strong><span>${escapeHtml(displayCellValue("dominant_corr", row.dominant_corr))}</span></div>
      <div class="detail-field"><strong>相关方向</strong><span>${escapeHtml(correlationDirection)}</span></div>
      <div class="detail-field"><strong>${escapeHtml(innovationDirectionLabel)}</strong><span>${escapeHtml(innovationDirection)}</span></div>
      <div class="detail-field"><strong>主筛查—变化量一致性</strong><span>${escapeHtml(innovationExplanation)}</span></div>
    </div>
    <p id="directionalityTimeExplanation">${escapeHtml(timeRelationshipExplanation(row.lag))}</p>
    <p>${escapeHtml(correlationDirectionExplanation(correlationDirection, preprocessMode))}</p>
    <p>${escapeHtml(directionInteractionExplanation(row.lag, correlationDirection))}</p>
    <p>${escapeHtml(correlationConsistencyMessage(row.pearson, row.spearman))}</p>
    <p class="help">时间领先和正负相关只表示当前数据中的时序关联，不等于因果方向。共同负荷、上游扰动和工况切换均可能产生类似结果。</p>
    <h4>滞后相关曲线</h4>
    <div id="lagProfilePanel" class="lag-profile-panel loading" aria-live="polite">正在加载滞后相关曲线……</div>
    <h4>解释说明</h4>
    <p>证据修正系数由证据覆盖度和数据质量共同计算，用于修正综合证据得分，不表示统计概率或因果置信度。</p>
  `;
}

function lagProfileCacheKey(runId, variable) {
  return JSON.stringify({ run_id: runId, variable });
}

function clearLagProfileCache() {
  lagProfileCache.clear();
  lagProfileRequestSerial += 1;
  lastLagProfile = null;
  clearTimeout(lagProfileResizeTimer);
}

async function loadLagProfile(row) {
  const panel = el("lagProfilePanel");
  if (!panel) return;
  const runId = currentRunId;
  const variable = String(row && row.variable || "");
  const key = lagProfileCacheKey(runId, variable);
  const requestId = ++lagProfileRequestSerial;
  lastLagProfile = null;
  panel.dataset.lagProfileKey = key;
  panel.className = "lag-profile-panel loading";
  panel.textContent = "正在加载滞后相关曲线……";
  if (!runId || !variable) {
    panel.className = "lag-profile-panel error";
    panel.textContent = "无法加载滞后相关曲线：缺少当前运行或变量信息。";
    return;
  }

  const cached = lagProfileCache.get(key);
  if (cached) {
    lastLagProfile = { key, payload: cached };
    renderLagProfile(cached, panel);
    return;
  }

  try {
    const params = new URLSearchParams({ run_id: runId, variable });
    const response = await fetch(`/api/lag_profile?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "滞后相关曲线加载失败");
    lagProfileCache.set(key, data);
    if (
      requestId !== lagProfileRequestSerial
      || currentRunId !== runId
      || !panel.isConnected
      || panel.dataset.lagProfileKey !== key
      || el("lagProfilePanel") !== panel
    ) return;
    lastLagProfile = { key, payload: data };
    renderLagProfile(data, panel);
  } catch (error) {
    if (
      requestId !== lagProfileRequestSerial
      || currentRunId !== runId
      || !panel.isConnected
      || panel.dataset.lagProfileKey !== key
      || el("lagProfilePanel") !== panel
    ) return;
    panel.className = "lag-profile-panel error";
    panel.textContent = `滞后相关曲线加载失败：${error.message || String(error)}`;
  }
}

function renderLagProfile(payload, panel = el("lagProfilePanel")) {
  if (!panel || !panel.isConnected) return;
  const points = (payload.points || [])
    .map((point) => ({ ...point, lag: Number(point.lag), pearson: lagProfileNumber(point.pearson), spearman: lagProfileNumber(point.spearman) }))
    .filter((point) => Number.isFinite(point.lag))
    .sort((left, right) => left.lag - right.lag);
  if (!points.length || !points.some((point) => point.pearson !== null || point.spearman !== null)) {
    panel.className = "lag-profile-panel error";
    panel.textContent = "该变量没有可绘制的滞后相关数据。";
    return;
  }

  const bestLag = Number(payload.best_lag);
  const bestPoint = points.find((point) => point.lag === bestLag) || null;
  const zeroPoint = points.find((point) => point.lag === 0) || null;
  const maxLag = Number(payload.max_lag);
  const boundary = Boolean(bestPoint && bestPoint.lag_boundary_flag)
    || (Number.isFinite(maxLag) && Math.abs(bestLag) === maxLag);
  const width = Math.max(560, Math.floor(panel.clientWidth || 760));
  const height = 300;
  const pad = { left: 54, right: 24, top: 28, bottom: 54 };
  let xMin = points[0].lag;
  let xMax = points[points.length - 1].lag;
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  const xScale = (lag) => pad.left + ((lag - xMin) / (xMax - xMin)) * (width - pad.left - pad.right);
  const yScale = (value) => pad.top + ((1 - value) / 2) * (height - pad.top - pad.bottom);
  const yTicks = [-1, -0.5, 0, 0.5, 1];
  const xTicks = Array.from(new Set([xMin, ...(xMin < 0 && xMax > 0 ? [0] : []), xMax]));
  const grid = yTicks.map((tick) => `
    <line x1="${pad.left}" y1="${yScale(tick)}" x2="${width - pad.right}" y2="${yScale(tick)}" stroke="var(--line-soft)"/>
    <text x="${pad.left - 8}" y="${yScale(tick) + 4}" text-anchor="end" fill="var(--muted)" font-size="11">${tick.toFixed(1)}</text>
  `).join("");
  const xLabels = xTicks.map((tick) => `
    <text x="${xScale(tick)}" y="${height - 30}" text-anchor="middle" fill="var(--muted)" font-size="11">${formatSignedLag(tick, false)}</text>
  `).join("");
  const zeroLine = xMin <= 0 && xMax >= 0 ? `
    <line x1="${xScale(0)}" y1="${pad.top}" x2="${xScale(0)}" y2="${height - pad.bottom}" stroke="#64748b" stroke-dasharray="4 4"/>
    <text x="${xScale(0) + 4}" y="${pad.top + 12}" fill="#64748b" font-size="11">lag = 0</text>
  ` : "";
  const bestAtRight = bestLag >= xMax;
  const bestLabelX = xScale(bestLag) + (bestAtRight ? -4 : 4);
  const bestLabelAnchor = bestAtRight ? "end" : "start";
  const bestLine = Number.isFinite(bestLag) ? `
    <line x1="${xScale(bestLag)}" y1="${pad.top}" x2="${xScale(bestLag)}" y2="${height - pad.bottom}" stroke="#d97706" stroke-width="1.5" stroke-dasharray="6 3"/>
    <text x="${bestLabelX}" y="${pad.top + 13}" text-anchor="${bestLabelAnchor}" fill="#b45309" font-size="11">当前最佳滞后 ${formatSignedLag(bestLag, false)}</text>
  ` : "";
  const bestMarkers = bestPoint ? [
    lagProfileBestMarker(bestPoint, "pearson", xScale, yScale, "#176b87", -22, "P", bestAtRight),
    lagProfileBestMarker(bestPoint, "spearman", xScale, yScale, "#c2410c", -8, "S", bestAtRight),
  ].join("") : "";

  updateDirectionalityTimeDetails(bestLag, payload.sampling_interval_minutes);
  panel.className = "lag-profile-panel";
  panel.innerHTML = `
    <div class="lag-profile-chart" role="img" aria-label="${escapeHtml(payload.variable)} 的 Pearson 与 Spearman 滞后相关曲线">
      <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
        ${grid}
        ${zeroLine}
        ${bestLine}
        <path d="${lagProfilePath(points, "pearson", xScale, yScale)}" fill="none" stroke="#176b87" stroke-width="2.3" vector-effect="non-scaling-stroke"/>
        <path d="${lagProfilePath(points, "spearman", xScale, yScale)}" fill="none" stroke="#c2410c" stroke-width="2.3" stroke-dasharray="7 4" vector-effect="non-scaling-stroke"/>
        ${lagProfileBoundaryMarkers(points, "pearson", xScale, yScale)}
        ${lagProfileBoundaryMarkers(points, "spearman", xScale, yScale)}
        ${bestMarkers}
        ${xLabels}
        <text x="${width / 2}" y="${height - 8}" text-anchor="middle" fill="var(--muted)" font-size="12">滞后点数</text>
        <text x="15" y="${height / 2}" text-anchor="middle" transform="rotate(-90 15 ${height / 2})" fill="var(--muted)" font-size="12">相关系数</text>
      </svg>
    </div>
    <div class="lag-profile-legend">
      <span><i class="lag-profile-line" style="border-color:#176b87"></i>Pearson 曲线</span>
      <span><i class="lag-profile-line spearman" style="border-color:#c2410c"></i>Spearman 曲线</span>
      <span>虚线标记：lag = 0 / 当前最佳滞后</span>
      <span>红圈：搜索边界点</span>
    </div>
    <div class="lag-profile-directions"><span>负值：变量滞后目标</span><span>0：同步变化</span><span>正值：变量领先目标</span></div>
    <div class="lag-profile-summary">
      ${lagProfileSummaryItem("同步 Pearson", zeroPoint && zeroPoint.pearson, "correlation")}
      ${lagProfileSummaryItem("同步 Spearman", zeroPoint && zeroPoint.spearman, "correlation")}
      ${lagProfileSummaryItem("最佳滞后", Number.isFinite(bestLag) ? formatSignedLag(bestLag) : "-")}
      ${lagProfileSummaryItem("最佳 Pearson", bestPoint && bestPoint.pearson, "correlation")}
      ${lagProfileSummaryItem("最佳 Spearman", bestPoint && bestPoint.spearman, "correlation")}
      ${lagProfileSummaryItem("主导方法", displayCellValue("method", payload.method))}
      ${lagProfileSummaryItem("是否触及边界", boundary ? "是" : "否")}
    </div>
    ${lagProfileTimeHint(bestLag, payload.sampling_interval_minutes)}
    <p class="lag-profile-message">${escapeHtml(correlationConsistencyMessage(bestPoint && bestPoint.pearson, bestPoint && bestPoint.spearman))}</p>
    ${boundary ? '<p class="lag-profile-warning">最佳滞后触及搜索边界，当前最大滞后点数可能偏小，建议结合工艺时间尺度复核。</p>' : ""}
    <p class="help">曲线来自本次主筛查已生成的 lag_scores.csv，仅用于人工观察，不参与评分、排序或因果判断。P/Q 值在大样本与时序自相关下仅供参考。</p>
  `;
}

function lagProfilePath(points, column, xScale, yScale) {
  let path = "";
  let drawing = false;
  for (const point of points) {
    const value = point[column];
    if (value === null || !Number.isFinite(value)) {
      drawing = false;
      continue;
    }
    path += `${drawing ? " L" : "M"}${xScale(point.lag).toFixed(2)},${yScale(value).toFixed(2)}`;
    drawing = true;
  }
  return path;
}

function lagProfileBoundaryMarkers(points, column, xScale, yScale) {
  return points.filter((point) => point.lag_boundary_flag && point[column] !== null).map((point) => `
    <circle cx="${xScale(point.lag)}" cy="${yScale(point[column])}" r="4.5" fill="var(--panel)" stroke="#dc2626" stroke-width="2"><title>搜索边界点；${columnLabel(column)} ${formatLagCorrelation(point[column])}</title></circle>
  `).join("");
}

function lagProfileBestMarker(point, column, xScale, yScale, color, labelOffset, prefix, alignRight) {
  const value = point[column];
  if (value === null || !Number.isFinite(value)) return "";
  const x = xScale(point.lag);
  const y = yScale(value);
  const labelX = x + (alignRight ? -6 : 6);
  const anchor = alignRight ? "end" : "start";
  return `<circle cx="${x}" cy="${y}" r="3" fill="${color}"/><text x="${labelX}" y="${y + labelOffset}" text-anchor="${anchor}" fill="${color}" font-size="11">${prefix} ${formatLagCorrelation(value)}</text>`;
}

function lagProfileSummaryItem(label, value, type = "text") {
  const text = type === "correlation" ? formatLagCorrelation(value) : String(value ?? "-");
  return `<div><strong>${escapeHtml(label)}</strong><span>${escapeHtml(text)}</span></div>`;
}

function formatLagCorrelation(value) {
  const number = lagProfileNumber(value);
  return number === null ? "-" : number.toFixed(3);
}

function lagProfileNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatSignedLag(value, includeUnit = true) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const signed = number > 0 ? `+${number}` : String(number);
  return includeUnit ? `${signed} 点` : signed;
}

function lagDirectionText(lag, missingText = "未计算") {
  const value = lagProfileNumber(lag);
  if (value === null) return missingText;
  if (value > 0) return "变量领先目标";
  if (value < 0) return "变量滞后目标";
  return "同步变化";
}

function lagProfileTimeHint(bestLag, intervalMinutes) {
  const interval = Number(intervalMinutes);
  if (!Number.isFinite(bestLag) || !Number.isFinite(interval) || interval <= 0) return "";
  const points = Math.abs(bestLag);
  const minutes = points * interval;
  return `<p class="lag-profile-message">时间换算：${points} 点 ≈ ${minutes} 分钟（${lagDirectionText(bestLag)}）。</p>`;
}

function correlationConsistencyMessage(pearson, spearman) {
  const p = lagProfileNumber(pearson);
  const s = lagProfileNumber(spearman);
  if (p === null || s === null) return "Pearson 或 Spearman 数据缺失，方法一致性暂无法判断。";
  const pStrength = Math.abs(p);
  const sStrength = Math.abs(s);
  const bothAwayFromZero = pStrength > 0.05 && sStrength > 0.05;
  if (bothAwayFromZero && Math.sign(p) !== Math.sign(s)) {
    return "Pearson 与 Spearman 的方向不一致，建议检查异常点、工况混合、分群和时间对齐。";
  }
  if (sStrength - pStrength >= 0.15) {
    return "Spearman 明显高于 Pearson，关系可能具有单调非线性、异常值影响或工况分群。";
  }
  if (pStrength - sStrength >= 0.15) {
    return "Pearson 明显高于 Spearman，结果可能受局部线性关系、极端值或数据分群影响。";
  }
  if (Math.sign(p) === Math.sign(s) && Math.abs(p - s) < 0.15) {
    return "Pearson 与 Spearman 方向和强度基本一致。";
  }
  return "Pearson 与 Spearman 存在一定差异，建议结合散点图、工况分层和时间对齐复核。";
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
  const elapsed = formatAnalysisSeconds(overview.analysis_elapsed_seconds);
  if (elapsed) metrics.push(["初步分析总耗时", elapsed]);
  el("overview").innerHTML = metrics.map(([label, value]) =>
    `<div class="metric-card"><span class="metric-value">${escapeHtml(formatValue(value))}</span><span class="metric-label">${escapeHtml(label)}</span></div>`
  ).join("");
}

function renderAnalysisTimingBreakdown(timings) {
  const node = el("analysisTimingBreakdown");
  const fields = [
    ["read_data_seconds", "读取数据"],
    ["analysis_core_seconds", "核心分析"],
    ["write_outputs_seconds", "写出结果"],
    ["result_payload_seconds", "结果加载"],
    ["task_total_seconds", "总耗时"],
  ];
  const parts = fields.flatMap(([field, label]) => {
    const formatted = formatAnalysisSeconds(timings[field]);
    return formatted ? [`${label}：${formatted}`] : [];
  });
  node.textContent = parts.join("｜");
  node.hidden = parts.length === 0;
}


const GENERIC_TABLE_CORE_COLUMNS = {
  overviewTop: ["variable", "driver_rank", "driver_priority_score", "pearson", "spearman", "method", "correlation_direction", "lag", "direction", "candidate_class", "risk_flags", "recommended_use"],
  nearMissTable: ["variable", "near_miss_score", "lag", "direction", "risk_flags", "recommended_use"],
  grangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "interpretation"],
  modelVariableImportanceTable: ["variable", "max_importance", "importance_rank", "best_model_feature", "best_model_lag", "recommended_use"],
  importanceTable: ["variable", "importance", "importance_rank", "feature", "lag", "method"],
  modelDiscoveredTable: ["variable", "max_importance", "importance_rank", "best_model_lag", "recommended_use", "discovery_reason"],
  enhancedSummaryTable: ["variable", "final_score", "lag", "direction", "status", "model_lift", "rolling_stability"],
  enhancedLiftTable: ["variable", "status", "model_lift_score", "median_fold_lift", "positive_fold_ratio", "model_lift", "ar_baseline_rmse", "candidate_rmse"],
  enhancedRollingTable: ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_stability"],
  conditionalGrangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "predictive_contribution"],
  xgbModelSummaryTable: ["model_name", "mean_rmse", "mean_mae", "mean_r2", "M2_vs_M1_rmse_improvement_pct"],
  xgbCandidateUpliftTable: ["variable", "median_rmse_improvement_pct", "median_mae_improvement_pct", "positive_rmse_fold_ratio", "validation_status"]
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
  if ("driver_priority_score" in (row || {}) && "final_score" in (row || {})) {
    loadLagProfile(row);
  }
  el("detailModalClose").focus();
}

function closeDetailModal() {
  const modal = el("detailModal");
  if (!modal) return;
  modal.classList.remove("open");
  modal.hidden = true;
  lagProfileRequestSerial += 1;
  lastLagProfile = null;
  clearTimeout(lagProfileResizeTimer);
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
  if (targetId === "finalReviewSummaryTable") return "未运行 最终推荐摘要。";
  if (targetId === "causalReviewEvidenceTable") return "未运行 逐变量综合证据复核表。";
  if (targetId === "xgbModelSummaryTable") return "未运行 XGB 四级验证。";
  if (targetId === "xgbCandidateUpliftTable") return "未运行 XGB 四级验证。";
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
  return ["variable", "status", "model_lift_score", "median_fold_lift", "positive_fold_ratio", "ar_baseline_rmse", "candidate_rmse", "model_lift"];
}

function rollingCorrColumns() {
  return ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_abs_corr_median", "rolling_corr_iqr", "rolling_sign_consistency", "valid_window_count", "rolling_stability"];
}

function conditionalGrangerColumns() {
  return ["variable", "status", "best_lag", "tested_lags", "lag_mode", "lag_window", "fallback_maxlag", "baseline_maxlag", "min_p_value", "fdr_q_value", "baseline_rmse", "full_rmse", "predictive_contribution", "condition_number", "base_condition_number", "full_condition_number", "control_columns", "n_rows", "interpretation"];
}

function xgbModelSummaryColumns() {
  return ["model_name", "mean_rmse", "mean_mae", "mean_r2", "M2_vs_M1_rmse_improvement_pct"];
}

function xgbCandidateUpliftColumns() {
  return ["variable", "median_rmse_improvement_pct", "median_mae_improvement_pct", "positive_rmse_fold_ratio", "validation_status"];
}

function renderXgbRunSummary(summary) {
  const container = el("xgbRunSummary");
  if (!container) return;
  if (!summary || summary.status !== "success") {
    container.innerHTML = "";
    return;
  }
  const timings = summary.timings_seconds || {};
  const requiredValues = [
    summary.row_count,
    summary.candidate_count,
    summary.fold_count,
    summary.m0_feature_count,
    summary.m1_feature_count,
    summary.m2_feature_count,
    summary.max_used_lag,
    timings.total,
  ];
  if (requiredValues.some(value => value === undefined || value === null)) {
    container.innerHTML = "";
    return;
  }
  const metrics = [
    ["样本行数", summary.row_count],
    ["候选数量", summary.candidate_count],
    ["时间折数", summary.fold_count],
    ["M0/M1/M2 特征数", `${summary.m0_feature_count}/${summary.m1_feature_count}/${summary.m2_feature_count}`],
    ["最大使用滞后", summary.max_used_lag],
    ["总耗时（秒）", timings.total],
  ];
  container.innerHTML = metrics.map(([label, value]) =>
    `<div class="metric-card"><span class="metric-value">${escapeHtml(formatValue(value))}</span><span class="metric-label">${escapeHtml(label)}</span></div>`
  ).join("");
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

function cellHtml(column, value, formatter = null) {
  const rendered = formatter ? formatter(column, value) : escapeHtml(formatCellValue(column, value));
  const title = cellTitle(column, value);
  return title ? `<td title="${escapeHtml(title)}">${rendered}</td>` : `<td>${rendered}</td>`;
}

function cellTitle(column, value) {
  if (column === "integrated_review_decision" && String(value ?? "") === "priority_review_with_statistical_limit") {
    return "统计证据支持较强，但统计检验受到高共线性、共同负荷或滞后边界限制；建议工程复核，但不是因果结论。";
  }
  return "";
}

const THREE_DECIMAL_SCORE_COLUMNS = new Set([
  "driver_priority_score", "final_score", "driver_priority_factor",
  "evidence_completeness", "evidence_confidence", "data_quality_score",
  "evidence_strength", "evidence_score", "evidence_score_low", "evidence_score_high",
]);
const THREE_DECIMAL_CORRELATION_COLUMNS = new Set([
  "dominant_corr", "pearson", "spearman", "pearson_r2", "spearman_r2",
]);
const SIGNIFICANCE_COLUMNS = new Set([
  "pearson_p", "spearman_p", "pearson_q", "spearman_q", "corr_q_value",
]);

function formatCellValue(column, value) {
  const scoreValue = value === null || value === undefined || value === "" ? NaN : Number(value);
  if (column === "lag_boundary_flag" && typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (THREE_DECIMAL_SCORE_COLUMNS.has(column) && Number.isFinite(scoreValue)) {
    return scoreValue.toFixed(3);
  }
  if (THREE_DECIMAL_CORRELATION_COLUMNS.has(column) && Number.isFinite(scoreValue)) {
    return scoreValue.toFixed(3);
  }
  if (SIGNIFICANCE_COLUMNS.has(column) && Number.isFinite(scoreValue)) {
    return scoreValue.toPrecision(3);
  }
  if (column === "n" && Number.isFinite(scoreValue)) {
    return String(Math.round(scoreValue));
  }
  const text = String(value ?? "");
  const maps = {
    method: {
      pearson: "Pearson",
      spearman: "Spearman",
    },
    innovation_sign: {
      "1": "正向",
      "-1": "负向",
      "0": "方向较弱",
    },
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
  renderDownloadTarget("finalReviewSummaryDownload", downloads, "final_review_summary.csv");
  renderDownloadTarget("causalEvidenceDownload", downloads, "causal_review_evidence.csv");
}

function renderXgbDownloads(downloads) {
  renderDownloadTarget("xgbModelSummaryDownload", downloads, "xgb_validation/xgb_model_summary.csv");
  renderDownloadTarget("xgbCandidateUpliftDownload", downloads, "xgb_validation/xgb_candidate_uplift.csv");
  renderDownloadTarget("xgbValidationSummaryDownload", downloads, "xgb_validation/xgb_validation_summary.json");
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
      non_predictive_lag: "非预测性滞后",
      innovation_verified: "变化量验证通过",
      innovation_lag_conflict: "变化量滞后冲突",
      innovation_sign_conflict: "变化量符号冲突",
      innovation_sign_unknown: "变化量符号未知",
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
      not_available: "未获得",
      supported: "支持",
      partially_supported: "部分支持",
      not_supported: "不支持",
      conflicting: "存在冲突",
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
      upstream_driver_candidate: "上游驱动候选",
      synchronous_association: "同步关联",
      downstream_response: "下游响应",
      formula_or_derived: "公式或派生变量",
      poor_quality: "低质量变量",
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
      unstable_candidate: "不稳定候选",
      poor_quality_variable: "低质量变量",
      manual_review_required: "需要人工复核",
      control_variable_reference: "控制变量参考",
      formula_like: "公式类变量",
      strong_formula_leakage: "强公式泄漏",
      common_capacity_driver: "共同负荷驱动",
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
    if (value.startsWith("predictive validation only; not a causal conclusion")) return "仅作预测验证；不是因果结论；解析式 p/q 值不能完全消除工业时序自相关影响";
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
  const addedLabels = {
    driver_rank: "驱动排名",
    driver_priority_score: "驱动优先得分",
    driver_priority_factor: "驱动优先系数",
    innovation_lag: "变化量滞后",
    innovation_direction: "变化量方向",
    innovation_sign: "变化量符号",
    innovation_status: "变化量验证状态",
  };
  if (addedLabels[column]) return addedLabels[column];
  const labels = {
    variable: "变量",
    trend_action: "趋势验证",
    final_score: "稳健综合得分",
    candidate_class: "候选类别",
    lag: "最佳滞后",
    direction: "时间关系",
    correlation_direction: "相关方向",
    raw_corr: "原始相关",
    association_score: "原始关联规范化得分",
    innovation_score: "变化量关联得分",
    residual_corr: "残差相关",
    independent_signal_score: "独立残差信号得分",
    correlation_evidence_score: "关联证据综合得分",
    correlation_evidence_status: "关联证据状态",
    prediction_score: "增量预测得分",
    stability_score: "综合稳定性",
    data_quality_score: "数据质量得分",
    evidence_strength: "证据强度",
    evidence_completeness: "证据覆盖度",
    evidence_confidence: "证据修正系数",
    evidence_coverage_status: "证据覆盖状态",
    layer1_association_status: "Layer 1 关联状态",
    layer2_temporal_status: "Layer 2 时间状态",
    layer3_independence_status: "Layer 3 独立性状态",
    layer4_model_status: "Layer 4 模型状态",
    stability_status: "稳定性状态",
    data_quality_status: "数据质量状态",
    evidence_support_items: "主要支持证据",
    evidence_against_items: "主要反对证据",
    evidence_missing_items: "缺失证据",
    evidence_conflict_items: "证据冲突",
    candidate_summary: "候选解释",
    evidence_score_low: "证据得分下界",
    evidence_score_high: "证据得分上界",
    score_method: "评分方法",
    residual_status: "残差状态",
    risk_flags: "风险标签",
    recommended_use: "建议用途",
    recommended_action: "建议动作",
    formula_like_flag: "公式类变量",
    strong_formula_leakage_flag: "强公式泄漏",
    common_capacity_driver_flag: "疑似共同负荷驱动",
    engineering_context: "工程上下文",
    target_leads_variable_flag: "变量滞后目标",
    unstable_across_regimes_flag: "跨工况不稳定",
    unstable_over_time_flag: "时序不稳定",
    lag_boundary_flag: "是否触及滞后边界",
    low_model_lift_flag: "低模型增益",
    poor_data_quality_flag: "数据质量较差",
    residual_collinearity_flag: "残差共线性风险",
    risk_count: "风险数量",
    strong_risk_count: "强风险数量",
    weak_risk_count: "弱风险数量",
    risk_level: "风险等级",
    human_reason: "风险说明",
    pearson: "Pearson 相关系数",
    spearman: "Spearman 相关系数",
    dominant_corr: "主导相关系数",
    pearson_p: "Pearson P 值",
    spearman_p: "Spearman P 值",
    pearson_q: "Pearson Q 值",
    spearman_q: "Spearman Q 值",
    corr_q_value: "主导方法 Q 值",
    pearson_r2: "Pearson R²",
    spearman_r2: "Spearman ρ²",
    n: "有效样本数",
    score: "得分",
    p_value: "P值",
    r2: "R²",
    method: "主导相关方法",
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
    median_fold_lift: "分段提升中位数",
    positive_fold_ratio: "正提升分段比例",
    model_lift_status: "模型提升状态",
    risk_penalty: "风险惩罚",
    risk_penalty_rate: "风险相对扣减率",
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
    model_name: "模型",
    mean_rmse: "平均RMSE",
    mean_mae: "平均MAE",
    mean_r2: "平均R²",
    M2_vs_M1_rmse_improvement_pct: "M2相对M1 RMSE改善(%)",
    median_rmse_improvement_pct: "RMSE改善中位数(%)",
    median_mae_improvement_pct: "MAE改善中位数(%)",
    positive_rmse_fold_ratio: "RMSE改善折占比",
    validation_status: "验证状态",
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
  clearLagProfileCache();
  fileId = "";
  currentRunId = "";
  currentAnalysisContext = {};
  recognizedColumns = [];
  recognizedNumericColumns = [];
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
  lastCausalEvidenceRows = [];
  lastFinalReviewSummaryRows = [];
  lastXgbModelSummaryRows = [];
  lastXgbCandidateUpliftRows = [];
  lastXgbValidationSummary = {};
  lastTrendSeries = [];
  lastTrendAxisMode = "shared";
  trendTimeRangeMode = "auto";
  trendSamplingIntervalMs = null;
  trendLatestTime = "";
  trendAutoWindowActive = false;
  lastScatterMatrixPayload = null;
  tableSortStates = { table: { column: "driver_rank", direction: "asc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
  el("fileInput").value = "";
  el("timeColumn").innerHTML = "";
  el("targetColumn").innerHTML = "";
  el("segmentColumn").innerHTML = "";
  el("excludedColumnsOptions").innerHTML = "";
  el("excludedColumnsSummary").textContent = "未选择剔除列";
  el("excludedColumnsDropdown").open = false;
  el("capacityOptions").innerHTML = "";
  el("capacitySummary").textContent = "请选择残差控制列";
  el("capacityDropdown").open = false;
  el("forceIncludeOptions").innerHTML = "";
  el("forceIncludeSummary").textContent = "请选择强制复核变量";
  el("forceIncludeDropdown").open = false;
  el("secondaryIncludeOptions").innerHTML = "";
  el("secondaryIncludeSummary").textContent = "请选择二次验证补充变量";
  el("secondaryIncludeDropdown").open = false;
  el("secondaryResampleMode").value = "raw";
  el("secondaryResampleRule").value = "";
  el("secondaryMaxLag").value = "";
  el("trendVar1").innerHTML = "";
  el("trendVar2").innerHTML = "";
  el("trendVar3").innerHTML = "";
  el("trendVar4").innerHTML = "";
  ["scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"].forEach((id) => { if (el(id)) el(id).value = ""; });
  el("trendStart").value = "";
  el("trendEnd").value = "";
  el("trendMaxPoints").value = "10000";
  el("analyze").disabled = true;
  el("runEnhancedScreening").disabled = true;
  el("runGranger").disabled = true;
  el("runModel").disabled = true;
  el("runCausalReview").disabled = true;
  el("enableXgbValidation").checked = false;
  el("runXgbValidation").disabled = true;
  el("drawTrend").disabled = true;
  el("drawScatterMatrix").disabled = true;
  el("downloads").innerHTML = "";
  llmPromptText = "";
  el("llmConnectionStatus").textContent = "尚未测试 API 连接。";
  setLlmReport("");
  el("llmReportDownload").innerHTML = "";
  el("llmApiKey").value = "";
  el("overview").innerHTML = "";
  el("analysisTimingBreakdown").textContent = "";
  el("analysisTimingBreakdown").hidden = true;
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
  clearScatterMatrix();
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
  clearOptionalElement("finalReviewQualityOverview");
  resetOptionalTable("finalReviewSummaryTable", "未运行 最终推荐摘要。");
  closeDetailModal();
  resetOptionalTable("causalReviewEvidenceTable", "未运行 逐变量综合证据复核表。");
  resetOptionalTable("xgbModelSummaryTable", "未运行 XGB 四级验证。");
  resetOptionalTable("xgbCandidateUpliftTable", "未运行 XGB 四级验证。");
  clearOptionalElement("xgbRunSummary");
  clearOptionalElement("xgbModelSummaryDownload");
  clearOptionalElement("xgbCandidateUpliftDownload");
  clearOptionalElement("xgbValidationSummaryDownload");
  el("xgbStatus").textContent = "XGB 四级验证未启用。";
  el("xgbTopN").value = "8";
  el("xgbMaxLag").value = "";
  el("xgbWhitelist").value = "";
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


if __name__ == "__main__":
    main()
