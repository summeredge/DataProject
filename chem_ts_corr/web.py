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
    exclude_window_stats,
    load_timeseries_csv,
    normalize_excluded_columns,
    read_timeseries_table,
)
from chem_ts_corr.pipeline import (
    _read_preprocessing_context,
    begin_downstream_stage,
    confirm_initial_screening_branch,
    DISCOVERY_CANDIDATES_FILENAME,
    run_causal_review_for_active_branch,
    run_enhanced_screening_for_active_branch,
    run_granger_for_active_branch,
    run_initial_screening_workflow,
    load_analysis_source_frame,
    run_model_for_active_branch,
    run_xgb_for_active_branch,
)
from chem_ts_corr.screening import CONTROL_REFERENCE_COLUMNS, order_initial_candidates
from chem_ts_corr.xgb_validation import validate_xgb_top_n
from chem_ts_corr.llm_api import LLMCallConfig, call_openai_compatible_chat, generate_llm_report, redact_secret
from chem_ts_corr.llm_report import build_llm_analysis_package, build_llm_prompt
from chem_ts_corr.validation_summary import (
    VALIDATION_SUMMARY_COLUMNS,
    VALIDATION_SUMMARY_FILENAME,
    VALIDATION_FIELDS_COLUMNS,
    build_validation_fields_from_output_dir,
    build_validation_summary_from_output_dir,
)
from chem_ts_corr.causal_review_evidence import (
    EVIDENCE_MATRIX_COLUMNS,
    evidence_matrix_status_labels,
)
from chem_ts_corr.verification_review_pool import (
    FILENAME as VERIFICATION_REVIEW_POOL_FILENAME,
    add_to_verification_review_pool,
    read_verification_review_pool,
    write_initial_verification_review_pool,
)


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
    DISCOVERY_CANDIDATES_FILENAME,
    "model_variable_importance.csv",
    "recommended_candidates.csv",
    "lag_peak_quality.csv",
    "rolling_corr_scores.csv",
    "causal_review_candidates.csv",
    "conditional_granger_scores.csv",
    "causal_review_report.csv",
    "final_review_summary.csv",
    "causal_review_evidence.csv",
    "evidence_matrix.csv",
    "enhanced_validation_summary.csv",
    VERIFICATION_REVIEW_POOL_FILENAME,
    VALIDATION_SUMMARY_FILENAME,
    "preprocessing_comparison.csv",
    "preprocessing_context.json",
    "llm_prompt.md",
    "llm_report.md",
    "xgb_validation/xgb_fold_metrics.csv",
    "xgb_validation/xgb_model_summary.csv",
    "xgb_validation/xgb_candidate_uplift.csv",
    "xgb_validation/xgb_candidate_fold_metrics.csv",
    "xgb_validation/xgb_predictions.csv",
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
MAX_TREND_TOTAL_POINTS = 300000
MAX_EXCLUDE_WINDOW_CONTEXTS = 4
EXCLUDE_WINDOW_CONTEXTS: dict[tuple[str, str], dict[str, Any]] = {}
EXCLUDE_WINDOW_CONTEXTS_LOCK = threading.Lock()
FINAL_REVIEW_SUMMARY_FIELD_NOTES = {
    "final_rank": "人工复核优先级（展示序号）；仅用于第三层人工复核，不参与初筛评分或排序。",
    "review_priority": "人工复核优先级。",
    "review_reason": "证据摘要。",
    "final_recommendation": "复核建议。",
}
INITIAL_SCREENING_COLUMNS = (
    "variable", "driver_rank", "final_score", "pearson", "spearman", "method", "dominant_corr",
    "correlation_direction", "lag", "direction", "lag_quality", "lag_quality_status",
    "lag_boundary_flag", "n", "data_quality_score", "risk_flags", "risk_level",
    "human_reason", "recommended_use", "recommended_action", "force_included",
    "variable_role", "is_residual_control", "is_capacity_reference", "is_segment_reference",
    "innovation_score", "innovation_lag", "innovation_direction", "innovation_sign",
    "innovation_status", "pearson_p", "spearman_p", "pearson_q", "spearman_q",
    "corr_q_value", "pearson_r2", "spearman_r2",
    "association_score", "near_peak_lag_min", "near_peak_lag_max", "near_peak_lag_count",
    "temporal_direction_status", "temporal_penalty_rate", "temporal_score_cap",
    *CONTROL_REFERENCE_COLUMNS,
)
RECOMMENDED_CANDIDATE_COLUMNS = (
    *(column for column in INITIAL_SCREENING_COLUMNS if column not in CONTROL_REFERENCE_COLUMNS),
    "candidate_source", "selected_by_raw", "selected_by_residual", "raw_candidate_rank",
    "residual_candidate_rank", "candidate_pool_rank", "common_capacity_candidate_flag",
    "residual_signal_score", "residual_evidence_status", "load_adjusted_relation_status",
    "candidate_priority_tier", "candidate_priority_score", "candidate_priority_rank",
)


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
                params = parse_qs(parsed.query, keep_blank_values=True)
                self._send_json(_trend_response(params))
            except Exception as exc:
                if _is_client_disconnect(exc):
                    return
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/scatter_matrix":
            try:
                params = parse_qs(parsed.query, keep_blank_values=True)
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
            if self.path == "/api/exclude_window":
                self._send_json(_exclude_window_response(self))
                return
            if self.path == "/api/restore_exclude_window":
                self._send_json(_restore_exclude_window_response(self))
                return
            if self.path == "/api/restore_all_exclude_windows":
                self._send_json(_restore_all_exclude_windows_response(self))
                return
            if self.path == "/api/confirm_initial_screening_branch":
                self._send_json(_confirm_initial_screening_branch_response(self))
                return
            if self.path == "/api/run_granger":
                self._send_json(_run_granger_response(self))
                return
            if self.path == "/api/run_model":
                self._send_json(_run_model_response(self))
                return
            if self.path == "/api/add_to_verification_review_pool":
                self._send_json(_add_to_verification_review_pool_response(self))
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


def _exclude_window_context(
    file_id: str,
    time_column: str,
    encoding: str,
) -> dict[str, Any]:
    file_id = _validate_file_id(file_id)
    context_key = (file_id, time_column)
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        existing = EXCLUDE_WINDOW_CONTEXTS.pop(context_key, None)
        if existing is not None:
            EXCLUDE_WINDOW_CONTEXTS[context_key] = existing
            return existing

    path = _resolve_upload(file_id)
    frame = load_timeseries_csv(
        path,
        time_column,
        encoding=_resolve_encoding(path, encoding),
    )
    context = {
        "time_column": time_column,
        "frame": frame,
        "exclude_windows": [],
    }
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        existing = EXCLUDE_WINDOW_CONTEXTS.pop(context_key, None)
        if existing is not None:
            EXCLUDE_WINDOW_CONTEXTS[context_key] = existing
            return existing
        EXCLUDE_WINDOW_CONTEXTS[context_key] = context
        while len(EXCLUDE_WINDOW_CONTEXTS) > MAX_EXCLUDE_WINDOW_CONTEXTS:
            EXCLUDE_WINDOW_CONTEXTS.pop(next(iter(EXCLUDE_WINDOW_CONTEXTS)))
        return context


def _existing_exclude_window_context(file_id: str, time_column: str) -> dict[str, Any]:
    file_id = _validate_file_id(file_id)
    context_key = (file_id, time_column)
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        context = EXCLUDE_WINDOW_CONTEXTS.pop(context_key, None)
        if context is None:
            raise ValueError("当前数据上下文没有排除窗口状态")
        EXCLUDE_WINDOW_CONTEXTS[context_key] = context
        return context


def _exclude_window_payload(context: dict[str, Any]) -> dict[str, Any]:
    windows = [dict(window) for window in context["exclude_windows"]]
    return {
        "excludeWindows": windows,
        "excludeWindowStats": exclude_window_stats(context["frame"], windows),
    }


def _exclude_window_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    context = _exclude_window_context(
        _field(form, "file_id"),
        _field(form, "time_column"),
        _field(form, "encoding", "utf-8-sig"),
    )
    window = {"start": _field(form, "start"), "end": _field(form, "end")}
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        windows = [*context["exclude_windows"], window]
        exclude_window_stats(context["frame"], windows)
        context["exclude_windows"] = windows
        return _exclude_window_payload(context)


def _restore_exclude_window_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    context = _existing_exclude_window_context(
        _field(form, "file_id"), _field(form, "time_column")
    )
    try:
        index = int(_field(form, "index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("排除窗口索引无效") from exc
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        windows = context["exclude_windows"]
        if index < 0 or index >= len(windows):
            raise ValueError("排除窗口不存在")
        context["exclude_windows"] = [
            window for position, window in enumerate(windows) if position != index
        ]
        return _exclude_window_payload(context)


def _restore_all_exclude_windows_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    context = _existing_exclude_window_context(
        _field(form, "file_id"), _field(form, "time_column")
    )
    with EXCLUDE_WINDOW_CONTEXTS_LOCK:
        context["exclude_windows"] = []
        return _exclude_window_payload(context)


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
    residual_control_columns = _list_field(form, "residual_control_columns")
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
        top_k=_int_field(form, "top_k", 20),
        preprocess_mode=_field(form, "preprocess_mode", "raw"),
        lowpass_tau_minutes=_float_field(form, "lowpass_tau_minutes", 5.0),
        diff_interval_minutes=_optional_float_field(form, "diff_interval_minutes"),
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
    try:
        exclude_context = _exclude_window_context(
            file_id, time_column, resolved_encoding
        )
    except ValueError as exc:
        if str(exc) != "Invalid file id":
            raise
        exclude_windows = []
    else:
        with EXCLUDE_WINDOW_CONTEXTS_LOCK:
            exclude_windows = [
                dict(window) for window in exclude_context["exclude_windows"]
            ]
    config = replace(config, exclude_windows=exclude_windows)
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

        workflow_result = run_initial_screening_workflow(
            config, progress_callback=progress
        )
        pipeline_timings = _workflow_timings(workflow_result)
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
            if "overview" in result:
                result["overview"].setdefault(
                    "analysis_elapsed_seconds", task_total_seconds
                )
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


def _workflow_timings(workflow_result: object) -> dict[str, float]:
    """Normalize timings from the unified initial-screening workflow.

    ``raw`` returns a flat ``timings`` dict; non-raw returns per-branch
    ``raw`` / ``processed`` dicts. The normalized shape keeps the legacy
    ``analysis_timings`` keys used by the UI.
    """
    if not isinstance(workflow_result, dict):
        return {}
    flat = workflow_result.get("timings")
    if isinstance(flat, dict):
        return {
            key: _non_negative_seconds(flat.get(key))
            for key in (
                "read_data_seconds",
                "analysis_core_seconds",
                "write_outputs_seconds",
                "pipeline_total_seconds",
            )
        }
    branches = [
        workflow_result.get(branch)
        for branch in ("raw", "processed")
        if isinstance(workflow_result.get(branch), dict)
    ]
    if not branches:
        return {}
    return {
        "read_data_seconds": sum(
            _non_negative_seconds(item.get("read_data_seconds")) for item in branches
        ),
        "analysis_core_seconds": sum(
            _non_negative_seconds(item.get("analysis_core_seconds"))
            for item in branches
        ),
        "write_outputs_seconds": sum(
            _non_negative_seconds(item.get("write_outputs_seconds"))
            for item in branches
        ),
        "pipeline_total_seconds": max(
            _non_negative_seconds(item.get("pipeline_total_seconds"))
            for item in branches
        ),
    }


def _non_negative_seconds(value: object) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(seconds) or seconds < 0:
        return 0.0
    return round(seconds, 6)


def _build_result_payload(run_id: str, output_dir: Path, config: AnalysisConfig) -> dict[str, Any]:
    context = _read_context_for_payload(output_dir)
    if (
        context is not None
        and context["branch_selection_status"] == "awaiting_confirmation"
    ):
        return _build_pending_payload(run_id, output_dir, config, context)

    ranked = order_initial_candidates(_safe_read_result_csv(output_dir / "ranked_features.csv"))
    recommended = _order_recommended_candidates(_safe_read_result_csv(output_dir / "recommended_candidates.csv"))
    display_ranked = _with_correlation_display_fields(_initial_screening_frame(ranked))
    risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
    residual = _safe_read_result_csv(output_dir / "residual_corr_scores.csv")
    regime = _safe_read_result_csv(output_dir / "regime_scores.csv")
    lift = _safe_read_result_csv(output_dir / "model_lift_scores.csv")
    rolling = _safe_read_result_csv(output_dir / "rolling_corr_scores.csv")
    enhanced = _safe_read_result_csv(output_dir / "enhanced_validation_summary.csv")
    validation = _validation_summary_for_payload(output_dir)
    evidence_matrix = _evidence_matrix_for_payload(output_dir)
    granger = _safe_read_result_csv(output_dir / "granger_tests.csv")
    importance = _safe_read_result_csv(output_dir / "shap_or_importance.csv")
    model_variable_importance = _safe_read_result_csv(output_dir / "model_variable_importance.csv")
    model_discovered = _safe_read_result_csv(output_dir / "model_discovered_candidates.csv")
    discovery_candidates = _safe_read_result_csv(output_dir / DISCOVERY_CANDIDATES_FILENAME)
    verification_review_pool = _verification_review_pool_for_payload(output_dir)
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    risky = risk[risk.get("risk_count", 0) > 0] if not risk.empty else risk
    comparison = _safe_read_result_csv(output_dir / "preprocessing_comparison.csv")
    active_mode = (
        context["active_preprocessing_mode"]
        if context is not None
        else config.preprocess_mode
    )
    analysis_context = {
        "preprocess_mode": active_mode,
        "lowpass_tau_minutes": (
            context["lowpass_tau_minutes"]
            if context is not None and context["lowpass_tau_minutes"] is not None
            else config.lowpass_tau_minutes
        ),
        "diff_interval_minutes": (
            context["requested_diff_interval_minutes"]
            if context is not None
            else config.diff_interval_minutes
        ),
        "detrend_window": config.detrend_window,
    }
    return {
        "run_id": run_id,
        "analysisContext": analysis_context,
        "preprocessingContext": context,
        "preprocessingComparison": _records(comparison) if context is not None else [],
        "branchSelectionStatus": (
            context["branch_selection_status"] if context is not None else None
        ),
        "activeScreeningBranch": (
            context["active_screening_branch"] if context is not None else None
        ),
        "activePreprocessingMode": active_mode if context is not None else None,
        "selectedPreprocessingMode": (
            context["selected_preprocessing_mode"] if context is not None else None
        ),
        "branchLocked": (output_dir / "screening_downstream.lock").exists(),
        "overview": _overview_payload(display_ranked, risk, config, _summary_metrics(summary), recommended),
        "rankedFeatures": _records(display_ranked),
        "recommendedCandidates": _records(_recommended_candidate_frame(recommended)),
        "riskFlags": _records(risky.head(50)),
        "lagScores": [],
        "residualScores": _records(residual),
        "regimeScores": _records(regime.head(50)),
        "modelLiftScores": _records(lift.head(50)),
        "rollingCorrScores": _records(rolling.head(50)),
        "enhancedValidationSummary": _records(enhanced.head(200)),
        "validationSummary": _records(validation.head(500)),
        "validationFields": _records(_validation_fields_for_payload(output_dir)),
        "evidenceMatrix": _records(evidence_matrix.head(500)),
        "evidenceMatrixStatusLabels": evidence_matrix_status_labels(),
        "finalReviewSummaryFieldNotes": dict(FINAL_REVIEW_SUMMARY_FIELD_NOTES),
        "grangerTests": _records(granger.head(200)),
        "importance": _records(importance.head(200)),
        "modelVariableImportance": _records(model_variable_importance.head(200)),
        "modelDiscoveredCandidates": _records(model_discovered.head(200)),
        "discoveryCandidates": _records(discovery_candidates.head(200)),
        "verificationReviewPool": _records(verification_review_pool),
        "downloads": _download_links(run_id, output_dir),
    }


def _read_context_for_payload(output_dir: Path) -> dict[str, Any] | None:
    """Read the frozen preprocessing context without failing on legacy runs.

    Legacy runs without ``preprocessing_context.json`` keep the historical
    payload behavior. Missing/invalid contexts for new workflow runs keep
    their frozen backend error tokens.
    """
    context_path = output_dir / "preprocessing_context.json"
    if not context_path.exists():
        return None
    return _read_preprocessing_context(output_dir)


def _verification_review_pool_for_payload(output_dir: Path) -> pd.DataFrame:
    pool = read_verification_review_pool(output_dir)
    return pool if pool is not None else pd.DataFrame()


def _validation_summary_for_payload(output_dir: Path) -> pd.DataFrame:
    """Read the frozen summary or derive it from already-produced evidence.

    Payload construction is read-only.  A missing summary is derived in
    memory, while an existing malformed summary is ignored rather than
    exposing fields outside the five-column contract.
    """
    path = output_dir / VALIDATION_SUMMARY_FILENAME
    if path.exists():
        stored = _safe_read_result_csv(path)
        if set(VALIDATION_SUMMARY_COLUMNS).issubset(stored.columns):
            return stored[VALIDATION_SUMMARY_COLUMNS]
    return build_validation_summary_from_output_dir(output_dir)


def _validation_fields_for_payload(output_dir: Path) -> pd.DataFrame:
    """Build stage-specific lag/lift metadata for API consumers.

    The V1 five-column ``validationSummary`` remains unchanged.  V3 fields are
    exposed separately so they cannot be mistaken for a unified conclusion or
    feed the initial-screening ranking.
    """
    fields = build_validation_fields_from_output_dir(output_dir)
    if not set(VALIDATION_FIELDS_COLUMNS).issubset(fields.columns):
        return pd.DataFrame(columns=VALIDATION_FIELDS_COLUMNS)
    return fields[VALIDATION_FIELDS_COLUMNS]


def _evidence_matrix_for_payload(output_dir: Path) -> pd.DataFrame:
    """Read the explanation-only matrix without deriving or rerunning stages."""
    path = output_dir / "evidence_matrix.csv"
    matrix = _safe_read_result_csv(path)
    if not path.exists() or not set(EVIDENCE_MATRIX_COLUMNS).issubset(matrix.columns):
        return pd.DataFrame(columns=EVIDENCE_MATRIX_COLUMNS)
    return matrix[EVIDENCE_MATRIX_COLUMNS]


def _build_pending_payload(
    run_id: str,
    output_dir: Path,
    config: AnalysisConfig,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimal payload for the awaiting-confirmation state.

    The formal root screening files do not exist yet, so only the frozen
    context, comparison and allowed downloads are returned. Branch-internal
    files are never wrapped as formal results here.
    """
    context = context or _read_context_for_payload(output_dir)
    comparison = _safe_read_result_csv(output_dir / "preprocessing_comparison.csv")
    allowed_downloads = {
        item["name"]
        for item in _download_links(run_id, output_dir)
        if item["name"] in {"preprocessing_comparison.csv", "preprocessing_context.json"}
    }
    downloads = [
        {"name": name, "url": f"/download?run_id={run_id}&file={name}"}
        for name in sorted(allowed_downloads)
    ]
    status = (
        context["branch_selection_status"]
        if context is not None
        else "awaiting_confirmation"
    )
    return {
        "run_id": run_id,
        "preprocessingContext": context,
        "preprocessingComparison": _records(comparison),
        "branchSelectionStatus": status,
        "selectedPreprocessingMode": (
            context["selected_preprocessing_mode"] if context is not None else None
        ),
        "activeScreeningBranch": None,
        "activePreprocessingMode": None,
        "branchLocked": (output_dir / "screening_downstream.lock").exists(),
        "downloads": downloads,
    }


def _branch_context_payload(output_dir: Path) -> dict[str, Any]:
    """Return the frozen branch/preprocessing context for endpoint responses."""
    context = _read_context_for_payload(output_dir)
    return {
        "branchSelectionStatus": (
            context["branch_selection_status"] if context is not None else None
        ),
        "activeScreeningBranch": (
            context["active_screening_branch"] if context is not None else None
        ),
        "activePreprocessingMode": (
            context["active_preprocessing_mode"] if context is not None else None
        ),
        "selectedPreprocessingMode": (
            context["selected_preprocessing_mode"] if context is not None else None
        ),
        "branchLocked": (output_dir / "screening_downstream.lock").exists(),
    }


def _confirm_initial_screening_branch_response(
    handler: BaseHTTPRequestHandler,
) -> dict[str, Any]:
    """Confirm an existing screening branch and return the refreshed payload.

    Confirmation only publishes the already-computed branch through the
    backend ``confirm_initial_screening_branch()``; it never re-runs
    screening or re-computes the comparison. Frozen backend error tokens are
    preserved on failure.
    """
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    branch = _field(form, "branch")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    confirm_initial_screening_branch(output_dir, branch=branch)
    return _build_result_payload(run_id, output_dir, config)


def _require_formal_branch(output_dir: Path) -> dict[str, Any] | None:
    """Gate report/LLM consumers behind a confirmed formal screening branch.

    Legacy runs without ``preprocessing_context.json`` are allowed (their
    formal root files are authoritative). New workflow runs must not be
    ``awaiting_confirmation``; the frozen backend error token is preserved.
    """
    context_path = output_dir / "preprocessing_context.json"
    if not context_path.exists():
        return None
    context = _read_preprocessing_context(output_dir)
    if context["branch_selection_status"] == "awaiting_confirmation":
        raise ValueError(
            "initial_screening_branch_not_confirmed: 请先确认正式初筛分支。"
        )
    return context


def _lock_formal_branch_for_llm(output_dir: Path) -> None:
    """Lock a confirmed PR-13 branch before writing LLM artifacts."""
    context = _require_formal_branch(output_dir)
    if context is not None:
        begin_downstream_stage(output_dir)


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
    total_started = time.perf_counter()
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    run_enhanced_screening_for_active_branch(
        output_dir,
        base_config=_read_run_config(output_dir),
    )
    lift = _safe_read_result_csv(output_dir / "model_lift_scores.csv")
    rolling = _safe_read_result_csv(output_dir / "rolling_corr_scores.csv")
    enhanced = _safe_read_result_csv(output_dir / "enhanced_validation_summary.csv")
    validation = _validation_summary_for_payload(output_dir)

    result = {
        "modelLiftScores": _records(lift.head(200)),
        "rollingCorrScores": _records(rolling.head(200)),
        "enhancedValidationSummary": _records(enhanced.head(200)),
        "validationSummary": _records(validation.head(500)),
        "validationFields": _records(_validation_fields_for_payload(output_dir)),
        "verificationReviewPool": _records(_verification_review_pool_for_payload(output_dir)),
        "downloads": _download_links(run_id, output_dir),
        "message": "增强筛选完成：结果用于补充验证预测增益和时间稳定性，不代表因果结论。",
        **_branch_context_payload(output_dir),
    }
    result["timings"] = {
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
    run_granger_for_active_branch(
        output_dir,
        base_config=_read_run_config(output_dir),
    )
    granger = _safe_read_result_csv(output_dir / "granger_tests.csv")
    return {
        "grangerTests": _records(granger.head(200)),
        "validationSummary": _records(_validation_summary_for_payload(output_dir).head(500)),
        "validationFields": _records(_validation_fields_for_payload(output_dir)),
        "verificationReviewPool": _records(_verification_review_pool_for_payload(output_dir)),
        "downloads": _download_links(run_id, output_dir),
        "message": "Granger 二级验证完成。",
        **_branch_context_payload(output_dir),
    }


def _run_model_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    result = run_model_for_active_branch(
        output_dir,
        base_config=_read_run_config(output_dir),
    )
    importance = _safe_read_result_csv(output_dir / "shap_or_importance.csv")
    model_variable_importance = _safe_read_result_csv(
        output_dir / "model_variable_importance.csv"
    )
    model_discovered = _safe_read_result_csv(
        output_dir / "model_discovered_candidates.csv"
    )
    discovery_candidates = _safe_read_result_csv(
        output_dir / DISCOVERY_CANDIDATES_FILENAME
    )
    return {
        "importance": _records(importance.head(200)),
        "modelVariableImportance": _records(model_variable_importance.head(200)),
        "modelDiscoveredCandidates": _records(model_discovered.head(200)),
        "discoveryCandidates": _records(discovery_candidates.head(200)),
        "validationSummary": _records(_validation_summary_for_payload(output_dir).head(500)),
        "validationFields": _records(_validation_fields_for_payload(output_dir)),
        "verificationReviewPool": _records(_verification_review_pool_for_payload(output_dir)),
        "modelMetrics": result.get("model_metrics") or {},
        "downloads": _download_links(run_id, output_dir),
        "message": "随机森林模型解释完成。",
        **_branch_context_payload(output_dir),
    }


def _add_to_verification_review_pool_response(
    handler: BaseHTTPRequestHandler,
) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    variable = _field(form, "variable")
    candidate_source = _field(form, "candidate_source")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    ranked = _safe_read_result_csv(output_dir / "ranked_features.csv")
    if ranked.empty:
        raise ValueError("verification_review_pool_initial_screening_missing")
    if read_verification_review_pool(output_dir) is None:
        write_initial_verification_review_pool(
            output_dir,
            ranked,
            top_k=config.top_k,
            manual_include=config.force_include_variables,
        )
    if candidate_source == "model_discovery":
        discovery_path = output_dir / DISCOVERY_CANDIDATES_FILENAME
        if not discovery_path.exists():
            discovery_path = output_dir / "model_discovered_candidates.csv"
        discovered = _safe_read_result_csv(discovery_path)
        discovered_variables = (
            set(discovered["variable"].dropna().astype(str))
            if "variable" in discovered.columns
            else set()
        )
        if variable not in discovered_variables:
            raise ValueError("verification_candidate_not_confirmed_model_discovery")
    pool = add_to_verification_review_pool(
        output_dir,
        ranked,
        variable=variable,
        candidate_source=candidate_source,
    )
    return {
        "verificationReviewPool": _records(pool),
        "downloads": _download_links(run_id, output_dir),
        "message": "已加入二级验证复核池。",
        **_branch_context_payload(output_dir),
    }


SECONDARY_CANDIDATE_CONTEXT_FILENAME = "secondary_candidate_context.csv"


def _secondary_candidate_context_path(output_dir: Path) -> Path:
    return output_dir / SECONDARY_CANDIDATE_CONTEXT_FILENAME


def _save_secondary_candidate_context(output_dir: Path, variables: list[str]) -> None:
    frame = pd.DataFrame({"variable": [v for v in (variables or []) if v]})
    frame.to_csv(_secondary_candidate_context_path(output_dir), index=False, encoding="utf-8-sig")


def _load_secondary_candidate_context(output_dir: Path) -> list[str]:
    frame = _safe_read_result_csv(_secondary_candidate_context_path(output_dir))
    if frame.empty or "variable" not in frame.columns:
        return []
    return [str(value) for value in frame["variable"].dropna() if str(value)]


def _build_causal_review_candidate_table(
    ranked: pd.DataFrame,
    variables: list[str],
    risk_flags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from chem_ts_corr.causal_review import build_causal_review_candidates

    candidate_variables = list(dict.fromkeys([v for v in (variables or []) if v]))
    if not candidate_variables:
        return build_causal_review_candidates(pd.DataFrame(columns=["variable"]))
    if ranked.empty or "variable" not in ranked.columns:
        return build_causal_review_candidates(pd.DataFrame({"variable": candidate_variables}))
    selected = ranked[ranked["variable"].astype(str).isin(candidate_variables)].copy(deep=True)
    selected["variable"] = selected["variable"].astype(str)
    missing_variables = [
        variable for variable in candidate_variables if variable not in set(selected["variable"])
    ]
    if missing_variables:
        selected = pd.concat(
            [selected, pd.DataFrame({"variable": missing_variables})],
            ignore_index=True,
        )
    if risk_flags is not None and not risk_flags.empty and "variable" in risk_flags.columns:
        merge_columns = [
            column
            for column in risk_flags.columns
            if column != "variable" and column not in selected.columns
        ]
        if merge_columns:
            risk_source = risk_flags[["variable", *merge_columns]].copy(deep=True)
            risk_source["variable"] = risk_source["variable"].astype(str)
            risk_source = risk_source.drop_duplicates(subset=["variable"], keep="first")
            selected = selected.merge(
                risk_source,
                on="variable",
                how="left",
                sort=False,
                suffixes=("", "__risk"),
            )
            for column in merge_columns:
                risk_column = f"{column}__risk"
                if risk_column in selected.columns:
                    selected[column] = selected[column].combine_first(selected[risk_column])
                    selected = selected.drop(columns=[risk_column])
    return build_causal_review_candidates(selected)



def _run_causal_review_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    config = _read_run_config(output_dir)
    run_causal_review_for_active_branch(
        output_dir,
        base_config=config,
        control_columns=_list_field(form, "control_columns") or None,
        maxlag=_optional_int_field(form, "maxlag"),
        min_rows=_int_field(form, "min_rows", 60),
        top_n=_optional_int_field(form, "top_n"),
        conditional_lag_mode=_field(form, "conditional_lag_mode", "ranked_window"),
        conditional_lag_window=_int_field(form, "conditional_lag_window", 5),
        conditional_fallback_maxlag=_int_field(form, "conditional_fallback_maxlag", 24),
        conditional_baseline_maxlag=_optional_int_field(form, "conditional_baseline_maxlag") or 24,
    )
    conditional = _safe_read_result_csv(output_dir / "conditional_granger_scores.csv")
    report = _safe_read_result_csv(output_dir / "causal_review_report.csv")
    evidence = _safe_read_result_csv(output_dir / "causal_review_evidence.csv")
    final_summary = _safe_read_result_csv(output_dir / "final_review_summary.csv")
    evidence_matrix = _evidence_matrix_for_payload(output_dir)
    risk_filter = _list_field(form, "risk_flag_filter")
    if risk_filter:
        risk = _safe_read_result_csv(output_dir / "risk_flags.csv")
        final_summary = _filter_candidates_by_risk_flags(
            final_summary, risk, risk_filter
        )
        report = _filter_candidates_by_risk_flags(report, risk, risk_filter)
        evidence_matrix = _filter_candidates_by_risk_flags(
            evidence_matrix, risk, risk_filter
        )
    return {
        "conditionalGrangerScores": _records(conditional.head(500)),
        "causalReviewReport": _records(report.head(500)),
        "finalReviewSummary": _records(final_summary.head(500)),
        "causalReviewEvidence": _records(evidence.head(500)),
        "evidenceMatrix": _records(evidence_matrix.head(500)),
        "evidenceMatrixStatusLabels": evidence_matrix_status_labels(),
        "finalReviewSummaryFieldNotes": dict(FINAL_REVIEW_SUMMARY_FIELD_NOTES),
        "validationSummary": _records(_validation_summary_for_payload(output_dir).head(500)),
        "validationFields": _records(_validation_fields_for_payload(output_dir)),
        "downloads": _download_links(run_id, output_dir),
        "message": "第三层可信度审查完成：结果用于解释独立预测贡献及其限制，不是因果结论，也不改变初筛结果。",
        **_branch_context_payload(output_dir),
    }


def _run_xgb_validation_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    if not _bool_field(form, "enable_xgb_validation"):
        return {
            "status": "skipped",
            "error_message": None,
            "xgbModelSummary": [],
            "xgbCandidateUplift": [],
            "xgbCandidateFoldMetrics": [],
            "xgbValidationSummary": {},
            "downloads": [],
            "message": "XGB 时间外预测验证未启用。",
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

    control_columns = _list_field(form, "control_columns") or None
    whitelist = _list_field(form, "whitelist") or None
    try:
        result = run_xgb_for_active_branch(
            output_dir,
            base_config=config,
            control_columns=control_columns,
            whitelist=whitelist,
            top_n=top_n,
            max_lag=max_lag,
        )
    except ValueError as exc:
        return _xgb_response_payload(
            run_id,
            output_dir,
            status="invalid_input",
            error_message=str(exc),
        )
    return _xgb_response_payload(
        run_id,
        output_dir,
        status=result["status"],
        error_message=result.get("error_message"),
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
    candidate_fold_metrics = pd.DataFrame()
    validation_summary: dict[str, Any] = {}
    downloads = _download_links(run_id, output_dir)
    if status == "success":
        model_summary = _safe_read_result_csv(
            output_dir / "xgb_validation" / "xgb_model_summary.csv"
        )
        candidate_uplift = _safe_read_result_csv(
            output_dir / "xgb_validation" / "xgb_candidate_uplift.csv"
        )
        candidate_fold_metrics = _safe_read_result_csv(
            output_dir / "xgb_validation" / "xgb_candidate_fold_metrics.csv"
        )
        summary_path = output_dir / "xgb_validation" / "xgb_validation_summary.json"
        validation_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
    messages = {
        "success": "XGB 时间外预测验证完成：已生成候选变量预测增量证据，仅供人工复核参考，不改变前三层结果。",
        "missing_dependency": "XGB 时间外预测验证缺少可选依赖。",
        "invalid_input": "XGB 时间外预测验证输入无效。",
        "failed": "XGB 时间外预测验证失败。",
    }
    return {
        "status": status,
        "error_message": error_message,
        "xgbModelSummary": _records(model_summary),
        "xgbCandidateUplift": _records(candidate_uplift),
        "xgbCandidateFoldMetrics": _records(candidate_fold_metrics),
        "xgbValidationSummary": validation_summary,
        "downloads": downloads,
        "message": messages.get(status, "XGB 时间外预测验证未运行。"),
        **_branch_context_payload(output_dir),
    }


def _llm_prompt_response(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    form = _multipart_form(handler)
    run_id = _field(form, "run_id")
    output_dir = _resolve_run_dir(run_id)
    _lock_formal_branch_for_llm(output_dir)
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
    _lock_formal_branch_for_llm(output_dir)
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
    ordered = ranked.copy()
    ordered["variable"] = ordered["variable"].astype(str)
    ordered["_secondary_final_score"] = pd.to_numeric(
        ordered.get("final_score", pd.Series(float("nan"), index=ordered.index)),
        errors="coerce",
    )
    ordered = ordered.sort_values(
        "_secondary_final_score",
        ascending=False,
        na_position="last",
        kind="stable",
    )
    top_k = max(0, int(config.top_k or 0))
    top = ordered["variable"].tolist()[:top_k]
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


def _scaled_frame_for_secondary_causal(
    config: AnalysisConfig, protected_columns: list[str] | None = None
) -> pd.DataFrame:
    from chem_ts_corr.preprocess import (
        operating_segment_mask,
        preprocess_frame_causal,
        standardize_frame,
        transform_frame_causal,
    )

    extra_protected = tuple(c for c in (protected_columns or []) if c)
    cache_key = ("causal", *_scaled_frame_cache_key(config, extra_protected))
    with SCALED_FRAME_CACHE_LOCK:
        cached = SCALED_FRAME_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy(deep=True)

    numeric = _numeric_frame(config, protected_columns)
    cleaned = preprocess_frame_causal(
        numeric,
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
        cleaned,
        config.preprocess_mode,
        config.detrend_window,
        lowpass_tau_minutes=config.lowpass_tau_minutes,
        diff_interval_minutes=config.diff_interval_minutes,
    )
    target_mask = target_mask.reindex(transformed.index).fillna(False).astype(bool)
    scaled = standardize_frame(transformed, fit_mask=target_mask)
    scaled.attrs[TARGET_SEGMENT_MASK_ATTR] = tuple(target_mask.tolist())
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

    raw = load_analysis_source_frame(
        config,
        load_frame=load_timeseries_csv,
        drop_columns=drop_excluded_columns,
        extra_protected_columns=protected_columns,
    )
    numeric = select_numeric_frame(raw, config.target)
    numeric.attrs = dict(raw.attrs)
    roles = load_roles(config, list(numeric.columns))
    numeric = apply_ignore_roles(numeric, roles, config.target)
    numeric.attrs = dict(raw.attrs)
    return numeric


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
        lowpass_tau_minutes=config.lowpass_tau_minutes,
        diff_interval_minutes=config.diff_interval_minutes,
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
    if isinstance(stored, tuple) and len(stored) == len(frame):
        resolved = pd.Series(stored, index=frame.index, dtype=bool)
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
        tuple((window["start"], window["end"]) for window in config.exclude_windows),
        protected_columns,
        config.resample_rule,
        config.min_valid_ratio,
        config.max_interpolate_gap_points,
        config.interpolate_limit_area,
        config.preprocess_mode,
        config.detrend_window,
        config.lowpass_tau_minutes,
        config.diff_interval_minutes,
    )


def _clear_scaled_frame_cache() -> None:
    with SCALED_FRAME_CACHE_LOCK:
        SCALED_FRAME_CACHE.clear()


def _chart_frame_from_params(
    params: dict[str, list[str]],
    variables: list[str],
    *,
    total_points_cap: int | None = None,
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
    if total_points_cap is not None and variables:
        max_points = min(max_points, total_points_cap // len(variables))
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
    lowpass_tau_minutes = _positive_query_float(
        params, "lowpass_tau_minutes", default=5.0
    )
    diff_interval_minutes = _optional_positive_query_float(
        params, "diff_interval_minutes"
    )
    transformed = transform_frame(
        segmented[columns],
        _single(params, "preprocess_mode", "raw"),
        int(_single(params, "detrend_window", "24") or 24),
        lowpass_tau_minutes=lowpass_tau_minutes,
        diff_interval_minutes=diff_interval_minutes,
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
    variables = list(dict.fromkeys(variables))
    if not variables:
        raise ValueError("请选择至少一个趋势变量")
    if len(variables) > 8:
        raise ValueError("最多选择 8 个趋势变量")

    exclude_window_payload = _exclude_window_payload(
        _exclude_window_context(
            _single(params, "file_id"),
            _single(params, "time_column"),
            _single(params, "encoding", "utf-8-sig"),
        )
    )

    transformed, raw_rows, max_points = _chart_frame_from_params(
        params,
        variables,
        total_points_cap=MAX_TREND_TOTAL_POINTS,
    )
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
        **exclude_window_payload,
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
    ranked: pd.DataFrame, risk: pd.DataFrame, config: AnalysisConfig, metrics: dict[str, str], recommended: pd.DataFrame | None = None
) -> dict[str, Any]:
    high_risk = int((risk.get("risk_count", pd.Series(dtype=float)) > 0).sum()) if not risk.empty else 0
    review = int((ranked.get("recommended_use", pd.Series(dtype=str)).astype(str) == "prediction_candidate").sum()) if not ranked.empty else 0
    overview_ranked = ranked
    reference_columns = ["is_residual_control", "is_capacity_reference", "is_segment_reference"]
    control_reference_count = int(ranked.reindex(columns=reference_columns, fill_value=False).fillna(False).astype(bool).any(axis=1).sum())
    return {
        "top10": _records(overview_ranked.head(10)),
        "effective_variables": int(len(ranked)),
        "recommended_candidate_count": int(len(recommended)) if recommended is not None else 0,
        "control_reference_count": control_reference_count,
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


def _initial_screening_frame(ranked: pd.DataFrame) -> pd.DataFrame:
    return ranked[[column for column in INITIAL_SCREENING_COLUMNS if column in ranked.columns]].copy()


def _recommended_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in RECOMMENDED_CANDIDATE_COLUMNS if column in frame.columns]].copy()


def _order_recommended_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rank_column = (
        "candidate_priority_rank"
        if "candidate_priority_rank" in frame.columns
        else "candidate_pool_rank"
        if "candidate_pool_rank" in frame.columns
        else None
    )
    if rank_column is None:
        return frame
    ordered = frame.copy()
    ordered["_candidate_order_rank"] = pd.to_numeric(ordered[rank_column], errors="coerce")
    ordered["_candidate_variable"] = ordered.get("variable", pd.Series("", index=ordered.index)).astype(str)
    return ordered.sort_values(["_candidate_order_rank", "_candidate_variable"], na_position="last", kind="stable").drop(columns=["_candidate_order_rank", "_candidate_variable"])


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


def _positive_query_float(
    params: dict[str, list[str]], name: str, *, default: float
) -> float:
    if name not in params:
        return default
    value = _single(params, name, "")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite value greater than 0") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a finite value greater than 0")
    return resolved


def _optional_positive_query_float(
    params: dict[str, list[str]], name: str
) -> float | None:
    value = _single(params, name, "")
    if value == "":
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite value greater than 0") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a finite value greater than 0")
    return resolved


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
    .row > label { align-self:start; width:100%; }
    label.checkbox-row { display:flex; align-items:center; align-self:end; gap:8px; min-height:31px; }
    label.checkbox-row input[type="checkbox"] { width:auto; margin:0; }
    .check { display:flex; align-items:center; gap:8px; color:var(--text); font-size:14px; }
    .check input { width:auto; }
    .actions { display:flex; gap:10px; flex-wrap:wrap; }
    .multi-dropdown { border:1px solid var(--line); border-radius:6px; background:var(--panel); }
    .multi-dropdown > summary { list-style:none; cursor:pointer; padding:6px 8px; font-size:var(--font-xs); text-align:left; }
    .multi-dropdown > summary::-webkit-details-marker { display:none; }
    .single-dropdown { min-width:0; border:1px solid var(--line); border-radius:6px; background:var(--panel); }
    .single-dropdown > summary { display:flex; align-items:center; justify-content:space-between; gap:8px; list-style:none; cursor:pointer; padding:6px 8px; color:var(--text); font-size:var(--font-xs); }
    .single-dropdown > summary::after { content:"⌄"; font-size:14px; line-height:1; }
    .single-dropdown[open] > summary::after { content:"⌃"; }
    .single-dropdown > summary::-webkit-details-marker { display:none; }
    .single-dropdown > .select-filter { width:calc(100% - 16px); margin:6px 8px 0; }
    .single-options { max-height:180px; overflow:auto; border-top:1px solid var(--line); padding:6px 8px; display:grid; gap:2px; }
    .single-option { border-radius:4px; padding:6px 8px; background:transparent; color:var(--text); font-size:var(--font-xs); font-weight:400; text-align:left; }
    .single-option:hover, .single-option[aria-selected="true"] { background:var(--surface-muted); }
    .variable-select-native { display:none; }
    .multi-filter { width:calc(100% - 16px); margin:6px 8px 0; }
    .select-filter-empty { padding:6px 8px; color:var(--muted); font-size:var(--font-xs); }
    .multi-options { max-height:180px; min-width:260px; overflow:auto; border-top:1px solid var(--line); padding:6px 8px; display:grid; gap:4px; }
    .multi-options label { display:grid; grid-template-columns:16px 1fr; align-items:center; column-gap:8px; font-size:var(--font-xs); color:var(--text); text-align:left; line-height:1.2; }
    .multi-options label[hidden] { display:none; }
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
    .chart-controls { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; align-items:end; }
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
    .exclude-windows { margin:12px 0; padding:12px; border:1px solid var(--line); border-radius:8px; background:var(--surface-muted); }
    .exclude-windows h3 { margin:0 0 6px; font-size:var(--font-sm); }
    .exclude-window-list { display:grid; gap:6px; margin:8px 0; }
    .exclude-window-item { display:flex; align-items:center; justify-content:space-between; gap:10px; font-size:var(--font-sm); }
    .exclude-window-item button { flex:0 0 auto; }
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
    @media (max-width:900px) { main { grid-template-columns:1fr; padding:12px; } .row { grid-template-columns:1fr; } .llm-config-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } .trend-stats { grid-template-columns:repeat(2, minmax(0, 1fr)); } .scatter-matrix-controls { grid-template-columns:repeat(2, minmax(0, 1fr)); } .chart-controls { grid-template-columns:repeat(2,minmax(120px,1fr)); } }
    @media (max-width:560px) { .grid { grid-template-columns:1fr; } .llm-config-grid { grid-template-columns:1fr; } .trend-stats { grid-template-columns:1fr; } .scatter-matrix-controls { grid-template-columns:1fr; } .chart-controls { grid-template-columns:1fr; } }
  </style>
  <style>
    /* ============================================================
       Apple 设计语言主题层（依据 DESIGN-apple.md）
       仅覆盖视觉表现：颜色 / 字体 / 圆角 / 间距 / 按钮 / 卡片
       不改变任何元素 ID、class 或交互逻辑
       ============================================================ */
    :root {
      --bg:#f5f5f7;
      --panel:#ffffff;
      --line:rgba(0,0,0,.08);
      --line-soft:#e8e8ed;
      --text:#1d1d1f;
      --muted:#7a7a7a;
      --text-subtle:#7a7a7a;
      --accent:#0066cc;
      --accent-bright:#0071e3;
      --accent-soft:rgba(0,102,204,.10);
      --ink-muted-80:#333333;
      --surface-pearl:#fafafc;
      --surface-black:#000000;
      --surface-muted:#f5f5f7;
      --focus:#0071e3;
      --green:#14633b;
      --warn:#b45309;
      --danger-bg:#fdecec; --danger-text:#9c1c1c;
      --warning-bg:#fdf3d7; --warning-text:#8f5900;
      --info-bg:#e8f0fe; --info-text:#0a5da8;
      --success-bg:#e6f6ec; --success-text:#14633b;
      --font-stack:"SF Pro Display","SF Pro Text",system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
    }
    body {
      font-family:var(--font-stack);
      font-size:15px;
      line-height:1.5;
      -webkit-font-smoothing:antialiased;
    }
    /* 全局导航：纯黑细条（global-nav） */
    .global-nav {
      position:sticky; top:0; z-index:60;
      display:flex; align-items:center; justify-content:center;
      height:44px; padding:0 20px;
      background:var(--surface-black); color:#fff;
    }
    .global-nav-brand {
      font-size:12px; font-weight:600; letter-spacing:-.12px;
      color:#fff; white-space:nowrap;
    }
    /* 页头 Hero：大标题 + 副标题 */
    header {
      padding:56px 24px 44px;
      background:var(--bg);
      border-bottom:none;
      text-align:center;
    }
    h1 {
      margin:0 0 10px;
      font-size:40px; font-weight:600; line-height:1.1; letter-spacing:-.374px;
      color:var(--text);
    }
    .subtitle {
      max-width:640px; margin:0 auto;
      color:var(--muted);
      font-size:17px; font-weight:400; line-height:1.47; letter-spacing:-.224px;
    }
    /* 主布局：羊皮纸画布 + 白色面板 */
    main {
      max-width:1560px; margin:0 auto;
      padding:20px 28px 56px;
      gap:24px;
      align-items:start;
    }
    section {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:18px;
      padding:20px;
    }
    .controls { gap:14px; font-size:13px; align-content:start; }
    .control-group {
      gap:10px; padding:14px;
      border:1px solid var(--line);
      border-radius:11px;
      background:var(--surface-muted);
    }
    .control-group-title {
      font-size:13px; font-weight:600; letter-spacing:-.12px;
      color:var(--ink-muted-80);
    }
    label { font-size:12px; color:var(--muted); letter-spacing:-.12px; }
    input[type="text"], input[type="number"], input[type="datetime-local"], input[type="password"], select {
      padding:8px 12px;
      border:1px solid var(--line);
      border-radius:11px;
      background:var(--panel);
      color:var(--text);
      font-size:13px;
      transition:border-color .15s ease;
    }
    input[type="text"]:hover, input[type="number"]:hover, input[type="datetime-local"]:hover, input[type="password"]:hover, select:hover {
      border-color:rgba(0,0,0,.16);
    }
    input[type="text"]:focus, input[type="number"]:focus, input[type="datetime-local"]:focus, input[type="password"]:focus, select:focus {
      border-color:var(--accent);
    }
    label.checkbox-row { color:var(--text); font-size:13px; }
    /* 自定义下拉：胶囊芯片 */
    .multi-dropdown, .single-dropdown {
      background:transparent; border:none; border-radius:0;
    }
    .multi-dropdown > summary, .single-dropdown > summary {
      border:1px solid var(--line);
      border-radius:9999px;
      background:var(--panel);
      padding:8px 14px;
      font-size:13px;
      color:var(--text);
    }
    .single-dropdown > summary::after { color:var(--muted); }
    .multi-options, .single-options {
      border:1px solid var(--line);
      border-radius:11px;
      background:var(--panel);
      box-shadow:0 8px 24px rgba(0,0,0,.08);
    }
    .single-option:hover, .single-option[aria-selected="true"] { background:var(--surface-muted); }
    /* 按钮：蓝色胶囊 + 按压微交互 */
    button {
      border:1px solid transparent;
      border-radius:9999px;
      padding:10px 22px;
      background:var(--accent);
      color:#fff;
      font-size:14px; font-weight:400; letter-spacing:-.12px;
      transition:background .15s ease, transform .12s ease;
    }
    button:hover { background:var(--accent-bright); }
    button:active { transform:scale(.95); }
    button:disabled { opacity:.4; }
    button.secondary {
      background:var(--panel);
      color:var(--ink-muted-80);
      border-color:var(--line);
    }
    button.secondary:hover { background:var(--surface-pearl); }
    /* 状态条与提示框 */
    .status { padding:10px 14px; border-radius:11px; font-size:13px; line-height:1.5; }
    .help {
      padding:10px 14px;
      border:1px solid var(--line);
      border-radius:11px;
      background:var(--surface-muted);
      font-size:13px; line-height:1.6; color:var(--muted);
    }
    .note { font-size:12px; line-height:1.5; }
    /* Tab 栏：悬浮胶囊分段控件 */
    .tabs {
      top:44px;
      background:rgba(245,245,247,.96);
      border-bottom:1px solid var(--line);
      border-radius:11px;
      padding:6px;
      gap:4px;
    }
    .tab-button {
      background:transparent;
      color:var(--muted);
      border-radius:9999px;
      padding:0 16px;
      font-size:13px;
    }
    .tab-button:hover { background:rgba(0,0,0,.05); color:var(--text); }
    .tab-button.active { background:var(--accent); color:#fff; }
    h2 {
      margin:4px 0 14px;
      font-size:24px; font-weight:600; line-height:1.25; letter-spacing:-.224px;
      color:var(--text);
    }
    h3 {
      margin:0 0 10px;
      font-size:17px; font-weight:600; line-height:1.3; letter-spacing:-.224px;
      color:var(--text);
    }
    h4 {
      margin:14px 0 8px;
      font-size:15px; font-weight:600; letter-spacing:-.12px;
      color:var(--text);
    }
    /* 指标卡 */
    .overview-grid { gap:12px; }
    .metric-card {
      width:auto; min-width:150px; min-height:84px;
      padding:14px 16px;
      border:1px solid var(--line);
      border-radius:18px;
      background:var(--panel);
    }
    .metric-value {
      font-size:28px; font-weight:600; line-height:1.1; letter-spacing:-.28px;
      color:var(--text);
    }
    .metric-label { font-size:12px; color:var(--muted); line-height:1.35; }
    /* 图表容器 */
    .chart { border-radius:11px; }
    .scatter-matrix-chart { border-radius:11px; }
    .lag-profile-panel { padding:12px; border-radius:11px; background:var(--surface-muted); }
    .lag-profile-chart { border-radius:8px; background:var(--panel); }
    .trend-stat-card { border-radius:11px; }
    /* 表格 */
    .table-wrap, .terms-help-table-wrap {
      border-radius:11px;
      box-shadow:none;
    }
    th, td { padding:9px 12px; }
    th {
      background:var(--surface-pearl);
      color:var(--ink-muted-80);
      font-size:12px; font-weight:600; letter-spacing:-.12px;
      box-shadow:0 1px 0 var(--line-soft);
    }
    th:first-child { background:var(--surface-pearl); }
    tbody tr:nth-child(even) { background:#f7f7f9; }
    tbody tr:hover { background:var(--surface-muted); }
    th.sortable:hover { background:#eef1f6; }
    .compact-result-table tbody tr.selected { background:var(--info-bg); }
    .clickable-row:hover { background:var(--surface-muted); }
    /* 详情面板 / 复核卡片 */
    .detail-panel, .review-card { border-radius:11px; }
    .detail-field, .metric-item { border-radius:8px; background:var(--surface-muted); }
    /* 弹窗 */
    .modal-backdrop { background:rgba(0,0,0,.4); }
    .modal-card { border-radius:18px; box-shadow:0 24px 64px rgba(0,0,0,.22); }
    .modal-close { background:var(--surface-muted); color:var(--text); }
    /* 下载链接：蓝色描边胶囊 */
    .download-buttons a {
      border:1px solid var(--accent);
      border-radius:9999px;
      background:transparent;
      color:var(--accent);
      padding:8px 16px;
      font-size:13px;
      transition:background .15s ease;
    }
    .download-buttons a:hover { background:var(--accent-soft); }
    /* 空状态 / 代码块 / Markdown 报告 */
    .empty { border-radius:11px; }
    pre { border-radius:11px; }
    .markdown-report {
      padding:28px 32px;
      border-radius:18px;
      font-size:15px; line-height:1.7;
    }
    .markdown-report h1 { font-size:28px; letter-spacing:-.28px; }
    .markdown-report h2 { font-size:22px; letter-spacing:-.224px; border-bottom-color:var(--line-soft); }
    .markdown-report h3 { font-size:17px; }
    .markdown-report code { background:#f0f2f5; }
    .small-button { padding:4px 12px; font-size:12px; }
    /* 页脚 */
    .site-footer {
      padding:28px 24px 40px;
      background:var(--bg);
      border-top:1px solid var(--line);
      text-align:center;
      color:var(--muted);
      font-size:12px; line-height:1.6; letter-spacing:-.12px;
    }
    .site-footer p { margin:0; }
    /* 滞后曲线：旧强调色映射为 Apple 蓝（纯展示层覆盖） */
    .lag-profile-chart svg path[stroke="#176b87"] { stroke:var(--accent) !important; }
    .lag-profile-chart svg circle[fill="#176b87"] { fill:var(--accent) !important; }
    .lag-profile-chart svg text[fill="#176b87"] { fill:var(--accent) !important; }
    .lag-profile-line:not(.spearman) { border-color:var(--accent) !important; }
    /* 响应式微调 */
    @media (max-width:900px) {
      main { padding:16px; gap:16px; }
      header { padding:40px 20px 28px; }
      h1 { font-size:32px; }
      .tabs { border-radius:16px; }
    }
    @media (max-width:640px) {
      h1 { font-size:28px; }
      .subtitle { font-size:15px; }
      .metric-value { font-size:24px; }
    }
  </style>
  <style>
    /* 工作台层级：保持现有组件语义，只收紧视觉层次与交互状态。 */
    :root {
      --radius-sm:4px;
      --radius-md:6px;
      --radius-panel:8px;
      --surface-control:#fbfcfd;
      --surface-log:#f8fafc;
      --focus-ring:rgba(0,113,227,.28);
    }
    [hidden] { display:none !important; }
    header {
      padding:28px 32px 24px;
      text-align:left;
    }
    header h1 { font-size:32px; }
    .subtitle { margin:0; max-width:760px; font-size:15px; }
    main { gap:20px; padding:18px 24px 44px; }
    section { border-radius:var(--radius-panel); padding:18px; }
    .controls { border-top:3px solid var(--accent); }
    .results { border-top:3px solid var(--accent); }
    .section-heading, .results-heading {
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:16px;
      margin-bottom:14px;
    }
    .section-heading h2, .results-heading h2 { margin:0 0 4px; font-size:20px; }
    .section-description, .results-description {
      margin:0;
      max-width:680px;
      color:var(--muted);
      font-size:var(--font-sm);
      line-height:1.45;
    }
    .results-priority {
      flex:0 0 auto;
      padding:5px 8px;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      color:var(--muted);
      background:var(--surface-muted);
      font-size:var(--font-xs);
      white-space:nowrap;
    }
    .control-group { border-radius:var(--radius-md); background:var(--surface-control); }
    .control-group.primary-group { background:var(--panel); border-color:var(--line); }
    .advanced-parameters {
      border:1px solid var(--line);
      border-radius:var(--radius-md);
      background:var(--surface-control);
    }
    .advanced-parameters > summary {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      cursor:pointer;
      padding:11px 12px;
      color:var(--text);
      font-size:var(--font-base);
      font-weight:650;
      list-style:none;
    }
    .advanced-parameters > summary::-webkit-details-marker { display:none; }
    .advanced-parameters > summary::after { content:"+"; color:var(--muted); font-size:16px; }
    .advanced-parameters[open] > summary::after { content:"−"; }
    .advanced-summary-note { color:var(--muted); font-size:var(--font-xs); font-weight:400; margin-left:auto; }
    .advanced-parameters-body { display:grid; gap:10px; padding:0 10px 10px; }
    .multi-dropdown, .single-dropdown { border-radius:var(--radius-md); }
    .multi-dropdown > summary, .single-dropdown > summary { border-radius:var(--radius-md); }
    button { border-radius:var(--radius-md); min-height:40px; }
    button:focus-visible, a:focus-visible, summary:focus-visible, input:focus-visible, select:focus-visible {
      outline:2px solid var(--focus);
      outline-offset:2px;
      box-shadow:0 0 0 3px var(--focus-ring);
    }
    button:disabled, input:disabled, select:disabled { cursor:not-allowed; }
    input:disabled, select:disabled { color:var(--muted); background:#eef1f4; border-color:var(--line-soft); }
    .status-panel {
      display:grid;
      gap:8px;
      padding:12px;
      border:1px solid var(--line);
      border-left:3px solid var(--accent);
      border-radius:var(--radius-md);
      background:var(--surface-log);
    }
    .status-panel-heading { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
    .status-panel-title { font-size:var(--font-sm); font-weight:700; color:var(--text); }
    .status-panel-caption { color:var(--muted); font-size:var(--font-xs); }
    .status { border-radius:var(--radius-sm); }
    .status.loading { display:flex; align-items:center; gap:8px; }
    .status.loading::before { content:"处理中"; flex:0 0 auto; padding:2px 5px; border:1px solid var(--line); border-radius:3px; color:var(--text); background:var(--panel); font-size:var(--font-xs); font-weight:700; }
    .tabs { border-radius:var(--radius-md); }
    .tab-button { border-radius:var(--radius-sm); }
    .metric-card { border-radius:var(--radius-md); }
    .empty {
      min-height:72px;
      display:grid;
      place-items:center;
      border-radius:var(--radius-md);
      background:var(--surface-log);
      line-height:1.45;
    }
    .detail-hint {
      padding:8px 10px;
      border-left:3px solid var(--accent);
      color:var(--muted);
      background:var(--surface-muted);
      font-size:var(--font-sm);
    }
    .table-wrap, .terms-help-table-wrap {
      width:100%;
      border-radius:var(--radius-md);
      box-shadow:none;
    }
    table { min-width:720px; }
    th { font-weight:700; }
    td.numeric, th.numeric { text-align:right; font-variant-numeric:tabular-nums; }
    .status-label {
      display:inline-flex;
      align-items:center;
      max-width:100%;
      padding:3px 6px;
      border:1px solid var(--line);
      border-radius:var(--radius-sm);
      background:var(--surface-muted);
      color:var(--text);
      font-size:var(--font-xs);
      line-height:1.25;
      white-space:normal;
    }
    .status-label-positive { border-color:#a7d7ba; background:#f0faf3; }
    .status-label-caution { border-color:#e8cf8d; background:#fffaf0; }
    .status-label-negative { border-color:#e7b5b5; background:#fff5f5; }
    .status-label-neutral { border-color:var(--line); background:var(--surface-muted); }
    .modal-card { border-radius:var(--radius-panel); box-shadow:0 10px 28px rgba(15,23,42,.18); }
    .detail-panel, .review-card, .detail-field, .metric-item, pre, .markdown-report { border-radius:var(--radius-md); }
    .download-buttons a { border-radius:var(--radius-md); }
    .results .help { border-radius:var(--radius-md); }
    @media (max-width:900px) {
      main { grid-template-columns:1fr; padding:14px; }
      .section-heading, .results-heading { display:block; }
      .results-priority { display:inline-block; margin-top:8px; }
    }
    @media (max-width:640px) {
      header { padding:24px 16px 20px; }
      header h1 { font-size:28px; }
      main { padding:10px; gap:12px; }
      section { padding:14px; }
      .tabs { flex-wrap:nowrap; overflow-x:auto; justify-content:flex-start; scrollbar-width:thin; }
      .tab-button { flex:0 0 auto; min-width:92px; }
      .controls .grid, .controls .row, .advanced-parameters-body .grid, .advanced-parameters-body .row { grid-template-columns:1fr; }
      .actions { display:grid; grid-template-columns:1fr; }
      .actions button { width:100%; }
      .overview-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }
      .metric-card { min-width:0; width:auto; }
      .status-panel-heading { display:block; }
      .status-panel-caption { display:block; margin-top:2px; }
      .chart-controls, .trend-options, .scatter-matrix-controls, .llm-config-grid { grid-template-columns:1fr; }
      .trend-stats { grid-template-columns:1fr; }
      table { min-width:680px; }
    }
  </style>
</head>
<body>
  <nav class="global-nav" aria-label="主导航">
    <span class="global-nav-brand">化工装置时序相关性分析</span>
  </nav>
  <header>
    <h1>化工装置时序相关性分析</h1>
    <div class="subtitle">浏览器负责上传和展示，Python 后台处理大数据并生成下载结果。</div>
  </header>
  <main>
    <section class="controls" aria-labelledby="controlsTitle">
      <div class="section-heading">
        <div>
          <h2 id="controlsTitle">分析参数</h2>
          <p class="section-description">先上传数据并确认基础参数，再运行主筛查。工况、剔除和复核设置收在高级参数中。</p>
        </div>
      </div>
      <div class="control-group primary-group">
        <div class="control-group-title">数据输入</div>
      <label>数据文件（CSV / Excel / TXT）
        <input id="fileInput" type="file" accept=".csv,.txt,.tsv,.xlsx,.xls,.xlsm,text/csv,text/plain,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel">
      </label>
      <div class="actions">
        <button id="upload">上传并识别列</button>
        <button id="reset" class="secondary">清空</button>
      </div>
      </div>
      <div class="control-group primary-group">
        <div class="control-group-title">基础分析参数</div>
        <div class="row">
          <label>时间列<select id="timeColumn"></select></label>
          <label>目标列<select id="targetColumn"></select></label>
        </div>
        <div class="row">
          <label>最大滞后点数<input id="maxLag" type="number" min="0" max="5000" value="12"></label>
          <label>输出前 K 个<input id="topK" type="number" min="1" max="2000" value="20"></label>
        </div>
        <div class="row">
          <label>最小有效比例<input id="minValidRatio" type="number" min="0.1" max="1" step="0.05" value="0.7"></label>
          <label>重采样间隔（分钟）<input id="resampleRule" type="number" min="1" step="1" inputmode="numeric" placeholder="可留空，例如 5"></label>
        </div>
        <div class="row">
          <label>预处理模式
            <select id="preprocessMode">
              <option value="raw">原始数据</option>
              <option value="lowpass">一阶低通平滑</option>
              <option value="lowpass_detrend">一阶低通 + 去趋势</option>
              <option value="lowpass_diff">一阶低通 + 差分</option>
            </select>
          </label>
          <label id="lowpassTauLabel" hidden>低通时间常数 τ（分钟）<input id="lowpassTauMinutes" type="number" min="0.1" step="0.1" value="5.0"></label>
          <label id="diffIntervalLabel" hidden>差分间隔（分钟）<input id="diffIntervalMinutes" type="number" min="1" step="1" placeholder="留空表示一个采样周期"></label>
          <label id="detrendWindowLabel" hidden>去趋势窗口点数<input id="detrendWindow" type="number" min="3" max="100000" value="24"></label>
        </div>
        <div class="help">选择 Raw：只运行 Raw，完成后直接发布正式初筛。选择任一预处理模式（一阶低通 / 一阶低通+去趋势 / 一阶低通+差分）：同时运行 Raw 和该预处理模式两个独立初筛，完成后需要人工确认正式分支。</div>
      </div>
      <details id="advancedParameters" class="advanced-parameters">
        <summary><span>高级参数</span><span class="advanced-summary-note">剔除列 · 工况与残差 · 复核</span></summary>
        <div class="advanced-parameters-body">
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
              <label>可信度审查候选数量<input id="causalTopN" type="number" min="1" max="1000" placeholder="可留空"></label>
              <label>风险标签包含过滤<input id="riskFlagFilter" placeholder="如 共同负荷驱动，留空表示不过滤"></label>
            </div>
          </div>
        </div>
      </details>
      <div class="status-panel" aria-label="运行日志">
        <div class="status-panel-heading">
          <span class="status-panel-title">运行状态与日志</span>
          <span class="status-panel-caption">后台任务、上传和验证状态会显示在这里</span>
        </div>
        <div id="status" class="status info" role="status" aria-live="polite" aria-busy="false"></div>
        <div class="note">大文件会由 Python 后台处理。分析期间请不要关闭启动服务的命令窗口。</div>
      </div>
    </section>

    <section class="results" aria-labelledby="resultsTitle">
      <div class="results-heading">
        <div>
          <h2 id="resultsTitle">分析结果</h2>
          <p class="results-description">默认先展示主筛查摘要和候选表格；趋势图、后续验证与下载结果按需查看。</p>
        </div>
        <span class="results-priority">主任务：筛选可复核候选</span>
      </div>
      <div class="tabs" role="tablist" aria-label="结果分类">
        <button class="tab-button active" role="tab" aria-selected="true" aria-controls="overviewTab" id="tab-overviewTab" data-tab="overviewTab" tabindex="0">初步分析</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="trendTab" id="tab-trendTab" data-tab="trendTab" tabindex="-1">趋势图</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="validationTab" id="tab-validationTab" data-tab="validationTab" tabindex="-1">二次验证</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="causalReviewTab" id="tab-causalReviewTab" data-tab="causalReviewTab" tabindex="-1">可信度审查</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="xgbValidationTab" id="tab-xgbValidationTab" data-tab="xgbValidationTab" tabindex="-1">时间外预测验证</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="llmReportTab" id="tab-llmReportTab" data-tab="llmReportTab" tabindex="-1">AI 综合解读</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="downloadsTab" id="tab-downloadsTab" data-tab="downloadsTab" tabindex="-1">下载</button>
        <button class="tab-button" role="tab" aria-selected="false" aria-controls="termsHelpTab" id="tab-termsHelpTab" data-tab="termsHelpTab" tabindex="-1">术语与标签说明</button>
      </div>

      <div id="overviewTab" class="tab-panel active" role="tabpanel" aria-labelledby="tab-overviewTab">
        <h2>初步分析</h2>
        <div class="actions"><button id="analyze" disabled>开始分析</button></div>
        <div id="branchSelectionSection" class="control-group" hidden>
          <div class="control-group-title">Raw vs Processed 对比</div>
          <div id="branchSelectionStatus" class="help"></div>
          <div id="preprocessingComparisonTable" class="empty">选择任一预处理模式完成双分支初筛后，此处显示冻结的预处理对比结果。</div>
          <div class="actions">
            <button id="confirmRawBranch" disabled>确认使用原始数据</button>
            <button id="confirmProcessedBranch" disabled>确认使用预处理数据</button>
          </div>
          <div id="branchLockedHint" class="help" hidden>后续验证已开始，当前初筛分支已锁定；如需切换请重新分析。</div>
        </div>
        <div id="overview" class="overview-grid"></div>
        <div id="analysisTimingBreakdown" class="help" hidden></div>
        <h2>初步分析 Top 10</h2>
        <div class="help">final_score 以基础关联强度 × 数据质量为基线，Residual 与稳定性只提供有限正向奖励；仅明确的目标领先时间证据会负向约束得分。</div>
        <div class="detail-hint">主表格中的行可点击查看变量详情；详情会在独立窗口中打开，不改变排序或分析结果。</div>
        <div id="overviewTop" class="empty">上传数据并点击“开始分析”后显示结果。</div>
        <div id="candidatesTab">
          <h2>去负荷(残差)验证候选</h2>
          <div class="help">基于原始关联和去负荷后的残差信号筛选得到，用于后续验证排序，不代表因果关系或独立驱动结论。</div>
          <div id="recommendedCandidateTable" class="empty">完成主筛查后显示去负荷(残差)验证候选。</div>
          <h2>控制/负荷参考变量</h2>
          <div class="help">由已配置的残差控制列、负荷列、分段列，以及位号末尾的.SV/.SP/.MV自动识别。此类变量保留原始统计得分和排名，但默认不进入普通验证候选排序。</div>
          <div class="help">位号包含FIC、TIC、PIC、LIC或AIC但以.PV结尾的变量，不会仅凭前缀自动识别为控制参考变量。</div>
          <div id="controlReferenceTable" class="empty">当前未识别到控制或负荷参考变量。</div>
          <h2>完整初步分析结果</h2>
          <div class="help">展示所有有效分析变量的初步统计结果。控制、负荷和工况参考变量不会从完整结果中删除；验证候选范围单独保存于 recommended_candidates.csv。</div>
          <h3>结果质量提示</h3>
          <div id="screeningQualityHints" class="empty">完成主筛查后显示结果质量提示。</div>
          <div id="table" class="empty">上传数据并点击“开始分析”后显示结果。</div>
        </div>
      </div>

      <div id="trendTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-trendTab" hidden>
        <h2>趋势图</h2>
        <div id="trendReviewHint" class="help">点击可信度审查摘要中的“查看趋势”后显示候选变量审查提示。</div>
        <div class="chart-controls">
          <label>数据 1<select id="trendVar1"></select></label>
          <label>数据 2<select id="trendVar2"></select></label>
          <label>数据 3<select id="trendVar3"></select></label>
          <label>数据 4<select id="trendVar4"></select></label>
          <label>数据 5<select id="trendVar5"></select></label>
          <label>数据 6<select id="trendVar6"></select></label>
          <label>数据 7<select id="trendVar7"></select></label>
          <label>数据 8<select id="trendVar8"></select></label>
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
        <div class="actions">
          <button id="clearTrendSelection" type="button" class="secondary" disabled>清除选择</button>
          <button id="addExcludeWindow" type="button" disabled>加入排除窗口</button>
        </div>
        <div id="trendSelectionInfo" class="help" aria-live="polite">可在趋势绘图区横向拖动选择时间窗口。</div>
        <section class="exclude-windows" aria-labelledby="excludeWindowsTitle">
          <h3 id="excludeWindowsTitle">已标记排除时间段</h3>
          <div class="help">排除窗口将在下一次分析时生效；修改不会重算已有分析结果。</div>
          <div id="excludeWindowList" class="exclude-window-list empty">尚未添加排除窗口。</div>
          <div id="excludeWindowStats" class="help">已标记：0 个窗口 / 0 点（0.0%）</div>
          <div class="actions"><button id="restoreAllExcludeWindows" type="button" class="secondary" disabled>恢复所有数据</button></div>
        </section>
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
          <span>下游阶段只使用已确认正式分支的预处理口径与主筛查滞后参数，不再提供独立的二次重采样或补充白名单切换。</span>
          <span id="downstreamGateHint" class="help" hidden>请先确认正式初筛分支。</span>
        </div>
        <div class="actions">
          <button id="runEnhancedScreening" disabled>运行增强筛选</button>
          <button id="runGranger" disabled>运行 Granger 验证</button>
          <button id="runModel" disabled>运行随机森林模型解释</button>
        </div>
        <section id="validationSummarySection" aria-labelledby="validationSummaryTitle">
          <h3 id="validationSummaryTitle">统一验证结论</h3>
          <div class="help">默认展示每个变量的验证状态、证据一致性、主要支持证据和限制因素。这里仅汇总已执行的二级验证；未执行、未计算或失败的分析会明确标记，不会被当作支持证据。</div>
          <div id="validationSummaryTable" class="empty">完成主筛查后显示统一验证结论。</div>
          <h3>阶段字段</h3>
          <div class="help">阶段字段仅用于区分初筛、普通验证和条件验证的 signed lag 与模型提升；缺失值保持缺失，不参与评分或排序。</div>
          <div id="validationFieldsTable" class="empty">完成主筛查后显示阶段字段。</div>
        </section>

        <details id="enhancedValidationDetails" class="validation-detail-section">
          <summary>Enhanced Validation 详细结果</summary>
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
        </details>

        <details id="grangerValidationDetails" class="validation-detail-section">
          <summary>Granger 详细结果</summary>
          <div class="help">Granger 显著表示历史预测信息，不等于因果成立。</div>
          <div id="grangerTable" class="empty">未启用 Granger 检验。</div>
        </details>

        <details id="modelExplanationDetails" class="validation-detail-section">
          <summary>Model Explanation 详细结果</summary>
          <div class="help">随机森林重要性表示模型依赖，不等于可操作性或因果结论。</div>
          <h3>随机森林模型解释变量排序</h3>
          <div class="help">该表按变量汇总随机森林/SHAP 重要性，每个变量仅显示最强 lag。结果表示预测模型依赖，不代表因果关系或可操作性。</div>
          <div id="modelVariableImportanceTable" class="empty">运行随机森林模型解释后显示变量排序。</div>
          <h3>随机森林模型解释特征明细</h3>
          <div id="importanceTable" class="empty">未启用随机森林模型解释。</div>
          <h3>随机森林模型遗漏探索</h3>
          <div class="help">该表仅检查初筛 Rank K+1~K+10 中的遗漏线索，最多显示 5 个并保持初筛顺序；不属于二级验证结论，不会自动加入推荐、候选池或任何排序。结果仅表示预测模型依赖，不代表因果关系或可操作性。</div>
          <div id="modelDiscoveredTable" class="empty">运行随机森林模型解释后显示遗漏探索线索。</div>
        </details>
        <section id="verificationReviewPoolSection" aria-labelledby="verificationReviewPoolTitle">
          <h3 id="verificationReviewPoolTitle">二级验证复核池</h3>
          <div class="help">复核池独立于一级初筛候选池。初筛 Top-K 自动进入；人工加入和模型遗漏探索变量只有点击“加入复核池”后，才会进入后续 Enhanced、Granger 和 Model Explanation。</div>
          <div class="actions">
            <label>人工加入变量<input id="manualReviewPoolVariable" placeholder="输入初筛变量名"></label>
            <button id="addManualReviewPool" disabled>加入复核池</button>
            <label>确认模型遗漏探索变量<select id="modelDiscoveryReviewPoolVariable"><option value="">请选择模型探索变量</option></select></label>
            <button id="addModelDiscoveryReviewPool" disabled>加入复核池</button>
          </div>
          <div id="verificationReviewPoolTable" class="empty">完成主筛查后显示二级验证复核池。</div>
        </section>
      </div>

      <div id="causalReviewTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-causalReviewTab" hidden>
        <h2>第三层可信度审查</h2>
        <div class="help">
          <span>本层是可信度审查：解释第二层预测价值是否可能受共同驱动、控制响应或统计限制影响；不是因果结论，也不改变初筛评分或排序。正式第三层候选来自已发布初筛的 causal_review_candidates.csv；风险标签包含过滤仅用于结果展示，不改变正式候选。</span>
          <span>可信度审查支持长滞后变量。默认围绕主筛查最佳滞后附近做条件 Granger 验证，避免对 1..maxlag 全量扫描造成计算过慢。如需完整扫描，可切换为 full_scan。</span>
          <span>高共线性和共同负荷风险不等于变量无效。对于统计证据支持较强的候选，平台会保留工程审查建议，同时标记统计检验受限。</span>
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
        <button id="runCausalReview" disabled>运行可信度审查</button>
        </div>
        <h2>条件 Granger 预测验证结果</h2>
        <div class="download-buttons" id="conditionalDownload"></div>
        <div id="conditionalGrangerTable" class="empty">未运行 条件 Granger 预测验证。</div>
        <h2>可信度审查摘要</h2>
        <div class="help">该表基于逐变量可信度审查证据生成，用于解释独立预测贡献、混杂风险、控制关系和统计限制。结果不是因果结论，也不改变初筛评分、排序或 Top-K。人工复核优先级仅用于第三层展示和复核建议，不参与算法评分或初筛排序；点击其它列排序仅用于辅助查看。点击“查看趋势”可自动带入目标变量和候选变量，用于人工检查滞后方向、响应形态和工艺合理性。</div>
        <div class="download-buttons" id="finalReviewSummaryDownload"></div>
        <h3>可信度审查概览</h3>
        <div id="finalReviewQualityOverview" class="overview-grid"></div>
        <div id="finalReviewSummaryTable" class="empty">未运行 可信度审查摘要。</div>
        <h2>逐变量可信度审查证据表</h2>
        <div class="help">该表整合主筛查、增强筛选、Granger、随机森林模型解释、条件 Granger 和风险标签，用于判断预测价值的独立性及其混杂、控制关系和统计限制。对于高共线性、共同负荷等限制，平台会保留工程审查建议并标记限制。该表不是因果结论，也不改变初筛结果。</div>
        <div class="download-buttons" id="causalEvidenceDownload"></div>
        <div id="causalReviewEvidenceTable" class="empty">未运行 逐变量可信度审查证据表。</div>
        <h2>人工复核证据矩阵</h2>
        <div class="help">按“变量 → 初筛结果 → 预测价值证据 → 独立性审查 → 混杂风险 → 控制关系 → 统计限制”组织展示。矩阵只引用已有初筛、二级验证、可信度审查和（如已执行）XGB字段，不产生新的评分或排序；人工复核优先级仅用于展示，不改变初筛顺序。</div>
        <div class="download-buttons" id="evidenceMatrixDownload"></div>
        <div id="evidenceMatrixTable" class="empty">未运行 可信度审查，暂无人工复核证据矩阵。</div>
      </div>


      <div id="xgbValidationTab" class="tab-panel" role="tabpanel" aria-labelledby="tab-xgbValidationTab" hidden>
        <h2>XGBoost 时间外预测验证</h2>
        <div class="help">第四层回答：候选变量在时间顺序隔离的数据中，是否仍提供额外预测信息。结果是候选变量预测增量证据和模型时间外表现，仅供人工复核参考；不用于因果结论、工艺根因判断或变量排名，不改变前三层结果。</div>
        <div class="help">Baseline（M1）：目标变量历史信息 + 配置的控制变量历史；Candidate：同一 M1 基线 + 单个候选变量历史信息。预测改善只表示候选变量提供额外预测信息，不表示候选变量决定目标变量。</div>
        <div class="row">
          <label class="checkbox-row"><input id="enableXgbValidation" type="checkbox">启用 XGB 时间外预测验证</label>
          <label>候选数量<input id="xgbTopN" type="number" min="1" max="10" value="8"></label>
          <label>最大滞后<input id="xgbMaxLag" type="number" min="1" max="5000" placeholder="自动"></label>
          <label>白名单<input id="xgbWhitelist" placeholder="变量名以逗号分隔"></label>
        </div>
        <div class="help">自动候选默认 8 个、最多 10 个；加入白名单后，总候选数量最多 12 个。</div>
        <div class="actions">
          <button id="runXgbValidation" disabled>运行 XGB 时间外预测验证</button>
        </div>
        <div id="xgbStatus" class="help" aria-live="polite">XGB 时间外预测验证未启用。</div>
        <div id="xgbRunSummary" class="overview-grid"></div>
        <h2>模型时间外验证摘要</h2>
        <div class="download-buttons" id="xgbModelSummaryDownload"></div>
        <div class="download-buttons" id="xgbFoldMetricsDownload"></div>
        <div id="xgbModelSummaryTable" class="empty">未运行 XGB 时间外预测验证。</div>
        <h2>候选变量增量验证</h2>
        <div class="help">
          <span>RMSE 改善中位数（%）：各时间外测试折中，相对 M1 基线模型的 RMSE 改善百分比中位数；改善率 = (baseline_error - candidate_error) / baseline_error × 100%。大于 0 表示加入该候选后预测误差下降，小于 0 表示预测误差上升。</span>
          <span>MAE 改善中位数（%）：各时间外测试折中，相对 M1 基线模型的 MAE 改善百分比中位数。大于 0 表示平均绝对误差下降。</span>
          <span>RMSE 改善折占比：RMSE 改善百分比大于 0 的时间折数占全部验证折数的比例，范围为 0～1；例如 0.67 表示约 67% 的时间折得到改善。数值越高，跨时间段改善越稳定。</span>
          <span>以上指标均为时间外预测增量证据，不代表工艺因果成立；不参与 ranking、scoring 或 candidate selection，也不修改 final_score、ranked_features.csv、Top-K、第二层 validation_summary 或第三层可信度审查结果。</span>
        </div>
        <div class="download-buttons" id="xgbCandidateUpliftDownload"></div>
        <div id="xgbCandidateUpliftTable" class="empty">未运行 XGB 时间外预测验证。</div>
        <details id="xgbCandidateFoldDetails">
          <summary>逐时间折验证明细</summary>
          <div class="help">每一折模拟“使用更早时间段训练，在后续未参与训练的时间段验证”。多个时间折出现正向改善，表示预测增量在当前数据范围内具有更好的跨时间一致性，但不代表长期稳定性或工艺因果关系。</div>
          <div id="xgbCandidateFoldMetricsTable" class="empty">点击候选变量汇总行后查看该变量的逐时间折明细。</div>
        </details>
        <div class="download-buttons" id="xgbCandidateFoldMetricsDownload"></div>
        <div class="download-buttons" id="xgbValidationSummaryDownload"></div>
        <div class="download-buttons" id="xgbPredictionsDownload"></div>
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

  <footer class="site-footer">
    <p>化工装置时序相关性分析 · 结果仅供工程复核参考，不构成因果结论</p>
  </footer>

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
let lastRecommendedRows = [];
let lastGrangerRows = [];
let lastImportanceRows = [];
let lastModelVariableRows = [];
let lastNearMissRows = [];
let lastModelDiscoveredRows = [];
let lastEnhancedSummaryRows = [];
let lastEnhancedLiftRows = [];
let lastEnhancedRollingRows = [];
let lastValidationSummaryRows = [];
let lastValidationFieldsRows = [];
let lastVerificationReviewPoolRows = [];
let lastConditionalRows = [];
let lastCausalEvidenceRows = [];
let lastEvidenceMatrixRows = [];
const DEFAULT_EVIDENCE_MATRIX_STATUS_LABELS = {
  validation_status: {not_run: "未执行", not_computed: "未计算", missing: "证据缺失", supported: "已有支持证据", limited: "证据有限或存在限制", failed: "执行失败"},
  evidence_consistency: {not_run: "未执行", not_computed: "未计算", missing: "证据缺失", consistent: "证据较一致", partial: "证据部分一致"},
  independent_predictive_support: {supported: "独立预测贡献证据较强", supported_with_limitations: "存在独立预测贡献证据，但存在限制", not_supported: "未形成独立预测贡献证据", not_computed: "未计算"},
  confounder_assessment: {not_assessed: "未审查", no_flagged_confounder: "未发现明显混杂风险", common_driver_risk: "可能存在共同驱动影响", shared_signal_risk: "可能存在共享信号影响", formula_relation_risk: "可能存在公式关系影响"},
  control_relation_assessment: {not_assessed: "未审查", no_control_relation_flagged: "未发现明显控制关系风险", control_reference: "控制参考变量", possible_control_response: "可能属于控制响应信号", shared_capacity_or_control_context: "可能存在负荷或控制背景影响"},
  statistical_limitation: {not_computed: "未计算", no_flagged_statistical_limitation: "未发现明显统计限制", high_collinearity_limitation: "高共线性限制", insufficient_sample_limitation: "样本不足限制", failed_statistical_limitation: "统计计算失败"},
  direction_assessment: {variable_leads_target: "变量领先目标", target_leads_variable: "目标领先变量", zero_lag: "零滞后", not_computed: "未计算"},
  xgb_status: {not_computed: "未计算", missing: "证据缺失", validated_incremental_signal: "时间外预测增量已支持", weak_incremental_value: "时间外预测增量较弱", redundant_with_baseline: "与基线信息重复", unstable_out_of_time: "时间外预测增量不稳定", insufficient_features: "有效特征不足"},
  generalization_status: {not_computed: "未计算", missing: "证据缺失", validated_incremental_signal: "时间外预测增量已支持", weak_incremental_value: "时间外预测增量较弱", redundant_with_baseline: "与基线信息重复", unstable_out_of_time: "时间外预测增量不稳定", insufficient_features: "有效特征不足"},
};
let evidenceMatrixStatusLabels = DEFAULT_EVIDENCE_MATRIX_STATUS_LABELS;
let lastFinalReviewSummaryRows = [];
let lastXgbModelSummaryRows = [];
let lastXgbCandidateUpliftRows = [];
let lastXgbCandidateFoldMetricRows = [];
let lastXgbValidationSummary = {};
let llmPromptText = "";
let llmReportMarkdown = "";
let lastModalTrigger = null;
let tableSortStates = { table: { column: "final_score", direction: "desc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
const el = (id) => document.getElementById(id);
const trendColors = ["#176b87", "#c2410c", "#6d28d9", "#15803d", "#b91c1c", "#ca8a04", "#a21caf", "#475569"];
const llmPromptEndpoint = "/api/llm_prompt";
let lastTrendSeries = [];
let lastTrendAxisMode = "shared";
let trendResizeTimer = null;
let trendTimeRangeMode = "auto";
let trendSamplingIntervalMs = null;
let trendLatestTime = "";
let trendAutoWindowActive = false;
let trendDefaultStart = "";
let trendDefaultEnd = "";
let trendSelection = null;
let excludeWindows = [];
let excludeWindowStats = { exclude_window_count: 0, excluded_rows: 0, excluded_ratio: 0 };
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
activateTab("overviewTab");
updatePreprocessControls();
el("drawTrend").addEventListener("click", drawTrend);
el("trendStart").addEventListener("input", markTrendTimeRangeManual);
el("trendEnd").addEventListener("input", markTrendTimeRangeManual);
el("trendMaxPoints").addEventListener("change", updateAutoTrendTimeRange);
el("clearTrendSelection").addEventListener("click", clearTrendSelection);
el("addExcludeWindow").addEventListener("click", addExcludeWindow);
el("restoreAllExcludeWindows").addEventListener("click", restoreAllExcludeWindows);
el("drawScatterMatrix").addEventListener("click", drawScatterMatrix);
el("preprocessMode").addEventListener("change", updatePreprocessControls);
el("confirmRawBranch").addEventListener("click", () => confirmInitialScreeningBranch("raw"));
el("confirmProcessedBranch").addEventListener("click", () => confirmInitialScreeningBranch("processed"));
el("runEnhancedScreening").addEventListener("click", runEnhancedScreening);
el("runGranger").addEventListener("click", runGranger);
el("runModel").addEventListener("click", runModel);
el("addManualReviewPool").addEventListener("click", () => addToVerificationReviewPool("manual_include"));
el("addModelDiscoveryReviewPool").addEventListener("click", () => addToVerificationReviewPool("model_discovery"));
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
  searchableMultiOptions(box);
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
  searchableMultiOptions(box);
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

function updatePreprocessControls() {
  const mode = el("preprocessMode").value || "raw";
  const lowpass = mode === "lowpass" || mode === "lowpass_detrend" || mode === "lowpass_diff";
  el("lowpassTauLabel").hidden = !lowpass;
  el("diffIntervalLabel").hidden = mode !== "lowpass_diff";
  el("detrendWindowLabel").hidden = mode !== "lowpass_detrend";
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
  searchableMultiOptions(box);
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
  syncSearchableSelect(select);
}

function refreshColumnSelectors() {
  const excluded = new Set(getExcludedColumnSelection());
  const available = recognizedNumericColumns.filter((name) => !excluded.has(name));
  const current = Object.fromEntries(
    ["targetColumn", "segmentColumn", "trendVar1", "trendVar2", "trendVar3", "trendVar4", "trendVar5", "trendVar6", "trendVar7", "trendVar8", "scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"]
      .map((id) => [id, el(id).value])
  );
  const capacity = getCapacitySelection().filter((name) => !excluded.has(name));
  const forced = getForceIncludeSelection().filter((name) => !excluded.has(name));

  restoreSelect("targetColumn", available, current.targetColumn);
  restoreSelect("segmentColumn", available, current.segmentColumn, true, "不分段");
  ["trendVar1", "trendVar2", "trendVar3", "trendVar4", "trendVar5", "trendVar6", "trendVar7", "trendVar8", "scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"].forEach((id) => {
    restoreSelect(id, available, current[id], true, "不选择");
  });
  fillCapacityOptions(available);
  setCapacitySelection(capacity);
  fillForceIncludeOptions(available);
  setForceIncludeSelection(forced);
  const whitelist = el("xgbWhitelist").value.split(/[,，]/).map((value) => value.trim()).filter((value) => value && !excluded.has(value));
  el("xgbWhitelist").value = whitelist.join(",");
  updateExcludedColumnDisabledState();
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

async function uploadFile() {
  const file = el("fileInput").files[0];
  if (!file) return setStatus("请选择 CSV、Excel 或 TXT 数据文件。");
  clearVariableFilters();
  clearLagProfileCache();
  currentAnalysisContext = {};
  try {
    setStatus("正在上传文件...", "loading");
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/upload", form);
    fileId = data.file_id;
    updateExcludeWindowState([], null);
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
    const url = `/api/columns?file_id=${encodeURIComponent(fileId)}&encoding=auto`;
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
  el("capacityDropdown").open = false;
  el("forceIncludeDropdown").open = false;
  fillSelect(el("trendVar1"), data.numericColumns);
  fillSelect(el("trendVar2"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar3"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar4"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar5"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar6"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar7"), data.numericColumns, true, "不选择");
  fillSelect(el("trendVar8"), data.numericColumns, true, "不选择");
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
  syncSearchableSelect(el("timeColumn"));
  trendTimeRangeMode = "auto";
  trendSamplingIntervalMs = Number(data.trendSamplingIntervalMs);
  trendLatestTime = data.timeEnd || "";
  trendAutoWindowActive = false;
  trendDefaultStart = data.trendStartDefault || "";
  trendDefaultEnd = data.trendEndDefault || "";
  trendSelection = null;
  if (data.trendStartDefault) el("trendStart").value = data.trendStartDefault;
  if (data.trendEndDefault) el("trendEnd").value = data.trendEndDefault;
  updateTrendSelectionInfo();
    const loadCandidate = data.numericColumns.find((name) => /load|负荷|进料|流量|feed|rate/i.test(name));
    if (loadCandidate) {
      el("segmentColumn").value = loadCandidate;
      syncSearchableSelect(el("segmentColumn"));
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
    form.append("encoding", "auto");
    form.append("time_column", el("timeColumn").value);
    form.append("target", el("targetColumn").value);
    form.append("max_lag", el("maxLag").value);
    form.append("top_k", el("topK").value);
    form.append("min_valid_ratio", el("minValidRatio").value);
    form.append("resample_rule", el("resampleRule").value.trim());
    form.append("preprocess_mode", el("preprocessMode").value);
    form.append("lowpass_tau_minutes", el("lowpassTauMinutes").value);
    form.append("diff_interval_minutes", el("diffIntervalMinutes").value.trim());
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
  if (data.branchSelectionStatus === "awaiting_confirmation") {
    renderPendingBranchResult(data);
    return;
  }
  lastRows = data.rankedFeatures || [];
  lastRecommendedRows = data.recommendedCandidates || [];
  lastGrangerRows = data.grangerTests || [];
  lastImportanceRows = data.importance || [];
  lastModelVariableRows = data.modelVariableImportance || [];
  lastModelDiscoveredRows = data.modelDiscoveredCandidates || [];
  lastEnhancedSummaryRows = data.enhancedValidationSummary || [];
  const hasEnhancedScreening = lastEnhancedSummaryRows.length > 0;
  lastEnhancedLiftRows = hasEnhancedScreening ? (data.modelLiftScores || []) : [];
  lastEnhancedRollingRows = hasEnhancedScreening ? (data.rollingCorrScores || []) : [];
  lastValidationSummaryRows = data.validationSummary || [];
  lastValidationFieldsRows = data.validationFields || [];
  lastVerificationReviewPoolRows = data.verificationReviewPool || [];
  lastConditionalRows = [];
  lastCausalEvidenceRows = [];
  lastEvidenceMatrixRows = data.evidenceMatrix || [];
  evidenceMatrixStatusLabels = data.evidenceMatrixStatusLabels || evidenceMatrixStatusLabels;
  lastFinalReviewSummaryRows = [];
  lastXgbModelSummaryRows = [];
  lastXgbCandidateUpliftRows = [];
  lastXgbCandidateFoldMetricRows = [];
  lastXgbValidationSummary = {};
  closeDetailModal();
  renderOverview(data.overview || {});
  renderAnalysisTimingBreakdown(data.analysis_timings || {});
  renderValidationSummaryTable(lastValidationSummaryRows);
  renderValidationFieldsTable(lastValidationFieldsRows);
  renderVerificationReviewPool(lastVerificationReviewPoolRows);
  renderScreeningQualityHints(lastRows);
  delete tableSortStates["table"];
  renderTable(lastRows);
  delete tableSortStates[recommendedCandidateTable];
  renderRecommendedCandidateTable(lastRecommendedRows);
  delete tableSortStates[controlReferenceTable];
  renderControlReferenceTable(
    lastRows.filter((row) => row.is_control_reference === true)
  );
  delete tableSortStates["overviewTop"];
  renderGenericTable("overviewTop", (data.overview && data.overview.top10) || [], coreCandidateColumns());
  renderGenericTable("grangerTable", lastGrangerRows);
  renderGenericTable("modelVariableImportanceTable", lastModelVariableRows, modelVariableImportanceColumns());
  renderGenericTable("importanceTable", lastImportanceRows);
  renderGenericTable("modelDiscoveredTable", lastModelDiscoveredRows, modelDiscoveredColumns());
  syncModelDiscoveryReviewPoolOptions(lastModelDiscoveredRows);
  renderGenericTable("enhancedSummaryTable", lastEnhancedSummaryRows, enhancedSummaryColumns());
  renderGenericTable("enhancedLiftTable", lastEnhancedLiftRows, modelLiftColumns());
  renderGenericTable("enhancedRollingTable", lastEnhancedRollingRows, rollingCorrColumns());
  renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
  renderFinalReviewQualityOverview(lastFinalReviewSummaryRows);
  renderFinalReviewSummaryTable(lastFinalReviewSummaryRows);
  renderCausalReviewEvidenceTable(lastCausalEvidenceRows);
  renderEvidenceMatrixTable(lastEvidenceMatrixRows);
  renderGenericTable("xgbModelSummaryTable", lastXgbModelSummaryRows, xgbModelSummaryColumns());
  renderXgbCandidateUpliftTable(lastXgbCandidateUpliftRows);
  clearXgbCandidateFoldDetails();
  renderXgbRunSummary(lastXgbValidationSummary);
  renderReviewDownloads(data.downloads || []);
  renderDownloads(data.downloads || []);
  el("runEnhancedScreening").disabled = !currentRunId;
  el("runGranger").disabled = !currentRunId;
  el("runModel").disabled = !currentRunId;
  el("addManualReviewPool").disabled = !currentRunId;
  el("addModelDiscoveryReviewPool").disabled = !currentRunId;
  el("runCausalReview").disabled = !currentRunId;
  updateBranchSelectionUi(data);
  setDownstreamGate(false);
  el("generateLlmReport").disabled = !currentRunId;
  updateXgbRunAvailability();
}

function renderPendingBranchResult(data) {
  lastRows = [];
  lastRecommendedRows = [];
  lastGrangerRows = [];
  lastImportanceRows = [];
  lastModelVariableRows = [];
  lastNearMissRows = [];
  lastModelDiscoveredRows = [];
  lastEnhancedSummaryRows = [];
  lastEnhancedLiftRows = [];
  lastEnhancedRollingRows = [];
  lastValidationSummaryRows = [];
  lastValidationFieldsRows = [];
  lastVerificationReviewPoolRows = [];
  lastConditionalRows = [];
  lastCausalEvidenceRows = [];
  lastEvidenceMatrixRows = [];
  lastFinalReviewSummaryRows = [];
  lastXgbModelSummaryRows = [];
  lastXgbCandidateUpliftRows = [];
  lastXgbCandidateFoldMetricRows = [];
  lastXgbValidationSummary = {};
  clearXgbCandidateFoldDetails();
  closeDetailModal();
  renderOverview({});
  renderAnalysisTimingBreakdown(data.analysis_timings || {});
  renderValidationSummaryTable(lastValidationSummaryRows);
  renderValidationFieldsTable(lastValidationFieldsRows);
  renderVerificationReviewPool(lastVerificationReviewPoolRows);
  renderGenericTable("preprocessingComparisonTable", data.preprocessingComparison || [], preprocessingComparisonColumns());
  resetOptionalTable("evidenceMatrixTable", "未运行 可信度审查，暂无人工复核证据矩阵。");
  clearOptionalElement("evidenceMatrixDownload");
  renderDownloads(data.downloads || []);
  updateBranchSelectionUi(data);
  setDownstreamGate(true);
  el("addManualReviewPool").disabled = true;
  el("addModelDiscoveryReviewPool").disabled = true;
  el("generateLlmReport").disabled = true;
  updateXgbRunAvailability();
}

function processedBranchLabel(mode) {
  const labels = {
    lowpass: "确认使用一阶低通",
    lowpass_detrend: "确认使用低通 + 去趋势",
    lowpass_diff: "确认使用低通 + 差分",
  };
  return labels[String(mode || "")] || "确认使用预处理数据";
}

function updateBranchSelectionUi(data) {
  const section = el("branchSelectionSection");
  const status = data.branchSelectionStatus;
  const activeBranch = data.activeScreeningBranch;
  const locked = Boolean(data.branchLocked);
  const selected = data.selectedPreprocessingMode || "";
  el("confirmProcessedBranch").textContent = processedBranchLabel(selected);
  el("branchLockedHint").hidden = !locked;
  if (!status) {
    section.hidden = true;
    el("confirmRawBranch").disabled = true;
    el("confirmProcessedBranch").disabled = true;
    return;
  }
  section.hidden = false;
  if (status === "awaiting_confirmation") {
    el("branchSelectionStatus").textContent =
      "双分支初筛已完成，正式初筛尚未发布。请查看 Raw vs Processed 对比后明确确认一个分支；确认前不会展示正式排名、Top-K 或推荐变量。";
    el("confirmRawBranch").disabled = locked;
    el("confirmProcessedBranch").disabled = locked;
    return;
  }
  if (status === "not_required") {
    el("branchSelectionStatus").textContent =
      "已选择原始数据：只运行 Raw 并自动发布正式初筛（branch_selection_status = not_required）。";
    el("confirmRawBranch").disabled = true;
    el("confirmProcessedBranch").disabled = true;
    return;
  }
  el("branchSelectionStatus").textContent =
    `已确认正式初筛分支：${activeBranch === "raw" ? "原始数据" : "预处理数据"}` +
    (data.activePreprocessingMode ? `（active preprocessing = ${data.activePreprocessingMode}）` : "") +
    (locked ? "。后续验证已开始，当前初筛分支已锁定；如需切换请重新分析。" : "。");
  el("confirmRawBranch").disabled = activeBranch === "raw" || locked;
  el("confirmProcessedBranch").disabled = activeBranch === "processed" || locked;
}

function setDownstreamGate(blocked) {
  el("downstreamGateHint").hidden = !blocked;
  el("runEnhancedScreening").disabled = blocked || !currentRunId;
  el("runGranger").disabled = blocked || !currentRunId;
  el("runModel").disabled = blocked || !currentRunId;
  el("runCausalReview").disabled = blocked || !currentRunId;
  el("runXgbValidation").disabled = blocked || !currentRunId;
}

async function confirmInitialScreeningBranch(branch) {
  if (!currentRunId) return setStatus("请先完成初筛。");
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在确认正式初筛分支...", startedAt);
  el("confirmRawBranch").disabled = true;
  el("confirmProcessedBranch").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    form.append("branch", branch);
    const data = await postForm("/api/confirm_initial_screening_branch", form);
    renderAnalysisResult(data);
    setStatus(appendElapsed("已确认正式初筛分支。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  } finally {
    stopStatusTimer(timerId);
  }
}

function preprocessingComparisonColumns() {
  return [
    "variable", "processed_mode", "raw_final_score", "processed_final_score",
    "final_score_delta", "raw_rank", "processed_rank", "rank_delta",
    "raw_pearson", "processed_pearson", "raw_spearman", "processed_spearman",
    "raw_best_lag", "processed_best_lag", "lag_direction_changed",
    "raw_in_top_k", "processed_in_top_k", "raw_candidate", "processed_candidate",
    "raw_risk_tags", "processed_risk_tags",
  ];
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
    lastValidationSummaryRows = data.validationSummary || lastValidationSummaryRows;
    lastValidationFieldsRows = data.validationFields || lastValidationFieldsRows;
  lastVerificationReviewPoolRows = data.verificationReviewPool || lastVerificationReviewPoolRows;
    renderValidationSummaryTable(lastValidationSummaryRows);
    renderValidationFieldsTable(lastValidationFieldsRows);
    renderGenericTable("enhancedSummaryTable", lastEnhancedSummaryRows, enhancedSummaryColumns());
    renderGenericTable("enhancedLiftTable", lastEnhancedLiftRows, modelLiftColumns());
    renderGenericTable("enhancedRollingTable", lastEnhancedRollingRows, rollingCorrColumns());
    renderDownloads(data.downloads || []);
    updateBranchSelectionUi(data);
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
    lastValidationSummaryRows = data.validationSummary || lastValidationSummaryRows;
    lastValidationFieldsRows = data.validationFields || lastValidationFieldsRows;
  lastVerificationReviewPoolRows = data.verificationReviewPool || lastVerificationReviewPoolRows;
    renderValidationSummaryTable(lastValidationSummaryRows);
    renderValidationFieldsTable(lastValidationFieldsRows);
    renderGenericTable("grangerTable", lastGrangerRows);
    renderDownloads(data.downloads || []);
    updateBranchSelectionUi(data);
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
    lastValidationSummaryRows = data.validationSummary || lastValidationSummaryRows;
    lastValidationFieldsRows = data.validationFields || lastValidationFieldsRows;
  lastVerificationReviewPoolRows = data.verificationReviewPool || lastVerificationReviewPoolRows;
    renderValidationSummaryTable(lastValidationSummaryRows);
    renderValidationFieldsTable(lastValidationFieldsRows);
    renderGenericTable("modelVariableImportanceTable", lastModelVariableRows, modelVariableImportanceColumns());
    renderGenericTable("importanceTable", lastImportanceRows);
    renderGenericTable("modelDiscoveredTable", lastModelDiscoveredRows, modelDiscoveredColumns());
    syncModelDiscoveryReviewPoolOptions(lastModelDiscoveredRows);
    renderDownloads(data.downloads || []);
    updateBranchSelectionUi(data);
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
  const timerId = startStatusTimer("正在运行可信度审查：解释预测价值的独立性及其限制，不是因果结论...", startedAt);
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
    lastEvidenceMatrixRows = data.evidenceMatrix || [];
    evidenceMatrixStatusLabels = data.evidenceMatrixStatusLabels || evidenceMatrixStatusLabels;
    lastFinalReviewSummaryRows = data.finalReviewSummary || [];
    lastValidationSummaryRows = data.validationSummary || lastValidationSummaryRows;
    lastValidationFieldsRows = data.validationFields || lastValidationFieldsRows;
  lastVerificationReviewPoolRows = data.verificationReviewPool || lastVerificationReviewPoolRows;
    tableSortStates["finalReviewSummaryTable"] = { column: "final_rank", direction: "asc" };
    renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
    renderValidationSummaryTable(lastValidationSummaryRows);
    renderValidationFieldsTable(lastValidationFieldsRows);
    renderFinalReviewQualityOverview(lastFinalReviewSummaryRows);
    renderFinalReviewSummaryTable(lastFinalReviewSummaryRows);
    renderCausalReviewEvidenceTable(lastCausalEvidenceRows);
    renderEvidenceMatrixTable(lastEvidenceMatrixRows);
    renderReviewDownloads(data.downloads || []);
    renderDownloads(data.downloads || []);
    updateBranchSelectionUi(data);
    updateXgbRunAvailability();
    setStatus(appendElapsed(data.message || "可信度审查完成。结果不是因果结论，也不改变初筛结果。", startedAt), "success");
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
  if (!enabled) el("xgbStatus").textContent = "XGB 时间外预测验证未启用。";
}

async function runXgbValidation() {
  if (!el("enableXgbValidation").checked) {
    el("xgbStatus").textContent = "请先启用 XGB 时间外预测验证。";
    return;
  }
  if (!currentRunId || !lastFinalReviewSummaryRows.length) {
    el("xgbStatus").textContent = "请先完成可信度审查。";
    return;
  }
  const startedAt = performance.now();
  const timerId = startStatusTimer("正在运行 XGB 时间外预测验证...", startedAt);
  el("runXgbValidation").disabled = true;
  el("xgbStatus").textContent = "正在运行 XGB 时间外预测验证...";
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
    lastXgbCandidateFoldMetricRows = data.xgbCandidateFoldMetrics || [];
    lastXgbValidationSummary = data.xgbValidationSummary || {};
    renderGenericTable("xgbModelSummaryTable", lastXgbModelSummaryRows, xgbModelSummaryColumns());
    renderXgbCandidateUpliftTable(lastXgbCandidateUpliftRows);
    clearXgbCandidateFoldDetails();
    renderXgbRunSummary(lastXgbValidationSummary);
    renderXgbDownloads(data.status === "success" ? (data.downloads || []) : []);
    renderDownloads(data.downloads || []);
    updateBranchSelectionUi(data);
    const message = data.error_message || data.message || "XGB 时间外预测验证失败。";
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
  { category: "参数设置说明", name: "预处理模式", signal: "原始 / 一阶低通 / 一阶低通+去趋势 / 一阶低通+差分。", reading: "选择 Raw 只运行 Raw 并直接发布正式初筛；选择任一预处理模式会同时运行 Raw 和该预处理模式两个独立初筛，完成后需要人工确认正式分支，系统不会自动选择“更优”模式。", action: "根据问题选择模式，确认分支前先查看 Raw vs Processed 对比。" },
  { category: "参数设置说明", name: "低通时间常数 τ（分钟）", signal: "一阶低通平滑的时间常数。", reading: "只在 lowpass / lowpass_detrend / lowpass_diff 下生效；Raw 不参与实际分析。", action: "结合工艺响应速度设置，默认 5.0 分钟。" },
  { category: "参数设置说明", name: "差分间隔（分钟）", signal: "lowpass_diff 使用的差分间隔；留空表示一个分析采样周期。", reading: "只影响 lowpass_diff 模式，非空时必须大于 0。", action: "需要多个采样周期差分时填写，否则留空。" },
  { category: "参数设置说明", name: "去趋势窗口点数", signal: "滑动去趋势时使用的窗口长度。", reading: "只在 lowpass_detrend 下有实际意义；窗口决定慢趋势被剔除的尺度，过短可能去掉真实响应，过长可能保留漂移。", action: "按班次、停留时间或主要扰动周期设置，并检查趋势图。" },
  { category: "参数设置说明", name: "负荷代表列", signal: "代表装置负荷或产量的变量。", reading: "用于识别共同负荷驱动和工况稳定性，影响风险标签解释。", action: "优先选择现场认可的负荷、进料或产量指标。" },
  { category: "参数设置说明", name: "工况分段", signal: "按低/中/高负荷或自定义阈值拆分工况。", reading: "用于判断候选关系是否跨工况稳定，影响复核优先级。", action: "先确认分段边界有工程含义，避免样本过少。" },
  { category: "参数设置说明", name: "自定义下限 / 自定义上限", signal: "工况分段或过滤时使用的自定义上下限。", reading: "会限定参与对比的运行区间，影响稳定性和风险标签判断。", action: "用装置负荷区间、牌号或操作窗口确定上下限。" },
  { category: "参数设置说明", name: "残差控制列", signal: "在残差相关或条件验证中需要控制的变量。", reading: "用于减弱共同负荷、已知干扰或强共线变量的影响。", action: "填入负荷、设定值、关键上游扰动等已知控制因素。" },
  { category: "参数设置说明", name: "强制复核变量", signal: "即使未进入主排序前列也要纳入复核的变量。", reading: "扩展复核范围，适合业务重点变量或专家指定变量。", action: "只加入有明确工艺理由的点位，避免复核清单过长。" },
  { category: "参数设置说明", name: "可信度审查候选数量", signal: "进入第三层可信度审查的候选变量数量。", reading: "数量越大覆盖越广但计算和人工解释成本越高。", action: "先用默认数量快速定位，再按需要扩大范围。" },
  { category: "参数设置说明", name: "风险标签包含过滤", signal: "按风险标签文本筛选复核或推荐结果。", reading: "只改变页面查看和复核聚焦范围，不表示未显示变量没有风险。", action: "用于定位共同负荷、数据质量等特定问题，留空表示不过滤。" },
  { category: "风险标签说明", name: "滞后边界风险", signal: "最佳滞后贴近扫描窗口边界，峰值可能尚未完全覆盖。", reading: "当前最大滞后点数可能偏小，真实响应时间可能更长。", action: "扩大最大滞后点数，结合趋势图确认峰值是否继续外移。" },
  { category: "风险标签说明", name: "变量滞后目标风险", signal: "页面显示为变量滞后目标。", reading: "变量变化晚于目标，更像响应量或受同一扰动影响。", action: "优先检查工艺方向，通常不直接作为前馈变量。" },
  { category: "风险标签说明", name: "公式泄漏 / 计算耦合风险", signal: "候选变量可能由目标或其上下游计算项派生。", reading: "高相关可能来自公式、软测量或报表口径耦合。", action: "核对 DCS/ historian 点位定义，剔除直接计算关系后再复核。" },
  { category: "风险标签说明", name: "数据质量风险", signal: "数据质量问题通过 data_quality_score 连续降低基础分；严重质量问题保留风险标记，但不再额外封顶。", reading: "统计指标可能受采样、坏点或仪表状态驱动。", action: "先清洗数据、确认仪表有效性，再重新运行分析。" },
  { category: "风险标签说明", name: "共线性风险", signal: "多个候选变量高度同步或代表同一工艺负荷。", reading: "模型可能难以区分真正贡献变量，单变量解释不稳定。", action: "做变量分组、残差控制或条件 Granger 预测验证。" },
  { category: "证据等级与复核建议", name: "强预测证据", signal: "相关、模型提升、预测贡献、稳定性等多类证据同时较好。", reading: "该变量对预测目标有较稳定信息量，但仍不是因果结论。", action: "进入优先复核，结合机理、趋势和现场操作记录确认。" },
  { category: "证据等级与复核建议", name: "风险受限证据", signal: "统计证据较强，但伴随共线性、共同负荷或数据质量等限制。", reading: "变量可能重要，但证据解释需要更谨慎。", action: "保留观察，先排除风险来源，再决定是否用于工程复核。" },
  { category: "证据等级与复核建议", name: "优先复核", signal: "可信度审查中独立预测支持较强且限制较少。", reading: "值得投入工程时间检查变量定义、方向和可操作性。", action: "查看趋势，核对滞后方向，并与班组/工艺专家确认。" },
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

function searchableSelect(select, values, allowEmpty = false, emptyLabel = "不分段") {
  select._searchableOptions = { values: [...values], allowEmpty, emptyLabel };
  let dropdown = document.querySelector(`[data-select-dropdown-for="${select.id}"]`);
  let filter = document.querySelector(`[data-select-filter-for="${select.id}"]`);
  if (!dropdown) {
    dropdown = document.createElement("details");
    dropdown.className = "single-dropdown";
    dropdown.dataset.selectDropdownFor = select.id;
    const summary = document.createElement("summary");
    summary.dataset.selectSummaryFor = select.id;
    filter = document.createElement("input");
    filter.type = "search";
    filter.className = "select-filter";
    filter.placeholder = "筛选位号（可选）";
    filter.setAttribute("aria-label", "筛选位号（可选）");
    filter.dataset.selectFilterFor = select.id;
    filter.addEventListener("input", () => renderSearchableSelect(select));
    const empty = document.createElement("div");
    empty.className = "select-filter-empty";
    empty.dataset.selectFilterEmptyFor = select.id;
    empty.textContent = "未找到匹配的位号";
    empty.hidden = true;
    const options = document.createElement("div");
    options.className = "single-options";
    options.dataset.selectOptionsFor = select.id;
    select.insertAdjacentElement("beforebegin", dropdown);
    select.classList.add("variable-select-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    select.addEventListener("change", () => syncSearchableSelect(select));
    dropdown.append(summary, filter, empty, options, select);
  }
  renderSearchableSelect(select);
}

function renderSearchableSelect(select) {
  const { values, allowEmpty, emptyLabel } = select._searchableOptions;
  const currentValue = select.value;
  const filter = document.querySelector(`[data-select-filter-for="${select.id}"]`);
  const query = (filter?.value || "").trim().toLowerCase();
  const filtered = query ? values.filter((value) => String(value).toLowerCase().includes(query)) : values;
  select.innerHTML = "";
  if (allowEmpty) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel;
    select.appendChild(option);
  }
  if (!values.length) {
    const option = document.createElement("option");
    option.disabled = true;
    option.textContent = "未找到匹配的位号";
    select.appendChild(option);
  }
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
  if (currentValue && !values.includes(currentValue)) {
    const option = document.createElement("option");
    option.value = currentValue;
    option.textContent = currentValue;
    option.hidden = true;
    select.appendChild(option);
  }
  if (currentValue || allowEmpty) select.value = currentValue;
  const options = document.querySelector(`[data-select-options-for="${select.id}"]`);
  const empty = document.querySelector(`[data-select-filter-empty-for="${select.id}"]`);
  options.innerHTML = "";
  const visibleValues = allowEmpty ? ["", ...filtered] : filtered;
  for (const value of visibleValues) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "single-option";
    option.dataset.value = value;
    option.textContent = value || emptyLabel;
    option.setAttribute("aria-selected", value === select.value ? "true" : "false");
    option.addEventListener("click", () => {
      select.value = value;
      filter.value = "";
      renderSearchableSelect(select);
      document.querySelector(`[data-select-dropdown-for="${select.id}"]`).open = false;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    options.appendChild(option);
  }
  empty.hidden = filtered.length > 0;
  syncSearchableSelect(select);
}

function syncSearchableSelect(select) {
  const summary = document.querySelector(`[data-select-summary-for="${select.id}"]`);
  if (!summary) return;
  summary.textContent = select.selectedOptions[0]?.textContent || "请选择";
  for (const option of document.querySelectorAll(`[data-select-options-for="${select.id}"] .single-option`)) {
    option.setAttribute("aria-selected", option.dataset.value === select.value ? "true" : "false");
  }
}

function fillSelect(select, values, allowEmpty = false, emptyLabel = "不分段") {
  searchableSelect(select, values, allowEmpty, emptyLabel);
}

function searchableMultiOptions(box) {
  let filter = document.querySelector(`[data-multi-filter-for="${box.id}"]`);
  let empty = document.querySelector(`[data-multi-filter-empty-for="${box.id}"]`);
  if (!filter) {
    filter = document.createElement("input");
    filter.type = "search";
    filter.className = "multi-filter";
    filter.placeholder = "筛选位号（可选）";
    filter.setAttribute("aria-label", "筛选位号（可选）");
    filter.dataset.multiFilterFor = box.id;
    filter.addEventListener("input", () => filterMultiOptions(box));
    box.insertAdjacentElement("beforebegin", filter);
    empty = document.createElement("div");
    empty.className = "select-filter-empty";
    empty.dataset.multiFilterEmptyFor = box.id;
    empty.textContent = "未找到匹配的位号";
    empty.hidden = true;
    box.insertAdjacentElement("beforebegin", empty);
  }
  filterMultiOptions(box);
}

function filterMultiOptions(box) {
  const query = (document.querySelector(`[data-multi-filter-for="${box.id}"]`)?.value || "").trim().toLowerCase();
  const rows = Array.from(box.querySelectorAll("label"));
  const visible = rows.filter((row) => {
    const matches = !query || row.textContent.toLowerCase().includes(query);
    row.hidden = !matches;
    return matches;
  });
  const empty = document.querySelector(`[data-multi-filter-empty-for="${box.id}"]`);
  if (empty) empty.hidden = visible.length > 0;
}

function clearVariableFilters() {
  for (const filter of document.querySelectorAll("[data-select-filter-for], [data-multi-filter-for]")) {
    filter.value = "";
  }
  for (const dropdown of document.querySelectorAll(".single-dropdown, .multi-dropdown")) {
    dropdown.open = false;
  }
}

function markTrendTimeRangeManual() {
  trendTimeRangeMode = "manual";
  trendAutoWindowActive = false;
  trendSelection = null;
  updateTrendSelectionInfo();
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
  const activeMode = currentAnalysisContext.preprocess_mode;
  const hasActiveContext = typeof activeMode === "string" && activeMode.length > 0;
  const activeTau = currentAnalysisContext.lowpass_tau_minutes ?? el("lowpassTauMinutes").value;
  const activeDetrendWindow = currentAnalysisContext.detrend_window ?? el("detrendWindow").value;
  params.set("file_id", fileId);
  params.set("encoding", "auto");
  params.set("time_column", el("timeColumn").value);
  params.set("trend_start", el("trendStart").value);
  params.set("trend_end", el("trendEnd").value);
  params.set("trend_max_points", el("trendMaxPoints").value || "10000");
  params.set("segment_column", el("segmentColumn").value);
  params.set("segment_mode", el("segmentMode").value);
  params.set("segment_min", el("segmentMin").value);
  params.set("segment_max", el("segmentMax").value);
  params.set("preprocess_mode", hasActiveContext ? activeMode : el("preprocessMode").value);
  params.set("lowpass_tau_minutes", hasActiveContext ? activeTau : el("lowpassTauMinutes").value);
  params.set("diff_interval_minutes", hasActiveContext ? (currentAnalysisContext.diff_interval_minutes ?? "") : el("diffIntervalMinutes").value.trim());
  params.set("detrend_window", hasActiveContext ? activeDetrendWindow : el("detrendWindow").value);
  params.set("excluded_columns", getExcludedColumnSelection().join(","));
}

function clearTrendSelection() {
  trendSelection = null;
  trendTimeRangeMode = "manual";
  trendAutoWindowActive = false;
  if (trendDefaultStart) el("trendStart").value = trendDefaultStart;
  if (trendDefaultEnd) el("trendEnd").value = trendDefaultEnd;
  el("trendMaxPoints").value = "10000";
  updateTrendSelectionInfo();
  if (fileId && lastTrendSeries.length) void drawTrend();
}

function timestampMilliseconds(value) {
  const milliseconds = Date.parse(value);
  return Number.isFinite(milliseconds) ? milliseconds : null;
}

function datetimeLocalValue(milliseconds) {
  const local = new Date(milliseconds - new Date(milliseconds).getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function formatTrendTimestamp(milliseconds) {
  return datetimeLocalValue(milliseconds).replace("T", " ");
}

function formatTrendDuration(milliseconds) {
  const totalMinutes = Math.max(0, Math.round(milliseconds / 60000));
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  const parts = [];
  if (days) parts.push(`${days}天`);
  if (hours) parts.push(`${hours}小时`);
  if (minutes || !parts.length) parts.push(`${minutes}分钟`);
  return parts.join("");
}

function updateTrendSelectionInfo() {
  const button = el("clearTrendSelection");
  const addButton = el("addExcludeWindow");
  const info = el("trendSelectionInfo");
  if (!button || !addButton || !info) return;
  button.disabled = !trendSelection;
  addButton.disabled = !trendSelection;
  if (!trendSelection) {
    info.textContent = "可在趋势绘图区横向拖动选择时间窗口。";
    return;
  }
  info.textContent = `已选择时间窗口：${formatTrendTimestamp(trendSelection.start)} ～ ${formatTrendTimestamp(trendSelection.end)}（${formatTrendDuration(trendSelection.end - trendSelection.start)}）`;
}

function setTrendWindowFromSelection(start, end) {
  const earlier = Math.min(start, end);
  const later = Math.max(start, end);
  trendSelection = { start: earlier, end: later };
  trendTimeRangeMode = "manual";
  trendAutoWindowActive = false;
  el("trendStart").value = datetimeLocalValue(earlier);
  el("trendEnd").value = datetimeLocalValue(later);
  updateTrendSelectionInfo();
}

function updateExcludeWindowState(windows, stats) {
  excludeWindows = Array.isArray(windows) ? windows : [];
  excludeWindowStats = stats || { exclude_window_count: 0, excluded_rows: 0, excluded_ratio: 0 };
  renderExcludeWindows();
}

function renderExcludeWindows() {
  const list = el("excludeWindowList");
  const stats = el("excludeWindowStats");
  const restoreAll = el("restoreAllExcludeWindows");
  if (!list || !stats || !restoreAll) return;
  const count = Number(excludeWindowStats.exclude_window_count || 0);
  const rows = Number(excludeWindowStats.excluded_rows || 0);
  const ratio = Number(excludeWindowStats.excluded_ratio || 0);
  stats.textContent = `已标记：${count} 个窗口 / ${rows.toLocaleString("zh-CN")} 点（${(ratio * 100).toFixed(1)}%）`;
  restoreAll.disabled = !excludeWindows.length;
  if (!excludeWindows.length) {
    list.className = "exclude-window-list empty";
    list.textContent = "尚未添加排除窗口。";
    return;
  }
  list.className = "exclude-window-list";
  list.innerHTML = excludeWindows.map((window, index) =>
    `<div class="exclude-window-item"><span>${escapeHtml(String(window.start))} ～ ${escapeHtml(String(window.end))}</span><button type="button" class="secondary" data-exclude-window-index="${index}">恢复</button></div>`
  ).join("");
  list.querySelectorAll("[data-exclude-window-index]").forEach((button) => {
    button.addEventListener("click", () => restoreExcludeWindow(Number(button.dataset.excludeWindowIndex)));
  });
}

function trendWindowTimestamp(milliseconds) {
  const date = new Date(milliseconds - new Date(milliseconds).getTimezoneOffset() * 60000);
  return date.toISOString().slice(0, 19);
}

async function addExcludeWindow() {
  if (!trendSelection || !fileId) return;
  try {
    const form = new FormData();
    form.append("file_id", fileId);
    form.append("time_column", el("timeColumn").value);
    form.append("encoding", "auto");
    form.append("start", trendWindowTimestamp(trendSelection.start));
    form.append("end", trendWindowTimestamp(trendSelection.end));
    const data = await postForm("/api/exclude_window", form);
    updateExcludeWindowState(data.excludeWindows, data.excludeWindowStats);
    trendSelection = null;
    updateTrendSelectionInfo();
    if (lastTrendSeries.length) renderTrendChart(lastTrendSeries, lastTrendAxisMode);
    setStatus("已加入排除窗口；将在下一次分析时生效。", "success");
  } catch (error) {
    setStatus(error.message || String(error), "error");
  }
}

async function restoreExcludeWindow(index) {
  try {
    const form = new FormData();
    form.append("file_id", fileId);
    form.append("time_column", el("timeColumn").value);
    form.append("index", String(index));
    const data = await postForm("/api/restore_exclude_window", form);
    updateExcludeWindowState(data.excludeWindows, data.excludeWindowStats);
    if (lastTrendSeries.length) renderTrendChart(lastTrendSeries, lastTrendAxisMode);
  } catch (error) {
    setStatus(error.message || String(error), "error");
  }
}

async function restoreAllExcludeWindows() {
  if (!excludeWindows.length || !window.confirm("确定恢复所有数据吗？\n这将清除当前全部排除窗口。")) return;
  try {
    const form = new FormData();
    form.append("file_id", fileId);
    form.append("time_column", el("timeColumn").value);
    const data = await postForm("/api/restore_all_exclude_windows", form);
    updateExcludeWindowState(data.excludeWindows, data.excludeWindowStats);
    if (lastTrendSeries.length) renderTrendChart(lastTrendSeries, lastTrendAxisMode);
  } catch (error) {
    setStatus(error.message || String(error), "error");
  }
}

async function drawTrend() {
  try {
    const variables = Array.from(new Set([el("trendVar1").value, el("trendVar2").value, el("trendVar3").value, el("trendVar4").value, el("trendVar5").value, el("trendVar6").value, el("trendVar7").value, el("trendVar8").value].filter(Boolean)));
    if (!variables.length) return setStatus("请至少选择一个趋势变量。");
    const params = new URLSearchParams();
    appendChartQueryParams(params);
    params.set("time_range_mode", trendAutoWindowActive ? "auto" : "manual");
    params.set("variables", variables.join(","));
    setStatus("正在生成趋势图...", "loading");
    const response = await fetch(`/api/trend?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "趋势图生成失败");
    updateExcludeWindowState(data.excludeWindows, data.excludeWindowStats);
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
  const sharedRange = trendSharedRange(series);
  const ranges = series.map((item) => axisMode === "shared" ? sharedRange : valueRange(item.points));
  let timeStart = Infinity;
  let timeEnd = -Infinity;
  for (const item of series) {
    for (const point of (item.points || [])) {
      const pointTime = timestampMilliseconds(point.x);
      if (pointTime === null) continue;
      timeStart = Math.min(timeStart, pointTime);
      timeEnd = Math.max(timeEnd, pointTime);
    }
  }
  if (!Number.isFinite(timeStart) || !Number.isFinite(timeEnd)) {
    lastTrendSeries = [];
    container.className = "chart empty";
    container.textContent = "趋势数据缺少可解析的时间戳。";
    el("trendLegend").innerHTML = "";
    clearTrendStats();
    return;
  }
  const plotWidth = width - pad.left - pad.right;
  const timeToX = (milliseconds) => pad.left + (milliseconds - timeStart) / Math.max(1, timeEnd - timeStart) * plotWidth;
  const xToTime = (position) => timeStart + (position - pad.left) / Math.max(1, plotWidth) * (timeEnd - timeStart);
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
    const points = (item.points || []).map((point) => {
      const value = trendFiniteValue(point);
      const pointTime = timestampMilliseconds(point.x);
      return Number.isFinite(value) && pointTime !== null
        ? `${timeToX(pointTime).toFixed(2)},${y(value, ranges[idx]).toFixed(2)}`
        : null;
    }).filter(Boolean).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${trendColors[idx % trendColors.length]}" stroke-width="2.2"/>`;
  }).join("");
  const excludeWindowMarkup = excludeWindows.map((window) => {
    const start = timestampMilliseconds(window.start);
    const end = timestampMilliseconds(window.end);
    if (start === null || end === null) return "";
    const leftTime = Math.max(timeStart, Math.min(start, end));
    const rightTime = Math.min(timeEnd, Math.max(start, end));
    if (leftTime > rightTime) return "";
    return `<rect data-exclude-window x="${timeToX(leftTime)}" y="${pad.top}" width="${Math.max(1, timeToX(rightTime) - timeToX(leftTime))}" height="${height - pad.top - pad.bottom}" fill="#b45309" fill-opacity=".16" pointer-events="none"/>`;
  }).join("");
  const currentSelection = trendSelection && trendSelection.start >= timeStart && trendSelection.end <= timeEnd
    ? trendSelection
    : null;
  const selectionMarkup = currentSelection
    ? `<g data-trend-selection pointer-events="none"><rect x="${timeToX(currentSelection.start)}" y="${pad.top}" width="${Math.max(0, timeToX(currentSelection.end) - timeToX(currentSelection.start))}" height="${height - pad.top - pad.bottom}" fill="#176b87" fill-opacity=".18"/><line data-trend-selection-edge="start" x1="${timeToX(currentSelection.start)}" x2="${timeToX(currentSelection.start)}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="#176b87" stroke-width="1.5"/><line data-trend-selection-edge="end" x1="${timeToX(currentSelection.end)}" x2="${timeToX(currentSelection.end)}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="#176b87" stroke-width="1.5"/></g>`
    : '<g data-trend-selection pointer-events="none" visibility="hidden"><rect y="0" height="0" fill="#176b87" fill-opacity=".18"/><line data-trend-selection-edge="start"/><line data-trend-selection-edge="end"/></g>';
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
    ${excludeWindowMarkup}
    ${paths}
    ${selectionMarkup}
    <rect id="trendSelectionHitbox" x="${pad.left}" y="${pad.top}" width="${plotWidth}" height="${height - pad.top - pad.bottom}" fill="transparent" style="cursor:crosshair;touch-action:none"/>
    <text x="${pad.left}" y="${height - 10}" font-size="10" fill="#5f6b7a">${escapeHtml(formatTrendTimestamp(timeStart))}</text>
    <text x="${width - pad.right}" y="${height - 10}" text-anchor="end" font-size="10" fill="#5f6b7a">${escapeHtml(formatTrendTimestamp(timeEnd))}</text>
  </svg>`;
  el("trendLegend").innerHTML = series.map((item, idx) =>
    `<span><i class="swatch" style="background:${trendColors[idx % trendColors.length]}"></i>${escapeHtml(item.name)}</span>`
  ).join("");
  renderTrendStats(series);
  updateTrendSelectionInfo();
  if (typeof container.querySelector !== "function") return;
  const svg = container.querySelector("svg");
  const hitbox = container.querySelector("#trendSelectionHitbox");
  const selectionGroup = svg?.querySelector("[data-trend-selection]");
  const selectionArea = selectionGroup?.querySelector("rect");
  const selectionEdges = selectionGroup?.querySelectorAll("[data-trend-selection-edge]");
  if (!svg || !hitbox || !selectionGroup || !selectionArea || !selectionEdges?.length) return;
  const positionFromEvent = (event) => {
    const bounds = svg.getBoundingClientRect();
    const position = (event.clientX - bounds.left) / Math.max(1, bounds.width) * width;
    return Math.min(width - pad.right, Math.max(pad.left, position));
  };
  const drawSelection = (start, end) => {
    const left = Math.min(timeToX(start), timeToX(end));
    const right = Math.max(timeToX(start), timeToX(end));
    selectionGroup.removeAttribute("visibility");
    selectionArea.setAttribute("x", left);
    selectionArea.setAttribute("y", pad.top);
    selectionArea.setAttribute("width", right - left);
    selectionArea.setAttribute("height", height - pad.top - pad.bottom);
    selectionEdges[0].setAttribute("x1", left);
    selectionEdges[0].setAttribute("x2", left);
    selectionEdges[0].setAttribute("y1", pad.top);
    selectionEdges[0].setAttribute("y2", height - pad.bottom);
    selectionEdges[1].setAttribute("x1", right);
    selectionEdges[1].setAttribute("x2", right);
    selectionEdges[1].setAttribute("y1", pad.top);
    selectionEdges[1].setAttribute("y2", height - pad.bottom);
  };
  const restoreSelection = () => {
    if (trendSelection && trendSelection.start >= timeStart && trendSelection.end <= timeEnd) {
      drawSelection(trendSelection.start, trendSelection.end);
    } else {
      selectionGroup.setAttribute("visibility", "hidden");
    }
  };
  let dragStart = null;
  hitbox.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    dragStart = positionFromEvent(event);
    hitbox.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });
  hitbox.addEventListener("pointermove", (event) => {
    if (dragStart === null) return;
    drawSelection(xToTime(dragStart), xToTime(positionFromEvent(event)));
  });
  hitbox.addEventListener("pointerup", (event) => {
    if (dragStart === null) return;
    const dragEnd = positionFromEvent(event);
    const start = dragStart;
    dragStart = null;
    if (Math.abs(dragEnd - start) < 3) return restoreSelection();
    setTrendWindowFromSelection(xToTime(start), xToTime(dragEnd));
    drawSelection(trendSelection.start, trendSelection.end);
  });
  hitbox.addEventListener("pointercancel", () => { dragStart = null; restoreSelection(); });
}

function trendChartWidth(container) {
  const measured = Math.floor(container.getBoundingClientRect().width || container.clientWidth || 0);
  return Math.max(320, measured || 960);
}

function trendRangeFromValues(count, min, max) {
  if (!count) return { min: 0, max: 1 };
  if (min === max) { min -= 1; max += 1; }
  const margin = (max - min) * 0.08;
  return { min: min - margin, max: max + margin };
}

function trendFiniteValue(point) {
  if (point === null || point === undefined) return NaN;
  const raw = point.y;
  if (raw === null || raw === undefined) return NaN;
  return Number(raw);
}

function trendSharedRange(series) {
  let count = 0, min = Infinity, max = -Infinity;
  for (const item of (series || [])) {
    for (const point of (item.points || [])) {
      const value = trendFiniteValue(point);
      if (!Number.isFinite(value)) continue;
      count += 1;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  return trendRangeFromValues(count, min, max);
}

function valueRange(points) {
  let count = 0, min = Infinity, max = -Infinity;
  for (const point of (points || [])) {
    const value = trendFiniteValue(point);
    if (!Number.isFinite(value)) continue;
    count += 1;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return trendRangeFromValues(count, min, max);
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

function trendNumericSummary(points) {
  const values = [];
  let count = 0, sum = 0, sumSquares = 0, min = Infinity, max = -Infinity;
  for (const point of (points || [])) {
    const value = trendFiniteValue(point);
    if (!Number.isFinite(value)) continue;
    values.push(value);
    count += 1;
    sum += value;
    sumSquares += value * value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return { values, count, sum, sumSquares, min, max };
}

function trendHistogram(points, requestedBinCount = 12) {
  const { values, count, sum, min, max } = trendNumericSummary(points);
  if (!count) return { bins: [], min: NaN, max: NaN, mean: NaN, stddev: NaN, count: 0 };
  const mean = sum / count;
  const stddev = Math.sqrt(values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / count);
  if (min === max) return { bins: [{ min, max, count }], min, max, mean, stddev, count };
  const binCount = Math.min(requestedBinCount, Math.max(1, Math.ceil(Math.sqrt(count))));
  const binWidth = (max - min) / binCount;
  const counts = Array(binCount).fill(0);
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.floor(((value - min) / (max - min)) * binCount));
    counts[index] += 1;
  }
  const bins = counts.map((binCountValue, index) => ({
    min: min + index * binWidth,
    max: index === binCount - 1 ? max : min + (index + 1) * binWidth,
    count: binCountValue,
  }));
  return { bins, min, max, mean, stddev, count };
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
  let maxCount = 0;
  for (const bin of histogram.bins) maxCount = Math.max(maxCount, bin.count);
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
  const { values, count, sum, min, max } = trendNumericSummary(points);
  if (!count) return { mean: NaN, stddev: NaN, max: NaN, min: NaN, range: NaN, median: NaN, count: 0, ratio: 0 };
  const mean = sum / count;
  const variance = values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / count;
  return {
    mean,
    stddev: Math.sqrt(variance),
    max,
    min,
    range: max - min,
    median: median(values),
    count,
    ratio: total ? count / total : 0,
  };
}

function median(values) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const candidateTable = "table";
const recommendedCandidateTable = "recommendedCandidateTable";
const controlReferenceTable = "controlReferenceTable";
const candidateCoreColumns = coreCandidateColumns;
const INITIAL_SCREENING_DETAIL_COLUMNS = [
  "dominant_corr", "lag_quality", "lag_quality_status", "lag_boundary_flag", "n",
  "data_quality_score", "risk_level", "human_reason", "recommended_action", "force_included",
  "is_residual_control", "is_capacity_reference", "is_segment_reference",
  "is_auto_control_reference", "is_control_reference", "control_reference_type", "control_reference_source",
  "innovation_score", "innovation_lag", "innovation_direction", "innovation_sign", "innovation_status",
  "association_score", "near_peak_lag_min", "near_peak_lag_max", "near_peak_lag_count",
  "temporal_direction_status", "temporal_penalty_rate", "temporal_score_cap",
  "pearson_p", "spearman_p", "pearson_q", "spearman_q", "corr_q_value", "pearson_r2", "spearman_r2",
];

function candidateDetailColumns(row) {
  const core = new Set(candidateCoreColumns());
  return INITIAL_SCREENING_DETAIL_COLUMNS.filter((column) => !core.has(column) && Object.prototype.hasOwnProperty.call(row || {}, column));
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
    aliases: ["poor_data_quality", "severe_data_quality", "poor_quality_variable"],
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

function renderRecommendedCandidateTable(rows) {
  renderCompactDetailTable({
    targetId: recommendedCandidateTable,
    rows,
    coreColumns: recommendedCandidateColumns(),
    detailColumns: candidateDetailColumns,
    emptyText: "没有可展示的重点候选。",
    modalTitle: (row) => `候选详情：${displayCellValue("variable", row.variable)}`,
  });
}

function renderControlReferenceTable(rows) {
  renderCompactDetailTable({
    targetId: controlReferenceTable,
    rows,
    coreColumns: controlReferenceColumns(),
    detailColumns: candidateDetailColumns,
    emptyText: "当前未识别到控制或负荷参考变量。",
    modalTitle: (row) => `参考变量详情：${displayCellValue("variable", row.variable)}`,
  });
}

function coreCandidateColumns() {
  return ["variable", "candidate_source", "variable_role", "final_score", "pearson", "spearman", "method", "correlation_direction", "lag", "direction", "lag_quality", "data_quality_score", "risk_flags", "risk_level", "recommended_use"];
}

function recommendedCandidateColumns() {
  return ["variable", "candidate_priority_rank", "candidate_source", "load_adjusted_relation_status", "candidate_priority_score", "residual_signal_score", "residual_evidence_status", "common_capacity_candidate_flag", "final_score"];
}

function controlReferenceColumns() {
  return ["driver_rank", "variable", "control_reference_type", "control_reference_source", "final_score", "temporal_direction_status", "data_quality_score", "risk_flags"];
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
  const preserveInputOrder = targetId === candidateTable || targetId === recommendedCandidateTable || targetId === controlReferenceTable || targetId === "overviewTop" || targetId === "validationSummaryTable" || targetId === "modelDiscoveredTable" || targetId === "evidenceMatrixTable";
  ensureTableSortState(targetId, preserveInputOrder ? null : columns[0]);
  const displayRows = sortedRowsForTable(targetId, rows);
  const table = document.createElement("table");
  table.className = "compact-result-table";
  table.setAttribute("aria-label", "核心列");
  table.innerHTML = `<thead><tr>${columns.map((c) => sortableHeaderHtml(targetId, c)).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  displayRows.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.dataset.rowIndex = String(index);
    tr.className = "clickable-row";
    for (const column of columns) {
      const value = getValue(row, column);
      const td = document.createElement("td");
      td.className = tableCellClass(column, value);
      td.innerHTML = formatter ? formatter(column, value, row) : renderTableCell(column, value);
      tr.appendChild(td);
    }
    tr.addEventListener("click", (event) => {
      if (!shouldOpenRowDetail(event)) return;
      selectCompactDetailRow(table, tr, row, detailColumns, valueGetter, modalTitle, event.currentTarget);
    });
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
  const isScreeningCandidate = "final_score" in (row || {});
  const groupedColumns = new Set(isScreeningCandidate ? [
      "final_score", "data_quality_score",
      "recommended_use", "recommended_action",
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

function shouldOpenRowDetail(event) {
  if (event.defaultPrevented || event.target.closest("button, a, input, select, textarea, label")) return false;
  const selection = window.getSelection?.();
  return !selection || selection.isCollapsed || !event.currentTarget.contains(selection.anchorNode) || !event.currentTarget.contains(selection.focusNode);
}

function timeRelationshipExplanation(row, intervalMinutes = null) {
  const status = String(row?.temporal_direction_status ?? "not_computed");
  const messages = {
    variable_leads_supported: "接近最优的滞后位置均显示变量领先目标，时间方向允许其作为上游候选，但不代表因果成立。",
    target_leads_supported: "接近最优的滞后位置均显示目标领先变量，该变量不适合作为上游原因候选；可能是下游响应、反馈动作或其他滞后结果，具体机制需工艺确认。",
    synchronous: "接近最优的滞后位置集中在同步区域，当前无法区分上游和下游。",
    direction_unresolved: "已完成滞后扫描，但接近最优的滞后范围无法可靠区分时间方向。",
    not_computed: "当前没有获得可用的时间方向结果，建议检查数据、时间轴和滞后扫描设置。",
  };
  const lag = lagProfileNumber(row?.lag);
  const bestPoint = lag === null ? "" : ` 最佳相关点位于${formatSignedLag(lag)}个采样点。`;
  const nearMin = lagProfileNumber(row?.near_peak_lag_min);
  const nearMax = lagProfileNumber(row?.near_peak_lag_max);
  const nearCount = lagProfileNumber(row?.near_peak_lag_count);
  const nearRange = nearMin === null || nearMax === null || nearCount === null
    ? ""
    : ` 接近最优的滞后范围为[${nearMin}, ${nearMax}]（${nearCount}个点）。`;
  const boundary = row?.lag_boundary_flag === true || String(row?.lag_boundary_flag) === "1"
    ? " 最佳滞后触及搜索边界，准确滞后长度可能尚未完全识别。"
    : "";
  return `${messages[status] || messages.not_computed}${bestPoint}${nearRange}${boundary}`;
}

function correlationDirectionExplanation(direction, preprocessMode) {
  const mode = String(preprocessMode ?? "");
  const messages = {
    raw: {
      正向: "在当前最佳滞后对齐下，候选变量水平较高时，目标变量水平通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量水平较高时，目标变量水平通常较低。",
      方向较弱: "当前最佳滞后点的原始水平相关系数接近零，相关方向较弱。",
    },
    lowpass: {
      正向: "在当前最佳滞后对齐下，候选变量一阶低通平滑后的水平较高时，目标变量低通后的水平通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量一阶低通平滑后的水平较高时，目标变量低通后的水平通常较低。",
      方向较弱: "当前最佳滞后点的一阶低通平滑后相关系数接近零，相关方向较弱。",
    },
    detrend: {
      正向: "在当前最佳滞后对齐下，候选变量去趋势后的偏离较高时，目标变量去趋势后的偏离通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量去趋势后的偏离较高时，目标变量去趋势后的偏离通常较低。",
      方向较弱: "当前最佳滞后点的去趋势后相关系数接近零，相关方向较弱。",
    },
    lowpass_detrend: {
      正向: "在当前最佳滞后对齐下，候选变量低通+去趋势后的偏离较高时，目标变量的偏离通常也较高。",
      负向: "在当前最佳滞后对齐下，候选变量低通+去趋势后的偏离较高时，目标变量的偏离通常较低。",
      方向较弱: "当前最佳滞后点的低通+去趋势后相关系数接近零，相关方向较弱。",
    },
    diff: {
      正向: "在当前最佳滞后对齐下，候选变量增加时，目标变量通常也呈增加趋势。",
      负向: "在当前最佳滞后对齐下，候选变量增加时，目标变量通常呈下降趋势。",
      方向较弱: "当前最佳滞后点的变化量相关系数接近零，变化方向关系较弱。",
    },
    lowpass_diff: {
      正向: "在当前最佳滞后对齐下，候选变量低通+差分后的变化增加时，目标变量的变化通常也增加。",
      负向: "在当前最佳滞后对齐下，候选变量低通+差分后的变化增加时，目标变量的变化通常下降。",
      方向较弱: "当前最佳滞后点的低通+差分变化量相关系数接近零，变化方向关系较弱。",
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
  if (mode === "diff" || mode === "detrend_diff" || mode === "lowpass_diff") {
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
    lowpass: "一阶低通",
    lowpass_detrend: "一阶低通 + 去趋势",
    lowpass_diff: "一阶低通 + 差分",
    detrend: "去趋势",
    diff: "一阶差分",
    detrend_diff: "去趋势后差分",
  };
  return labels[String(mode ?? "")] || "未知预处理口径";
}

function directionalitySummary(row, correlationDirection, intervalMinutes = null) {
  const labels = {
    variable_leads_supported: "变量领先方向获得近峰范围支持",
    target_leads_supported: "目标领先方向获得近峰范围支持",
    synchronous: "近峰滞后集中于同步区域",
    direction_unresolved: "近峰范围方向未能可靠区分",
    not_computed: "时间方向未计算",
  };
  const correlation = correlationDirection === "方向较弱"
    ? "相关方向较弱"
    : correlationDirection === "未计算" ? "相关方向未计算" : `${correlationDirection}相关`;
  const status = String(row?.temporal_direction_status ?? "not_computed");
  return `${labels[status] || labels.not_computed}，${correlation}。`;
}

function directionInteractionExplanation(row, correlationDirection) {
  if (row?.temporal_direction_status === "variable_leads_supported" && correlationDirection === "负向") {
    return "候选变量先变化，并与之后的目标变化呈反向关系。";
  }
  return "时间关系与相关方向是两种独立信息，建议结合工艺机理和时间对齐复核。";
}

function updateDirectionalityTimeDetails(lag, intervalMinutes) {
  const explanation = el("directionalityTimeExplanation");
  const summary = el("directionalitySummary");
  if (!explanation) return;
  const row = JSON.parse(explanation.dataset.directionEvidence || "{}");
  row.lag = lag;
  explanation.textContent = timeRelationshipExplanation(row, intervalMinutes);
  if (summary) summary.textContent = directionalitySummary(row, summary.dataset.correlationDirection, intervalMinutes);
}

function renderScreeningScoreDetails(row) {
  if (!("final_score" in (row || {}))) return "";
  const scoreColumns = [
    "final_score", "association_score", "innovation_score", "innovation_status", "lag_quality",
    "temporal_direction_status", "temporal_penalty_rate",
    "data_quality_score", "risk_flags", "recommended_use", "recommended_action",
  ];
  const renderFields = (columns, labels = {}) => columns.map((column) => `
    <div class="detail-field">
      <strong>${escapeHtml(labels[column] || columnLabel(column))}</strong>
      <span>${escapeHtml(displayCellValue(column, row[column]))}</span>
    </div>
  `).join("");
  const preprocessMode = currentAnalysisContext.preprocess_mode;
  const timeRelationship = displayCellValue("temporal_direction_status", row.temporal_direction_status);
  const correlationDirection = row.correlation_direction || "未计算";
  const innovationDirection = innovationDirectionText(row.innovation_sign);
  const innovationDirectionLabel = preprocessMode === "diff" || preprocessMode === "detrend_diff" || preprocessMode === "lowpass_diff"
    ? "当前分析变化方向"
    : "变化量相关方向";
  const innovationExplanation = innovationDirectionExplanation(row.innovation_status, preprocessMode);
  return `
    <h4>初步筛选得分</h4>
    <div class="detail-grid">${renderFields(scoreColumns)}</div>
    <div class="detail-field"><strong>近峰滞后范围</strong><span>${escapeHtml(`[${displayCellValue("near_peak_lag_min", row.near_peak_lag_min)}, ${displayCellValue("near_peak_lag_max", row.near_peak_lag_max)}]（${displayCellValue("near_peak_lag_count", row.near_peak_lag_count)} 个点）`)}</span></div>
    <p>final_score 以基础关联强度 × 数据质量为基线，Residual 与稳定性只提供有限正向奖励；风险标签本身不扣分，只有明确的目标领先时间证据会负向约束得分。</p>
    <h4>相关性证据</h4>
    <div class="detail-grid">${renderFields(CORRELATION_OVERVIEW_COLUMNS, { lag: "最佳滞后点", direction: "滞后方向" })}</div>
    <details class="correlation-evidence-details">
      <summary>展开 P/Q、R² 与样本数</summary>
      <p class="help">大样本与时序自相关下，P/Q 值、R² 和样本数仅供参考，不参与评分、筛选、排序或颜色强调。</p>
      <div class="detail-grid">${renderFields(CORRELATION_DETAIL_COLUMNS)}</div>
    </details>
    ${renderLoadAdjustmentValidation(row)}
    <h4>方向性解释</h4>
    <p><strong>组合摘要：</strong><span id="directionalitySummary" data-correlation-direction="${escapeHtml(correlationDirection)}">${escapeHtml(directionalitySummary(row, correlationDirection))}</span></p>
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
    <p id="directionalityTimeExplanation" data-direction-evidence="${escapeHtml(JSON.stringify({ temporal_direction_status: row.temporal_direction_status, lag: row.lag, near_peak_lag_min: row.near_peak_lag_min, near_peak_lag_max: row.near_peak_lag_max, near_peak_lag_count: row.near_peak_lag_count, lag_boundary_flag: row.lag_boundary_flag }))}">${escapeHtml(timeRelationshipExplanation(row))}</p>
    <p>${escapeHtml(correlationDirectionExplanation(correlationDirection, preprocessMode))}</p>
    <p>${escapeHtml(directionInteractionExplanation(row, correlationDirection))}</p>
    <p>${escapeHtml(correlationConsistencyMessage(row.pearson, row.spearman))}</p>
    <p class="help">时间领先和正负相关只表示当前数据中的时序关联，不等于因果方向。共同负荷、上游扰动和工况切换均可能产生类似结果。</p>
    <h4>滞后相关曲线</h4>
    <div id="lagProfilePanel" class="lag-profile-panel loading" aria-live="polite">正在加载滞后相关曲线……</div>
    <h4>解释说明</h4>
    <p>初步分析仅描述当前阶段的统计筛选结果，不对未执行的后续分析作解释。</p>
  `;
}

function hasDisplayValue(value) {
  return value !== null && value !== undefined && value !== "";
}

function loadAdjustmentChannelText(selected, channel) {
  if (selected === true || selected === "true") return "已获得支持";
  if (selected === false || selected === "false") return "未获得支持";
  return `${channel}结果未提供`;
}

function loadAdjustmentExplanation(row) {
  const explanations = {
    dual_channel_supported: "该变量在全量数据和去负荷数据中均显示关联支持。",
    residual_only_supported: "该变量仅在去负荷后发现独立关联。",
    raw_only_supported: "该变量的关联目前主要表现于全量数据中。",
    raw_only_common_load_risk: "全量数据关联在去负荷后明显减弱，可能受共同负荷影响。",
    raw_only_residual_weak: "该变量的关联主要表现于全量数据中，去除负荷共同变化后独立关联有限。",
    force_included_only: "该变量由人工强制包含，未形成可展示的去负荷验证结论。",
    control_reference: "该变量作为控制或负荷参考展示，不适用去负荷验证。",
  };
  const relation = String(row?.load_adjusted_relation_status ?? "");
  if (explanations[relation]) return explanations[relation];
  if (relation === "raw_only_residual_missing") {
    if (row?.residual_evidence_status === "insufficient") return "去负荷验证不可计算。";
    if (row?.residual_evidence_status === "missing") return "本次分析未执行去负荷验证。";
    return "未提供去负荷验证结果。";
  }
  const residualStatus = String(row?.residual_status ?? "");
  if (["not_run", "not_computed"].includes(residualStatus)) return "本次分析未执行去负荷验证。";
  if (residualStatus && residualStatus !== "ok") return "去负荷验证不可计算。";
  if (row?.residual_evidence_status === "insufficient") return "去负荷验证不可计算。";
  if (row?.residual_evidence_status === "missing") return "未提供去负荷验证结果。";
  return "未提供去负荷验证结果。";
}

function renderLoadAdjustmentValidation(row) {
  if (![
    "candidate_source", "selected_by_raw", "selected_by_residual",
    "load_adjusted_relation_status", "residual_evidence_status", "residual_signal_score",
    "residual_status",
  ].some((column) => column in (row || {}))) return "";
  const fields = [];
  if (hasDisplayValue(row.candidate_source)) fields.push(["候选来源", displayCellValue("candidate_source", row.candidate_source)]);
  fields.push(["全量数据关联", loadAdjustmentChannelText(row.selected_by_raw, "全量数据关联")]);
  fields.push(["去负荷后关联", loadAdjustmentChannelText(row.selected_by_residual, "去负荷后关联")]);
  if (hasDisplayValue(row.load_adjusted_relation_status)) fields.push(["去负荷验证状态", displayCellValue("load_adjusted_relation_status", row.load_adjusted_relation_status)]);
  if (hasDisplayValue(row.residual_evidence_status)) fields.push(["去负荷验证证据", displayCellValue("residual_evidence_status", row.residual_evidence_status)]);
  if (hasDisplayValue(row.residual_status)) fields.push(["去负荷结果状态", displayCellValue("residual_status", row.residual_status)]);
  if (hasDisplayValue(row.residual_signal_score)) fields.push(["去负荷后独立关联强度", displayCellValue("residual_signal_score", row.residual_signal_score)]);
  const grid = fields.map(([label, value]) => `
    <div class="detail-field"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>
  `).join("");
  return `
    <h4>负荷调整验证</h4>
    <div class="detail-grid">${grid}</div>
    <p><strong>验证说明：</strong>${escapeHtml(loadAdjustmentExplanation(row))}</p>
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
    ["有效分析变量数量", overview.effective_variables ?? ""],
    ["重点候选数量", overview.recommended_candidate_count ?? ""],
    ["控制/负荷参考变量数量", overview.control_reference_count ?? ""],
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
  overviewTop: ["variable", "final_score", "lag", "direction", "variable_role", "pearson", "spearman", "method", "correlation_direction", "lag_quality", "data_quality_score", "risk_flags", "risk_level", "recommended_use"],
  validationSummaryTable: ["variable", "validation_status", "evidence_consistency", "supporting_methods", "limiting_factors"],
  grangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "interpretation"],
  modelVariableImportanceTable: ["variable", "max_importance", "importance_rank", "best_model_feature", "best_model_lag", "recommended_use"],
  importanceTable: ["variable", "importance", "importance_rank", "feature", "lag", "method"],
  modelDiscoveredTable: ["variable", "max_importance", "importance_rank", "best_model_lag", "missing_from_screening_top_n", "discovery_reason"],
  enhancedSummaryTable: ["variable", "final_score", "lag", "direction", "status", "model_lift", "rolling_stability"],
  validationFieldsTable: ["variable", "initial_screening_lag", "validation_lag", "conditional_validation_lag", "screening_model_lift", "validation_model_lift"],
  verificationReviewPoolTable: ["variable", "candidate_source", "source_rank", "include_reason"],
  enhancedLiftTable: ["variable", "status", "model_lift_score", "median_fold_lift", "positive_fold_ratio", "model_lift", "ar_baseline_rmse", "candidate_rmse"],
  enhancedRollingTable: ["variable", "best_lag", "best_score", "rolling_corr_median", "rolling_stability"],
  conditionalGrangerTable: ["variable", "status", "best_lag", "min_p_value", "fdr_q_value", "predictive_contribution"],
  xgbModelSummaryTable: ["model_name", "mean_rmse", "mean_mae", "mean_r2", "M2_vs_M1_rmse_improvement_pct"],
  xgbCandidateUpliftTable: ["variable", "median_rmse_improvement_pct", "median_mae_improvement_pct", "positive_rmse_fold_ratio", "validation_status"],
  xgbCandidateFoldMetricsTable: ["variable", "fold", "test_time_range", "rmse_improvement_pct", "mae_improvement_pct", "test_rows"]
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

function validationSummaryColumns() {
  return ["variable", "validation_status", "evidence_consistency", "supporting_methods", "limiting_factors"];
}

function renderValidationSummaryTable(rows) {
  renderGenericTable("validationSummaryTable", rows || [], validationSummaryColumns());
}

function validationFieldsColumns() {
  return ["variable", "initial_screening_lag", "validation_lag", "conditional_validation_lag", "screening_model_lift", "validation_model_lift"];
}

function renderValidationFieldsTable(rows) {
  renderGenericTable("validationFieldsTable", rows || [], validationFieldsColumns());
}

function renderVerificationReviewPool(rows) {
  renderGenericTable("verificationReviewPoolTable", rows || []);
}

function syncModelDiscoveryReviewPoolOptions(rows) {
  const select = el("modelDiscoveryReviewPoolVariable");
  if (!select) return;
  const current = select.value;
  const poolVariables = new Set((lastVerificationReviewPoolRows || []).map((row) => String(row.variable || "")));
  const options = (rows || [])
    .map((row) => String(row.variable || ""))
    .filter((variable) => variable && !poolVariables.has(variable));
  select.innerHTML = '<option value="">请选择模型探索变量</option>' + options
    .map((variable) => `<option value="${escapeHtml(variable)}">${escapeHtml(variable)}</option>`)
    .join("");
  if (options.includes(current)) select.value = current;
}

async function addToVerificationReviewPool(candidateSource) {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  const isModelDiscovery = candidateSource === "model_discovery";
  const variable = (isModelDiscovery
    ? el("modelDiscoveryReviewPoolVariable").value
    : el("manualReviewPoolVariable").value.trim());
  if (!variable) {
    return setStatus(isModelDiscovery ? "请选择模型遗漏探索变量。" : "请输入要人工加入的初筛变量。", "error");
  }
  const startedAt = performance.now();
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    form.append("variable", variable);
    form.append("candidate_source", candidateSource);
    const data = await postForm("/api/add_to_verification_review_pool", form);
    lastVerificationReviewPoolRows = data.verificationReviewPool || [];
    renderVerificationReviewPool(lastVerificationReviewPoolRows);
    syncModelDiscoveryReviewPoolOptions(lastModelDiscoveredRows);
    if (!isModelDiscovery) el("manualReviewPoolVariable").value = "";
    renderDownloads(data.downloads || []);
    setStatus(appendElapsed(data.message || "已加入二级验证复核池。", startedAt), "success");
  } catch (error) {
    setStatus(appendElapsed(error.message || String(error), startedAt), "error");
  }
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
  const columns = finalReviewSummaryColumns().filter((column) => column === "trend_action" || finalSummaryValue(rows[0], column) !== undefined);
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
      } else {
        td.innerHTML = renderTableCell(column, finalSummaryValue(row, column));
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
    tr.addEventListener("click", (event) => {
      if (!shouldOpenRowDetail(event)) return;
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
    "independent_predictive_support",
    "confounder_assessment",
    "control_relation_assessment",
    "statistical_limitation",
    "direction_assessment",
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
      <p>该卡片汇总该变量在主筛查、验证和可信度审查中的证据，用于解释独立预测贡献及其混杂、控制和统计限制；不改变初筛结果。</p>
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
  if ("final_score" in (row || {})) {
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

function validationSummaryStateLabel(value) {
  const text = String(value ?? "").trim().toLowerCase();
  const labels = {
    not_run: "未执行",
    variable_missing: "变量缺失",
    zero_evidence: "零支持证据",
    computed_no_support: "已计算但未形成支持",
    missing: "证据缺失",
    not_computed: "不可计算",
    skipped: "已跳过",
    failed: "执行失败",
    support: "支持",
  };
  if (labels[text]) return labels[text];
  if (text.startsWith("failed")) return "执行失败";
  if (text.startsWith("skipped")) return "已跳过";
  if (text.startsWith("not_computed") || text.startsWith("unavailable")) return "不可计算";
  return text ? "状态未知" : "";
}

function validationSummarySupportingMethods(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "无支持证据";
  const labels = {enhanced_screening: "增强筛选", granger: "Granger", model_explanation: "模型解释"};
  const methods = raw
    .split(/[;,]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => labels[item] || "其他验证");
  return methods.length ? methods.join("、") : "无支持证据";
}

function displayCellValue(column, value) {
  const matrixLabels = evidenceMatrixStatusLabels[column];
  if (matrixLabels && Object.prototype.hasOwnProperty.call(matrixLabels, String(value ?? ""))) {
    return matrixLabels[String(value ?? "")];
  }
  if (column === "supporting_methods") {
    return validationSummarySupportingMethods(value);
  }
  if (column === "limiting_factors") {
    const methodLabels = {enhanced_screening: "增强筛选", granger: "Granger", model_explanation: "模型解释"};
    return String(value ?? "")
      .split(/[;,]/)
      .map((item) => {
        const [method, state] = item.trim().split(":", 2);
        if (!state) return item.trim() ? "状态未知" : "";
        return `${methodLabels[method] || "验证方法"}：${validationSummaryStateLabel(state)}`;
      })
      .filter(Boolean)
      .join("；");
  }
  if (column === "control_reference_type") {
    const labels = {residual_control: "去负荷控制参考", capacity_reference: "负荷参考", segment_reference: "工况分段参考", pid_setpoint: "PID设定值参考", pid_output: "PID输出参考"};
    return labels[value] || value;
  }
  if (column === "control_reference_source") {
    const labels = {configured_residual_control: "参数配置：去负荷控制列", configured_capacity: "参数配置：负荷列", configured_segment: "参数配置：分段列", tag_suffix_sv: "位号后缀.SV", tag_suffix_sp: "位号后缀.SP", tag_suffix_mv: "位号后缀.MV"};
    return labels[value] || value;
  }
  if (column === "candidate_source") {
    const labels = {raw_only: "全量数据", residual_only: "去负荷数据", raw_and_residual: "全量数据和去负荷数据", force_included: "人工强制包含", control_reference: "控制/负荷参考", initial_screening: "初筛 Top-K", manual_include: "人工加入", model_discovery: "模型发现确认"};
    return labels[value] || value;
  }
  if (column === "residual_evidence_status") {
    const labels = {strong: "去负荷后独立关联明显", weak: "去负荷后独立关联较弱", insufficient: "去负荷验证不可计算", missing: "未提供去负荷验证结果", control_reference: "控制/负荷参考"};
    return labels[value] || value;
  }
  if (column === "residual_status") {
    const labels = {ok: "结果可用", not_run: "未执行", not_computed: "未执行", no_valid_controls: "未找到可用的去负荷控制列", insufficient_joint_samples: "有效样本不足", no_valid_residual_lag: "未找到可用去负荷滞后结果", rank_deficient: "控制列秩不足", rank_deficient_no_valid_residual_lag: "控制列秩不足且未找到可用去负荷滞后结果", fit_failed: "去负荷拟合失败", control_reference_not_residualized: "控制/负荷参考未参与去负荷"};
    return labels[value] || value;
  }
  if (column === "load_adjusted_relation_status") {
    const labels = {dual_channel_supported: "全量数据和去负荷后关联均有支持", residual_only_supported: "仅在去负荷后发现独立关联", raw_only_supported: "全量数据关联明显", raw_only_common_load_risk: "全量数据关联明显，去负荷后明显减弱", raw_only_residual_weak: "全量数据关联明显，去负荷后独立关联较弱", raw_only_residual_missing: "全量数据关联明显，本次分析未执行去负荷验证", force_included_only: "人工强制包含", control_reference: "控制/负荷参考"};
    return labels[value] || value;
  }
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
  setSelectValueIfExists("trendVar5", "");
  setSelectValueIfExists("trendVar6", "");
  setSelectValueIfExists("trendVar7", "");
  setSelectValueIfExists("trendVar8", "");
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
  if (option) {
    node.value = targetValue;
    syncSearchableSelect(node);
  }
}

function missingText(targetId) {
  if (targetId === "validationSummaryTable") return "完成主筛查后显示统一验证结论。";
  if (targetId === "validationFieldsTable") return "完成主筛查后显示阶段字段。";
  if (targetId === "grangerTable") return "未启用 Granger 检验，或没有可展示结果。";
  if (targetId === "enhancedSummaryTable") return "点击“运行增强筛选”后显示增强筛选摘要。";
  if (targetId === "enhancedLiftTable") return "点击“运行增强筛选”后显示模型提升评分。";
  if (targetId === "enhancedRollingTable") return "点击“运行增强筛选”后显示滚动稳定性评分。";
  if (targetId === "modelVariableImportanceTable") return "运行随机森林模型解释后显示变量排序。";
  if (targetId === "importanceTable") return "未启用随机森林模型解释，或没有可展示结果。";
  if (targetId === "modelDiscoveredTable") return "运行随机森林模型解释后显示遗漏探索线索。";
  if (targetId === "conditionalGrangerTable") return "未运行 条件 Granger 预测验证。";
  if (targetId === "finalReviewSummaryTable") return "未运行 可信度审查摘要。";
  if (targetId === "causalReviewEvidenceTable") return "未运行 逐变量可信度审查证据表。";
  if (targetId === "xgbModelSummaryTable") return "未运行 XGB 时间外预测验证。";
  if (targetId === "xgbCandidateUpliftTable") return "未运行 XGB 时间外预测验证。";
  if (targetId === "xgbCandidateFoldMetricsTable") return "当前候选没有可展示的逐时间折明细。";
  if (targetId === "overviewTop") return "暂无初步分析 Top 10。";
  return "无可展示结果。";
}

function modelVariableImportanceColumns() {
  return ["variable", "best_model_feature", "best_model_lag", "max_importance", "total_importance", "feature_count", "importance_rank", "method", "ranked_feature_rank", "ranked_final_score", "risk_flags", "recommended_use", "recommended_action", "interpretation"];
}

function modelDiscoveredColumns() {
  return ["variable", "best_model_feature", "best_model_lag", "max_importance", "importance_rank", "model_feature_count", "nearby_lag_count", "ranked_feature_rank", "ranked_final_score", "missing_from_screening_top_n", "risk_flags", "discovery_reason", "interpretation"];
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

function xgbCandidateFoldMetricColumns() {
  return ["variable", "fold", "test_time_range", "rmse_improvement_pct", "mae_improvement_pct", "test_rows"];
}

function renderXgbCandidateUpliftTable(rows) {
  renderGenericTable("xgbCandidateUpliftTable", rows || [], xgbCandidateUpliftColumns());
  const container = el("xgbCandidateUpliftTable");
  if (!container) return;
  container.onclick = (event) => {
    const rowElement = event.target.closest?.("tbody tr");
    if (!rowElement || !shouldOpenRowDetail(event)) return;
    const displayRows = sortedRowsForTable("xgbCandidateUpliftTable", rows || []);
    const row = displayRows[Number(rowElement.dataset.rowIndex)];
    if (row) renderXgbCandidateFoldDetails(row.variable);
  };
}

function renderXgbCandidateFoldDetails(variable) {
  const details = el("xgbCandidateFoldDetails");
  const rows = lastXgbCandidateFoldMetricRows.filter(
    (row) => String(row.variable ?? "") === String(variable ?? "")
  );
  if (details) {
    details.open = true;
    const summary = details.querySelector("summary");
    if (summary) summary.textContent = `逐时间折验证明细：${displayCellValue("variable", variable)}`;
  }
  if (!rows.length) {
    resetOptionalTable(
      "xgbCandidateFoldMetricsTable",
      "该候选没有可展示的逐时间折明细（可能为有效特征不足或尚未计算）。"
    );
    return;
  }
  const displayRows = rows.map((row) => ({
    variable: row.variable,
    fold: row.fold,
    test_time_range: `${row.test_start ?? ""} ~ ${row.test_end ?? ""}`,
    rmse_improvement_pct: row.rmse_improvement_pct,
    mae_improvement_pct: row.mae_improvement_pct,
    test_rows: row.test_rows,
  }));
  renderGenericTable(
    "xgbCandidateFoldMetricsTable",
    displayRows,
    xgbCandidateFoldMetricColumns()
  );
}

function clearXgbCandidateFoldDetails() {
  const details = el("xgbCandidateFoldDetails");
  if (details) {
    details.open = false;
    const summary = details.querySelector("summary");
    if (summary) summary.textContent = "逐时间折验证明细";
  }
  resetOptionalTable(
    "xgbCandidateFoldMetricsTable",
    "点击候选变量汇总行后查看该变量的逐时间折明细。"
  );
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
  "independent_predictive_support",
  "confounder_assessment",
  "control_relation_assessment",
  "statistical_limitation",
  "direction_assessment",
  "statistical_limit_level",
  "risk_constraint_level",
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
    "independent_predictive_support", "confounder_assessment", "control_relation_assessment", "statistical_limitation", "direction_assessment",
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
  if (column === "trend_action") return "";
  const aliases = FINAL_SUMMARY_COLUMN_ALIASES[column] || [column];
  for (const key of aliases) {
    if (Object.prototype.hasOwnProperty.call(row, key)) return row[key];
  }
  return undefined;
}

function causalReviewEvidenceColumns() {
  return ["variable", "candidate_grade", "final_score", "data_priority", "evidence_score", "evidence_level", "independent_predictive_support", "confounder_assessment", "control_relation_assessment", "statistical_limitation", "direction_assessment", "statistical_limit_level", "risk_constraint_level", "integrated_review_decision", "integrated_review_reason", "statistical_limit_reason", "evidence_reason", "conditional_granger_status", "conditional_fdr_q_value", "predictive_contribution", "model_lift", "rolling_stability", "model_importance_rank", "risk_flags", "interpretation"];
}


const causalEvidenceCoreColumns = () => ["variable", "candidate_grade", "final_score", "data_priority", "evidence_level", "independent_predictive_support", "confounder_assessment", "statistical_limitation", "integrated_review_decision"];

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
    modalTitle: (row) => `逐变量可信度审查证据表：${displayCellValue("variable", row.variable)}`,
  });
}

function evidenceMatrixColumns() {
  return [
    "variable", "initial_rank", "final_score",
    "validation_status", "evidence_consistency", "supporting_methods",
    "independent_predictive_support", "confounder_assessment",
    "control_relation_assessment", "statistical_limitation",
    "direction_assessment", "xgb_status", "generalization_status",
  ];
}

function renderEvidenceMatrixTable(rows) {
  renderGenericTable("evidenceMatrixTable", rows || [], evidenceMatrixColumns());
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
  "final_score",
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
  renderDownloadTarget("evidenceMatrixDownload", downloads, "evidence_matrix.csv");
}

function renderXgbDownloads(downloads) {
  renderDownloadTarget("xgbFoldMetricsDownload", downloads, "xgb_validation/xgb_fold_metrics.csv");
  renderDownloadTarget("xgbModelSummaryDownload", downloads, "xgb_validation/xgb_model_summary.csv");
  renderDownloadTarget("xgbCandidateUpliftDownload", downloads, "xgb_validation/xgb_candidate_uplift.csv");
  renderDownloadTarget("xgbCandidateFoldMetricsDownload", downloads, "xgb_validation/xgb_candidate_fold_metrics.csv");
  renderDownloadTarget("xgbValidationSummaryDownload", downloads, "xgb_validation/xgb_validation_summary.json");
  renderDownloadTarget("xgbPredictionsDownload", downloads, "xgb_validation/xgb_predictions.csv");
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

const STATUS_COLUMNS = new Set([
  "status", "validation_status", "evidence_consistency", "innovation_status", "conditional_status",
  "conditional_granger_status", "residual_status", "residual_evidence_status",
  "load_adjusted_relation_status", "evidence_level", "risk_level", "data_quality_status",
  "rolling_status", "stability_status", "recommended_use", "final_decision",
  "final_recommendation", "integrated_review_decision", "candidate_grade", "lag_quality",
  "lag_quality_status", "risk_constraint_level", "statistical_limit_level",
  "validation_status", "evidence_consistency", "independent_predictive_support",
  "confounder_assessment", "control_relation_assessment", "statistical_limitation",
  "direction_assessment", "xgb_status", "generalization_status",
]);

function statusTone(column, value) {
  const text = String(value ?? "").toLowerCase();
  if (!text) return "neutral";
  if (["risk_level", "risk_constraint_level", "statistical_limit_level"].includes(column) && /high|strong|risk/.test(text)) return "negative";
  if (/failed|error|not_recommended|not_supported|poor|conflict|insufficient|target_leads|negative/.test(text)) return "negative";
  if (/risk|warning|limited|manual|secondary|weak|partial|not_run|not_computed|skipped|unknown/.test(text)) return "caution";
  if (/success|supported|strong|consistent|ok|normal|priority|candidate|positive/.test(text)) return "positive";
  return "neutral";
}

function renderTableCell(column, value) {
  const text = displayCellValue(column, value);
  if (!STATUS_COLUMNS.has(column)) return escapeHtml(text);
  const tone = statusTone(column, value);
  return `<span class="status-label status-label-${tone}">${escapeHtml(text || "-")}</span>`;
}

function tableCellClass(column, value) {
  const name = String(column || "");
  const number = typeof value === "number" ? value : Number(value);
  const numericColumn = /(?:^|_)(score|lag|rmse|mae|r2|p_value|q_value|rank|count|rows|fold|condition_number|importance|contribution|iteration)(?:$|_)/i.test(name);
  if (Number.isFinite(number) || numericColumn) return "numeric";
  const wrapColumn = /interpretation|reason|action|risk_flags|control_columns|evidence_reason|statistical_limit_reason|key_reason|suggested_next_action|lag_boundary_hint|supporting_methods|limiting_factors/i.test(name);
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
      candidate: "普通候选",
      residual_control: "残差控制参考",
      capacity_reference: "负荷参考",
      segment_reference: "工况分段参考",
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
      partial: "部分一致",
      consistent: "一致",
      limited: "有限支持",
      enhanced_screening: "增强筛选",
      model_explanation: "模型解释",
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
      variable_leads_supported: "变量领先目标，时间方向允许作为上游候选",
      variable_leads_target: "变量领先目标",
      synchronous: "变量与目标基本同步，无法区分上游和下游",
      target_leads_supported: "目标明显领先该变量，不适合作为上游原因候选",
      direction_unresolved: "时间方向无法可靠确认",
      conditional_granger_supported: "条件 Granger 显示存在独立预测贡献证据",
      independent_predictive_evidence: "独立预测贡献证据",
      independent_predictive_evidence_limited: "有限独立预测贡献证据",
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
      conditional_granger_supported: "条件 Granger 显示存在独立预测贡献证据",
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
      poor_data_quality: "数据质量需关注",
      severe_data_quality: "数据质量严重不足",
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
      "confounder review summary only": "仅作可信度审查摘要",
      supported: "独立预测支持",
      supported_with_limitations: "独立预测支持但存在限制",
      supported_without_positive_contribution: "历史兼容值：独立预测支持但增量不为正",
      limited_by_collinearity: "历史兼容值：独立预测受共线性限制",
      limited_by_lag_fallback: "历史兼容值：独立预测受滞后回退限制",
      not_computed: "未计算",
      not_assessed: "未审查",
      formula_relation_risk: "公式关系风险",
      common_driver_risk: "共同驱动风险",
      shared_signal_risk: "共享信号风险",
      no_flagged_confounder: "未标记混杂风险",
      control_reference: "控制变量参考",
      formula_coupled_reference: "公式耦合参考",
      possible_control_response: "可能控制响应",
      shared_capacity_or_control_context: "共同负荷或控制背景",
      no_control_relation_flagged: "未标记控制关系",
      no_flagged_statistical_limitation: "未标记统计限制",
      high_collinearity_limitation: "高共线性限制",
      insufficient_sample_limitation: "样本不足限制",
      failed_statistical_limitation: "统计计算失败",
      weak_statistical_limitation: "历史兼容值：弱统计限制",
      medium_statistical_limitation: "历史兼容值：中等统计限制",
      strong_statistical_limitation: "历史兼容值：强统计限制",
      variable_leads_target: "变量领先目标",
      zero_lag: "零滞后",
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
      lag_reaches_boundary: "滞后触及边界",
    };
    if (map[value]) return map[value];
    if (value === "enhanced screening only; not a causal conclusion") return "仅作增强筛查；不是因果结论";
    if (value.startsWith("confounder review of predictive evidence only; not a causal conclusion")) return "仅作可信度审查；不是因果结论；解析式 p/q 值不能完全消除工业时序自相关影响";
    if (value === "model explanation only; not a causal conclusion") return "仅作模型解释；不是因果结论";
    if (value === "model discovery exploration only; not a validation conclusion or causal conclusion") return "仅作模型遗漏探索；不是验证结论或因果结论";
    if (value === "confounder review evidence only; not a causal conclusion") return "仅作可信度审查证据；不是因果结论";
    if (value === "confounder review summary only; not a causal conclusion") return "仅作可信度审查摘要；不是因果结论";
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
    innovation_lag: "变化量滞后",
    innovation_direction: "变化量方向",
    innovation_sign: "变化量符号",
    innovation_status: "变化量验证状态",
  };
  if (addedLabels[column]) return addedLabels[column];
  const labels = {
    variable: "变量",
    driver_rank: "初筛排名",
    initial_rank: "初筛结果（原始排名）",
    candidate_source: "候选来源",
    source_rank: "初筛排名",
    include_reason: "加入原因",
    candidate_priority_rank: "候选优先级",
    candidate_priority_score: "候选综合优先分",
    residual_signal_score: "去负荷后独立关联强度",
    residual_evidence_status: "去负荷验证证据",
    load_adjusted_relation_status: "去负荷验证",
    common_capacity_candidate_flag: "共同负荷风险",
    variable_role: "变量角色",
    control_reference_type: "参考类型",
    control_reference_source: "识别来源",
    trend_action: "趋势验证",
    final_score: "初步筛选得分",
    lag: "最佳滞后",
    direction: "时间关系",
    correlation_direction: "相关方向",
    raw_corr: "全量数据关联强度",
    association_score: "全量数据关联规范化得分",
    innovation_score: "变化量关联得分",
    residual_corr: "去负荷后关联强度",
    independent_signal_score: "去负荷后独立关联得分",
    correlation_evidence_score: "关联证据综合得分",
    correlation_evidence_status: "关联证据状态",
    temporal_direction_status: "时间方向状态",
    temporal_penalty_rate: "时间方向扣分",
    temporal_score_cap: "时间方向得分上限",
    near_peak_lag_min: "近峰滞后最小值",
    near_peak_lag_max: "近峰滞后最大值",
    near_peak_lag_count: "近峰滞后点数",
    prediction_score: "增量预测得分",
    stability_score: "综合稳定性",
    data_quality_score: "数据质量得分",
    evidence_strength: "证据强度",
    evidence_completeness: "证据覆盖度",
    evidence_confidence: "证据修正系数",
    evidence_coverage_status: "证据覆盖状态",
    stability_status: "稳定性状态",
    data_quality_status: "数据质量状态",
    evidence_missing_items: "缺失证据",
    evidence_score_low: "证据得分下界",
    evidence_score_high: "证据得分上界",
    score_method: "评分方法",
    residual_status: "去负荷结果状态",
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
    poor_data_quality_flag: "数据质量需关注",
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
    fold: "时间折",
    train_start: "训练开始",
    train_end: "训练结束",
    validation_start: "验证开始",
    validation_end: "验证结束",
    test_start: "测试开始",
    test_end: "测试结束",
    test_time_range: "测试时间范围",
    train_rows: "训练样本数",
    validation_rows: "验证样本数",
    test_rows: "测试样本数",
    mean_rmse: "平均RMSE",
    mean_mae: "平均MAE",
    mean_r2: "平均R²",
    M2_vs_M1_rmse_improvement_pct: "M2相对M1 RMSE改善(%)",
    rmse_improvement_pct: "RMSE改善(%)",
    mae_improvement_pct: "MAE改善(%)",
    candidate_r2: "候选R²",
    best_iteration: "最佳迭代轮数",
    median_rmse_improvement_pct: "RMSE改善中位数(%)",
    median_mae_improvement_pct: "MAE改善中位数(%)",
    positive_rmse_fold_ratio: "RMSE改善折占比",
    validation_status: "验证状态",
    evidence_consistency: "证据一致性",
    supporting_methods: "主要支持证据",
    independent_predictive_support: "独立预测贡献证据",
    confounder_assessment: "混杂风险审查",
    control_relation_assessment: "控制关系审查",
    statistical_limitation: "统计限制审查",
    direction_assessment: "方向审查（带符号滞后）",
    xgb_status: "XGB状态（已有结果）",
    generalization_status: "泛化状态（已有结果）",
    limiting_factors: "限制因素",
    initial_screening_lag: "初筛滞后（signed）",
    validation_lag: "验证滞后（signed）",
    conditional_validation_lag: "条件验证滞后（signed）",
    screening_model_lift: "初筛模型提升",
    validation_model_lift: "验证模型提升",
    candidate_grade: "候选等级",
    review_tier: "复核层级",
    review_priority: "人工复核优先级",
    review_reason: "证据摘要",
    final_review_decision: "可信度审查建议",
    final_review_reason: "证据摘要",
    conditional_granger_status: "条件Granger状态",
    final_rank: "人工复核优先级（展示序号；不参与初筛评分或排序）",
    final_recommendation: "可信度审查建议",
    final_decision: "可信度审查建议",
    key_reason: "主要原因（证据摘要）",
    main_reason: "主要原因（证据摘要）",
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
    integrated_review_decision: "可信度审查建议",
    integrated_review_reason: "证据摘要",
    independent_predictive_support: "独立预测贡献审查",
    confounder_assessment: "混杂风险审查",
    control_relation_assessment: "控制关系审查",
    statistical_limitation: "统计限制审查",
    direction_assessment: "方向审查（带符号滞后）",
    model_importance_rank: "模型重要性排名",
    model_explanation_support: "模型解释支持",
    causalReviewEvidence: "逐变量可信度审查证据表",
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
    raw_score: "原始滞后得分",
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
  if (!node) return;
  node.className = `status ${type}`;
  node.dataset.state = type;
  node.setAttribute("aria-busy", type === "loading" ? "true" : "false");
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
  clearVariableFilters();
  clearLagProfileCache();
  fileId = "";
  currentRunId = "";
  currentAnalysisContext = {};
  recognizedColumns = [];
  recognizedNumericColumns = [];
  lastRows = [];
  lastRecommendedRows = [];
  lastGrangerRows = [];
  lastImportanceRows = [];
  lastModelVariableRows = [];
  lastNearMissRows = [];
  lastModelDiscoveredRows = [];
  lastEnhancedSummaryRows = [];
  lastEnhancedLiftRows = [];
  lastEnhancedRollingRows = [];
  lastVerificationReviewPoolRows = [];
  lastConditionalRows = [];
  lastCausalEvidenceRows = [];
  lastEvidenceMatrixRows = [];
  evidenceMatrixStatusLabels = DEFAULT_EVIDENCE_MATRIX_STATUS_LABELS;
  lastFinalReviewSummaryRows = [];
  lastXgbModelSummaryRows = [];
  lastXgbCandidateUpliftRows = [];
  lastXgbCandidateFoldMetricRows = [];
  lastXgbValidationSummary = {};
  lastTrendSeries = [];
  lastTrendAxisMode = "shared";
  trendTimeRangeMode = "auto";
  trendSamplingIntervalMs = null;
  trendLatestTime = "";
  trendAutoWindowActive = false;
  trendDefaultStart = "";
  trendDefaultEnd = "";
  trendSelection = null;
  lastScatterMatrixPayload = null;
  tableSortStates = { table: { column: "final_score", direction: "desc" }, finalReviewSummaryTable: { column: "final_rank", direction: "asc" } };
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
  el("preprocessMode").value = "raw";
  el("lowpassTauMinutes").value = "5.0";
  el("diffIntervalMinutes").value = "";
  el("detrendWindow").value = "24";
  updatePreprocessControls();
  el("branchSelectionSection").hidden = true;
  el("branchSelectionStatus").textContent = "";
  el("branchLockedHint").hidden = true;
  el("confirmRawBranch").disabled = true;
  el("confirmProcessedBranch").disabled = true;
  resetOptionalTable("preprocessingComparisonTable", "选择任一预处理模式完成双分支初筛后，此处显示冻结的预处理对比结果。");
  el("downstreamGateHint").hidden = true;
  el("trendVar1").innerHTML = "";
  el("trendVar2").innerHTML = "";
  el("trendVar3").innerHTML = "";
  el("trendVar4").innerHTML = "";
  el("trendVar5").innerHTML = "";
  el("trendVar6").innerHTML = "";
  el("trendVar7").innerHTML = "";
  el("trendVar8").innerHTML = "";
  ["scatterX1", "scatterX2", "scatterX3", "scatterY1", "scatterY2", "scatterY3"].forEach((id) => { if (el(id)) el(id).value = ""; });
  for (const select of document.querySelectorAll(".variable-select-native")) {
    const options = document.querySelector(`[data-select-options-for="${select.id}"]`);
    if (options) options.innerHTML = "";
    syncSearchableSelect(select);
  }
  el("trendStart").value = "";
  el("trendEnd").value = "";
  el("trendMaxPoints").value = "10000";
  updateTrendSelectionInfo();
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
  el("controlReferenceTable").className = "empty";
  el("controlReferenceTable").textContent = "当前未识别到控制或负荷参考变量。";
  el("screeningQualityHints").className = "empty";
  el("screeningQualityHints").textContent = "完成主筛查后显示结果质量提示。";
  el("table").className = "empty";
  el("table").textContent = "上传数据并点击“开始分析”后显示结果。";
  el("trendChart").className = "chart empty";
  el("trendChart").textContent = "选择 1 到 4 个数据后点击“显示趋势”。";
  el("trendReviewHint").textContent = "点击可信度审查摘要中的“查看趋势”后显示候选变量审查提示。";
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
  el("modelDiscoveredTable").textContent = "运行随机森林模型解释后显示遗漏探索线索。";
  el("verificationReviewPoolTable").className = "empty";
  el("verificationReviewPoolTable").textContent = "完成主筛查后显示二级验证复核池。";
  el("manualReviewPoolVariable").value = "";
  el("modelDiscoveryReviewPoolVariable").innerHTML = '<option value="">请选择模型探索变量</option>';
  el("addManualReviewPool").disabled = true;
  el("addModelDiscoveryReviewPool").disabled = true;
  el("enhancedSummaryTable").className = "empty";
  el("enhancedSummaryTable").textContent = "点击“运行增强筛选”后显示增强筛选摘要。";
  el("enhancedLiftTable").className = "empty";
  el("enhancedLiftTable").textContent = "点击“运行增强筛选”后显示模型提升评分。";
  el("enhancedRollingTable").className = "empty";
  el("enhancedRollingTable").textContent = "点击“运行增强筛选”后显示滚动稳定性评分。";
  resetOptionalTable("conditionalGrangerTable", "未运行 条件 Granger 预测验证。");
  clearOptionalElement("finalReviewQualityOverview");
  resetOptionalTable("finalReviewSummaryTable", "未运行 可信度审查摘要。");
  closeDetailModal();
  resetOptionalTable("causalReviewEvidenceTable", "未运行 逐变量可信度审查证据表。");
  resetOptionalTable("evidenceMatrixTable", "未运行 可信度审查，暂无人工复核证据矩阵。");
  resetOptionalTable("xgbModelSummaryTable", "未运行 XGB 时间外预测验证。");
  resetOptionalTable("xgbCandidateUpliftTable", "未运行 XGB 时间外预测验证。");
  clearXgbCandidateFoldDetails();
  clearOptionalElement("xgbRunSummary");
  clearOptionalElement("xgbModelSummaryDownload");
  clearOptionalElement("xgbFoldMetricsDownload");
  clearOptionalElement("xgbCandidateUpliftDownload");
  clearOptionalElement("xgbCandidateFoldMetricsDownload");
  clearOptionalElement("xgbValidationSummaryDownload");
  clearOptionalElement("xgbPredictionsDownload");
  el("xgbStatus").textContent = "XGB 时间外预测验证未启用。";
  el("xgbTopN").value = "8";
  el("xgbMaxLag").value = "";
  el("xgbWhitelist").value = "";
  clearOptionalElement("conditionalDownload");
  clearOptionalElement("finalReviewSummaryDownload");
  clearOptionalElement("causalEvidenceDownload");
  clearOptionalElement("evidenceMatrixDownload");
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
