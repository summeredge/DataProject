from __future__ import annotations

import argparse
from dataclasses import asdict
import json
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

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.data import EXCEL_SUFFIXES, TEXT_SUFFIXES, load_timeseries_csv, read_timeseries_table
from chem_ts_corr.causality import run_granger_tests
from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.modeling import fit_explainable_model
from chem_ts_corr.model_discovery import build_model_discovered_candidates, build_model_variable_importance
from chem_ts_corr.pipeline import run_analysis


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
    "causal_review_evidence.csv",
    "enhanced_validation_summary.csv",
}
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


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
    with TASKS_LOCK:
        TASKS[task_id] = {
            "status": "running",
            "message": "等待后台分析启动",
            "run_id": run_id,
            "start_time": time.time(),
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

        run_analysis(config, progress_callback=progress)
        with TASKS_LOCK:
            TASKS[task_id].update(
                {
                    "status": "done",
                    "message": "分析完成",
                    "end_time": time.time(),
                    "result": _build_result_payload(config.output_dir.name, config.output_dir, config),
                }
            )
    except Exception as exc:
        with TASKS_LOCK:
            TASKS[task_id].update({"status": "error", "message": str(exc), "error": str(exc), "end_time": time.time()})


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
    conditional.to_csv(output_dir / "conditional_granger_scores.csv", index=False, encoding="utf-8-sig")
    report.to_csv(output_dir / "causal_review_report.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(output_dir / "causal_review_evidence.csv", index=False, encoding="utf-8-sig")
    return {
        "conditionalGrangerScores": _records(conditional.head(500)),
        "causalReviewReport": _records(report.head(500)),
        "causalReviewEvidence": _records(evidence.head(500)),
        "downloads": _download_links(run_id, output_dir),
        "message": "三层复核完成：结果仅为预测验证/人工复核建议，不是因果结论。",
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
    return standardize_frame(transformed)


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
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
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
    matches = sorted(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise FileNotFoundError("上传文件不存在，请重新上传")
    path = matches[0].resolve()
    if UPLOADS_DIR.resolve() not in path.parents:
        raise ValueError("Invalid upload path")
    if path.suffix.lower() not in TEXT_SUFFIXES | EXCEL_SUFFIXES:
        raise ValueError("Unsupported upload file type")
    return path


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
    return int(value) if value else default


def _float_field(form: dict[str, Any], name: str, default: float) -> float:
    value = _field(form, name, "")
    return float(value) if value else default


def _optional_float_field(form: dict[str, Any], name: str) -> float | None:
    value = _field(form, name, "")
    return float(value) if value else None


def _optional_int_field(form: dict[str, Any], name: str) -> int | None:
    value = _field(form, name, "")
    return int(value) if value else None


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
    :root { --bg:#f6f7f9; --panel:#fff; --line:#d9dee7; --text:#16202a; --muted:#5f6b7a; --accent:#176b87; --green:#0f5132; --warn:#8a5a00; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; color:var(--text); background:var(--bg); }
    header { padding:22px 28px 14px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0 0 8px; font-size:24px; }
    .subtitle { color:var(--muted); font-size:14px; }
    main { display:grid; grid-template-columns:minmax(320px,430px) 1fr; gap:18px; padding:18px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    .controls { display:grid; gap:8px; align-content:start; font-size:80%; }
    label { display:grid; gap:3px; font-size:10px; line-height:1.2; color:var(--muted); }
    input, select { width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:6px; color:var(--text); background:#fff; font-size:11px; line-height:1.2; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .check { display:flex; align-items:center; gap:8px; color:var(--text); font-size:14px; }
    .check input { width:auto; }
    .actions { display:flex; gap:10px; flex-wrap:wrap; }
    .multi-dropdown { border:1px solid var(--line); border-radius:6px; background:#fff; }
    .multi-dropdown > summary { list-style:none; cursor:pointer; padding:6px 8px; font-size:11px; text-align:left; }
    .multi-dropdown > summary::-webkit-details-marker { display:none; }
    .multi-options { max-height:180px; min-width:260px; overflow:auto; border-top:1px solid var(--line); padding:6px 8px; display:grid; gap:4px; }
    .multi-options label { display:grid; grid-template-columns:16px 1fr; align-items:center; column-gap:8px; font-size:11px; color:var(--text); text-align:left; line-height:1.2; }
    .multi-options input[type="checkbox"] { margin:0; }
    .multi-options span { display:block; text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    button { border:0; border-radius:6px; padding:10px 14px; font-weight:650; cursor:pointer; background:var(--accent); color:#fff; }
    button.secondary { background:#e8edf3; color:var(--text); }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .status { min-height:22px; color:var(--muted); font-size:13px; white-space:pre-wrap; }
    .note { color:var(--warn); font-size:13px; line-height:1.5; }
    .results { display:grid; gap:16px; min-width:0; align-content:start; position:relative; }
    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    h2 { margin:0; font-size:18px; }
    .download-buttons { display:flex; gap:8px; flex-wrap:wrap; }
    .download-buttons a { display:inline-block; border-radius:6px; padding:8px 10px; background:var(--green); color:#fff; text-decoration:none; font-size:13px; }
    .help { display:grid; gap:6px; color:var(--muted); font-size:13px; line-height:1.5; background:#f8fafc; border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
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
      background:#f8fafc;
    }
    .metric-value { display:block; font-size:20px; line-height:1.15; font-weight:700; color:var(--text); }
    .metric-label { color:var(--muted); font-size:12px; line-height:1.25; }
    .chart { min-height:280px; border:1px solid var(--line); border-radius:6px; background:#fff; overflow:hidden; }
    .chart svg { width:100%; height:320px; display:block; }
    .chart-controls { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)) 150px auto; gap:10px; align-items:end; }
    .trend-options { display:grid; grid-template-columns:repeat(3,minmax(160px,1fr)); gap:10px; align-items:end; }
    .legend { display:flex; justify-content:center; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
    .swatch { width:18px; height:3px; border-radius:2px; display:inline-block; vertical-align:middle; margin-right:6px; }
    .table-wrap { overflow:auto; max-height:560px; width:100%; min-width:320px; max-width:100%; resize:horizontal; border:1px solid var(--line); border-radius:6px; }
    .table-wrap::after { content:"拖动右下角可调整表格宽度"; display:block; padding:4px 8px; color:var(--muted); font-size:11px; background:#f8fafc; border-top:1px solid var(--line); }
    table { width:max-content; min-width:100%; border-collapse:collapse; font-size:13px; }
    th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
    th { position:sticky; top:0; background:#eef2f6; z-index:1; }
    th.sortable { cursor:pointer; user-select:none; }
    th.sortable:hover { background:#dde6ef; }
    th .sort-mark { color:var(--muted); margin-left:6px; font-size:11px; }
    .decision-badge { display:inline-block; border-radius:999px; padding:3px 8px; font-weight:700; font-size:12px; }
    .decision-risk_limited_review { background:#fef3c7; color:#92400e; }
    .decision-priority_review { background:#fee2e2; color:#991b1b; }
    .decision-secondary_review { background:#ffedd5; color:#9a3412; }
    .decision-not_recommended { background:#e5e7eb; color:#374151; }
    .decision-insufficient_evidence { background:#dbeafe; color:#1e40af; }
    .decision-manual_review_only { background:#7f1d1d; color:#fff; }
    .empty { color:var(--muted); padding:24px; text-align:center; border:1px dashed var(--line); border-radius:6px; }
    pre { margin:0; padding:12px; background:#f8fafc; border:1px solid var(--line); border-radius:6px; max-height:260px; overflow:auto; white-space:pre-wrap; font-size:12px; }
    @media (max-width:900px) { main { grid-template-columns:1fr; padding:12px; } .row { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>化工装置时序相关性分析</h1>
    <div class="subtitle">浏览器负责上传和展示，Python 后台处理大数据并生成下载结果。</div>
  </header>
  <main>
    <section class="controls">
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
      <div id="status" class="status"></div>
      <div class="note">大文件会由 Python 后台处理。分析期间请不要关闭启动服务的命令窗口。</div>
    </section>

    <section class="results">
      <div class="tabs">
        <button class="tab-button active" data-tab="overviewTab">总览</button>
        <button class="tab-button" data-tab="candidatesTab">候选变量</button>
        <button class="tab-button" data-tab="trendTab">趋势图</button>
        <button class="tab-button" data-tab="validationTab">二次验证</button>
        <button class="tab-button" data-tab="causalReviewTab">三层复核</button>
        <button class="tab-button" data-tab="downloadsTab">下载</button>
      </div>

      <div id="overviewTab" class="tab-panel active">
        <h2>总览</h2>
        <div id="overview" class="overview-grid"></div>
        <h2>前 10 个推荐变量</h2>
        <div id="overviewTop" class="empty">上传数据并点击“开始分析”后显示结果。</div>
      </div>

      <div id="candidatesTab" class="tab-panel">
        <h2>候选变量</h2>
        <div class="help">默认只展示候选排序结果的核心列和前 50 行，完整结果请到下载页获取。</div>
        <div id="table" class="empty">上传数据并点击“开始分析”后显示结果。</div>
        <h2>轻量遗漏候选</h2>
        <div class="help">该表基于已有滞后相关、残差相关、峰值质量和风险标签生成，用于提示主筛查前 K 个外可能遗漏的候选。结果不代表因果结论。</div>
        <div id="nearMissTable" class="empty">完成主筛查后显示轻量遗漏候选。</div>
      </div>

      <div id="trendTab" class="tab-panel">
        <h2>趋势图</h2>
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
      </div>

      <div id="validationTab" class="tab-panel">
        <h2>二次验证</h2>
        <div class="help">先完成主筛查，再按需运行增强筛选、Granger 预测验证或随机森林模型解释。结果会同步写入下载文件。</div>
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

      <div id="causalReviewTab" class="tab-panel">
        <h2>三层复核</h2>
        <div class="help">所有结果仅作为“预测验证/人工复核建议”，不是因果结论。可在左侧设置前 N 个候选变量和风险标签包含过滤后运行。</div>
        <div class="help">三层复核支持长滞后变量。默认围绕主筛查最佳滞后附近做条件 Granger 验证，避免对 1..maxlag 全量扫描造成计算过慢。如需完整扫描，可切换为 full_scan。baseline 最大滞后用于控制目标自身惯性和控制列影响；候选变量仍可验证主筛查发现的长滞后。若需要完全复现旧逻辑，可将 baseline 最大滞后设置为与 maxlag 相同，并选择 full_scan。</div>
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
        <h2>三层复核报告</h2>
        <div class="download-buttons" id="causalReportDownload"></div>
        <div id="causalReviewTable" class="empty">未运行 三层复核。</div>
        <h2>综合证据复核</h2>
        <div class="help">综合证据复核会整合已生成的增强筛选、Granger 和随机森林模型解释结果；如果这些结果尚未运行，则对应证据为空。该表仍不是因果结论。</div>
        <div class="download-buttons" id="causalEvidenceDownload"></div>
        <div id="causalReviewEvidenceTable" class="empty">未运行 综合证据复核。</div>
      </div>

      <div id="downloadsTab" class="tab-panel">
        <h2>下载</h2>
        <div id="downloads" class="download-buttons"></div>
      </div>
    </section>
  </main>

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
let sortState = { column: "score", direction: "desc" };
const el = (id) => document.getElementById(id);
const trendColors = ["#176b87", "#c2410c", "#6d28d9", "#15803d"];

for (const button of document.querySelectorAll(".tab-button")) {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
}
el("drawTrend").addEventListener("click", drawTrend);
el("runEnhancedScreening").addEventListener("click", runEnhancedScreening);
el("runGranger").addEventListener("click", runGranger);
el("runModel").addEventListener("click", runModel);
el("runCausalReview").addEventListener("click", runCausalReview);

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
    setStatus("正在上传文件...");
    const form = new FormData();
    form.append("file", file);
    const data = await postForm("/api/upload", form);
    fileId = data.file_id;
    setStatus(`已上传：${data.filename}\n正在识别列...`);
    await loadColumns();
  } catch (error) {
    setStatus(error.message || String(error));
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
    setStatus(`列识别完成。编码：${data.encoding}。采样读取 ${data.sampleRows} 行，识别到 ${data.columns.length} 列。`);
  } catch (error) {
    el("analyze").disabled = true;
    setStatus(error.message || String(error));
  }
}

async function analyze() {
  if (!fileId) return setStatus("请先上传文件。");
  setStatus("Python 后台正在分析，数据较大时请等待...");
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
    setStatus(`分析完成。运行 ID：${result.run_id}`);
  } catch (error) {
    setStatus(error.message || String(error));
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
    setStatus(formatTaskStatus(statusData));
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
  renderOverview(data.overview || {});
  renderTable(applySort(lastRows));
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
  renderCausalReviewTable("causalReviewTable", lastCausalReportRows);
  renderGenericTable("causalReviewEvidenceTable", lastCausalEvidenceRows, causalReviewEvidenceColumns());
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

async function runEnhancedScreening() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  setStatus("正在运行增强筛选：补充验证预测增益和时间稳定性...");
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
    setStatus(data.message || "增强筛选完成。结果不代表因果结论。");
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    el("runEnhancedScreening").disabled = !currentRunId;
  }
}

async function runGranger() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  setStatus("正在运行 Granger 二级验证...");
  el("runGranger").disabled = true;
  try {
    const form = new FormData();
    form.append("run_id", currentRunId);
    const data = await postForm("/api/run_granger", form);
    lastGrangerRows = data.grangerTests || [];
    renderGenericTable("grangerTable", lastGrangerRows);
    renderDownloads(data.downloads || []);
    setStatus("Granger 二级验证完成。");
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    el("runGranger").disabled = !currentRunId;
  }
}

async function runModel() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  setStatus("正在运行随机森林模型解释...");
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
    setStatus(`随机森林模型解释完成。${metrics}`);
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    el("runModel").disabled = !currentRunId;
    el("runCausalReview").disabled = !currentRunId;
  }
}

async function runCausalReview() {
  if (!currentRunId) return setStatus("请先完成主筛查。");
  setStatus("正在运行三层复核：结果仅为预测验证/人工复核建议，不是因果结论...");
  el("runCausalReview").disabled = true;
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
    renderGenericTable("conditionalGrangerTable", lastConditionalRows, conditionalGrangerColumns());
    renderCausalReviewTable("causalReviewTable", lastCausalReportRows);
    renderGenericTable("causalReviewEvidenceTable", lastCausalEvidenceRows, causalReviewEvidenceColumns());
    renderReviewDownloads(data.downloads || []);
    renderDownloads(data.downloads || []);
    setStatus(data.message || "三层复核完成。结果不是因果结论。");
  } catch (error) {
    setStatus(error.message || String(error));
  } finally {
    el("runCausalReview").disabled = !currentRunId;
  }
}

function activateTab(tabId) {
  for (const button of document.querySelectorAll(".tab-button")) {
    button.classList.toggle("active", button.dataset.tab === tabId);
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    panel.classList.toggle("active", panel.id === tabId);
  }
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
    setStatus("正在生成趋势图...");
    const response = await fetch(`/api/trend?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "趋势图生成失败");
    renderTrendChart(data.series || [], el("trendAxisMode").value);
    setStatus(`趋势图已生成，原始 ${data.raw_rows} 点，显示 ${data.rows} 点，最大点数 ${data.max_points}。`);
  } catch (error) {
    el("trendChart").className = "chart empty";
    el("trendChart").textContent = error.message || String(error);
    el("trendLegend").innerHTML = "";
  }
}

function renderTrendChart(series, axisMode) {
  const container = el("trendChart");
  if (!series.length) {
    container.className = "chart empty";
    container.textContent = "没有可绘制的趋势数据。";
    return;
  }
  const width = 960, height = 320, pad = { left: 58, right: axisMode === "independent" ? 58 : 20, top: 24, bottom: 44 };
  const maxLen = Math.max(...series.map((item) => item.points.length));
  const allValues = series.flatMap((item) => item.points.map((point) => point.y).filter((value) => value !== null));
  const sharedRange = valueRange(allValues);
  const ranges = series.map((item) => axisMode === "shared" ? sharedRange : valueRange(item.points.map((point) => point.y).filter((value) => value !== null)));
  const x = (index) => pad.left + (index / Math.max(1, maxLen - 1)) * (width - pad.left - pad.right);
  const y = (value, range) => pad.top + (1 - (value - range.min) / Math.max(1e-12, range.max - range.min)) * (height - pad.top - pad.bottom);
  const paths = series.map((item, idx) => {
    const points = item.points.map((point, index) => point.y === null ? null : `${x(index).toFixed(2)},${y(point.y, ranges[idx]).toFixed(2)}`).filter(Boolean).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${trendColors[idx % trendColors.length]}" stroke-width="2.2"/>`;
  }).join("");
  container.className = "chart";
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="#fff"/>
    <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>
    <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>
    ${axisMode === "independent" ? `<line x1="${width - pad.right}" y1="${pad.top}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#9aa4b2"/>` : ""}
    <text x="${pad.left}" y="18" font-size="12" fill="#5f6b7a">${axisMode === "independent" ? "独立 Y 轴：每条曲线按自身范围缩放" : "同一 Y 轴：所有曲线使用同一数值范围"}</text>
    ${paths}
  </svg>`;
  el("trendLegend").innerHTML = series.map((item, idx) =>
    `<span><i class="swatch" style="background:${trendColors[idx % trendColors.length]}"></i>${escapeHtml(item.name)}</span>`
  ).join("");
}

function valueRange(values) {
  if (!values.length) return { min: 0, max: 1 };
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const margin = (max - min) * 0.08;
  return { min: min - margin, max: max + margin };
}

function renderTable(rows) {
  if (!rows.length) {
    el("table").className = "empty";
    el("table").textContent = "没有可展示的候选变量。";
    return;
  }
  const columns = coreCandidateColumns();
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${columns.map((c) => {
    const mark = sortState.column === c ? (sortState.direction === "asc" ? "↑" : "↓") : "";
    return `<th class="sortable" data-column="${escapeHtml(c)}">${escapeHtml(columnLabel(c))}<span class="sort-mark">${mark}</span></th>`;
  }).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const row of rows) {
    body.innerHTML += `<tr>${columns.map((c) => `<td>${escapeHtml(formatValue(row[c]))}</td>`).join("")}</tr>`;
  }
  table.appendChild(body);
  for (const header of table.querySelectorAll("th.sortable")) {
    header.addEventListener("click", () => sortByColumn(header.dataset.column));
  }
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  el("table").className = "";
  el("table").replaceChildren(wrap);
}

function coreCandidateColumns() {
  return ["variable", "final_score", "lag", "direction", "raw_corr", "residual_corr", "risk_flags", "recommended_use", "recommended_action"];
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

function renderGenericTable(targetId, rows, preferredColumns = null) {
  const container = el(targetId);
  if (!rows.length) {
    container.className = "empty";
    container.textContent = missingText(targetId);
    return;
  }
  const columns = (preferredColumns || Object.keys(rows[0])).filter((column) => column in rows[0]);
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${columns.map((c) => `<th>${escapeHtml(columnLabel(c))}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const row of rows) {
    body.innerHTML += `<tr>${columns.map((c) => `<td>${escapeHtml(formatValue(row[c]))}</td>`).join("")}</tr>`;
  }
  table.appendChild(body);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
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
  if (targetId === "causalReviewEvidenceTable") return "未运行 综合证据复核。";
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

function causalReviewEvidenceColumns() {
  return ["variable", "candidate_grade", "final_score", "evidence_score", "evidence_level", "risk_constraint_level", "integrated_review_decision", "integrated_review_reason", "evidence_reason", "conditional_granger_status", "conditional_fdr_q_value", "predictive_contribution", "model_lift", "rolling_stability", "model_importance_rank", "risk_flags", "interpretation"];
}

function renderCausalReviewTable(targetId, rows) {
  const container = el(targetId);
  if (!rows.length) {
    container.className = "empty";
    container.textContent = missingText(targetId);
    return;
  }
  const columns = causalReviewColumns().filter((column) => column in rows[0]);
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>${columns.map((c) => `<th>${escapeHtml(columnLabel(c))}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  for (const row of rows) {
    body.innerHTML += `<tr>${columns.map((c) => `<td>${formatReviewCell(c, row[c])}</td>`).join("")}</tr>`;
  }
  table.appendChild(body);
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  wrap.appendChild(table);
  container.className = "";
  container.replaceChildren(wrap);
}

function formatReviewCell(column, value) {
  if (column === "final_review_decision") {
    const raw = String(value || "");
    return `<span class="decision-badge decision-${escapeHtml(raw)}">${escapeHtml(formatValue(raw))}</span>`;
  }
  return escapeHtml(formatValue(value));
}

function renderReviewDownloads(downloads) {
  renderDownloadTarget("conditionalDownload", downloads, "conditional_granger_scores.csv");
  renderDownloadTarget("causalReportDownload", downloads, "causal_review_report.csv");
  renderDownloadTarget("causalEvidenceDownload", downloads, "causal_review_evidence.csv");
}

function renderDownloadTarget(targetId, downloads, fileName) {
  const container = el(targetId);
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

function sortByColumn(column) {
  if (sortState.column === column) {
    sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
  } else {
    sortState = { column, direction: "asc" };
  }
  renderTable(applySort(lastRows));
}

function applySort(rows) {
  const direction = sortState.direction === "asc" ? 1 : -1;
  const column = sortState.column;
  return rows.slice().sort((a, b) => compareValues(a[column], b[column]) * direction);
}

function compareValues(a, b) {
  const numberA = typeof a === "number" ? a : Number(a);
  const numberB = typeof b === "number" ? b : Number(b);
  if (Number.isFinite(numberA) && Number.isFinite(numberB)) return numberA - numberB;
  return String(a ?? "").localeCompare(String(b ?? ""), "zh-CN", { numeric: true });
}

function formatValue(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "";
    if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(3);
    return Number.isInteger(value) ? String(value) : value.toFixed(6);
  }
  if (typeof value === "string") {
    const map = {
      unstable_over_time: "时序不稳定",
      low_model_lift: "低模型增益",
      lag_boundary: "滞后边界命中",
      lag_boundary_flag: "滞后边界命中",
      formula_coupled_reference: "公式耦合参考",
      strong_screening_candidate: "强初筛候选",
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
      target_leads_variable: "目标领先变量",
      unstable_across_regimes: "跨工况不稳定",
      poor_data_quality: "数据质量差",
      residual_collinearity: "残差共线性高",
      none: "无",
      weak: "弱",
      medium: "中",
      strong: "强",
      ok: "正常",
      skipped: "已跳过",
      risk_limited_review: "风险受限复核",
      priority_review: "优先复核",
      secondary_review: "二级复核",
      not_recommended: "暂不推荐",
      insufficient_evidence: "证据不足",
      manual_review_only: "仅人工复核",
      candidate_leads_target: "变量领先目标",
      target_leads_candidate: "目标领先变量",
      target_leads_variable: "目标领先变量",
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
      target_lead_risk: "目标领先风险",
      stability_risk: "稳定性风险",
      model_supported_screening_candidate: "模型支持的初筛候选",
      raw_lag_signal: "滞后相关线索",
      residual_signal: "残差相关线索",
      clear_lag_peak: "滞后峰值清晰",
      lag_boundary_risk: "滞后边界风险",
      data_or_formula_risk: "数据质量或公式泄漏风险",
      near_miss_candidate: "遗漏候选线索",
      "screening near-miss only": "仅作轻量遗漏筛查",
    };
    if (map[value]) return map[value];
    if (value === "predictive validation only; not a causal conclusion") return "仅作预测验证；不是因果结论";
    if (value === "model explanation only; not a causal conclusion") return "仅作模型解释；不是因果结论";
    if (value === "screening near-miss only; not a causal conclusion") return "仅作轻量遗漏筛查；不是因果结论";
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
    final_score: "综合得分",
    lag: "滞后",
    direction: "方向",
    raw_corr: "原始相关",
    residual_corr: "残差相关",
    risk_flags: "风险标签",
    recommended_use: "建议用途",
    recommended_action: "建议动作",
    formula_like_flag: "公式类变量",
    strong_formula_leakage_flag: "强公式泄漏",
    common_capacity_driver_flag: "疑似共同负荷驱动",
    closed_loop_suspect_flag: "疑似闭环反馈",
    target_leads_variable_flag: "目标领先变量",
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
    residual_p_value: "残差P值",
    residual_r2: "残差R²",
    regime: "工况",
    regime_stability: "工况稳定性",
    regime_sign_consistency: "符号一致性",
    regime_lag_consistency: "滞后一致性",
    regime_count: "工况数量",
    rolling_stability: "滚动稳定性",
    rolling_corr_median: "滚动相关中位数",
    rolling_sign_consistency: "滚动符号一致性",
    model_lift: "模型提升",
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
    fallback_maxlag: "Fallback最大滞后",
    baseline_maxlag: "基准滞后上限",
    interpretation: "解释边界",
    candidate_grade: "候选等级",
    review_tier: "复核层级",
    review_priority: "复核优先级",
    review_reason: "复核原因",
    final_review_decision: "最终复核建议",
    final_review_reason: "最终复核原因",
    conditional_granger_status: "条件Granger状态",
    conditional_best_lag: "条件最佳滞后",
    conditional_min_p_value: "条件最小P值",
    conditional_fdr_q_value: "条件FDR Q值",
    evidence_score: "证据得分",
    evidence_level: "证据等级",
    evidence_reason: "证据说明",
    risk_constraint_level: "风险约束等级",
    integrated_review_decision: "综合复核建议",
    integrated_review_reason: "综合复核原因",
    model_importance_rank: "模型重要性排名",
    model_explanation_support: "模型解释支持",
    causalReviewEvidence: "综合证据复核",
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

function setStatus(message) { el("status").textContent = message; }

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
  sortState = { column: "score", direction: "desc" };
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
  el("overview").innerHTML = "";
  el("overviewTop").className = "empty";
  el("overviewTop").textContent = "上传数据并点击“开始分析”后显示结果。";
  el("table").className = "empty";
  el("table").textContent = "上传数据并点击“开始分析”后显示结果。";
  el("nearMissTable").className = "empty";
  el("nearMissTable").textContent = "完成主筛查后显示轻量遗漏候选。";
  el("trendChart").className = "chart empty";
  el("trendChart").textContent = "选择 1 到 4 个数据后点击“显示趋势”。";
  el("trendLegend").innerHTML = "";
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
  el("conditionalGrangerTable").className = "empty";
  el("conditionalGrangerTable").textContent = "未运行 条件 Granger 预测验证。";
  el("causalReviewTable").className = "empty";
  el("causalReviewTable").textContent = "未运行 三层复核。";
  el("causalReviewEvidenceTable").className = "empty";
  el("causalReviewEvidenceTable").textContent = "未运行 综合证据复核。";
  el("conditionalDownload").innerHTML = "";
  el("causalReportDownload").innerHTML = "";
  el("causalEvidenceDownload").innerHTML = "";
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
