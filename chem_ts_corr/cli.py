from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import (
    confirm_initial_screening_branch,
    run_causal_review_for_active_branch,
    run_enhanced_screening_for_active_branch,
    run_granger_for_active_branch,
    run_initial_screening_workflow,
    run_model_for_active_branch,
    run_xgb_for_active_branch,
)


FORMAL_PREPROCESS_MODES = ["raw", "lowpass", "lowpass_detrend", "lowpass_diff"]
PROCESSED_PREPROCESS_MODES = ["lowpass", "lowpass_detrend", "lowpass_diff"]
SCREENING_BRANCHES = ["raw", "processed"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chem-ts-corr",
        description="Analyze correlations and lag relationships in industrial time-series data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="run an analysis")
    analyze.add_argument("--input", required=True, type=Path, help="input CSV path")
    analyze.add_argument("--time-column", required=True, help="timestamp column name")
    analyze.add_argument("--target", required=True, help="target variable column name")
    analyze.add_argument("--output", required=True, type=Path, help="output folder")
    analyze.add_argument("--encoding", default="utf-8-sig", help="CSV encoding, e.g. utf-8-sig or gb18030")
    analyze.add_argument("--max-lag", type=int, default=12, help="max lag points to search")
    analyze.add_argument("--resample-rule", default=None, help="optional pandas resample rule, e.g. 5min")
    analyze.add_argument("--min-valid-ratio", type=float, default=0.7)
    analyze.add_argument("--top-k", type=int, default=20)
    analyze.add_argument(
        "--preprocess-mode",
        choices=FORMAL_PREPROCESS_MODES,
        default="raw",
        help="preprocess before correlation",
    )
    analyze.add_argument(
        "--lowpass-tau-minutes",
        type=float,
        default=5.0,
        help="lowpass time constant in minutes (lowpass* modes only)",
    )
    analyze.add_argument(
        "--diff-interval-minutes",
        type=float,
        default=None,
        help="diff interval in minutes (lowpass_diff only); empty/None means one analysis sampling period",
    )
    analyze.add_argument("--detrend-window", type=int, default=24)
    analyze.add_argument("--segment-column", default=None, help="load column for operating segmentation")
    analyze.add_argument(
        "--capacity-columns",
        default="",
        help="backward-compatible alias of residual control columns",
    )
    analyze.add_argument(
        "--residual-control-columns",
        default="",
        help="comma-separated control columns used for residual correlation control",
    )
    analyze.add_argument(
        "--force-include-variables",
        default="",
        help="comma-separated variables to force include in rolling stability review",
    )
    analyze.add_argument("--roles", dest="roles_path", type=Path, default=None, help="optional CSV with columns: variable,role")
    analyze.add_argument(
        "--segment-mode",
        choices=["all", "low", "mid", "high", "custom"],
        default="all",
    )
    analyze.add_argument("--segment-min", type=float, default=None)
    analyze.add_argument("--segment-max", type=float, default=None)
    analyze.add_argument("--enable-granger", action="store_true", help="run optional Granger tests after formal screening")
    analyze.add_argument("--enable-model", action="store_true", help="run optional model explanation after formal screening")
    analyze.add_argument("--granger-maxlag", type=int, default=None)
    analyze.add_argument("--max-model-features", type=int, default=300)
    analyze.add_argument("--max-interpolate-gap-points", type=int, default=5)
    analyze.add_argument("--interpolate-limit-area", choices=["inside", "outside"], default="inside")
    analyze.add_argument(
        "--exclude-control-columns-from-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude residual/capacity control columns from top-k candidates by default",
    )

    confirm = subparsers.add_parser(
        "confirm-branch", help="confirm an existing screening branch as the formal result"
    )
    confirm.add_argument("--output", required=True, type=Path, help="run directory")
    confirm.add_argument(
        "--branch",
        required=True,
        choices=SCREENING_BRANCHES,
        help="branch to confirm: raw or processed",
    )

    run_enhanced = subparsers.add_parser(
        "run-enhanced", help="run enhanced screening on the active formal branch"
    )
    run_enhanced.add_argument("--output", required=True, type=Path, help="run directory")

    run_granger = subparsers.add_parser(
        "run-granger", help="run ordinary Granger on the active formal branch"
    )
    run_granger.add_argument("--output", required=True, type=Path, help="run directory")

    run_model = subparsers.add_parser(
        "run-model", help="run RF/SHAP model explanation on the active formal branch"
    )
    run_model.add_argument("--output", required=True, type=Path, help="run directory")

    run_causal = subparsers.add_parser(
        "run-causal-review", help="run three-tier causal review on the active formal branch"
    )
    run_causal.add_argument("--output", required=True, type=Path, help="run directory")
    run_causal.add_argument(
        "--control-columns",
        default="",
        help="comma-separated control columns",
    )
    run_causal.add_argument("--maxlag", type=int, default=None)
    run_causal.add_argument("--min-rows", type=int, default=60)
    run_causal.add_argument("--top-n", type=int, default=None)

    run_xgb = subparsers.add_parser(
        "run-xgb", help="run fold-safe XGB validation on the active formal branch"
    )
    run_xgb.add_argument("--output", required=True, type=Path, help="run directory")
    run_xgb.add_argument("--control-columns", default="", help="comma-separated control columns")
    run_xgb.add_argument("--whitelist", default="", help="comma-separated whitelist variables")
    run_xgb.add_argument("--top-n", type=int, default=None)
    run_xgb.add_argument("--max-lag", type=int, default=None)

    serve = subparsers.add_parser("serve", help="run the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _write_run_config(config: AnalysisConfig) -> None:
    """Persist the run config so downstream commands can reuse it.

    The schema mirrors the Web writer: paths are stored as strings and the
    ``file_id`` key is optional (the backend reader tolerates its absence).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    data["input_path"] = str(config.input_path)
    data["output_dir"] = str(config.output_dir)
    data["roles_path"] = str(config.roles_path) if config.roles_path else None
    (config.output_dir / "run_config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analyze":
        config = AnalysisConfig(
            input_path=args.input,
            time_column=args.time_column,
            target=args.target,
            output_dir=args.output,
            encoding=args.encoding,
            max_lag=args.max_lag,
            resample_rule=args.resample_rule,
            min_valid_ratio=args.min_valid_ratio,
            top_k=args.top_k,
            preprocess_mode=args.preprocess_mode,
            lowpass_tau_minutes=args.lowpass_tau_minutes,
            diff_interval_minutes=args.diff_interval_minutes,
            detrend_window=args.detrend_window,
            segment_column=args.segment_column,
            segment_mode=args.segment_mode,
            segment_min=args.segment_min,
            segment_max=args.segment_max,
            capacity_columns=_split_csv(args.capacity_columns),
            residual_control_columns=_split_csv(args.residual_control_columns),
            force_include_variables=_split_csv(args.force_include_variables),
            roles_path=args.roles_path,
            enable_granger=args.enable_granger,
            enable_model=args.enable_model,
            granger_maxlag=args.granger_maxlag,
            max_model_features=args.max_model_features,
            max_interpolate_gap_points=args.max_interpolate_gap_points,
            interpolate_limit_area=args.interpolate_limit_area,
            exclude_control_columns_from_candidates=args.exclude_control_columns_from_candidates,
        )
        _write_run_config(config)
        result = run_initial_screening_workflow(config)
        status = result.get("branch", "")
        if args.preprocess_mode == "raw":
            print(f"正式初筛完成：branch={status}，状态 not_required")
        else:
            print(
                "双分支初筛完成：raw + "
                f"{args.preprocess_mode} 已生成 preprocessing_comparison.csv；"
                "状态 awaiting_confirmation，请先运行 confirm-branch 确认正式分支。"
            )
        if args.enable_granger or args.enable_model:
            _run_legacy_enable_flags(args)
    elif args.command == "confirm-branch":
        confirm_initial_screening_branch(args.output, branch=args.branch)
        print(f"已确认正式初筛分支：{args.branch}")
    elif args.command == "run-enhanced":
        run_enhanced_screening_for_active_branch(args.output)
        print("增强筛选完成。")
    elif args.command == "run-granger":
        run_granger_for_active_branch(args.output)
        print("普通 Granger 完成。")
    elif args.command == "run-model":
        run_model_for_active_branch(args.output)
        print("模型解释完成。")
    elif args.command == "run-causal-review":
        run_causal_review_for_active_branch(
            args.output,
            control_columns=_split_csv(args.control_columns) or None,
            maxlag=args.maxlag,
            min_rows=args.min_rows,
            top_n=args.top_n,
        )
        print("三级复核完成。")
    elif args.command == "run-xgb":
        result = run_xgb_for_active_branch(
            args.output,
            control_columns=_split_csv(args.control_columns) or None,
            whitelist=_split_csv(args.whitelist) or None,
            top_n=args.top_n,
            max_lag=args.max_lag,
        )
        if result.get("status") == "success":
            print("XGB 四级验证完成。")
        else:
            print("XGB 四级验证失败。")
            print(f"status={result.get('status')}")
            print(f"error={result.get('error_message')}")
    elif args.command == "serve":
        from chem_ts_corr.web import run_server

        run_server(args.host, args.port, open_browser=not args.no_open)


def _run_legacy_enable_flags(args: argparse.Namespace) -> None:
    """Handle legacy ``--enable-granger`` / ``--enable-model`` on ``analyze``.

    Raw workflows may run the corresponding formal runners immediately after
    promotion. Processed workflows must wait for ``confirm-branch`` because
    the formal branch is not selected yet; the CLI never picks one
    automatically.
    """
    if args.preprocess_mode != "raw":
        raise SystemExit(
            "请先 confirm-branch，再运行对应 downstream 命令。"
        )
    if args.enable_granger:
        run_granger_for_active_branch(args.output)
        print("Granger 二级验证完成。")
    if args.enable_model:
        run_model_for_active_branch(args.output)
        print("模型解释完成。")


if __name__ == "__main__":
    main()
