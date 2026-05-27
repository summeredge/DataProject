from __future__ import annotations

import argparse
from pathlib import Path

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import run_analysis


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
    analyze.add_argument("--top-k", type=int, default=30)
    analyze.add_argument(
        "--preprocess-mode",
        choices=["raw", "detrend", "diff", "detrend_diff"],
        default="raw",
        help="preprocess before correlation",
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
    analyze.add_argument("--enable-granger", action="store_true", help="run optional Granger tests")
    analyze.add_argument("--enable-model", action="store_true", help="run optional model explanation")
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

    serve = subparsers.add_parser("serve", help="run the local web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true")
    return parser


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
            detrend_window=args.detrend_window,
            segment_column=args.segment_column,
            segment_mode=args.segment_mode,
            segment_min=args.segment_min,
            segment_max=args.segment_max,
            capacity_columns=[item for item in args.capacity_columns.split(",") if item],
            residual_control_columns=[item for item in args.residual_control_columns.split(",") if item] or [item for item in args.capacity_columns.split(",") if item],
            force_include_variables=[item for item in args.force_include_variables.split(",") if item],
            roles_path=args.roles_path,
            enable_granger=args.enable_granger,
            enable_model=args.enable_model,
            granger_maxlag=args.granger_maxlag,
            max_model_features=args.max_model_features,
            max_interpolate_gap_points=args.max_interpolate_gap_points,
            interpolate_limit_area=args.interpolate_limit_area,
            exclude_control_columns_from_candidates=args.exclude_control_columns_from_candidates,
        )
        run_analysis(config)
    elif args.command == "serve":
        from chem_ts_corr.web import run_server

        run_server(args.host, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
