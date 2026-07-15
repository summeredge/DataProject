from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr.xgb_runner import XGBRegressor, run_xgb_validation
from chem_ts_corr.xgb_validation import (
    DEFAULT_OUTER_SPLITS,
    DEFAULT_XGB_TOP_N,
    MAX_XGB_AUTO_TOP_N,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark XGB fourth-level validation.")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--variables", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=DEFAULT_XGB_TOP_N)
    parser.add_argument("--max-lag", type=int, default=360)
    return parser


def _synthetic_inputs(
    rows: int, variables: int, candidates: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(42)
    target = np.zeros(rows, dtype=float)
    noise = rng.normal(0, 1, rows)
    for index in range(1, rows):
        target[index] = 0.8 * target[index - 1] + noise[index]

    data: dict[str, np.ndarray] = {"target": target}
    for number in range(variables):
        lag = number % 5 + 1
        signal = np.roll(target, lag)
        signal[:lag] = signal[lag]
        data[f"x{number}"] = signal + rng.normal(0, 1 + number / variables, rows)
    frame = pd.DataFrame(
        data,
        index=pd.date_range("2025-01-01", periods=rows, freq="min"),
    )
    control_columns = ["x0"]
    candidate_variables = [f"x{number}" for number in range(1, candidates + 1)]
    final_review = pd.DataFrame(
        [
            {
                "final_rank": order,
                "variable": variable,
                "final_recommendation": "priority_review",
                "screening_lag": order % 5 + 1,
            }
            for order, variable in enumerate(candidate_variables, 1)
        ]
    )
    ranked = pd.DataFrame(
        [
            {
                "variable": variable,
                "lag": order % 5 + 1,
                "candidate_class": "upstream_driver_candidate",
                "risk_flags": "",
                "recommended_use": "strong_screening_candidate",
            }
            for order, variable in enumerate(candidate_variables, 1)
        ]
    )
    return frame, final_review, ranked, control_columns


def main() -> int:
    args = _parser().parse_args()
    if XGBRegressor is None:
        print('xgboost is not installed. Install with: pip install -e ".[xgb]"', file=sys.stderr)
        return 2
    if args.rows < 1 or args.variables < 2:
        print("rows must be positive and variables must be at least 2", file=sys.stderr)
        return 2
    if not 1 <= args.candidates <= min(MAX_XGB_AUTO_TOP_N, args.variables - 1):
        print("candidates must be between 1 and 10 and less than variables", file=sys.stderr)
        return 2

    frame, final_review, ranked, controls = _synthetic_inputs(
        args.rows, args.variables, args.candidates
    )
    tracemalloc.start()
    started_at = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="xgb-benchmark-") as temp_dir:
        result = run_xgb_validation(
            run_dir=Path(temp_dir),
            target="target",
            data=frame,
            final_review_summary=final_review,
            ranked_features=ranked,
            control_columns=controls,
            top_n=args.candidates,
            max_lag=args.max_lag,
        )
        summary_path = Path(temp_dir) / "xgb_validation" / "xgb_validation_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
    elapsed = time.perf_counter() - started_at
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fold_count = int(summary.get("fold_count", DEFAULT_OUTER_SPLITS))
    candidate_count = int(summary.get("candidate_count", 0))
    payload = {
        "rows": args.rows,
        "variables": args.variables,
        "candidates": args.candidates,
        "folds": fold_count,
        "status": result.status,
        "elapsed_seconds": round(elapsed, 6),
        "fit_count": (3 + candidate_count) * fold_count if result.status == "success" else 0,
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 3),
    }
    if result.error_message:
        payload["error_message"] = result.error_message
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
