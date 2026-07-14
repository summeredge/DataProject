from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chem_ts_corr.causality import (  # noqa: E402
    _GrangerDiagnostics,
    _fast_granger_ssr_ftests,
    _lagged_arrays,
    _ols_ssr_and_rank,
    _restricted_mask_key,
)


CASES = {
    "well_conditioned",
    "high_collinearity",
    "different_missing_masks",
    "near_rank_deficient",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Granger full-lag scan")
    parser.add_argument("--rows", type=int, default=45_000)
    parser.add_argument("--variables", type=int, default=12)
    parser.add_argument("--maxlag", type=int, default=380)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--case", choices=sorted(CASES), default="well_conditioned")
    parser.add_argument("--compare-old", action="store_true")
    parser.add_argument("--old-timeout", type=float, default=1_800.0)
    parser.add_argument("--worker", choices=["new", "old"], help=argparse.SUPPRESS)
    return parser.parse_args()


def _synthetic_frame(rows: int, variables: int, seed: int, case: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=rows)
    for index in range(1, rows):
        target[index] += 0.25 * target[index - 1]
    data = {"Y": target}
    for number in range(variables):
        noise = rng.normal(size=rows)
        if case == "well_conditioned":
            candidate = noise
            if number % 3 == 0:
                candidate = noise + 0.15 * np.roll(target, number % 5 + 1)
        elif case == "high_collinearity":
            candidate = target + rng.normal(scale=1e-8, size=rows)
        elif case == "near_rank_deficient":
            candidate = np.linspace(-1.0, 1.0, rows) + rng.normal(scale=1e-10, size=rows)
        else:
            candidate = noise
            missing = np.arange(number % 7, rows, max(17, variables + 5))
            candidate[missing] = np.nan
        data[f"X{number + 1}"] = candidate
    return pd.DataFrame(data)


def _new_scan(frame: pd.DataFrame, variables: list[str], maxlag: int):
    diagnostics = _GrangerDiagnostics()
    restricted_cache = {}
    valid_results = 0
    started = time.perf_counter()
    for variable in variables:
        mask = frame["Y"].notna() & frame[variable].notna()
        pair = frame[["Y", variable]].dropna()
        results = _fast_granger_ssr_ftests(
            pair,
            "Y",
            variable,
            maxlag,
            diagnostics=diagnostics,
            restricted_cache=restricted_cache,
            mask_key=_restricted_mask_key("Y", mask.to_numpy(dtype=bool)),
        )
        valid_results += len(results)
    elapsed = time.perf_counter() - started
    return elapsed, diagnostics, valid_results


def _old_scan(frame: pd.DataFrame, variables: list[str], maxlag: int):
    valid_results = 0
    started = time.perf_counter()
    for variable in variables:
        pair = frame[["Y", variable]].dropna()
        target_values = pair["Y"].to_numpy(dtype=float)
        variable_values = pair[variable].to_numpy(dtype=float)
        for lag in range(1, maxlag + 1):
            y, y_lags, x_lags = _lagged_arrays(target_values, variable_values, lag)
            restricted_ssr, restricted_rank = _ols_ssr_and_rank(y_lags, y)
            unrestricted = np.column_stack([y_lags, x_lags])
            unrestricted_ssr, unrestricted_rank = _ols_ssr_and_rank(unrestricted, y)
            df_num = unrestricted_rank - restricted_rank
            df_den = len(y) - unrestricted_rank
            if (
                df_num > 0
                and df_den > 0
                and np.isfinite(restricted_ssr)
                and np.isfinite(unrestricted_ssr)
                and unrestricted_ssr > 0
            ):
                valid_results += 1
    return time.perf_counter() - started, valid_results


def _peak_working_set_mb() -> float | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if succeeded:
            return float(counters.PeakWorkingSetSize / (1024 * 1024))
        return None

    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage / 1024.0 if sys.platform != "darwin" else usage / (1024 * 1024)
    except (ImportError, OSError):
        return None


def _blas_info() -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.__config__.show()
    return " ".join(output.getvalue().split())


def _blas_thread_count() -> int | None:
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value and value.isdigit():
            return int(value)
    return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _worker_result(args: argparse.Namespace) -> dict[str, object]:
    import scipy

    frame = _synthetic_frame(args.rows, args.variables, args.seed, args.case)
    variables = [f"X{number + 1}" for number in range(args.variables)]
    if args.worker == "new":
        elapsed, diagnostics, valid_results = _new_scan(frame, variables, args.maxlag)
        counters = {
            "fallback_count": diagnostics.fallback_count,
            "lstsq_count": diagnostics.lstsq_count,
            "initial_qr_count": diagnostics.initial_qr_count,
            "qr_rebuild_count": diagnostics.qr_rebuild_count,
            "restricted_cache_entries": diagnostics.restricted_cache_entries,
            "matrix_build_count": diagnostics.matrix_build_count,
        }
    else:
        elapsed, valid_results = _old_scan(frame, variables, args.maxlag)
        counters = {}

    return {
        "implementation": args.worker,
        "rows": args.rows,
        "variables": args.variables,
        "maxlag": args.maxlag,
        "seed": args.seed,
        "case": args.case,
        "elapsed_seconds": elapsed,
        "seconds_per_variable": elapsed / max(args.variables, 1),
        "peak_working_set_mb": _peak_working_set_mb(),
        "valid_lag_results": valid_results,
        **counters,
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas_info": _blas_info(),
        "blas_thread_count": _blas_thread_count(),
        "commit": _git_commit(),
        "operating_system": platform.platform(),
        "process_isolated": True,
    }


def _run_worker(args: argparse.Namespace, implementation: str, timeout: float | None = None):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--rows",
        str(args.rows),
        "--variables",
        str(args.variables),
        "--maxlag",
        str(args.maxlag),
        "--seed",
        str(args.seed),
        "--case",
        args.case,
        "--worker",
        implementation,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(completed.stdout)


def main() -> None:
    args = _arguments()
    if args.rows <= args.maxlag or args.variables <= 0 or args.maxlag <= 0:
        raise SystemExit("rows, variables, and maxlag must define a non-empty lag scan")
    if args.worker:
        print(json.dumps(_worker_result(args), ensure_ascii=False))
        return

    result = _run_worker(args, "new")
    if args.compare_old:
        try:
            old = _run_worker(args, "old", timeout=args.old_timeout)
        except subprocess.TimeoutExpired:
            result["old_timeout_seconds"] = args.old_timeout
            result["speedup_lower_bound"] = args.old_timeout / result["elapsed_seconds"]
        else:
            result["old_elapsed_seconds"] = old["elapsed_seconds"]
            result["old_peak_working_set_mb"] = old["peak_working_set_mb"]
            result["speedup"] = old["elapsed_seconds"] / result["elapsed_seconds"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
