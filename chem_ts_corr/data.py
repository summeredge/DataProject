from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
TEXT_SUFFIXES = {".csv", ".txt", ".tsv"}


def _exclude_window_mask(
    frame: pd.DataFrame,
    exclude_windows: list[dict[str, str]],
) -> tuple[pd.Series, int]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("apply_exclude_windows requires a DatetimeIndex")
    if not isinstance(exclude_windows, list):
        raise ValueError("exclude_windows must be a list")

    excluded = pd.Series(False, index=frame.index, dtype=bool)
    for position, window in enumerate(exclude_windows):
        if not isinstance(window, dict):
            raise ValueError(f"exclude_windows[{position}] must be an object with start and end")
        if set(window) != {"start", "end"}:
            raise ValueError(f"exclude_windows[{position}] must contain only start and end")

        start = _parse_exclude_window_time(window["start"], position, "start")
        end = _parse_exclude_window_time(window["end"], position, "end")
        try:
            invalid_range = start > end
        except TypeError as exc:
            raise ValueError(
                f"exclude_windows[{position}] start and end must use compatible timezones"
            ) from exc
        if invalid_range:
            raise ValueError(f"exclude_windows[{position}] start must be before or equal to end")
        try:
            excluded |= (frame.index >= start) & (frame.index <= end)
        except TypeError as exc:
            raise ValueError(
                f"exclude_windows[{position}] timestamps must match the DataFrame index timezone"
            ) from exc
    return excluded, len(exclude_windows)


def _parse_exclude_window_time(value: Any, position: int, field: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"exclude_windows[{position}].{field} must be an ISO-8601 timestamp string")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"exclude_windows[{position}].{field} is not a valid timestamp: {value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise ValueError(f"exclude_windows[{position}].{field} is not a valid timestamp: {value!r}")
    return timestamp


def apply_exclude_windows(
    frame: pd.DataFrame,
    exclude_windows: list[dict[str, str]],
) -> pd.DataFrame:
    """Return a copy of ``frame`` without rows in inclusive exclusion windows."""
    excluded, _ = _exclude_window_mask(frame, exclude_windows)
    result = frame.loc[~excluded].copy(deep=True)
    result.attrs = dict(frame.attrs)
    return result


def exclude_window_stats(
    frame: pd.DataFrame,
    exclude_windows: list[dict[str, str]],
) -> dict[str, int | float]:
    """Summarize rows selected out by the shared exclusion-window rule."""
    excluded, window_count = _exclude_window_mask(frame, exclude_windows)
    original_rows = len(frame)
    excluded_rows = int(excluded.sum())
    return {
        "original_rows": original_rows,
        "excluded_rows": excluded_rows,
        "remaining_rows": original_rows - excluded_rows,
        "excluded_ratio": excluded_rows / original_rows if original_rows else 0.0,
        "exclude_window_count": window_count,
    }


def normalize_excluded_columns(excluded_columns: list[str] | None) -> list[str]:
    return list(
        dict.fromkeys(
            str(column).strip()
            for column in (excluded_columns or [])
            if column is not None and str(column).strip()
        )
    )


def drop_excluded_columns(
    frame: pd.DataFrame,
    excluded_columns: list[str] | None,
    *,
    protected_columns: list[str] | None = None,
) -> pd.DataFrame:
    excluded = normalize_excluded_columns(excluded_columns)
    protected = set(normalize_excluded_columns(protected_columns))
    conflicts = [column for column in excluded if column in protected]
    if conflicts:
        raise ValueError(f"剔除列与受保护参数冲突：{'、'.join(conflicts)}")

    missing = [column for column in excluded if column not in frame.columns]
    if missing:
        raise ValueError(f"剔除列不存在：{'、'.join(missing)}")

    result = frame.drop(columns=excluded).copy(deep=True)
    result.attrs = dict(frame.attrs)
    return result


def read_timeseries_table(
    path: Path,
    encoding: str = "utf-8-sig",
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        try:
            return pd.read_excel(path, nrows=nrows, usecols=usecols), "excel"
        except ImportError as exc:
            raise ValueError("读取 Excel 文件需要安装 openpyxl 或 xlrd") from exc

    if suffix not in TEXT_SUFFIXES:
        raise ValueError(f"Unsupported input file type: {suffix or 'unknown'}")

    sep = "\t" if suffix == ".tsv" else (None if suffix == ".txt" else ",")
    encodings = ["utf-8-sig", "gb18030"] if encoding == "auto" else [encoding]
    last_error: Exception | None = None
    for candidate in encodings:
        try:
            kwargs = {
                "encoding": candidate,
                "nrows": nrows,
                "usecols": usecols,
            }
            if sep is None:
                kwargs.update({"sep": None, "engine": "python"})
            else:
                kwargs.update({"sep": sep, "low_memory": False})
            return pd.read_csv(path, **kwargs), candidate
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise ValueError("文本文件编码自动识别失败，请将文件转换为 UTF-8 或 GBK / GB18030 后重试") from last_error
    raise ValueError("文本文件读取失败")


def load_timeseries_csv(path: Path, time_column: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    frame, _ = read_timeseries_table(path, encoding=encoding)
    if time_column not in frame.columns:
        raise ValueError(f"Time column '{time_column}' was not found in input data")

    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    duplicate_timestamps = int(frame[time_column].duplicated().sum())
    frame = frame.dropna(subset=[time_column]).sort_values(time_column)
    frame = frame.set_index(time_column)
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.attrs["duplicate_timestamps"] = duplicate_timestamps
    return frame


def select_numeric_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found in input data")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is not numeric")
    valid_target = int(numeric[target].notna().sum())
    if valid_target < 10:
        raise ValueError(f"Target column '{target}' has insufficient numeric points: {valid_target} < 10")
    return numeric
