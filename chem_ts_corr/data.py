from __future__ import annotations

from pathlib import Path

import pandas as pd


EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
TEXT_SUFFIXES = {".csv", ".txt", ".tsv"}


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
        raise ValueError("文本文件编码识别失败，请手动选择 UTF-8 或 GBK / GB18030") from last_error
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
