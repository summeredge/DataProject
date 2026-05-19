from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_timeseries_csv(path: Path, time_column: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    frame = pd.read_csv(path, encoding=encoding)
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
    return numeric
