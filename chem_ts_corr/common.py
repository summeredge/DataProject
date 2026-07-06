from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        return default
    return str(value)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return default if np.isnan(numeric) else float(numeric)


def to_int(value: Any, default: int = 0) -> int:
    numeric = to_float(value, default=float("nan"))
    if np.isnan(numeric):
        return default
    try:
        return int(numeric)
    except (TypeError, ValueError, OverflowError):
        return default


def left_join_missing(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    on: str = "variable",
    columns: Sequence[str] | Mapping[str, str] | None = None,
    rename: Mapping[str, str] | None = None,
    fill_empty_strings: bool = False,
) -> pd.DataFrame:
    if not isinstance(on, str):
        columns = on  # Backward-compatible positional columns argument.
        on = "variable"
    result = left.copy(deep=True)
    rename_map = dict(rename or {})
    if isinstance(columns, Mapping):
        rename_map = {**dict(columns), **rename_map}
        requested = [on, *columns.keys()]
        fill_empty_strings = True
    elif columns is None:
        requested = list(right.columns) if right is not None and on in right.columns else [on]
    else:
        requested = list(columns)

    output_cols = [rename_map.get(col, col) for col in requested if col != on]
    for col in output_cols:
        if col not in result.columns:
            result[col] = pd.NA

    if right is None or right.empty or on not in right.columns or on not in result.columns:
        return result

    available = [col for col in requested if col in right.columns]
    if on not in available:
        available.insert(0, on)
    value_cols = [col for col in available if col != on]
    if not value_cols:
        return result

    side = right[[on, *value_cols]].copy(deep=True).rename(columns=rename_map)
    side = side.drop_duplicates(subset=[on], keep="first")
    joined_value_cols = [col for col in side.columns if col != on]
    merged = result.merge(side, on=on, how="left", sort=False, suffixes=("", "__joined"))
    for col in joined_value_cols:
        joined_col = f"{col}__joined"
        if joined_col in merged.columns:
            existing = merged[col]
            missing = existing.isna()
            if fill_empty_strings:
                missing = missing | existing.astype(str).str.strip().eq("")
            merged[col] = existing.where(~missing, merged[joined_col])
            merged = merged.drop(columns=[joined_col])
    return merged


def benjamini_hochberg(p_values: Sequence[float] | pd.Series) -> pd.Series:
    if isinstance(p_values, pd.Series):
        pvals = pd.to_numeric(p_values, errors="coerce")
        index = p_values.index
    else:
        pvals = pd.to_numeric(pd.Series(p_values), errors="coerce")
        index = pvals.index
    qvals = pd.Series(np.nan, index=index, dtype=float)
    valid = pvals.dropna().clip(lower=0.0, upper=1.0).sort_values()
    m = len(valid)
    if m == 0:
        return qvals
    ranked_items = list(valid.items())
    running = 1.0
    for rank in range(m, 0, -1):
        original_idx, original_p = ranked_items[rank - 1]
        value = min(running, float(original_p) * m / rank, 1.0)
        running = value
        qvals.loc[original_idx] = value
    return qvals.clip(lower=0.0, upper=1.0)
