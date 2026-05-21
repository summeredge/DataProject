from __future__ import annotations

import pandas as pd


def segment_by_load(
    frame: pd.DataFrame,
    segment_column: str | None,
    segment_mode: str,
    segment_min: float | None,
    segment_max: float | None,
) -> pd.DataFrame:
    if not segment_column or segment_mode == "all":
        return frame
    if segment_column not in frame.columns:
        raise ValueError(f"Segment column '{segment_column}' was not found in input data")

    load = pd.to_numeric(frame[segment_column], errors="coerce")
    valid_load = load.dropna()
    if valid_load.empty:
        raise ValueError(f"Segment column '{segment_column}' has no numeric values")

    lower = float("-inf")
    upper = float("inf")
    if segment_mode == "low":
        upper = float(valid_load.quantile(1 / 3))
    elif segment_mode == "mid":
        lower = float(valid_load.quantile(1 / 3))
        upper = float(valid_load.quantile(2 / 3))
    elif segment_mode == "high":
        lower = float(valid_load.quantile(2 / 3))
    elif segment_mode == "custom":
        if segment_min is not None:
            lower = segment_min
        if segment_max is not None:
            upper = segment_max
    else:
        raise ValueError(f"Unknown segment mode: {segment_mode}")

    segmented = frame.loc[load.between(lower, upper, inclusive="both")]
    if len(segmented) < 10:
        raise ValueError("Not enough rows in selected operating segment; at least 10 are required")
    return segmented


def preprocess_frame(
    frame: pd.DataFrame,
    target: str,
    resample_rule: str | None,
    min_valid_ratio: float,
    protected_columns: list[str] | None = None,
    max_interpolate_gap_points: int = 5,
    interpolate_limit_area: str = "inside",
) -> pd.DataFrame:
    if resample_rule:
        frame = frame.resample(resample_rule).median()

    valid_ratio = frame.notna().mean()
    keep_columns = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
    keep = set(keep_columns)
    keep.add(target)
    for col in (protected_columns or []):
        if col in frame.columns:
            keep.add(col)
    keep_columns = [c for c in frame.columns if c in keep]

    cleaned = frame[keep_columns].interpolate(method="time", limit=max_interpolate_gap_points, limit_area=interpolate_limit_area)
    cleaned = cleaned.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if target not in cleaned.columns:
        raise ValueError("Target column was removed during preprocessing")
    if cleaned.shape[0] < 10:
        raise ValueError("Not enough usable rows after preprocessing; at least 10 are required")

    low_variance = cleaned.nunique(dropna=True) <= 1
    protected = set(protected_columns or [])
    removable = [col for col in low_variance[low_variance].index if col != target and col not in protected]
    protected_low_var = [col for col in low_variance[low_variance].index if col in protected]
    if protected_low_var:
        cleaned.attrs["protected_low_variance_columns"] = protected_low_var
    return cleaned.drop(columns=removable)


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(ddof=0).replace(0, 1)
    return (frame - frame.mean()) / std


def transform_frame(frame: pd.DataFrame, mode: str, detrend_window: int) -> pd.DataFrame:
    if mode == "raw":
        return frame
    if mode == "detrend":
        return detrend_moving_average(frame, detrend_window)
    if mode == "diff":
        return frame.diff().dropna()
    if mode == "detrend_diff":
        return detrend_moving_average(frame, detrend_window).diff().dropna()
    raise ValueError(f"Unknown preprocess mode: {mode}")


def detrend_moving_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(3, int(window))
    trend = frame.rolling(window=window, center=True, min_periods=max(2, window // 4)).mean()
    detrended = frame - trend
    return detrended.interpolate(method="time", limit_direction="both").dropna()
