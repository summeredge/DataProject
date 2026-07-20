from __future__ import annotations

import pandas as pd

from chem_ts_corr.time_axis import infer_sample_period_ns, preserve_sample_period, sample_period_ns


def segment_by_load(
    frame: pd.DataFrame,
    segment_column: str | None,
    segment_mode: str,
    segment_min: float | None,
    segment_max: float | None,
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    mask = operating_segment_mask(
        frame,
        segment_column,
        segment_mode,
        segment_min,
        segment_max,
    )
    segmented = frame.loc[mask]
    return preserve_sample_period(segmented, period_ns)


def operating_segment_mask(
    frame: pd.DataFrame,
    segment_column: str | None,
    segment_mode: str,
    segment_min: float | None,
    segment_max: float | None,
) -> pd.Series:
    if not segment_column or segment_mode == "all":
        return pd.Series(True, index=frame.index, dtype=bool)
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

    mask = load.between(lower, upper, inclusive="both")
    if int(mask.sum()) < 10:
        raise ValueError("Not enough rows in selected operating segment; at least 10 are required")
    return mask


def preprocess_frame(
    frame: pd.DataFrame,
    target: str,
    resample_rule: str | None,
    min_valid_ratio: float,
    protected_columns: list[str] | None = None,
    max_interpolate_gap_points: int = 5,
    interpolate_limit_area: str = "inside",
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if resample_rule:
        frame = frame.resample(resample_rule).median()
        period_ns = infer_sample_period_ns(frame.index)

    valid_ratio = frame.notna().mean()
    keep_columns = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
    keep = set(keep_columns)
    keep.add(target)
    for col in (protected_columns or []):
        if col in frame.columns:
            keep.add(col)
    keep_columns = [c for c in frame.columns if c in keep]

    cleaned = frame[keep_columns].interpolate(method="time", limit=max_interpolate_gap_points, limit_area=interpolate_limit_area)
    cleaned = cleaned.dropna(axis=1, how="all")
    rows_before_dropna = int(len(cleaned))
    cleaned = cleaned.dropna(axis=0, how="any")
    rows_after_dropna = int(len(cleaned))
    rows_dropped_by_dropna = rows_before_dropna - rows_after_dropna
    cleaned.attrs["rows_before_dropna"] = rows_before_dropna
    cleaned.attrs["rows_after_dropna"] = rows_after_dropna
    cleaned.attrs["rows_dropped_by_dropna"] = rows_dropped_by_dropna
    if rows_dropped_by_dropna:
        warnings = [str(cleaned.attrs.get("preprocess_warnings", "")).strip()] if cleaned.attrs.get("preprocess_warnings") else []
        warnings.append(f"rows_dropped_by_dropna={rows_dropped_by_dropna}")
        cleaned.attrs["preprocess_warnings"] = "; ".join(warnings)

    if target not in cleaned.columns:
        raise ValueError("Target column was removed during preprocessing")
    if len(cleaned) < 10:
        raise ValueError("Not enough usable rows after preprocessing; at least 10 are required")

    low_variance = cleaned.nunique(dropna=True) <= 1
    protected = set(protected_columns or [])
    removable = [col for col in low_variance[low_variance].index if col != target and col not in protected]
    protected_low_var = [col for col in low_variance[low_variance].index if col in protected]
    if protected_low_var:
        cleaned.attrs["protected_low_variance_columns"] = protected_low_var
    return preserve_sample_period(cleaned.drop(columns=removable), period_ns)


def preprocess_frame_causal(
    frame: pd.DataFrame,
    target: str,
    resample_rule: str | None,
    max_forward_fill_gap_points: int = 5,
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if resample_rule:
        frame = frame.resample(resample_rule).median()
        period_ns = infer_sample_period_ns(frame.index)
    if target not in frame.columns:
        raise ValueError("Target column was removed during preprocessing")

    cleaned = frame.copy().dropna(subset=[target])
    predictor_columns = [column for column in cleaned.columns if column != target]
    if max_forward_fill_gap_points > 0 and predictor_columns:
        groups = _contiguous_segment_ids(cleaned.index, period_ns)
        cleaned[predictor_columns] = cleaned[predictor_columns].groupby(groups).ffill(
            limit=max_forward_fill_gap_points
        )
    if len(cleaned) < 10:
        raise ValueError("Not enough usable rows after preprocessing; at least 10 are required")
    return preserve_sample_period(cleaned, period_ns)


def standardize_frame(
    frame: pd.DataFrame,
    fit_mask: pd.Series | None = None,
) -> pd.DataFrame:
    fit_frame = frame
    if fit_mask is not None:
        resolved_mask = fit_mask.reindex(frame.index).fillna(False).astype(bool)
        fit_frame = frame.loc[resolved_mask]
    std = fit_frame.std(ddof=0).replace(0, 1)
    return preserve_sample_period((frame - fit_frame.mean()) / std, sample_period_ns(frame))


def transform_frame(
    frame: pd.DataFrame,
    mode: str,
    detrend_window: int,
    max_interpolate_gap_points: int = 5,
    interpolate_limit_area: str = "inside",
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if mode == "raw":
        return preserve_sample_period(frame, period_ns)
    if mode == "detrend":
        return detrend_moving_average(frame, detrend_window, max_interpolate_gap_points, interpolate_limit_area)
    if mode == "diff":
        return preserve_sample_period(frame.diff().dropna(), period_ns)
    if mode == "detrend_diff":
        transformed = detrend_moving_average(
            frame, detrend_window, max_interpolate_gap_points, interpolate_limit_area
        ).diff().dropna()
        return preserve_sample_period(transformed, period_ns)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def transform_frame_causal(
    frame: pd.DataFrame,
    mode: str,
    detrend_window: int,
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if mode == "raw":
        return preserve_sample_period(frame, period_ns)
    if mode == "detrend":
        return detrend_trailing_average(frame, detrend_window)
    if mode == "diff":
        return preserve_sample_period(_causal_difference(frame, period_ns).dropna(how="all"), period_ns)
    if mode == "detrend_diff":
        detrended = detrend_trailing_average(frame, detrend_window)
        transformed = _causal_difference(detrended, period_ns)
        return preserve_sample_period(transformed.dropna(how="all"), period_ns)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def detrend_moving_average(
    frame: pd.DataFrame,
    window: int,
    max_interpolate_gap_points: int = 5,
    interpolate_limit_area: str = "inside",
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    window = max(3, int(window))
    trend = frame.rolling(window=window, center=True, min_periods=max(2, window // 4)).mean()
    detrended = frame - trend
    transformed = detrended.interpolate(
        method="time", limit=max_interpolate_gap_points, limit_area=interpolate_limit_area
    ).dropna()
    return preserve_sample_period(transformed, period_ns)


def detrend_trailing_average(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    window = max(3, int(window))
    min_periods = max(2, window // 4)
    groups = _contiguous_segment_ids(frame.index, period_ns)
    trend = pd.concat(
        [
            group.rolling(window=window, center=False, min_periods=min_periods).mean()
            for _, group in frame.groupby(groups, sort=False)
        ]
    ).reindex(frame.index)
    detrended = frame - trend
    return preserve_sample_period(detrended.dropna(how="all"), period_ns)


def _causal_difference(frame: pd.DataFrame, period_ns: int | None) -> pd.DataFrame:
    groups = _contiguous_segment_ids(frame.index, period_ns)
    return frame.groupby(groups).diff()


def _contiguous_segment_ids(index: pd.Index, period_ns: int | None) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex) or period_ns is None or len(index) < 2:
        return pd.Series(0, index=index, dtype=int)
    breaks = index.to_series().diff().ne(pd.to_timedelta(period_ns, unit="ns"))
    breaks.iloc[0] = False
    return breaks.cumsum().astype(int)
