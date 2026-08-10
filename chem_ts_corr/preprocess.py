from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from chem_ts_corr.config import NOT_IMPLEMENTED_PREPROCESS_MODES
from chem_ts_corr.time_axis import infer_sample_period_ns, preserve_sample_period, sample_period_ns


LOWPASS_PHYSICAL_GAP_FACTOR = 1.5


@dataclass(frozen=True)
class FrameScaler:
    mean_: pd.Series
    scale_: pd.Series
    feature_names: tuple[object, ...]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(frame.columns) != self.feature_names:
            raise ValueError(
                "feature alignment mismatch: "
                f"X.columns={list(frame.columns)!r}, "
                f"model.feature_names={list(self.feature_names)!r}"
            )
        return (frame - self.mean_) / self.scale_


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
    scaler = FrameScaler(
        mean_=fit_frame.mean(),
        scale_=fit_frame.std(ddof=0).replace(0, 1),
        feature_names=tuple(fit_frame.columns),
    )
    return preserve_sample_period(scaler.transform(frame), sample_period_ns(frame))


def transform_frame(
    frame: pd.DataFrame,
    mode: str,
    detrend_window: int,
    max_interpolate_gap_points: int = 5,
    interpolate_limit_area: str = "inside",
    lowpass_tau_minutes: float = 5.0,
    diff_interval_minutes: float | None = None,
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if mode == "lowpass":
        return lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
    if mode == "lowpass_detrend":
        smoothed = lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
        return detrend_moving_average(
            smoothed,
            detrend_window,
            max_interpolate_gap_points,
            interpolate_limit_area,
        )
    if mode == "lowpass_diff":
        smoothed = lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
        transformed = difference_by_physical_interval(
            smoothed,
            diff_interval_minutes=diff_interval_minutes,
        ).dropna()
        return preserve_sample_period(transformed, period_ns)
    if mode in NOT_IMPLEMENTED_PREPROCESS_MODES:
        raise ValueError(
            f"Preprocess mode {mode!r} is defined in the contract but is not implemented yet"
        )
    if mode == "raw":
        return preserve_sample_period(frame, period_ns)
    if mode == "detrend":
        return detrend_moving_average(frame, detrend_window, max_interpolate_gap_points, interpolate_limit_area)
    if mode == "diff":
        return preserve_sample_period(
            difference_by_contiguous_segment(frame).dropna(), period_ns
        )
    if mode == "detrend_diff":
        detrended = detrend_moving_average(
            frame, detrend_window, max_interpolate_gap_points, interpolate_limit_area
        )
        transformed = difference_by_contiguous_segment(detrended).dropna()
        return preserve_sample_period(transformed, period_ns)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def transform_frame_causal(
    frame: pd.DataFrame,
    mode: str,
    detrend_window: int,
    lowpass_tau_minutes: float = 5.0,
    diff_interval_minutes: float | None = None,
) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    if mode == "lowpass":
        return lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
    if mode == "lowpass_detrend":
        smoothed = lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
        return detrend_trailing_average(
            smoothed,
            detrend_window,
        )
    if mode == "lowpass_diff":
        smoothed = lowpass_filter_frame(
            frame,
            tau_minutes=lowpass_tau_minutes,
        )
        transformed = difference_by_physical_interval(
            smoothed,
            diff_interval_minutes=diff_interval_minutes,
        )
        return preserve_sample_period(
            transformed.dropna(how="all"),
            period_ns,
        )
    if mode == "raw":
        return preserve_sample_period(frame, period_ns)
    if mode == "detrend":
        return detrend_trailing_average(frame, detrend_window)
    if mode == "diff":
        return preserve_sample_period(
            difference_by_contiguous_segment(frame).dropna(how="all"), period_ns
        )
    if mode == "detrend_diff":
        detrended = detrend_trailing_average(frame, detrend_window)
        transformed = difference_by_contiguous_segment(detrended)
        return preserve_sample_period(transformed.dropna(how="all"), period_ns)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def lowpass_filter_frame(
    frame: pd.DataFrame,
    tau_minutes: float,
) -> pd.DataFrame:
    """Apply a time-aware first-order low-pass filter to every column.

    alpha = 1 - exp(-dt / tau_minutes)
    y(t) = y(t-1) + alpha * [x(t) - y(t-1)]

    dt is the physical time between adjacent valid samples of a column.
    Each contiguous physical segment filters independently: an interval only
    starts a new segment when it exceeds LOWPASS_PHYSICAL_GAP_FACTOR times
    the sample period, so irregular-but-contiguous sampling keeps one filter
    state. The first valid value of a segment is the initial state, and
    segments never share state. Missing inputs stay missing, are never
    backfilled or written as 0.0, and the filter state carries across missing
    rows inside a segment.
    """
    try:
        tau_minutes = float(tau_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("tau_minutes must be a finite value greater than 0") from exc
    if not math.isfinite(tau_minutes) or tau_minutes <= 0:
        raise ValueError("tau_minutes must be a finite value greater than 0")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("lowpass_filter_frame requires a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(
            "lowpass_filter_frame requires a monotonically increasing DatetimeIndex "
            "with unique timestamps"
        )

    period_ns = sample_period_ns(frame)
    segment_ids = _physical_segment_ids(frame.index, period_ns)
    times_ns = frame.index.asi8
    filtered = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        output: list[float] = []
        state: float | None = None
        state_time_ns: int | None = None
        current_segment = -1
        for value, time_ns, segment_id in zip(values, times_ns, segment_ids):
            if int(segment_id) != current_segment:
                current_segment = int(segment_id)
                state = None
                state_time_ns = None
            value = float(value)
            if math.isnan(value):
                output.append(math.nan)
                continue
            time_ns = int(time_ns)
            if state is None:
                state = value
                state_time_ns = time_ns
                output.append(state)
                continue
            delta_minutes = (time_ns - state_time_ns) / 60_000_000_000.0
            alpha = 1.0 - math.exp(-delta_minutes / tau_minutes)
            state = state + alpha * (value - state)
            state_time_ns = time_ns
            output.append(state)
        filtered[column] = output
    filtered.attrs = dict(frame.attrs)
    return preserve_sample_period(filtered, period_ns)


def _physical_segment_ids(
    index: pd.DatetimeIndex,
    nominal_period_ns: int | None,
) -> pd.Series:
    if nominal_period_ns is None or len(index) < 2:
        return pd.Series(0, index=index, dtype=int)
    gap_threshold_ns = LOWPASS_PHYSICAL_GAP_FACTOR * float(nominal_period_ns)
    diffs_ns = index.to_series().diff().dt.total_seconds().mul(1_000_000_000.0)
    breaks = diffs_ns.gt(gap_threshold_ns)
    breaks.iloc[0] = False
    return breaks.cumsum().astype(int)


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


def difference_by_contiguous_segment(frame: pd.DataFrame) -> pd.DataFrame:
    period_ns = sample_period_ns(frame)
    groups = _contiguous_segment_ids(frame.index, period_ns)
    return preserve_sample_period(frame.groupby(groups).diff(), period_ns)


def _validate_diff_interval_minutes(
    diff_interval_minutes: float | None,
) -> float | None:
    if diff_interval_minutes is not None:
        try:
            diff_interval_minutes = float(diff_interval_minutes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "diff_interval_minutes must be a finite value greater than 0"
            ) from exc
        if not math.isfinite(diff_interval_minutes) or diff_interval_minutes <= 0:
            raise ValueError(
                "diff_interval_minutes must be a finite value greater than 0; "
                "use None for automatic interval"
            )
    return diff_interval_minutes


def resolve_diff_interval(
    frame: pd.DataFrame,
    diff_interval_minutes: float | None,
) -> tuple[int, float]:
    """Resolve a requested difference interval into fixed diff parameters.

    Returns (effective_diff_points, effective_diff_interval_minutes). None
    means one analysis sampling period; a specified interval is converted
    with max(1, round(diff_interval_minutes / sampling_interval_minutes)) and
    the effective interval is always an exact multiple of the sampling
    interval.
    """
    diff_interval_minutes = _validate_diff_interval_minutes(diff_interval_minutes)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("resolve_diff_interval requires a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(
            "resolve_diff_interval requires a monotonically increasing "
            "DatetimeIndex with unique timestamps"
        )

    period_ns = sample_period_ns(frame)
    if period_ns is None:
        if len(frame) < 2 and diff_interval_minutes is None:
            return (1, float("nan"))
        raise ValueError(
            "resolve_diff_interval could not determine a valid positive "
            "sampling period for the given time axis"
        )
    sampling_interval_minutes = period_ns / 60_000_000_000.0
    if diff_interval_minutes is None:
        return (1, float(sampling_interval_minutes))
    effective_diff_points = max(
        1, round(diff_interval_minutes / sampling_interval_minutes)
    )
    return (
        effective_diff_points,
        float(effective_diff_points * sampling_interval_minutes),
    )


def difference_by_physical_interval(
    frame: pd.DataFrame,
    diff_interval_minutes: float | None,
) -> pd.DataFrame:
    """Difference every column by a fixed physical interval within segments.

    difference(t) = x(t) - x(t - effective_diff_points), computed
    independently inside each contiguous physical segment so a real gap never
    borrows history from an earlier segment. Missing inputs and uncomputable
    positions stay missing and are never filled with 0.0.
    """
    diff_interval_minutes = _validate_diff_interval_minutes(diff_interval_minutes)
    if len(frame) == 0:
        result = frame.copy()
        result.attrs = dict(frame.attrs)
        return result
    effective_diff_points, _ = resolve_diff_interval(frame, diff_interval_minutes)
    period_ns = sample_period_ns(frame)
    if len(frame.columns) == 0:
        result = frame.copy()
        result.attrs = dict(frame.attrs)
        return preserve_sample_period(result, period_ns)
    segment_ids = _physical_segment_ids(frame.index, period_ns)
    result = frame.groupby(segment_ids, sort=False).diff(
        periods=effective_diff_points
    )
    result.attrs = dict(frame.attrs)
    return preserve_sample_period(result, period_ns)


def _contiguous_segment_ids(index: pd.Index, period_ns: int | None) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex) or period_ns is None or len(index) < 2:
        return pd.Series(0, index=index, dtype=int)
    breaks = index.to_series().diff().ne(pd.to_timedelta(period_ns, unit="ns"))
    breaks.iloc[0] = False
    return breaks.cumsum().astype(int)
