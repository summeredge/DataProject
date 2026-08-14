from __future__ import annotations

import numpy as np
import pandas as pd


SAMPLE_PERIOD_NS_ATTR = "lag_sample_period_ns"
PHYSICAL_GAP_STARTS_ATTR = "resample_physical_gap_starts"


def infer_sample_period_ns(index: pd.Index) -> int | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    values = np.sort(index.asi8)
    differences = np.diff(values)
    positive = differences[differences > 0]
    if not len(positive):
        return None
    periods, counts = np.unique(positive, return_counts=True)
    return int(periods[np.argmax(counts)])


def sample_period_ns(frame: pd.DataFrame) -> int | None:
    stored = frame.attrs.get(SAMPLE_PERIOD_NS_ATTR)
    try:
        stored_int = int(stored)
    except (TypeError, ValueError, OverflowError):
        stored_int = 0
    return stored_int if stored_int > 0 else infer_sample_period_ns(frame.index)


def preserve_sample_period(
    frame: pd.DataFrame,
    period_ns: int | None,
) -> pd.DataFrame:
    if period_ns is not None and int(period_ns) > 0:
        frame.attrs[SAMPLE_PERIOD_NS_ATTR] = int(period_ns)
    return frame


def preserve_time_axis_metadata(
    source: pd.DataFrame | pd.Series,
    result: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Copy only explicit time-axis semantics to a derived object."""
    period_ns = source.attrs.get(SAMPLE_PERIOD_NS_ATTR)
    try:
        period_ns = int(period_ns)
    except (TypeError, ValueError, OverflowError):
        period_ns = None
    if period_ns is not None and period_ns > 0:
        preserve_sample_period(result, period_ns)
    forced_starts = physical_gap_starts(source)
    if forced_starts:
        result.attrs[PHYSICAL_GAP_STARTS_ATTR] = forced_starts
    return result


def physical_gap_starts(frame: pd.DataFrame | pd.Series) -> tuple[pd.Timestamp, ...]:
    values = frame.attrs.get(PHYSICAL_GAP_STARTS_ATTR, ())
    return tuple(pd.Timestamp(value) for value in values)


def physical_segment_ids(
    index: pd.DatetimeIndex,
    period_ns: int | None,
    forced_starts: tuple[object, ...] | list[object] = (),
) -> pd.Series:
    if period_ns is None or len(index) < 2:
        return pd.Series(0, index=index, dtype=int)
    breaks = index.to_series().diff().ne(pd.to_timedelta(period_ns, unit="ns"))
    if forced_starts:
        breaks |= index.isin(pd.DatetimeIndex(forced_starts))
    breaks.iloc[0] = False
    return breaks.cumsum().astype(int)


def lagged_series(
    series: pd.Series,
    target_index: pd.Index,
    lag: int,
    *,
    period_ns: int | None = None,
    forced_starts: tuple[object, ...] | list[object] | None = None,
) -> pd.Series:
    lag = int(lag)
    if (
        isinstance(series.index, pd.DatetimeIndex)
        and isinstance(target_index, pd.DatetimeIndex)
        and series.index.is_unique
        and target_index.is_unique
    ):
        resolved_period = period_ns or infer_sample_period_ns(target_index)
        if resolved_period is not None and resolved_period > 0:
            source_index = target_index - pd.to_timedelta(lag * resolved_period, unit="ns")
            shifted = series.reindex(source_index)
            shifted.index = target_index
            starts = physical_gap_starts(series) if forced_starts is None else forced_starts
            for start in starts:
                start = pd.Timestamp(start)
                crosses_break = (
                    ((source_index < start) & (target_index >= start))
                    | ((target_index < start) & (source_index >= start))
                )
                shifted = shifted.mask(crosses_break)
            return shifted
    return series.reindex(target_index).shift(lag)
