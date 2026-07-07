from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from chem_ts_corr.common import as_text, benjamini_hochberg, left_join_missing, to_float


def test_to_float_handles_scalars_without_series_creation():
    assert to_float("1.2") == 1.2
    assert to_float(1) == 1.0
    assert to_float(None, default=-1.0) == -1.0
    assert to_float(float("nan"), default=-1.0) == -1.0
    assert to_float("bad", default=-1.0) == -1.0
    assert "pd.Series([value])" not in inspect.getsource(to_float)


def test_as_text_handles_nulls_and_scalars():
    assert as_text(None, default="missing") == "missing"
    assert as_text(float("nan"), default="missing") == "missing"
    assert as_text("abc") == "abc"
    assert as_text(12) == "12"


def test_left_join_missing_handles_empty_missing_key_columns_and_order():
    left = pd.DataFrame({"variable": ["b", "a"], "existing": [pd.NA, "keep"]})
    empty = pd.DataFrame(columns=["variable", "score"])
    result = left_join_missing(left, empty, columns=["variable", "score"])
    assert result["variable"].tolist() == ["b", "a"]
    assert "score" in result.columns
    assert result["score"].isna().all()

    no_key = left_join_missing(left, pd.DataFrame({"score": [1]}), columns=["variable", "score"])
    assert no_key["variable"].tolist() == ["b", "a"]
    assert "score" in no_key.columns

    right = pd.DataFrame({"variable": ["a", "b"], "score": [1.0, 2.0], "extra": [9, 9]})
    joined = left_join_missing(left, right, columns=["variable", "score"])
    assert joined["variable"].tolist() == ["b", "a"]
    assert joined["score"].tolist() == [2.0, 1.0]
    assert "extra" not in joined.columns
    assert pd.isna(joined.loc[0, "existing"])
    assert joined.loc[1, "existing"] == "keep"


def test_benjamini_hochberg_edges_index_range_and_monotonicity():
    assert benjamini_hochberg([]).empty

    all_nan = benjamini_hochberg(pd.Series([np.nan, np.nan], index=["x", "y"]))
    assert all_nan.index.tolist() == ["x", "y"]
    assert all_nan.isna().all()

    values = pd.Series([0.04, np.nan, 0.01, 0.03], index=["a", "b", "c", "d"])
    adjusted = benjamini_hochberg(values)
    assert adjusted.index.tolist() == ["a", "b", "c", "d"]
    assert adjusted.loc["b"] != adjusted.loc["b"]
    assert adjusted.dropna().between(0, 1).all()
    ordered = adjusted.loc[values.dropna().sort_values().index]
    assert ordered.is_monotonic_increasing
    assert adjusted.loc["c"] == 0.03
    assert adjusted.loc["d"] == 0.04
    assert adjusted.loc["a"] == 0.04
