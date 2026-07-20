from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from chem_ts_corr import causality, conditional_granger, lag
from chem_ts_corr.common import benjamini_hochberg


def test_pearson_and_spearman_share_one_complete_test_family():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"target": rng.normal(size=80), "x": rng.normal(size=80)})

    result = lag.compute_lag_scores(frame, "target", max_lag=2)
    family = pd.concat([result["pearson_p"], result["spearman_p"]], ignore_index=True)
    expected = benjamini_hochberg(family)

    np.testing.assert_allclose(result["pearson_q"], expected.iloc[: len(result)], equal_nan=True)
    np.testing.assert_allclose(result["spearman_q"], expected.iloc[len(result) :], equal_nan=True)
    assert len(expected.dropna()) == len(result) * 2


def test_granger_bh_family_contains_every_variable_and_lag(monkeypatch):
    frame = pd.DataFrame(
        {
            "Y": np.arange(100, dtype=float),
            "X1": np.arange(100, dtype=float),
            "X2": np.arange(100, dtype=float),
        }
    )
    results = {
        "X1": {1: (3.0, 0.001), 2: (2.0, 0.020)},
        "X2": {1: (4.0, 0.010), 2: (1.0, 0.500)},
    }

    def fake_fast(pair, target, variable, maxlag, **kwargs):
        return results[variable]

    monkeypatch.setattr(causality, "_fast_granger_ssr_ftests", fake_fast)
    monkeypatch.setattr(causality, "_predictive_contribution", lambda *args: 0.1)

    output = causality.run_granger_tests(frame, "Y", ["X1", "X2"], maxlag=2)
    expected = benjamini_hochberg([0.001, 0.020, 0.010, 0.500])

    indexed = output.set_index("variable")
    assert indexed.loc["X1", "fdr_q_value"] == expected.iloc[0]
    assert indexed.loc["X2", "fdr_q_value"] == expected.iloc[2]
    assert indexed.loc["X1", "fdr_q_value"] > indexed.loc["X1", "min_p_value"]


def test_conditional_granger_corrects_all_tested_lags_before_selection(monkeypatch):
    captured: list[list[float]] = []
    original = conditional_granger.benjamini_hochberg

    def capture(values):
        materialized = list(values)
        captured.append(materialized)
        return original(materialized)

    monkeypatch.setattr(conditional_granger, "benjamini_hochberg", capture)
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "target": rng.normal(size=240),
            "x1": rng.normal(size=240),
            "x2": rng.normal(size=240),
        }
    )

    output = conditional_granger.run_conditional_granger_tests(
        frame,
        "target",
        ["x1", "x2"],
        maxlag=3,
        min_rows=60,
        candidate_lags={"x1": [1, 2, 3], "x2": [1, 2, 3]},
    )

    assert [len(values) for values in captured] == [6]
    assert output["fdr_q_value"].notna().all()
    assert (output["fdr_q_value"] >= output["min_p_value"]).all()


def test_old_post_selection_bh_patterns_are_absent():
    granger_source = inspect.getsource(causality.run_granger_tests)
    conditional_source = inspect.getsource(conditional_granger.run_conditional_granger_tests)
    lag_source = inspect.getsource(lag.compute_lag_scores)

    assert 'benjamini_hochberg(result_frame["min_p_value"])' not in granger_source
    assert 'benjamini_hochberg(out["min_p_value"])' not in conditional_source
    assert 'benjamini_hochberg(result["pearson_p"])' not in lag_source
    assert 'benjamini_hochberg(result["spearman_p"])' not in lag_source
