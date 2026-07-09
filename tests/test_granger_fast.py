from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def test_fast_granger_matches_statsmodels_ssr_ftest():
    pytest.importorskip("statsmodels")
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.stattools import grangercausalitytests

    from chem_ts_corr.causality import _fast_granger_ssr_ftests

    rng = np.random.default_rng(42)
    n = 120
    x = rng.normal(size=n)
    y = np.zeros(n)
    noise = rng.normal(scale=0.2, size=n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.7 * x[t - 2] + noise[t]

    pair = pd.DataFrame({"Y": y, "X": x})
    maxlag = 4

    fast = _fast_granger_ssr_ftests(pair, "Y", "X", maxlag)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = grangercausalitytests(pair[["Y", "X"]], maxlag=maxlag, verbose=False)

    for lag in range(1, maxlag + 1):
        sm_f = float(sm[lag][0]["ssr_ftest"][0])
        sm_p = float(sm[lag][0]["ssr_ftest"][1])
        fast_f, fast_p = fast[lag]
        assert fast_f == pytest.approx(sm_f, rel=1e-6, abs=1e-8)
        assert fast_p == pytest.approx(sm_p, rel=1e-6, abs=1e-8)


def test_run_granger_tests_preserves_variable_predicts_target_direction():
    import numpy as np
    import pandas as pd

    from chem_ts_corr.causality import run_granger_tests

    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.4 * y[t - 1] + 0.9 * x[t - 2] + rng.normal(scale=0.1)

    frame = pd.DataFrame({"Y": y, "X": x})
    result = run_granger_tests(frame, target="Y", variables=["X"], maxlag=4)

    row = result[result["variable"] == "X"].iloc[0]
    assert row["status"] == "ok"
    assert int(row["best_granger_lag"]) in {1, 2, 3, 4}
    assert float(row["min_p_value"]) < 0.05


def test_fast_granger_scans_all_lags_up_to_maxlag():
    import numpy as np
    import pandas as pd

    from chem_ts_corr.causality import _fast_granger_ssr_ftests

    rng = np.random.default_rng(11)
    n = 100
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    pair = pd.DataFrame({"Y": y, "X": x})

    result = _fast_granger_ssr_ftests(pair, "Y", "X", maxlag=6)

    assert set(result.keys()) == {1, 2, 3, 4, 5, 6}


def test_run_granger_tests_does_not_call_statsmodels_runtime():
    source = Path("chem_ts_corr/causality.py").read_text(encoding="utf-8")
    run_start = source.index("def run_granger_tests")
    next_def = source.index("\ndef ", run_start + 1)
    body = source[run_start:next_def]

    assert "grangercausalitytests" not in body
    assert "verbose=False" not in body


def test_run_granger_tests_output_schema_is_stable():
    import numpy as np
    import pandas as pd

    from chem_ts_corr.causality import run_granger_tests

    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "Y": rng.normal(size=80),
            "X": rng.normal(size=80),
        }
    )

    result = run_granger_tests(frame, target="Y", variables=["X"], maxlag=3)

    expected = {
        "variable",
        "status",
        "best_granger_lag",
        "min_p_value",
        "f_statistic",
        "predictive_contribution",
        "interpretation",
        "fdr_q_value",
    }
    assert expected.issubset(set(result.columns))


def test_fast_granger_uses_unrestricted_model_rank_for_df_den():
    import numpy as np
    import pandas as pd

    from chem_ts_corr.causality import _fast_granger_ssr_ftests

    rng = np.random.default_rng(31)
    n = 80
    y = rng.normal(size=n)
    pair = pd.DataFrame({"Y": y, "X": y})

    result = _fast_granger_ssr_ftests(pair, "Y", "X", maxlag=1)

    assert 1 in result
    fast_f, fast_p = result[1]

    design = pd.DataFrame(
        {
            "target": pair["Y"],
            "target_lag_1": pair["Y"].shift(1),
            "variable_lag_1": pair["X"].shift(1),
        }
    ).dropna()
    target_values = design["target"].to_numpy()
    restricted_x = design[["target_lag_1"]].to_numpy()
    unrestricted_x = design[["target_lag_1", "variable_lag_1"]].to_numpy()

    restricted_matrix = np.column_stack([np.ones(len(restricted_x)), restricted_x])
    unrestricted_matrix = np.column_stack([np.ones(len(unrestricted_x)), unrestricted_x])
    restricted_coef, *_ = np.linalg.lstsq(restricted_matrix, target_values, rcond=None)
    unrestricted_coef, *_ = np.linalg.lstsq(unrestricted_matrix, target_values, rcond=None)
    restricted_residual = target_values - restricted_matrix @ restricted_coef
    ssr_r = float(np.dot(restricted_residual, restricted_residual))
    ssr_u = float(
        np.dot(
            target_values - unrestricted_matrix @ unrestricted_coef,
            target_values - unrestricted_matrix @ unrestricted_coef,
        )
    )
    df_den = len(target_values) - np.linalg.matrix_rank(unrestricted_matrix)
    expected_f = max(0.0, ssr_r - ssr_u) / (ssr_u / df_den)

    assert df_den == len(target_values) - 2
    assert fast_f == pytest.approx(expected_f)
    assert 0.0 <= fast_p <= 1.0


def test_fast_granger_rejects_near_perfect_unrestricted_fit():
    import numpy as np
    import pandas as pd

    from chem_ts_corr.causality import _fast_granger_ssr_ftests

    n = 60
    x = np.arange(float(n))
    y = np.zeros(n)
    y[1:] = x[:-1]
    pair = pd.DataFrame({"Y": y, "X": x})

    result = _fast_granger_ssr_ftests(pair, "Y", "X", maxlag=1)

    assert result == {}
