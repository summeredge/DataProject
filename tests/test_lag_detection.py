import numpy as np
import pandas as pd

from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags


def test_compute_lag_scores_detects_known_lag_and_q_columns():
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = np.sin(np.linspace(0, 8, n))
    target = pd.Series(base, index=idx, name="target")
    # variable leads target by 3 points => best lag should be +3
    lead3 = target.shift(-3).fillna(0).rename("lead3")
    frame = pd.concat([target, lead3], axis=1)

    lag_scores = compute_lag_scores(frame, target="target", max_lag=6)
    assert not lag_scores.empty
    assert {"pearson_q", "spearman_q", "corr_q_value", "lag_boundary_flag", "n"}.issubset(lag_scores.columns)

    best = summarize_best_lags(lag_scores)
    row = best.loc[best["variable"] == "lead3"].iloc[0]
    assert int(row["lag"]) == 3
    assert 0 <= float(row["score"]) <= 1
