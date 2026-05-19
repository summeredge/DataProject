import pandas as pd

from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags


def test_best_lag_detects_leading_variable():
    target = pd.Series([0, 1, 0, 2, 0, 3, 0, 4, 0, 5], name="target")
    lead = target.shift(-2).fillna(0).rename("lead")
    frame = pd.concat([target, lead], axis=1)

    scores = compute_lag_scores(frame, target="target", max_lag=3)
    best = summarize_best_lags(scores)

    assert best.iloc[0]["variable"] == "lead"
    assert best.iloc[0]["lag"] == 2
