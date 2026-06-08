import pandas as pd

from chem_ts_corr.web import _merge_near_miss_lags


def test_merge_near_miss_lags_adds_missing_near_miss_variable_lag():
    best_lags = {"ranked_x": 2}
    near_miss = pd.DataFrame(
        [
            {"variable": "near_x", "lag": 5},
            {"variable": "ranked_x", "lag": 9},
            {"variable": "bad_lag", "lag": "not-a-number"},
            {"variable": "", "lag": 3},
        ]
    )

    merged = _merge_near_miss_lags(best_lags, near_miss)

    assert merged == {"ranked_x": 2, "near_x": 5}
    assert best_lags == {"ranked_x": 2}
