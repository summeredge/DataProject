from __future__ import annotations

import pandas as pd

from chem_ts_corr.screening import final_ranked_features


def test_recommended_use_never_contains_closed_loop_status():
    ranked = pd.DataFrame(
        [
            {"variable": "A", "score": 0.9, "innovation_score": 0.9, "lag": 1, "direction": ""},
            {"variable": "B", "score": 0.8, "innovation_score": 0.8, "lag": 1, "direction": ""},
        ]
    )
    values = ranked[["variable", "score"]]
    output = final_ranked_features(
        ranked,
        pd.DataFrame(columns=["variable"]),
        pd.DataFrame(columns=["variable"]),
        values.rename(columns={"score": "model_lift_score"}),
        pd.DataFrame(columns=["variable"]),
        values.rename(columns={"score": "lag_quality"}),
        values.rename(columns={"score": "rolling_stability"}),
    )

    assert not set(output["recommended_use"]).intersection({"closed_loop_confirmed", "closed_loop_conflict", "closed_loop_suspect"})
