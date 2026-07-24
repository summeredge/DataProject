from __future__ import annotations

import pandas as pd

from chem_ts_corr.report import build_recommended_candidates


def test_historical_closed_loop_status_does_not_filter_recommendations():
    ranked = pd.DataFrame(
        [
            {"variable": "closed", "candidate_grade": "A", "final_score": 0.99, "recommended_use": "closed_loop_confirmed"},
            {"variable": "normal", "candidate_grade": "B", "final_score": 0.80, "recommended_use": "upstream_driver_candidate"},
        ]
    )

    assert build_recommended_candidates(ranked)["variable"].tolist() == ["closed", "normal"]
