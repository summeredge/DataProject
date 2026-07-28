from __future__ import annotations

import pandas as pd

from chem_ts_corr.screening import build_recommended_candidates


def _ranked() -> pd.DataFrame:
    return pd.DataFrame([
        {"variable": "both", "final_score": .95, "association_score": .9, "lag_quality": .8, "raw_corr": .9, "variable_role": "candidate", "force_included": False},
        {"variable": "raw_only", "final_score": .90, "association_score": .8, "lag_quality": .7, "raw_corr": .8, "variable_role": "candidate", "force_included": False},
        {"variable": "residual_only", "final_score": .30, "association_score": .3, "lag_quality": .2, "raw_corr": .3, "variable_role": "candidate", "force_included": False},
        {"variable": "weak", "final_score": .20, "association_score": .2, "lag_quality": .1, "raw_corr": .2, "variable_role": "candidate", "force_included": False},
        {"variable": "forced", "final_score": .10, "association_score": .1, "lag_quality": .1, "raw_corr": .1, "variable_role": "candidate", "force_included": False},
        {"variable": "control", "final_score": .99, "association_score": .9, "lag_quality": .9, "raw_corr": .9, "variable_role": "residual_control", "is_residual_control": True, "force_included": False},
    ])


def test_dual_channel_union_preserves_sources_and_is_not_retruncated():
    residual = pd.DataFrame([
        {"variable": "both", "residual_corr": .9, "residual_lag_quality": .8, "residual_n": 100, "residual_lag": 2, "residual_status": "ok"},
        {"variable": "residual_only", "residual_corr": .8, "residual_lag_quality": .7, "residual_n": 90, "residual_lag": 3, "residual_status": "ok"},
        {"variable": "raw_only", "residual_corr": .1, "residual_lag_quality": .1, "residual_n": 80, "residual_lag": 1, "residual_status": "ok"},
        {"variable": "weak", "residual_corr": .1, "residual_lag_quality": .1, "residual_n": 80, "residual_lag": 1, "residual_status": "fit_failed"},
    ])
    out = build_recommended_candidates(_ranked(), 2, ["forced", "control"], residual_corr_scores=residual, residual_top_k=2)
    rows = out.set_index("variable")

    assert set(out["variable"]) == {"both", "raw_only", "residual_only", "forced", "control"}
    assert "weak" not in set(out["variable"])
    assert rows.loc["both", "candidate_source"] == "raw_and_residual"
    assert rows.loc["raw_only", "candidate_source"] == "raw_only"
    assert bool(rows.loc["raw_only", "common_capacity_candidate_flag"])
    assert rows.loc["residual_only", "candidate_source"] == "residual_only"
    assert rows.loc["forced", "candidate_source"] == "force_included"
    assert rows.loc["control", "candidate_source"] == "control_reference"
    assert bool(rows.loc["both", "selected_by_raw"]) and bool(rows.loc["both", "selected_by_residual"])
    assert out["candidate_pool_rank"].tolist() == list(range(1, 6))


def test_dual_channel_pool_order_is_input_order_independent():
    residual = pd.DataFrame([
        {"variable": "both", "residual_corr": .9, "residual_lag_quality": .8, "residual_n": 100, "residual_lag": 2, "residual_status": "ok"},
        {"variable": "residual_only", "residual_corr": .8, "residual_lag_quality": .7, "residual_n": 90, "residual_lag": 3, "residual_status": "ok"},
    ])
    first = build_recommended_candidates(_ranked(), 2, ["forced"], residual_corr_scores=residual, residual_top_k=2)
    second = build_recommended_candidates(_ranked().sample(frac=1, random_state=1), 2, ["forced"], residual_corr_scores=residual.sample(frac=1, random_state=2), residual_top_k=2)
    columns = ["variable", "candidate_pool_rank", "candidate_source", "selected_by_raw", "selected_by_residual", "raw_candidate_rank", "residual_candidate_rank", "force_included"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
