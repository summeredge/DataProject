import numpy as np
import pandas as pd

from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.causal_review_service import REPORT_COLUMNS
from chem_ts_corr.conditional_granger import OUT_COLS


def _frame_with_lagged_signal(n: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.55 * y[t - 1] + 0.4 * x[t - 1] + 0.05 * rng.normal()
    return pd.DataFrame({"target": y, "x": x, "x1": x, "x2": rng.normal(size=n)}, index=idx)


def test_causal_review_runner_returns_expected_tables():
    frame = _frame_with_lagged_signal()
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])
    ranked = pd.DataFrame([{"variable": "x", "candidate_grade": "A", "final_score": 0.9}])

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
    )

    assert set(out) == {"conditional_granger_scores", "causal_review_report"}
    assert not out["conditional_granger_scores"].empty
    assert not out["causal_review_report"].empty


def test_causal_review_runner_handles_empty_candidates():
    frame = _frame_with_lagged_signal()
    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=pd.DataFrame(),
        causal_review_candidates=pd.DataFrame(columns=["variable"]),
        maxlag=3,
        min_rows=80,
    )

    assert out["conditional_granger_scores"].empty
    assert out["causal_review_report"].empty
    assert list(out["conditional_granger_scores"].columns) == OUT_COLS
    assert list(out["causal_review_report"].columns) == REPORT_COLUMNS


def test_causal_review_runner_respects_top_n():
    frame = _frame_with_lagged_signal()
    candidates = pd.DataFrame([{"variable": "x1"}, {"variable": "x2"}])

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=pd.DataFrame(),
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
        top_n=1,
    )

    assert out["conditional_granger_scores"]["variable"].tolist() == ["x1"]


def test_causal_review_runner_does_not_mutate_inputs():
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])
    original_ranked = ranked.copy(deep=True)
    original_candidates = candidates.copy(deep=True)

    run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
    )

    pd.testing.assert_frame_equal(ranked, original_ranked)
    pd.testing.assert_frame_equal(candidates, original_candidates)
