import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.causal_review_evidence import EVIDENCE_COLUMNS
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

    assert set(out) == {"conditional_granger_scores", "causal_review_report", "causal_review_evidence", "final_review_summary"}
    assert not out["conditional_granger_scores"].empty
    assert not out["causal_review_report"].empty
    assert not out["final_review_summary"].empty


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
    assert list(out["causal_review_evidence"].columns) == EVIDENCE_COLUMNS
    assert list(out["final_review_summary"].columns)


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


def test_causal_review_runner_reads_optional_evidence_files(tmp_path):
    frame = _frame_with_lagged_signal()
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])
    ranked = pd.DataFrame([{"variable": "x", "candidate_grade": "C", "final_score": 0.5}])
    pd.DataFrame([{"variable": "x", "model_lift": 0.06, "rolling_stability": 0.8}]).to_csv(
        tmp_path / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"variable": "x", "importance_rank": 2, "max_importance": 0.3}]).to_csv(
        tmp_path / "model_variable_importance.csv", index=False, encoding="utf-8-sig"
    )

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
        output_dir=tmp_path,
    )

    evidence = out["causal_review_evidence"].iloc[0]
    assert evidence["model_lift"] == 0.06
    assert evidence["rolling_stability"] == 0.8
    assert evidence["model_importance_rank"] == 2


def test_causal_review_runner_ranked_window_passes_window_candidate_lags(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "lag": 80, "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])

    run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=100,
        min_rows=80,
        conditional_lag_mode="ranked_window",
        conditional_lag_window=5,
        conditional_fallback_maxlag=24,
    )

    assert captured["candidate_lags"] == {"x": list(range(75, 86))}


def test_causal_review_runner_best_only_passes_single_ranked_lag(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "lag": 80, "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])

    run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=100,
        min_rows=80,
        conditional_lag_mode="best_only",
    )

    assert captured["candidate_lags"] == {"x": [80]}


def test_causal_review_runner_full_scan_passes_no_candidate_lags(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "lag": 80, "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])

    run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=100,
        min_rows=80,
        conditional_lag_mode="full_scan",
    )

    assert captured["candidate_lags"] is None


def test_causal_review_runner_rejects_unknown_conditional_lag_mode():
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "lag": 2, "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])

    with pytest.raises(ValueError):
        run_causal_review_stage(
            frame=frame,
            target="target",
            ranked_features=ranked,
            causal_review_candidates=candidates,
            maxlag=10,
            min_rows=80,
            conditional_lag_mode="unknown",
        )


def test_causal_review_runner_passes_conditional_baseline_maxlag(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured.update(kwargs)
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "lag": 4, "candidate_grade": "A", "final_score": 0.9}])
    candidates = pd.DataFrame([{"variable": "x", "review_priority": 1, "review_tier": "tier_1"}])

    run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=20,
        min_rows=80,
        conditional_lag_mode="ranked_window",
        conditional_lag_window=2,
        conditional_fallback_maxlag=6,
        conditional_baseline_maxlag=3,
    )

    assert captured["baseline_maxlag"] == 3
    assert captured["lag_mode"] == "ranked_window"
    assert captured["lag_window"] == 2
    assert captured["fallback_maxlag"] == 6
