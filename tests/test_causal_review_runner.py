import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.causal_review_runner import run_causal_review_stage
from chem_ts_corr.causal_review_evidence import EVIDENCE_COLUMNS, EVIDENCE_MATRIX_COLUMNS
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


def test_third_layer_snapshot_uses_only_current_valid_xgb(tmp_path):
    import json
    from chem_ts_corr.xgb_runner import record_xgb_execution

    frame = _frame_with_lagged_signal()
    ranked = pd.DataFrame([{"variable": "x", "candidate_grade": "A", "final_score": 0.9, "lag": 1}])
    candidates = pd.DataFrame([{"variable": "x"}])
    def review():
        return run_causal_review_stage(frame=frame, target="target", ranked_features=ranked,
                                      causal_review_candidates=candidates, maxlag=3, min_rows=80,
                                      output_dir=tmp_path)["evidence_matrix"]
    assert review().iloc[0]["xgb_status"] == "not_computed"
    directory = tmp_path / "xgb_validation"
    directory.mkdir()
    pd.DataFrame([{"model_name": name, "fold_count": 3, "mean_rmse": 1, "mean_mae": 1,
                  "median_rmse": 1, "median_mae": 1, "mean_r2": 0}
                  for name in ("M0", "M1", "M2")]).to_csv(directory / "xgb_model_summary.csv", index=False)
    pd.DataFrame([{"variable": "x", "fold_count": 3, "validation_status": "validated_incremental_signal",
                  "positive_rmse_fold_count": 3, "positive_mae_fold_count": 3, "positive_rmse_fold_ratio": 1,
                  "median_rmse_improvement_pct": 5, "median_mae_improvement_pct": 5,
                  "mean_rmse_improvement_pct": 5, "mean_mae_improvement_pct": 5,
                  "worst_fold_rmse_improvement_pct": 5}]).to_csv(
        directory / "xgb_candidate_uplift.csv", index=False)
    (directory / "xgb_validation_summary.json").write_text(json.dumps({"status": "success", "candidate_count": 1}))
    from xgb_output_helpers import write_fold_outputs
    write_fold_outputs(directory)
    record_xgb_execution(tmp_path, "success")
    snapshot = review()
    assert snapshot.iloc[0]["xgb_status"] == "validated_incremental_signal"
    snapshot.to_csv(tmp_path / "evidence_matrix.csv", index=False)
    before = (tmp_path / "evidence_matrix.csv").read_bytes()
    record_xgb_execution(tmp_path, "failed", "training failed")
    assert (tmp_path / "evidence_matrix.csv").read_bytes() == before
    assert review().iloc[0]["xgb_status"] == "missing"
    record_xgb_execution(tmp_path, "success")
    (directory / "xgb_fold_context.csv").unlink()
    assert review().iloc[0]["xgb_status"] == "missing"


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

    assert set(out) == {
        "conditional_granger_scores",
        "causal_review_report",
        "causal_review_evidence",
        "final_review_summary",
        "evidence_matrix",
    }
    assert not out["conditional_granger_scores"].empty
    assert not out["causal_review_report"].empty
    assert not out["final_review_summary"].empty
    assert list(out["evidence_matrix"].columns) == EVIDENCE_MATRIX_COLUMNS


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


def test_causal_review_runner_prefers_causal_ranked_lag_for_window(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    run_causal_review_stage(
        frame=_frame_with_lagged_signal(),
        target="target",
        ranked_features=pd.DataFrame([{"variable": "x", "lag": 2}]),
        causal_review_candidates=pd.DataFrame([{"variable": "x", "lag": 20}]),
        maxlag=24,
        min_rows=80,
        conditional_lag_mode="ranked_window",
        conditional_lag_window=1,
        prefer_ranked_lag=True,
    )

    assert captured["candidate_lags"] == {"x": [1, 2, 3]}


def test_causal_review_runner_causal_missing_lag_does_not_fallback_to_candidate_lag(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        captured["candidate_lag_status"] = kwargs["candidate_lag_status"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    run_causal_review_stage(
        frame=_frame_with_lagged_signal(),
        target="target",
        ranked_features=pd.DataFrame([{"variable": "x", "lag": np.nan}]),
        causal_review_candidates=pd.DataFrame([{"variable": "x", "lag": 20}]),
        maxlag=10,
        min_rows=80,
        conditional_lag_mode="ranked_window",
        conditional_fallback_maxlag=4,
        prefer_ranked_lag=True,
    )

    assert captured["candidate_lags"] == {"x": [1, 2, 3, 4]}
    assert captured["candidate_lag_status"] == {"x": "fallback_missing_ranked_lag"}


def test_causal_review_runner_default_keeps_candidate_lag_precedence(monkeypatch):
    captured = {}

    def fake_run_conditional_granger_tests(**kwargs):
        captured["candidate_lags"] = kwargs["candidate_lags"]
        row = {col: np.nan for col in OUT_COLS}
        row["variable"] = "x"
        row["status"] = "ok"
        return pd.DataFrame([row])

    import chem_ts_corr.causal_review_runner as runner_module

    monkeypatch.setattr(runner_module, "run_conditional_granger_tests", fake_run_conditional_granger_tests)
    run_causal_review_stage(
        frame=_frame_with_lagged_signal(),
        target="target",
        ranked_features=pd.DataFrame([{"variable": "x", "lag": 2}]),
        causal_review_candidates=pd.DataFrame([{"variable": "x", "lag": 20}]),
        maxlag=24,
        min_rows=80,
        conditional_lag_mode="ranked_window",
        conditional_lag_window=1,
    )

    assert captured["candidate_lags"] == {"x": [19, 20, 21]}


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


def test_causal_review_runner_limits_evidence_and_summary_to_top_n():
    frame = _frame_with_lagged_signal()
    candidates = pd.DataFrame([{"variable": "x1"}, {"variable": "x2"}, {"variable": "x"}])
    ranked = pd.DataFrame(
        [
            {"variable": "x1", "candidate_grade": "A", "final_score": 0.9},
            {"variable": "x2", "candidate_grade": "B", "final_score": 0.8},
            {"variable": "x", "candidate_grade": "C", "final_score": 0.7},
        ]
    )

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
        top_n=2,
    )

    expected = ["x1", "x2"]
    assert out["conditional_granger_scores"]["variable"].tolist() == expected
    assert out["causal_review_report"]["variable"].tolist() == expected
    assert out["causal_review_evidence"]["variable"].tolist() == expected
    assert set(out["final_review_summary"]["variable"]) == set(expected)


def test_causal_review_runner_keeps_unranked_selected_candidate_across_review_tables():
    frame = _frame_with_lagged_signal()
    frame["x_model_only"] = np.roll(frame["x"].to_numpy(), 1)
    candidates = pd.DataFrame([{"variable": "x1"}, {"variable": "x_model_only"}])
    ranked = pd.DataFrame([{"variable": "x1", "candidate_grade": "A", "final_score": 0.9, "lag": 1}])

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=ranked,
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
    )

    expected = ["x1", "x_model_only"]
    assert out["conditional_granger_scores"]["variable"].tolist() == expected
    assert out["causal_review_report"]["variable"].tolist() == expected
    assert out["causal_review_evidence"]["variable"].tolist() == expected
    assert out["final_review_summary"]["variable"].tolist() == expected
    unranked = out["causal_review_evidence"].set_index("variable").loc["x_model_only"]
    assert pd.isna(unranked["candidate_grade"])
    assert pd.isna(unranked["final_score"])


def test_causal_review_runner_marks_unranked_ranked_window_fallback():
    frame = _frame_with_lagged_signal()
    candidates = pd.DataFrame([{"variable": "x"}])

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=pd.DataFrame(columns=["variable", "lag"]),
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
        conditional_lag_mode="ranked_window",
    )

    assert "fallback" in str(out["conditional_granger_scores"].iloc[0]["status"])


def test_causal_review_runner_preserves_selected_candidate_metadata_without_ranked_match():
    frame = _frame_with_lagged_signal()
    frame["x_model_only"] = np.roll(frame["x"].to_numpy(), 1)
    candidates = pd.DataFrame([
        {"variable": "x_model_only", "candidate_grade": "B", "final_score": 0.66, "lag": 3}
    ])

    out = run_causal_review_stage(
        frame=frame,
        target="target",
        ranked_features=pd.DataFrame(columns=["variable", "candidate_grade", "final_score", "lag"]),
        causal_review_candidates=candidates,
        maxlag=3,
        min_rows=80,
    )

    evidence = out["causal_review_evidence"].iloc[0]
    summary = out["final_review_summary"].iloc[0]
    assert evidence["candidate_grade"] == "B"
    assert float(evidence["final_score"]) == 0.66
    assert int(evidence["lag"]) == 3
    assert summary["screening_grade"] == "B"
    assert float(summary["screening_score"]) == 0.66
    assert int(summary["screening_lag"]) == 3
