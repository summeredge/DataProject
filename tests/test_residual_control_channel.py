from __future__ import annotations

import numpy as np
import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.report import write_outputs
from chem_ts_corr.screening import residual_corr_scores
from chem_ts_corr.service import analyze_numeric_frame
from chem_ts_corr.web import _build_result_payload


def _frame(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = pd.date_range("2026-01-01", periods=n, freq="min")
    load = rng.normal(size=n)
    target = 1.8 * load + rng.normal(scale=0.15, size=n)
    return pd.DataFrame(
        {"target": target, "load": load, "common": 1.6 * load + rng.normal(scale=0.15, size=n)},
        index=index,
    )


def test_residual_channel_common_load_is_removed_and_control_is_reference():
    scores = residual_corr_scores(_frame(), "target", ["load"], 4)

    common = scores.set_index("variable").loc["common"]
    control = scores.set_index("variable").loc["load"]
    assert abs(float(common["residual_signed_corr"])) < 0.3
    assert common["residual_status"] == "ok"
    assert control["residual_status"] == "control_reference_not_residualized"
    assert pd.isna(control["residual_corr"])


def test_residual_channel_keeps_independent_lag_and_target_mask():
    frame = _frame()
    rng = np.random.default_rng(18)
    signal = rng.normal(size=len(frame))
    frame["target"] = 1.8 * frame["load"] + signal
    frame["lagged_signal"] = pd.Series(signal, index=frame.index).shift(-2) + rng.normal(scale=0.05, size=len(frame))
    mask = pd.Series(False, index=frame.index)
    mask.iloc[60:] = True

    scores = residual_corr_scores(frame, "target", ["load"], 4, target_mask=mask)
    row = scores.set_index("variable").loc["lagged_signal"]
    assert int(row["residual_lag"]) == 2
    assert float(row["residual_corr"]) > 0.8
    assert int(row["residual_n"]) < int(mask.sum())


def test_residual_channel_reports_no_valid_controls_and_rank_deficiency():
    frame = _frame()
    frame["constant"] = 1.0
    no_valid = residual_corr_scores(frame, "target", ["missing", "constant"], 3)
    assert no_valid.set_index("variable").loc["common", "residual_status"] == "no_valid_controls"

    frame["load_copy"] = frame["load"] * 2
    deficient = residual_corr_scores(frame, "target", ["load", "load_copy"], 3)
    row = deficient.set_index("variable").loc["common"]
    assert row["residual_status"] == "rank_deficient"
    assert int(row["control_matrix_rank"]) < int(row["control_count"]) + 1


def test_residual_channel_preserves_physical_time_gaps():
    frame = _frame(160)
    frame = pd.concat([frame.iloc[:80], frame.iloc[100:]])
    frame["lagged_signal"] = frame["target"].shift(-2)
    scores = residual_corr_scores(frame, "target", ["load"], 3)
    row = scores.set_index("variable").loc["lagged_signal"]
    assert row["residual_status"] == "ok"
    assert int(row["residual_n"]) == len(frame) - 2


def test_residual_output_is_reported_without_changing_initial_outputs(tmp_path):
    frame = _frame()
    common = dict(
        input_path=tmp_path / "input.csv", time_column="time", target="target",
        output_dir=tmp_path, max_lag=3, top_k=2, enable_model=False,
        skip_model_lift=True, skip_rolling_corr=True,
    )
    without = analyze_numeric_frame(frame, AnalysisConfig(**common, residual_control_columns=[]))
    with_controls = analyze_numeric_frame(frame, AnalysisConfig(**common, residual_control_columns=["load"]))

    ranked_columns = ["variable", "driver_rank", "final_score", "candidate_grade", "recommended_use"]
    pd.testing.assert_frame_equal(
        without.ranked_features[ranked_columns], with_controls.ranked_features[ranked_columns]
    )
    assert set(without.recommended_candidates["variable"]) - {"load"} == set(with_controls.recommended_candidates["variable"])
    pd.testing.assert_frame_equal(without.risk_flags, with_controls.risk_flags)
    assert not with_controls.residual_corr_scores.empty

    write_outputs(
        tmp_path, "target", with_controls.ranked_features, with_controls.lag_scores,
        with_controls.granger_tests, with_controls.importance, with_controls.metrics,
        diagnostics=with_controls.diagnostics, residual_corr_scores=with_controls.residual_corr_scores,
        risk_flags=with_controls.risk_flags, lag_peak_quality=with_controls.lag_peak_quality,
        recommended_candidates=with_controls.recommended_candidates,
    )
    payload = _build_result_payload("run", tmp_path, AnalysisConfig(**common, residual_control_columns=["load"]))
    residual_csv = pd.read_csv(tmp_path / "residual_corr_scores.csv", encoding="utf-8-sig")
    assert len(payload["residualScores"]) == len(residual_csv) > 0
