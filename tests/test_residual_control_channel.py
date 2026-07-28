from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr import screening
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.lag import compute_lag_scores, summarize_best_lags
from chem_ts_corr.report import write_outputs
from chem_ts_corr.screening import residual_corr_scores
from chem_ts_corr.service import analyze_numeric_frame
from chem_ts_corr.web import _build_result_payload


def _frame(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = pd.date_range("2026-01-01", periods=n, freq="min")
    load = rng.normal(size=n)
    return pd.DataFrame(
        {"target": 1.8 * load + rng.normal(scale=0.15, size=n), "load": load,
         "common": 1.6 * load + rng.normal(scale=0.15, size=n)}, index=index,
    )


MISSING_EVIDENCE_FIELDS = [
    "residual_pearson", "residual_spearman", "residual_signed_corr",
    "residual_corr", "residual_lag", "residual_lag_quality",
]


def _assert_missing_residual_evidence(row: pd.Series) -> None:
    assert row[MISSING_EVIDENCE_FIELDS].isna().all()


def test_residual_channel_common_load_is_removed_and_control_is_reference():
    scores = residual_corr_scores(_frame(), "target", ["load"], 4)
    common = scores.set_index("variable").loc["common"]
    control = scores.set_index("variable").loc["load"]

    assert abs(float(common["residual_signed_corr"])) < 0.3
    assert common["residual_method"] in {"pearson", "spearman"}
    assert common["residualization_method"] == "ols"
    assert common["residual_signed_corr"] == common[
        "residual_pearson" if common["residual_method"] == "pearson" else "residual_spearman"
    ]
    assert common["residual_corr"] == abs(common["residual_signed_corr"])
    assert control["residual_status"] == "control_reference_not_residualized"
    _assert_missing_residual_evidence(control)


def test_residual_channel_finds_distant_lag_independently_of_raw_lag():
    rng = np.random.default_rng(91)
    n = 480
    index = pd.date_range("2026-01-01", periods=n, freq="min")
    load = rng.normal(size=n)
    signal = rng.normal(size=n)
    frame = pd.DataFrame({
        "target": -3 * load + signal + rng.normal(scale=0.02, size=n),
        "load": load,
        "candidate": 3 * load + pd.Series(signal).shift(-20).to_numpy(),
    }, index=index)
    raw = summarize_best_lags(compute_lag_scores(frame[["target", "candidate"]], "target", 30)).iloc[0]
    residual = residual_corr_scores(frame, "target", ["load"], 30).set_index("variable").loc["candidate"]

    assert abs(int(raw["lag"])) <= 1
    assert int(residual["residual_lag"]) == 20
    assert float(residual["residual_corr"]) > 0.95


def test_residual_hints_do_not_change_simultaneous_control_or_full_scan():
    frame = _frame()
    without_hint = residual_corr_scores(frame, "target", ["load"], 12)
    wrong_hint = residual_corr_scores(frame, "target", ["load"], 12, best_lags={"common": -12, "load": 9})
    fields = [
        "residual_pearson", "residual_spearman", "residual_signed_corr", "residual_corr",
        "residual_method", "residual_lag", "residual_direction", "residual_lag_quality", "residual_status",
    ]
    pd.testing.assert_frame_equal(without_hint[fields], wrong_hint[fields])
    source = inspect.getsource(screening.residual_corr_scores)
    assert "lagged_series(" not in source
    assert "_best_lag_review_scores(" not in source
    assert "compute_lag_scores(" in source
    assert "summarize_best_lags(" in source
    assert "build_lag_peak_quality(" in source
    assert "lag_values=" not in source
    assert "fillna(0)" not in source
    assert "fillna(0.0)" not in source
    assert 'fillna({"residual_corr": 0})' not in source
    assert "abs(best_lag)" not in source

    service_source = Path("chem_ts_corr/service.py").read_text(encoding="utf-8")
    assert 'best_lags = raw_ranked.set_index("variable")["lag"].to_dict()' not in service_source
    residual_call = inspect.getsource(analyze_numeric_frame).split("residual_output = residual_corr_scores(", 1)[1].split(")\n", 1)[0]
    assert "best_lags=" not in residual_call
    assert '["condition_number"]' not in inspect.getsource(screening.risk_flags)


def test_residual_channel_reports_no_valid_controls_rank_deficiency_and_infinities(monkeypatch):
    frame = _frame()
    frame["constant"] = 1.0
    no_valid = residual_corr_scores(frame, "target", ["missing", "constant"], 3)
    no_valid_row = no_valid.set_index("variable").loc["common"]
    assert no_valid_row["residual_status"] == "no_valid_controls"
    _assert_missing_residual_evidence(no_valid_row)

    insufficient = residual_corr_scores(_frame(8), "target", ["load"], 3).set_index("variable").loc["common"]
    assert insufficient["residual_status"] == "insufficient_joint_samples"
    _assert_missing_residual_evidence(insufficient)

    frame["flat_candidate"] = frame["load"]
    with monkeypatch.context() as patch:
        patch.setattr(screening, "compute_lag_scores", lambda *args, **kwargs: pd.DataFrame())
        no_lag = residual_corr_scores(frame, "target", ["load"], 3).set_index("variable").loc["flat_candidate"]
    assert no_lag["residual_status"] == "no_valid_residual_lag"
    _assert_missing_residual_evidence(no_lag)

    frame["load_copy"] = frame["load"] * 2
    deficient = residual_corr_scores(frame, "target", ["load", "load_copy"], 3)
    row = deficient.set_index("variable").loc["common"]
    assert row["residual_status"] == "rank_deficient"
    assert row["residualization_method"] == "ols_rank_deficient"

    frame.loc[frame.index[:2], "common"] = [np.inf, -np.inf]
    finite = residual_corr_scores(frame, "target", ["load"], 3).set_index("variable").loc["common"]
    assert int(finite["residual_n"]) == len(frame) - 2 - 3
    assert not np.isinf(pd.to_numeric(finite.drop(labels=["requested_control_columns", "effective_control_columns", "residual_method", "residualization_method", "residual_direction", "residual_status"]), errors="coerce")).any()


def test_residual_fit_failure_does_not_stop_other_candidates(monkeypatch):
    frame = _frame()
    frame["good_candidate"] = frame["target"] + 0.1
    frame["bad_candidate"] = frame["target"] - 0.1
    frame["another_good_candidate"] = frame["target"] * 0.8
    original = screening._residualize

    def fail_bad(y, x, fit_mask=None):
        if y.name == "bad_candidate":
            raise np.linalg.LinAlgError("synthetic failure")
        return original(y, x, fit_mask)

    monkeypatch.setattr(screening, "_residualize", fail_bad)
    scores = residual_corr_scores(frame, "target", ["load"], 3).set_index("variable")
    bad = scores.loc["bad_candidate"]
    assert bad["residual_status"] == "fit_failed"
    _assert_missing_residual_evidence(bad)
    assert scores.loc["good_candidate", "residual_status"] == "ok"
    assert scores.loc["another_good_candidate", "residual_status"] == "ok"
    assert {"load", "good_candidate", "bad_candidate", "another_good_candidate"}.issubset(scores.index)


def test_residual_output_isolated_from_same_configuration_initial_outputs(tmp_path, monkeypatch):
    frame = _frame()
    config = AnalysisConfig(
        input_path=tmp_path / "input.csv", time_column="time", target="target", output_dir=tmp_path,
        max_lag=3, top_k=2, residual_control_columns=["load"], enable_model=False,
        skip_model_lift=True, skip_rolling_corr=True,
    )
    original = screening.residual_corr_scores
    monkeypatch.setattr(screening, "residual_corr_scores", lambda *args, **kwargs: pd.DataFrame(columns=["variable"]))
    without_output = analyze_numeric_frame(frame, config)
    monkeypatch.setattr(screening, "residual_corr_scores", original)
    with_output = analyze_numeric_frame(frame, config)

    for name in ["ranked_features", "risk_flags"]:
        pd.testing.assert_frame_equal(getattr(without_output, name), getattr(with_output, name))
    assert set(without_output.recommended_candidates["variable"]).issubset(set(with_output.recommended_candidates["variable"]))
    assert with_output.residual_corr_scores.empty is False

    write_outputs(
        tmp_path, config.target, with_output.ranked_features, with_output.lag_scores,
        with_output.granger_tests, with_output.importance, with_output.metrics,
        diagnostics=with_output.diagnostics, residual_corr_scores=with_output.residual_corr_scores,
        risk_flags=with_output.risk_flags, lag_peak_quality=with_output.lag_peak_quality,
        recommended_candidates=with_output.recommended_candidates,
    )
    payload = _build_result_payload("run", tmp_path, config)
    assert len(payload["residualScores"]) == len(with_output.residual_corr_scores) > 0
