from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.screening import prioritize_recommended_candidates


def _candidate(
    variable: str,
    source: str,
    *,
    final_score: float = .5,
    pool_rank: int = 1,
    raw: bool = False,
    residual: bool = False,
    common_load: bool = False,
    forced: bool = False,
) -> dict[str, object]:
    return {
        "variable": variable,
        "final_score": final_score,
        "candidate_source": source,
        "selected_by_raw": raw,
        "selected_by_residual": residual,
        "common_capacity_candidate_flag": common_load,
        "force_included": forced,
        "candidate_pool_rank": pool_rank,
    }


def _residual(
    variable: str,
    corr: object,
    *,
    quality: object = .8,
    n: object = 100,
    lag: object = 1,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "variable": variable,
        "residual_corr": corr,
        "residual_lag_quality": quality,
        "residual_n": n,
        "residual_lag": lag,
        "residual_status": status,
    }


def test_dual_channel_is_prioritized_and_channel_scores_use_exact_formulas():
    candidates = pd.DataFrame([
        _candidate("raw", "raw_only", final_score=.95, pool_rank=1, raw=True),
        _candidate("residual", "residual_only", final_score=.1, pool_rank=2, residual=True),
        _candidate("dual", "raw_and_residual", final_score=.7, pool_rank=3, raw=True, residual=True),
    ])
    residual = pd.DataFrame([
        _residual("residual", .7, quality=.9),
        _residual("dual", .8, quality=.6),
    ])

    out = prioritize_recommended_candidates(candidates, residual)
    rows = out.set_index("variable")

    assert out["variable"].tolist() == ["dual", "raw", "residual"]
    assert rows.loc["dual", "candidate_priority_tier"] == 0
    assert rows.loc["dual", "residual_signal_score"] == pytest.approx(.7)
    assert rows.loc["dual", "candidate_priority_score"] == pytest.approx(.60 * .7 + .40 * .7)
    assert rows.loc["raw", "candidate_priority_score"] == pytest.approx(.95)
    assert rows.loc["residual", "candidate_priority_score"] == pytest.approx(.8)
    assert rows.loc["residual", "load_adjusted_relation_status"] == "residual_only_supported"


def test_raw_strong_residual_weak_common_load_risk_is_retained_in_tier_two():
    candidates = pd.DataFrame([
        _candidate("x", "raw_only", final_score=.9, raw=True, common_load=True),
    ])

    row = prioritize_recommended_candidates(
        candidates, pd.DataFrame([_residual("x", .1)])
    ).iloc[0]

    assert row["residual_evidence_status"] == "weak"
    assert row["load_adjusted_relation_status"] == "raw_only_common_load_risk"
    assert row["candidate_priority_tier"] == 2
    assert row["candidate_priority_score"] == pytest.approx(.9)


@pytest.mark.parametrize(
    ("corr", "quality", "status", "expected"),
    [
        (.4, .6, "ok", .5),
        (0.0, 0.0, "ok", 0.0),
        (1.0, 1.0, "ok", 1.0),
        (-.5, 1.5, "ok", .5),
        (np.nan, .8, "ok", np.nan),
        (.8, .8, "fit_failed", np.nan),
    ],
)
def test_residual_signal_score_clips_only_valid_evidence(
    corr: float, quality: float, status: str, expected: float,
):
    candidate = pd.DataFrame([
        _candidate("x", "residual_only", final_score=.1, residual=True),
    ])
    row = prioritize_recommended_candidates(
        candidate, pd.DataFrame([_residual("x", corr, quality=quality, status=status)])
    ).iloc[0]

    if np.isnan(expected):
        assert pd.isna(row["residual_signal_score"])
    else:
        assert row["residual_signal_score"] == pytest.approx(expected)


@pytest.mark.parametrize("residual", [None, _residual("x", .8, status="fit_failed")])
def test_missing_or_invalid_residual_is_not_zero_weighted(residual: dict[str, object] | None):
    candidate = pd.DataFrame([
        _candidate("x", "raw_only", final_score=.8, raw=True),
    ])
    residual_frame = pd.DataFrame() if residual is None else pd.DataFrame([residual])

    row = prioritize_recommended_candidates(candidate, residual_frame).iloc[0]

    assert pd.isna(row["residual_signal_score"])
    assert row["candidate_priority_score"] == pytest.approx(.8)
    assert row["candidate_priority_score"] != pytest.approx(.6 * .8)
    assert row["residual_evidence_status"] == ("missing" if residual is None else "insufficient")


def test_force_inclusion_combinations_keep_source_score_and_tier_contracts():
    candidates = pd.DataFrame([
        _candidate("force", "force_included", final_score=.1, pool_rank=1, forced=True),
        _candidate("raw_force", "raw_only", final_score=.8, pool_rank=2, raw=True, forced=True),
        _candidate("residual_force", "residual_only", final_score=.1, pool_rank=3, residual=True, forced=True),
        _candidate("dual_force", "raw_and_residual", final_score=.7, pool_rank=4, raw=True, residual=True, forced=True),
        _candidate("reference_force", "control_reference", final_score=.9, pool_rank=5, forced=True),
    ])
    residual = pd.DataFrame([
        _residual("residual_force", .8),
        _residual("dual_force", .8),
        _residual("reference_force", .9),
    ])

    rows = prioritize_recommended_candidates(candidates, residual).set_index("variable")

    assert rows["force_included"].astype(bool).all()
    assert rows["candidate_source"].to_dict() == {
        "dual_force": "raw_and_residual",
        "raw_force": "raw_only",
        "residual_force": "residual_only",
        "force": "force_included",
        "reference_force": "control_reference",
    }
    assert rows["candidate_priority_tier"].to_dict() == {
        "dual_force": 0, "raw_force": 1, "residual_force": 1, "force": 3, "reference_force": 4,
    }
    assert pd.isna(rows.loc["force", "candidate_priority_score"])
    assert pd.isna(rows.loc["reference_force", "candidate_priority_score"])
    assert rows["candidate_priority_rank"].tolist() == [1, 2, 3, 4, 5]


def test_priority_output_is_deterministic_with_duplicate_and_shuffled_residual_rows():
    candidates = pd.DataFrame([
        _candidate("a", "residual_only", final_score=.1, pool_rank=8, residual=True),
        _candidate("b", "raw_only", final_score=.8, pool_rank=4, raw=True),
    ])
    residual = pd.DataFrame([
        _residual("a", .5, quality=.99, n=120),
        _residual("a", .8, quality=.2, n=120),
        _residual("a", .8, quality=.9, n=100),
    ])
    expected = prioritize_recommended_candidates(candidates, residual)

    assert expected.set_index("variable").loc["a", "residual_signal_score"] == pytest.approx(.85)
    assert expected.set_index("variable")["candidate_pool_rank"].to_dict() == {"a": 8, "b": 4}
    assert expected["candidate_priority_rank"].tolist() == [1, 2]
    for seed in range(5):
        actual = prioritize_recommended_candidates(
            candidates.sample(frac=1, random_state=seed),
            residual.sample(frac=1, random_state=seed),
        )
        pd.testing.assert_frame_equal(expected, actual)


def test_priority_source_contract_preserves_missing_values_and_initial_scores():
    source = inspect.getsource(screening.prioritize_recommended_candidates)

    for forbidden in ["fillna(0)", "fillna(0.0)", "nan_to_num", "default=0"]:
        assert forbidden not in source
    assert 'frame["final_score"] = frame["candidate_priority_score"]' not in source
    assert 'frame["driver_rank"] = frame["candidate_priority_rank"]' not in source
    for field in screening.CANDIDATE_PRIORITY_COLUMNS:
        assert field in source


def test_priority_calculation_has_one_owner_and_service_calls_it_after_pr3_union():
    sources = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in Path("chem_ts_corr").glob("*.py")
    }
    for field in screening.CANDIDATE_PRIORITY_COLUMNS:
        assignment = f'frame["{field}"] ='
        owners = [path for path, source in sources.items() if assignment in source]
        assert owners == ["chem_ts_corr/screening.py"]

    service_source = sources["chem_ts_corr/service.py"]
    assert service_source.index("recommended = build_recommended_candidates(") < service_source.index(
        "recommended = prioritize_recommended_candidates("
    )
    report_source = sources["chem_ts_corr/report.py"]
    assert "candidate_source = prioritize_recommended_candidates(" in report_source
