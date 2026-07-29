from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr import screening
from chem_ts_corr.screening import (
    RAW_CANDIDATE_MIN_SCORE,
    RESIDUAL_CANDIDATE_MIN_CORR,
    RESIDUAL_CANDIDATE_MIN_N,
    build_recommended_candidates,
)


def _ranked() -> pd.DataFrame:
    return pd.DataFrame([
        {"variable": "both", "final_score": .95, "association_score": .9, "lag_quality": .8, "raw_corr": .9, "variable_role": "candidate", "force_included": False},
        {"variable": "raw_only", "final_score": .90, "association_score": .8, "lag_quality": .7, "raw_corr": .8, "variable_role": "candidate", "force_included": False},
        {"variable": "residual_only", "final_score": .30, "association_score": .3, "lag_quality": .2, "raw_corr": .3, "variable_role": "candidate", "force_included": False},
        {"variable": "weak", "final_score": .20, "association_score": .2, "lag_quality": .1, "raw_corr": .2, "variable_role": "candidate", "force_included": False},
        {"variable": "forced", "final_score": .10, "association_score": .1, "lag_quality": .1, "raw_corr": .1, "variable_role": "candidate", "force_included": False},
        {"variable": "control", "final_score": .99, "association_score": .9, "lag_quality": .9, "raw_corr": .9, "variable_role": "residual_control", "is_residual_control": True, "force_included": False},
    ])


def _residual(
    variable: str,
    corr: float,
    *,
    quality: float = .8,
    n: int = 100,
    lag: int = 1,
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


def test_dual_channel_union_preserves_sources_and_is_not_retruncated():
    residual = pd.DataFrame([
        _residual("both", .9, quality=.8, lag=2),
        _residual("residual_only", .8, quality=.7, n=90, lag=3),
        _residual("raw_only", .1, quality=.1, n=80),
        _residual("weak", .1, quality=.1, n=80),
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
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("final_score", "residual_corr", "residual_n", "quality", "lag", "expected"),
    [
        (RAW_CANDIDATE_MIN_SCORE, .1, 100, .8, 1, True),
        (RAW_CANDIDATE_MIN_SCORE - .001, .1, 100, .8, 1, False),
        (.1, RESIDUAL_CANDIDATE_MIN_CORR, RESIDUAL_CANDIDATE_MIN_N, .8, 1, True),
        (.1, RESIDUAL_CANDIDATE_MIN_CORR - .001, RESIDUAL_CANDIDATE_MIN_N, .8, 1, False),
        (.1, RESIDUAL_CANDIDATE_MIN_CORR, RESIDUAL_CANDIDATE_MIN_N - 1, .8, 1, False),
        (.1, RESIDUAL_CANDIDATE_MIN_CORR, RESIDUAL_CANDIDATE_MIN_N, np.nan, 1, False),
        (.1, RESIDUAL_CANDIDATE_MIN_CORR, RESIDUAL_CANDIDATE_MIN_N, .8, np.nan, False),
    ],
)
def test_candidate_thresholds_are_inclusive_only_at_the_contract_boundary(
    final_score: float,
    residual_corr: float,
    residual_n: int,
    quality: float,
    lag: float,
    expected: bool,
):
    ranked = _ranked().loc[lambda frame: frame["variable"].eq("weak")].copy()
    ranked["final_score"] = final_score
    result = build_recommended_candidates(
        ranked,
        top_k=20,
        residual_corr_scores=pd.DataFrame([
            _residual("weak", residual_corr, n=residual_n, quality=quality, lag=lag),
        ]),
    )

    assert (not result.empty) is expected


def test_force_inclusion_preserves_each_channel_source_and_reference_role():
    ranked = pd.DataFrame(
        [
            {"variable": "weak_force", "final_score": .1, "raw_corr": .1, "variable_role": "candidate"},
            {"variable": "raw_force", "final_score": .8, "raw_corr": .8, "variable_role": "candidate"},
            {"variable": "residual_force", "final_score": .1, "raw_corr": .1, "variable_role": "candidate"},
            {"variable": "both_force", "final_score": .9, "raw_corr": .9, "variable_role": "candidate"},
            {"variable": "reference_force", "final_score": .9, "raw_corr": .9, "variable_role": "residual_control", "is_residual_control": True},
        ]
    )
    residual = pd.DataFrame([
        _residual("raw_force", .1),
        _residual("residual_force", .8),
        _residual("both_force", .9),
    ])
    forced = ranked["variable"].tolist()

    rows = build_recommended_candidates(
        ranked, top_k=20, force_include_variables=forced, residual_corr_scores=residual,
    ).set_index("variable")

    assert rows["force_included"].astype(bool).all()
    assert rows["candidate_source"].to_dict() == {
        "both_force": "raw_and_residual",
        "raw_force": "raw_only",
        "residual_force": "residual_only",
        "weak_force": "force_included",
        "reference_force": "control_reference",
    }


def test_duplicate_residual_rows_select_the_best_row_deterministically():
    ranked = pd.DataFrame([
        {"variable": "duplicate", "final_score": .1, "raw_corr": .1, "variable_role": "candidate"},
        {"variable": "rival", "final_score": .1, "raw_corr": .1, "variable_role": "candidate"},
    ])
    residual = pd.DataFrame([
        _residual("duplicate", .5, quality=.99, n=120),
        _residual("duplicate", .8, quality=.2, n=120),
        _residual("duplicate", .8, quality=.9, n=100),
        _residual("rival", .8, quality=.5, n=120),
    ])

    expected = build_recommended_candidates(
        ranked, top_k=20, residual_corr_scores=residual, residual_top_k=2,
    )
    assert expected["variable"].tolist() == ["duplicate", "rival"]
    assert expected["residual_candidate_rank"].tolist() == [1, 2]

    for seed in range(5):
        actual = build_recommended_candidates(
            ranked.sample(frac=1, random_state=seed),
            top_k=20,
            residual_corr_scores=residual.sample(frac=1, random_state=seed),
            residual_top_k=2,
        )
        pd.testing.assert_frame_equal(expected, actual)


def test_residual_deduplication_source_contract_sorts_before_dropping_duplicates():
    source = inspect.getsource(screening.build_recommended_candidates)
    residual_block = source.split("valid_residual_rows = residual.", 1)[1].split("selected_residual_rows =", 1)[0]
    sort_block = residual_block.split("drop_duplicates(", 1)[0]

    assert residual_block.index("sort_values(") < residual_block.index("drop_duplicates(")
    assert residual_block.count("sort_values(") == 1
    sort_columns = ["_residual_corr", "_quality_sort", "_residual_n", "_status_priority", "variable"]
    assert [sort_block.index(f'"{column}"') for column in sort_columns] == sorted(
        sort_block.index(f'"{column}"') for column in sort_columns
    )
    assert "ascending=[False, False, False, True, True]" in sort_block
    assert "RAW_CANDIDATE_MIN_SCORE" in source
    assert "RESIDUAL_CANDIDATE_MIN_CORR" in source
    assert "RESIDUAL_CANDIDATE_MIN_N" in source
    assert "locals()" not in source
    assert "globals()" not in source


def test_candidate_union_selection_is_only_implemented_in_screening():
    production_sources = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in Path("chem_ts_corr").glob("*.py")
    }

    for assignment in [
        '["selected_by_raw"] = pool["variable"].isin',
        '["selected_by_residual"] = pool["variable"].isin',
        '["candidate_source"] = np.select',
    ]:
        owners = [path for path, source in production_sources.items() if assignment in source]
        assert owners == ["chem_ts_corr/screening.py"]
