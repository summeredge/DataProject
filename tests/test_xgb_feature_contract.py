from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.xgb_validation import (
    AUTO_ALLOWED_RECOMMENDATIONS,
    DEFAULT_BASELINE_LAGS,
    DEFAULT_CANDIDATE_LAG_RADIUS,
    DEFAULT_OUTER_SPLITS,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_XGB_TOP_N,
    MAX_XGB_AUTO_TOP_N,
    MAX_XGB_TOTAL_CANDIDATES,
    XGB_CANDIDATE_COLUMNS,
    XGBFeatureSets,
    XGBTimeSplit,
    build_expanding_time_splits,
    build_xgb_candidate_pool,
    build_xgb_feature_sets,
    candidate_lag_window,
    normalize_positive_lags,
    prepare_xgb_validation_frame,
    validate_xgb_top_n,
)


def _summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "final_rank": 1,
        "final_recommendation": "priority_review",
        "screening_lag": 3,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _ranked(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "lag": 3,
        "candidate_class": "upstream_driver_candidate",
        "risk_flags": "",
        "recommended_use": "strong_screening_candidate",
        "role": "PV",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _pool(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {"candidate_order": 1, "screening_lag": 3}
    return pd.DataFrame([{**defaults, **row} for row in rows])


@pytest.mark.parametrize(
    "summary",
    [
        pd.DataFrame(),
        pd.DataFrame({"final_recommendation": ["priority_review"]}),
        pd.DataFrame({"variable": ["x"]}),
    ],
)
def test_candidate_pool_requires_complete_final_review_summary(summary: pd.DataFrame):
    with pytest.raises(ValueError, match="XGBoost fourth-level validation requires final_review_summary"):
        build_xgb_candidate_pool(summary, target="y")


@pytest.mark.parametrize("recommendation", sorted(AUTO_ALLOWED_RECOMMENDATIONS))
def test_all_allowed_recommendations_are_auto_eligible(recommendation: str):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x", "final_recommendation": recommendation}]), target="y"
    )

    assert result["variable"].tolist() == ["x"]
    assert bool(result.loc[0, "auto_eligible"]) is True


@pytest.mark.parametrize("recommendation", ["manual_review_only", "insufficient_evidence", "not_recommended"])
def test_disallowed_recommendations_are_auto_excluded(recommendation: str):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x", "final_recommendation": recommendation}]),
        target="y",
        whitelist=["x"],
    )

    assert result.loc[0, "selection_source"] == "whitelist"
    assert "recommendation_not_eligible" in result.loc[0, "auto_exclusion_reasons"]


@pytest.mark.parametrize(("lag", "eligible"), [(2, True), (0, False), (-2, False)])
def test_auto_candidate_requires_positive_screening_lag(lag: int, eligible: bool):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x", "screening_lag": lag}]),
        target="y",
        whitelist=[] if eligible else ["x"],
    )

    assert bool(result.loc[0, "auto_eligible"]) is eligible
    assert ("non_positive_screening_lag" in result.loc[0, "auto_exclusion_reasons"]) is not eligible


@pytest.mark.parametrize(
    ("variable", "controls", "recommended_use", "reason"),
    [
        ("y", [], "strong_screening_candidate", "target_variable"),
        ("c", ["c"], "strong_screening_candidate", "control_variable"),
        ("x", [], "control_variable_reference", "control_reference"),
    ],
)
def test_auto_candidate_excludes_target_and_control_references(
    variable: str, controls: list[str], recommended_use: str, reason: str
):
    summary = _summary([{"variable": variable}])
    ranked = _ranked([{"variable": variable, "recommended_use": recommended_use}])
    whitelist = [] if variable == "y" else [variable]
    result = build_xgb_candidate_pool(
        summary, ranked, target="y", control_columns=controls, whitelist=whitelist
    )

    if variable == "y":
        assert result.empty
    else:
        assert reason in result.loc[0, "auto_exclusion_reasons"]


@pytest.mark.parametrize("candidate_class", ["downstream_response", "formula_or_derived", "poor_quality"])
def test_excluded_candidate_classes_are_recorded(candidate_class: str):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x"}]),
        _ranked([{"variable": "x", "candidate_class": candidate_class}]),
        target="y",
        whitelist=["x"],
    )

    assert candidate_class in result.loc[0, "auto_exclusion_reasons"].split(";")


@pytest.mark.parametrize(
    "risk", ["strong_formula_leakage", "poor_data_quality", "target_leads_variable"]
)
def test_excluded_risk_tokens_are_recorded(risk: str):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x"}]),
        _ranked([{"variable": "x", "risk_flags": risk}]),
        target="y",
        whitelist=["x"],
    )

    assert risk in result.loc[0, "auto_exclusion_reasons"].split(";")


def test_multiple_risk_exclusion_reasons_have_deterministic_order():
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x"}]),
        _ranked(
            [{
                "variable": "x",
                "risk_flags": "target_leads_variable;poor_data_quality;strong_formula_leakage",
            }]
        ),
        target="y",
        whitelist=["x"],
    )

    assert result.loc[0, "auto_exclusion_reasons"] == (
        "strong_formula_leakage;poor_data_quality;target_leads_variable"
    )


def test_risk_tokens_use_exact_semicolon_matching():
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x"}]),
        _ranked([{"variable": "x", "risk_flags": "poor_data_quality_reviewed;other"}]),
        target="y",
    )

    assert bool(result.loc[0, "auto_eligible"]) is True


@pytest.mark.parametrize("candidate_class", ["closed_loop_related", "capacity_driven"])
def test_risk_limited_classes_are_not_directly_excluded(candidate_class: str):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x", "final_recommendation": "risk_limited_review"}]),
        _ranked([{"variable": "x", "candidate_class": candidate_class}]),
        target="y",
    )

    assert bool(result.loc[0, "auto_eligible"]) is True


def test_auto_topn_filters_before_selecting():
    summary = _summary(
        [
            {"variable": "bad", "final_rank": 1, "screening_lag": -1},
            {"variable": "good1", "final_rank": 2},
            {"variable": "good2", "final_rank": 3},
        ]
    )

    result = build_xgb_candidate_pool(summary, target="y", top_n=2)

    assert result["variable"].tolist() == ["good1", "good2"]


def test_final_rank_sort_is_numeric_stable_and_missing_last():
    summary = _summary(
        [
            {"variable": "missing1", "final_rank": None},
            {"variable": "ten", "final_rank": "10"},
            {"variable": "two1", "final_rank": 2},
            {"variable": "two2", "final_rank": "2"},
            {"variable": "missing2", "final_rank": "bad"},
        ]
    )

    result = build_xgb_candidate_pool(summary, target="y", top_n=10)

    assert result["variable"].tolist() == ["two1", "two2", "ten", "missing1", "missing2"]


@pytest.mark.parametrize(
    ("summary_row", "ranked_row", "controls", "reason"),
    [
        ({"final_recommendation": "not_recommended"}, {}, [], "recommendation_not_eligible"),
        ({"screening_lag": 0}, {}, [], "non_positive_screening_lag"),
        ({}, {"risk_flags": "poor_data_quality"}, [], "poor_data_quality"),
        ({}, {}, ["x"], "control_variable"),
    ],
)
def test_whitelist_overrides_auto_exclusions_but_preserves_reasons(
    summary_row: dict[str, object],
    ranked_row: dict[str, object],
    controls: list[str],
    reason: str,
):
    result = build_xgb_candidate_pool(
        _summary([{"variable": "x", **summary_row}]),
        _ranked([{"variable": "x", **ranked_row}]),
        target="y",
        whitelist=["x"],
        control_columns=controls,
    )

    assert result["variable"].tolist() == ["x"]
    assert result.loc[0, "selection_source"] == "whitelist"
    assert bool(result.loc[0, "force_included"]) is True
    assert bool(result.loc[0, "auto_eligible"]) is False
    assert reason in result.loc[0, "auto_exclusion_reasons"].split(";")


def test_whitelist_keeps_user_order_without_duplicates_or_target():
    result = build_xgb_candidate_pool(
        _summary([{"variable": "auto", "final_recommendation": "manual_review_only"}]),
        _ranked([{"variable": "w2"}, {"variable": "w1"}]),
        target="y",
        top_n=1,
        whitelist=["", "w2", "w1", "w2", "y"],
    )

    assert result["variable"].tolist() == ["w2", "w1"]
    assert result["candidate_order"].tolist() == [1, 2]
    assert result["selection_source"].tolist() == ["whitelist", "whitelist"]


def test_top_n_ten_selects_first_ten_eligible_candidates_in_rank_order():
    summary = _summary(
        [{"variable": f"x{number}", "final_rank": number} for number in range(1, 13)]
    )

    result = build_xgb_candidate_pool(summary, target="y", top_n=10)

    assert result["variable"].tolist() == [f"x{number}" for number in range(1, 11)]
    assert result["candidate_order"].tolist() == list(range(1, 11))


@pytest.mark.parametrize("top_n", [0, 11, -1, True, 8.5])
def test_candidate_pool_rejects_invalid_top_n_without_direct_call_bypass(top_n: object):
    with pytest.raises(ValueError, match="top_n must be an integer between 1 and 10"):
        build_xgb_candidate_pool(
            _summary([{"variable": "x"}]), target="y", top_n=top_n
        )


@pytest.mark.parametrize(
    ("top_n", "whitelist_count"),
    [(8, 4), (10, 2)],
)
def test_final_candidate_pool_allows_twelve_unique_candidates(
    top_n: int, whitelist_count: int
):
    summary = _summary(
        [{"variable": f"auto{number}", "final_rank": number} for number in range(top_n)]
    )
    whitelist = [f"manual{number}" for number in range(whitelist_count)]

    result = build_xgb_candidate_pool(
        summary, target="y", top_n=top_n, whitelist=whitelist
    )

    assert len(result) == 12
    assert result["variable"].tolist()[-whitelist_count:] == whitelist
    assert result["candidate_order"].tolist() == list(range(1, 13))


@pytest.mark.parametrize(
    ("top_n", "whitelist_count"),
    [(8, 5), (10, 3)],
)
def test_final_candidate_pool_rejects_thirteen_unique_candidates(
    top_n: int, whitelist_count: int
):
    summary = _summary(
        [{"variable": f"auto{number}", "final_rank": number} for number in range(top_n)]
    )

    with pytest.raises(
        ValueError, match="XGB total candidate count including whitelist must not exceed 12"
    ):
        build_xgb_candidate_pool(
            summary,
            target="y",
            top_n=top_n,
            whitelist=[f"manual{number}" for number in range(whitelist_count)],
        )


def test_total_candidate_limit_counts_unique_non_target_variables_only():
    summary = _summary(
        [{"variable": f"x{number}", "final_rank": number} for number in range(1, 11)]
    )

    result = build_xgb_candidate_pool(
        summary,
        target="y",
        top_n=10,
        whitelist=["x1", "x1", "manual", "manual", "y"],
    )

    assert len(result) == 11
    assert result["variable"].value_counts().max() == 1
    assert "y" not in result["variable"].tolist()
    assert result.loc[result["variable"].eq("x1"), "selection_source"].item() == (
        "final_review+whitelist"
    )
    assert bool(result.loc[result["variable"].eq("x1"), "force_included"].item()) is True


def test_auto_and_whitelist_overlap_is_one_forced_row_in_auto_position():
    result = build_xgb_candidate_pool(
        _summary([{"variable": "a", "final_rank": 1}, {"variable": "b", "final_rank": 2}]),
        target="y",
        whitelist=["b", "outside"],
    )

    assert result["variable"].tolist() == ["a", "b", "outside"]
    assert result.loc[1, "selection_source"] == "final_review+whitelist"
    assert bool(result.loc[1, "force_included"]) is True
    assert bool(result.loc[1, "auto_eligible"]) is True
    assert result.loc[2, "selection_source"] == "whitelist"


def test_candidate_metadata_precedence_and_output_contract():
    summary = _summary(
        [{"variable": "x", "screening_lag": None, "variable_role": "summary_role"}]
    )
    ranked = _ranked(
        [{"variable": "x", "lag": 7, "variable_role": "MV", "role": "DV", "risk_flags": "lag_boundary"}]
    )

    result = build_xgb_candidate_pool(summary, ranked, target="y")

    assert tuple(result.columns) == XGB_CANDIDATE_COLUMNS
    assert result.loc[0, "screening_lag"] == 7
    assert result.loc[0, "variable_role"] == "MV"
    assert result.loc[0, "risk_flags"] == "lag_boundary"


def test_candidate_pool_does_not_modify_inputs():
    summary = _summary([{"variable": "x"}])
    ranked = _ranked([{"variable": "x"}])
    before_summary = summary.copy(deep=True)
    before_ranked = ranked.copy(deep=True)

    build_xgb_candidate_pool(summary, ranked, target="y", whitelist=["x"])

    pd.testing.assert_frame_equal(summary, before_summary)
    pd.testing.assert_frame_equal(ranked, before_ranked)


def test_prepare_requires_target():
    with pytest.raises(ValueError, match="target column not found"):
        prepare_xgb_validation_frame(pd.DataFrame({"x": [1]}), "y", _pool([]))


def test_prepare_column_order_deduplication_and_missing_columns():
    frame = pd.DataFrame({"y": [1], "c": [2], "x": [3], "unused": [4]})
    pool = _pool([{"variable": "x"}, {"variable": "c", "candidate_order": 2}, {"variable": "missing", "candidate_order": 3}])

    result = prepare_xgb_validation_frame(frame, "y", pool, ["c", "c", "missing"])

    assert result.columns.tolist() == ["y", "c", "x"]


def test_prepare_numeric_conversion_infinities_and_missing_values():
    frame = pd.DataFrame(
        {"y": ["1", "bad", "3"], "x": [np.inf, -np.inf, "4.5"]}, index=[1, 2, 3]
    )

    result = prepare_xgb_validation_frame(frame, "y", _pool([{"variable": "x"}]))

    assert result.loc[1, "y"] == 1.0
    assert pd.isna(result.loc[2, "y"])
    assert pd.isna(result.loc[1, "x"])
    assert pd.isna(result.loc[2, "x"])
    assert result.loc[3, "x"] == 4.5
    assert result.index.tolist() == [1, 2, 3]


def test_prepare_stably_sorts_non_monotonic_index():
    frame = pd.DataFrame({"y": [30, 10, 20], "x": [3, 1, 2]}, index=[3, 1, 2])

    result = prepare_xgb_validation_frame(frame, "y", _pool([{"variable": "x"}]))

    assert result.index.tolist() == [1, 2, 3]
    assert result["y"].tolist() == [10, 20, 30]


def test_prepare_preserves_scale_rows_and_nan_without_modifying_input():
    frame = pd.DataFrame({"y": [1000.0, np.nan], "x": [0.01, 200.0]}, index=[2, 1])
    before = frame.copy(deep=True)

    result = prepare_xgb_validation_frame(frame, "y", _pool([{"variable": "x"}]))

    assert len(result) == 2
    assert pd.isna(result.loc[1, "y"])
    assert result.loc[2, "y"] == 1000.0
    assert result.loc[2, "x"] == 0.01
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("lags", "max_lag", "expected"),
    [
        ([5, 1, 5, 2], 10, (1, 2, 5)),
        ([0, -1, 1], 10, (1,)),
        ([1, 10, 11], 10, (1, 10)),
        ([1.0, 2, True], 10, (2,)),
    ],
)
def test_normalize_positive_lags(lags: list[object], max_lag: int, expected: tuple[int, ...]):
    assert normalize_positive_lags(lags, max_lag) == expected


def test_normalize_positive_lags_rejects_invalid_max():
    with pytest.raises(ValueError, match="max_lag"):
        normalize_positive_lags([1], 0)


@pytest.mark.parametrize(
    ("best_lag", "max_lag", "radius", "expected"),
    [
        (5, 10, 2, (3, 4, 5, 6, 7)),
        (1, 10, 2, (1, 2, 3)),
        (10, 10, 2, (8, 9, 10)),
        (None, 10, 2, ()),
        (np.nan, 10, 2, ()),
        (0, 10, 2, ()),
        (-3, 10, 2, ()),
        ("bad", 10, 2, ()),
        (2.5, 10, 2, ()),
    ],
)
def test_candidate_lag_window(
    best_lag: object, max_lag: int, radius: int, expected: tuple[int, ...]
):
    assert candidate_lag_window(best_lag, max_lag, radius) == expected


@pytest.mark.parametrize(("max_lag", "radius", "match"), [(0, 2, "max_lag"), (10, -1, "radius")])
def test_candidate_lag_window_rejects_invalid_bounds(max_lag: int, radius: int, match: str):
    with pytest.raises(ValueError, match=match):
        candidate_lag_window(3, max_lag, radius)


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": np.arange(20, dtype=float) + 100,
            "control": np.arange(20, dtype=float) * 10,
            "x": np.arange(20, dtype=float),
        }
    )


def test_feature_sets_define_m0_m1_m2_and_actual_max_lag():
    result = build_xgb_feature_sets(
        _feature_frame(),
        "y",
        _pool([{"variable": "x", "screening_lag": 4}]),
        control_columns=["control"],
        max_lag=6,
        baseline_lags=[2, 5],
        candidate_lag_radius=1,
    )

    assert isinstance(result, XGBFeatureSets)
    assert result.m0_features == ("y__lag_1",)
    assert result.m1_features == (
        "y__lag_1", "y__lag_2", "y__lag_5",
        "control__lag_1", "control__lag_2", "control__lag_5",
    )
    assert result.candidate_feature_map == {"x": ("x__lag_3", "x__lag_4", "x__lag_5")}
    assert result.m2_features == (*result.m1_features, "x__lag_3", "x__lag_4", "x__lag_5")
    assert set(result.m0_features) <= set(result.m1_features) < set(result.m2_features)
    assert result.max_used_lag == 5


@pytest.mark.parametrize("lag", [0, -2, None])
def test_non_positive_or_missing_candidate_lag_keeps_empty_mapping(lag: object):
    result = build_xgb_feature_sets(
        _feature_frame(), "y", _pool([{"variable": "x", "screening_lag": lag}]),
        max_lag=5, baseline_lags=[2]
    )

    assert result.candidate_feature_map == {"x": ()}
    assert result.m2_features == result.m1_features


def test_missing_candidate_column_keeps_empty_mapping():
    result = build_xgb_feature_sets(
        _feature_frame(), "y", _pool([{"variable": "missing"}]), max_lag=5, baseline_lags=[2]
    )

    assert result.candidate_feature_map == {"missing": ()}
    assert result.m2_features == result.m1_features


def test_candidate_control_does_not_duplicate_existing_features():
    result = build_xgb_feature_sets(
        _feature_frame(), "y", _pool([{"variable": "control", "screening_lag": 2}]),
        control_columns=["control"], max_lag=3, baseline_lags=[1, 2, 3], candidate_lag_radius=1
    )

    assert result.candidate_feature_map == {"control": ()}
    assert len(result.m2_features) == len(set(result.m2_features))


def test_feature_order_follows_candidate_order():
    frame = _feature_frame().assign(z=np.arange(20, dtype=float) + 50)
    pool = _pool(
        [
            {"variable": "x", "candidate_order": 2, "screening_lag": 2},
            {"variable": "z", "candidate_order": 1, "screening_lag": 3},
        ]
    )

    result = build_xgb_feature_sets(
        frame, "y", pool, max_lag=4, baseline_lags=[1], candidate_lag_radius=0
    )

    assert result.m2_features == ("y__lag_1", "z__lag_3", "x__lag_2")
    assert tuple(result.features.columns) == result.m2_features


def test_shift_uses_past_values_and_never_future_values():
    frame = _feature_frame()
    result = build_xgb_feature_sets(
        frame, "y", _pool([{"variable": "x", "screening_lag": 2}]),
        max_lag=2, baseline_lags=[1], candidate_lag_radius=0
    )
    timestamp = result.features.index[0]

    assert result.features.loc[timestamp, "x__lag_2"] == frame.loc[timestamp - 2, "x"]
    assert result.features.loc[timestamp, "y__lag_1"] == frame.loc[timestamp - 1, "y"]
    assert result.target.loc[timestamp] == frame.loc[timestamp, "y"]


def test_all_models_share_one_complete_case_index():
    frame = _feature_frame()
    frame.loc[10, "x"] = np.nan
    result = build_xgb_feature_sets(
        frame, "y", _pool([{"variable": "x", "screening_lag": 2}]),
        max_lag=3, baseline_lags=[1, 3], candidate_lag_radius=0
    )

    assert result.features.index.equals(result.target.index)
    assert result.features.loc[:, result.m0_features].index.equals(result.features.index)
    assert result.features.loc[:, result.m1_features].index.equals(result.features.index)
    assert not result.features.isna().any().any()
    assert not result.target.isna().any()


def test_feature_builder_does_not_modify_inputs():
    frame = _feature_frame()
    pool = _pool([{"variable": "x"}])
    before_frame = frame.copy(deep=True)
    before_pool = pool.copy(deep=True)

    build_xgb_feature_sets(frame, "y", pool, max_lag=5)

    pd.testing.assert_frame_equal(frame, before_frame)
    pd.testing.assert_frame_equal(pool, before_pool)


def test_default_constants_match_contract():
    assert DEFAULT_XGB_TOP_N == 8
    assert MAX_XGB_AUTO_TOP_N == 10
    assert MAX_XGB_TOTAL_CANDIDATES == 12
    assert validate_xgb_top_n(np.int64(10)) == 10
    assert DEFAULT_BASELINE_LAGS == (1, 2, 5, 10, 30, 60)
    assert DEFAULT_CANDIDATE_LAG_RADIUS == 2
    assert DEFAULT_OUTER_SPLITS == 3
    assert DEFAULT_VALIDATION_FRACTION == pytest.approx(0.15)


def test_default_time_split_returns_three_typed_folds():
    splits = build_expanding_time_splits(1000)

    assert len(splits) == 3
    assert all(isinstance(split, XGBTimeSplit) for split in splits)
    assert [split.fold for split in splits] == [0, 1, 2]


def test_expanding_time_split_exact_boundaries():
    splits = build_expanding_time_splits(
        100,
        n_splits=3,
        gap=2,
        validation_fraction=0.20,
        min_train_rows=5,
        min_validation_rows=4,
        min_test_rows=20,
    )

    assert splits == [
        XGBTimeSplit(0, slice(0, 17), slice(19, 23), slice(25, 50), 2),
        XGBTimeSplit(1, slice(0, 37), slice(39, 48), slice(50, 75), 2),
        XGBTimeSplit(2, slice(0, 57), slice(59, 73), slice(75, 100), 2),
    ]


def test_expanding_time_split_invariants_and_determinism():
    kwargs = dict(
        n_splits=4, gap=3, validation_fraction=0.1,
        min_train_rows=10, min_validation_rows=5, min_test_rows=20,
    )
    first = build_expanding_time_splits(200, **kwargs)
    second = build_expanding_time_splits(200, **kwargs)

    assert first == second
    assert [split.train_slice.stop for split in first] == sorted(
        split.train_slice.stop for split in first
    )
    assert all(split.train_slice.start == 0 for split in first)
    assert all(split.validation_slice.start - split.train_slice.stop == 3 for split in first)
    assert all(split.test_slice.start - split.validation_slice.stop == 3 for split in first)
    assert all(left.test_slice.stop <= right.test_slice.start for left, right in zip(first, first[1:]))
    assert first[-1].test_slice.stop == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 100, "n_splits": 0}, "n_splits"),
        ({"n_samples": 100, "gap": -1}, "gap"),
        ({"n_samples": 100, "validation_fraction": 0}, "validation_fraction"),
        ({"n_samples": 100, "validation_fraction": 1}, "validation_fraction"),
        ({"n_samples": 100, "min_train_rows": 0}, "min_train_rows"),
        ({"n_samples": 100, "min_validation_rows": 0}, "min_validation_rows"),
        ({"n_samples": 100, "min_test_rows": 0}, "min_test_rows"),
    ],
)
def test_time_split_rejects_invalid_parameters(kwargs: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        build_expanding_time_splits(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"n_samples": 100, "n_splits": 3, "min_train_rows": 30,
             "min_validation_rows": 4, "min_test_rows": 20},
            "train rows",
        ),
        (
            {"n_samples": 20, "n_splits": 3, "min_train_rows": 1,
             "min_validation_rows": 1, "min_test_rows": 6},
            "test_size",
        ),
        (
            {"n_samples": 100, "n_splits": 3, "gap": 20, "min_train_rows": 5,
             "min_validation_rows": 20, "min_test_rows": 20},
            "train rows",
        ),
    ],
)
def test_time_split_reports_specific_data_shortage(kwargs: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        build_expanding_time_splits(**kwargs)


@pytest.mark.parametrize(
    "forbidden",
    [
        "train_test_split",
        "KFold",
        "ShuffleSplit",
        "shuffle=True",
        "np.random",
        "random.",
        "shift(-",
        "abs(best_lag)",
        "abs(int(best_lag))",
        "standardize_frame",
        "StandardScaler",
        "MinMaxScaler",
        "zscore",
        "final_score",
        "driver_rank",
    ],
)
def test_xgb_contract_source_avoids_training_leakage_and_scaling(forbidden: str):
    source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    assert forbidden not in source


def test_xgb_model_import_is_optional_and_isolated_to_validation_module():
    source = Path("chem_ts_corr/xgb_validation.py").read_text(encoding="utf-8")
    production_sources = {
        path: path.read_text(encoding="utf-8")
        for path in Path("chem_ts_corr").glob("*.py")
    }

    assert "try:\n    from xgboost import XGBRegressor" in source
    assert "except ImportError:\n    XGBRegressor = None" in source
    assert "\nimport xgboost" not in source
    assert [
        path for path, text in production_sources.items() if "from xgboost import XGBRegressor" in text
    ] == [Path("chem_ts_corr/xgb_validation.py")]


def test_xgb_extra_is_independent_and_preserves_existing_extras():
    source = Path("pyproject.toml").read_text(encoding="utf-8")
    base = source.split("[project.optional-dependencies]", 1)[0]
    optional = source.split("[project.optional-dependencies]", 1)[1]

    assert "xgboost" not in base.lower()
    assert "xgb = [" in optional
    assert '"xgboost>=2.0"' in optional
    assert '"scikit-learn>=1.3"' in optional
    assert "full = [" in optional
    assert "dev = [" in optional
