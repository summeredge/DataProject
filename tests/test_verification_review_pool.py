from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.pipeline import _verification_review_variables
from chem_ts_corr.verification_review_pool import (
    COLUMNS,
    add_to_verification_review_pool,
    build_initial_verification_review_pool,
    read_verification_review_pool,
    write_initial_verification_review_pool,
)


def _ranked() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "v2", "driver_rank": 2, "final_score": 0.8},
            {"variable": "v1", "driver_rank": 1, "final_score": 0.9},
            {"variable": "v3", "driver_rank": 3, "final_score": 0.7},
            {"variable": "v4", "driver_rank": 4, "final_score": 0.6},
        ]
    )


def test_initial_review_pool_is_separate_and_tracks_sources_without_mutating_screening():
    ranked = _ranked()
    before = ranked.copy(deep=True)

    pool = build_initial_verification_review_pool(
        ranked, top_k=2, manual_include=["v4"]
    )

    assert list(pool.columns) == COLUMNS
    assert pool.to_dict(orient="records") == [
        {
            "variable": "v1",
            "candidate_source": "initial_screening",
            "source_rank": 1,
            "include_reason": "Top-K进入",
        },
        {
            "variable": "v2",
            "candidate_source": "initial_screening",
            "source_rank": 2,
            "include_reason": "Top-K进入",
        },
        {
            "variable": "v4",
            "candidate_source": "manual_include",
            "source_rank": 4,
            "include_reason": "初始人工指定",
        },
    ]
    pd.testing.assert_frame_equal(ranked, before)


def test_review_pool_accepts_manual_and_confirmed_model_discovery_sources(tmp_path: Path):
    ranked = _ranked()
    write_initial_verification_review_pool(tmp_path, ranked, top_k=1)

    manual = add_to_verification_review_pool(
        tmp_path,
        ranked,
        variable="v3",
        candidate_source="manual_include",
    )
    discovered = add_to_verification_review_pool(
        tmp_path,
        ranked,
        variable="v4",
        candidate_source="model_discovery",
    )

    assert manual.loc[1, "candidate_source"] == "manual_include"
    assert discovered["candidate_source"].tolist() == [
        "initial_screening",
        "manual_include",
        "model_discovery",
    ]
    assert discovered["source_rank"].tolist() == [1, 3, 4]
    assert read_verification_review_pool(tmp_path).equals(discovered)


def test_review_pool_rejects_unknown_variable_or_source(tmp_path: Path):
    ranked = _ranked()

    with pytest.raises(ValueError, match="verification_candidate_source_invalid"):
        add_to_verification_review_pool(
            tmp_path, ranked, variable="v3", candidate_source="initial_screening"
        )
    with pytest.raises(ValueError, match="verification_candidate_variable_not_in_initial_screening"):
        add_to_verification_review_pool(
            tmp_path, ranked, variable="missing", candidate_source="manual_include"
        )


def test_downstream_variable_resolution_prefers_the_separate_review_pool(tmp_path: Path):
    ranked = _ranked()
    write_initial_verification_review_pool(tmp_path, ranked, top_k=3)
    add_to_verification_review_pool(
        tmp_path, ranked, variable="v4", candidate_source="manual_include"
    )
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path, top_k=3)

    assert _verification_review_variables(tmp_path, ranked, config, web) == [
        "v1", "v2", "v3", "v4"
    ]


def test_model_discovery_api_requires_an_explicit_confirmed_discovery(tmp_path: Path, monkeypatch):
    ranked = _ranked()
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"variable": "v4"}]).to_csv(
        tmp_path / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig"
    )
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path, top_k=1)
    monkeypatch.setattr(web, "_multipart_form", lambda handler: {
        "run_id": "run-1",
        "variable": "v4",
        "candidate_source": "model_discovery",
    })
    monkeypatch.setattr(web, "_resolve_run_dir", lambda run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda output_dir: config)
    monkeypatch.setattr(web, "_download_links", lambda *args, **kwargs: [])
    monkeypatch.setattr(web, "_branch_context_payload", lambda output_dir: {})

    result = web._add_to_verification_review_pool_response(object())

    assert result["verificationReviewPool"][0]["candidate_source"] == "initial_screening"
    assert result["verificationReviewPool"][1] == {
        "variable": "v4",
        "candidate_source": "model_discovery",
        "source_rank": 4,
        "include_reason": "模型发现后人工确认",
    }
