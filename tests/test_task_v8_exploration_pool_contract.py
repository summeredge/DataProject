from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.model_discovery import build_model_discovered_candidates
from chem_ts_corr.pipeline import (
    _limited_model_exploration_variables,
    _limit_model_exploration_candidates,
)
from chem_ts_corr.verification_review_pool import (
    COLUMNS,
    read_verification_review_pool,
    write_initial_verification_review_pool,
)


def _ranked_features(count: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": [f"variable_{rank}" for rank in range(1, count + 1)],
            "driver_rank": list(range(1, count + 1)),
            "final_score": [1.0 - rank / 100 for rank in range(1, count + 1)],
            "lag": [rank % 3 - 1 for rank in range(1, count + 1)],
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_review_pool_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: AnalysisConfig,
    *,
    variable: str,
) -> None:
    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda _handler: {
            "run_id": "run-1",
            "variable": variable,
            "candidate_source": "model_discovery",
        },
    )
    monkeypatch.setattr(web, "_resolve_run_dir", lambda _run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda _output_dir: config)
    monkeypatch.setattr(web, "_download_links", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(web, "_branch_context_payload", lambda _output_dir: {})


def test_v8_review_pool_is_independent_and_preserves_ranked_top_k(tmp_path: Path):
    ranked = _ranked_features()
    ranked_path = tmp_path / "ranked_features.csv"
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    before_hash = _sha256(ranked_path)
    before_final_scores = ranked["final_score"].tolist()
    before_top_k = ranked.head(20)["variable"].tolist()

    pool = write_initial_verification_review_pool(tmp_path, ranked, top_k=20)

    pool_path = tmp_path / "verification_review_pool.csv"
    assert pool_path.exists()
    assert pool_path != ranked_path
    assert list(pool.columns) == COLUMNS
    assert pool["variable"].tolist() == before_top_k
    assert _sha256(ranked_path) == before_hash

    after_ranked = pd.read_csv(ranked_path, encoding="utf-8-sig")
    assert after_ranked["final_score"].tolist() == pytest.approx(before_final_scores)
    assert after_ranked.head(20)["variable"].tolist() == before_top_k


def test_v8_model_exploration_is_limited_to_rank_k_plus_one_through_k_plus_ten():
    ranked = _ranked_features(count=100)
    ranked_before = ranked.copy(deep=True)
    final_scores_before = ranked["final_score"].tolist()

    exploration_candidates = _limited_model_exploration_variables(ranked, top_k=20)
    exploration_ranks = ranked.set_index("variable").loc[
        exploration_candidates, "driver_rank"
    ].tolist()

    assert len(exploration_candidates) <= 10
    assert exploration_candidates == [f"variable_{rank}" for rank in range(21, 31)]
    assert all(21 <= rank <= 30 for rank in exploration_ranks)
    assert set(exploration_candidates).isdisjoint(set(ranked.head(20)["variable"]))
    assert not any(rank > 30 for rank in exploration_ranks)
    pd.testing.assert_frame_equal(ranked, ranked_before)
    assert ranked["driver_rank"].tolist() == list(range(1, 101))
    assert ranked["final_score"].tolist() == final_scores_before


def test_v8_model_discovery_output_is_capped_and_cannot_create_a_second_ranking(
    tmp_path: Path,
):
    ranked = _ranked_features(count=60)
    ranked_before = ranked.copy(deep=True)
    review_pool_before = write_initial_verification_review_pool(
        tmp_path, ranked, top_k=20
    ).copy(deep=True)
    importance = pd.DataFrame(
        [
            {
                "feature": f"variable_{rank}__lag_1",
                "variable": f"variable_{rank}",
                "importance": float(100 - rank),
                "lag": 1,
            }
            for rank in range(21, 36)
        ]
    )
    assert len(importance["variable"].unique()) > 5

    discovered = build_model_discovered_candidates(
        importance,
        ranked,
        screening_top_n=20,
    )
    limited = _limit_model_exploration_candidates(discovered, ranked, top_k=20)

    assert len(limited) <= 5
    assert limited["variable"].tolist() == [
        f"variable_{rank}" for rank in range(21, 26)
    ]
    assert set(limited["variable"]).isdisjoint(set(ranked.head(20)["variable"]))
    assert not {"validation_score", "validation_rank", "discovery_rank"}.intersection(
        limited.columns
    )
    pd.testing.assert_frame_equal(ranked, ranked_before)
    pd.testing.assert_frame_equal(read_verification_review_pool(tmp_path), review_pool_before)
    assert set(read_verification_review_pool(tmp_path)["variable"]) == set(
        ranked.head(20)["variable"]
    )


def test_v8_model_discovery_requires_manual_confirmation_before_pool_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranked = _ranked_features()
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    write_initial_verification_review_pool(tmp_path, ranked, top_k=20)
    config = AnalysisConfig(
        tmp_path / "input.csv",
        "time",
        "target",
        tmp_path,
        top_k=20,
    )
    _patch_review_pool_request(
        monkeypatch,
        tmp_path,
        config,
        variable="variable_25",
    )

    with pytest.raises(ValueError, match="verification_candidate_not_confirmed_model_discovery"):
        web._add_to_verification_review_pool_response(object())
    assert read_verification_review_pool(tmp_path)["variable"].tolist() == [
        f"variable_{rank}" for rank in range(1, 21)
    ]

    pd.DataFrame({"variable": ["variable_25"]}).to_csv(
        tmp_path / "model_discovered_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pool = web._add_to_verification_review_pool_response(object())

    assert pool["verificationReviewPool"][-1]["variable"] == "variable_25"
    assert pool["verificationReviewPool"][-1]["candidate_source"] == "model_discovery"


def test_v8_candidate_source_stays_scoped_to_review_pool(tmp_path: Path):
    ranked = _ranked_features()
    ranked_path = tmp_path / "ranked_features.csv"
    recommended_path = tmp_path / "recommended_candidates.csv"
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    ranked.assign(candidate_source="raw_only").to_csv(
        recommended_path, index=False, encoding="utf-8-sig"
    )
    before_ranked_hash = _sha256(ranked_path)
    before_recommended_hash = _sha256(recommended_path)

    pool = write_initial_verification_review_pool(tmp_path, ranked, top_k=20)

    assert "candidate_source" not in ranked.columns
    assert set(pool["candidate_source"]) == {"initial_screening"}
    assert not {"validation_score", "validation_rank", "discovery_rank"}.intersection(
        pool.columns
    )
    assert _sha256(ranked_path) == before_ranked_hash
    assert _sha256(recommended_path) == before_recommended_hash
    assert pd.read_csv(recommended_path, encoding="utf-8-sig")["candidate_source"].tolist() == [
        "raw_only"
    ] * len(ranked)
