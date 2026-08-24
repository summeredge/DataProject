from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.model_discovery import (
    DISCOVERY_CANDIDATE_COLUMNS,
    build_discovery_candidates,
    build_exploration_candidate_pool,
)
from chem_ts_corr.verification_review_pool import (
    read_verification_review_pool,
    write_initial_verification_review_pool,
)


def _ranked_features(count: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": [f"variable_{rank}" for rank in range(1, count + 1)],
            "driver_rank": list(range(1, count + 1)),
            "final_score": [1.0 - rank / 100 for rank in range(1, count + 1)],
        }
    )


def _model_discovered(variables: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variable": variables,
            "discovery_reason": ["model_only_signal"] * len(variables),
        }
    )


def test_v5_1_exploration_pool_is_exactly_rank_k_plus_one_to_k_plus_ten():
    ranked = _ranked_features()

    pool = build_exploration_candidate_pool(ranked, top_k=20)

    assert pool["source_rank"].tolist() == list(range(21, 31))
    assert pool["variable"].tolist() == [f"variable_{rank}" for rank in range(21, 31)]
    assert not set(pool["variable"]).intersection(ranked.head(20)["variable"])
    assert pool["source_rank"].max() == 30


def test_v5_1_config_defaults_and_hard_caps_are_explicit():
    config = AnalysisConfig(Path("input.csv"), "time", "target", Path("out"))

    assert config.discovery_candidate_window == 10
    assert config.max_discovery_candidates == 5
    assert len(
        build_exploration_candidate_pool(
            _ranked_features(), top_k=20, discovery_candidate_window=50
        )
    ) == 10
    assert len(
        build_discovery_candidates(
            _model_discovered([f"variable_{rank}" for rank in range(21, 31)]),
            _ranked_features(),
            top_k=20,
            max_discovery_candidates=50,
        )
    ) == 5


def test_v5_1_discovery_candidates_are_capped_without_a_second_ranking():
    ranked = _ranked_features()
    discovered = _model_discovered([f"variable_{rank}" for rank in range(21, 31)])

    output = build_discovery_candidates(discovered, ranked, top_k=20)

    assert list(output.columns) == DISCOVERY_CANDIDATE_COLUMNS
    assert len(output) <= 5
    assert output["variable"].tolist() == [
        f"variable_{rank}" for rank in range(21, 26)
    ]
    assert "discovery_rank" not in output.columns


def test_v5_1_exploration_does_not_mutate_initial_screening():
    ranked = _ranked_features()
    before = ranked.copy(deep=True)

    build_discovery_candidates(
        _model_discovered([f"variable_{rank}" for rank in range(21, 31)]),
        ranked,
        top_k=20,
    )

    pd.testing.assert_frame_equal(ranked, before)
    assert ranked["final_score"].tolist() == before["final_score"].tolist()
    assert ranked.head(20)["variable"].tolist() == [
        f"variable_{rank}" for rank in range(1, 21)
    ]


def test_v5_1_manual_confirmation_uses_discovery_candidates_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranked = _ranked_features()
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    write_initial_verification_review_pool(tmp_path, ranked, top_k=20)
    pd.DataFrame(
        {
            "variable": ["variable_25"],
            "source_rank": [25],
            "discovery_reason": ["model_only_signal"],
        }
    ).to_csv(tmp_path / "discovery_candidates.csv", index=False, encoding="utf-8-sig")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path, top_k=20)

    monkeypatch.setattr(web, "_resolve_run_dir", lambda _run_id: tmp_path)
    monkeypatch.setattr(web, "_read_run_config", lambda _output_dir: config)
    monkeypatch.setattr(web, "_download_links", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(web, "_branch_context_payload", lambda _output_dir: {})

    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda _handler: {
            "run_id": "run-1",
            "variable": "variable_26",
            "candidate_source": "model_discovery",
        },
    )
    with pytest.raises(ValueError, match="verification_candidate_not_confirmed_model_discovery"):
        web._add_to_verification_review_pool_response(object())
    assert read_verification_review_pool(tmp_path)["variable"].tolist() == [
        f"variable_{rank}" for rank in range(1, 21)
    ]

    monkeypatch.setattr(
        web,
        "_multipart_form",
        lambda _handler: {
            "run_id": "run-1",
            "variable": "variable_25",
            "candidate_source": "model_discovery",
        },
    )

    result = web._add_to_verification_review_pool_response(object())

    assert result["verificationReviewPool"][-1]["variable"] == "variable_25"
    assert result["verificationReviewPool"][-1]["candidate_source"] == "model_discovery"
    assert read_verification_review_pool(tmp_path)["variable"].tolist()[-1] == "variable_25"
