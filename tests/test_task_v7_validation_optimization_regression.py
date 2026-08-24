from pathlib import Path

import pandas as pd

from chem_ts_corr.validation_summary import (
    build_validation_fields,
    build_validation_summary,
    write_validation_summary,
)
from chem_ts_corr.verification_review_pool import (
    COLUMNS,
    add_to_verification_review_pool,
    read_verification_review_pool,
    write_initial_verification_review_pool,
)


def _ranked() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "v1", "driver_rank": 1, "final_score": 0.9, "lag": -3},
            {"variable": "v2", "driver_rank": 2, "final_score": 0.8, "lag": 0},
            {"variable": "v3", "driver_rank": 3, "final_score": 0.7, "lag": 2},
        ]
    )


def test_v7_review_pool_is_explicit_and_does_not_mutate_initial_screening(tmp_path: Path):
    ranked = _ranked()
    ranked_path = tmp_path / "ranked_features.csv"
    recommended_path = tmp_path / "recommended_candidates.csv"
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    ranked.assign(candidate_source="legacy_initial").to_csv(
        recommended_path, index=False, encoding="utf-8-sig"
    )
    before = {path: path.read_bytes() for path in (ranked_path, recommended_path)}

    write_initial_verification_review_pool(tmp_path, ranked, top_k=1)
    pd.DataFrame([{"variable": "v3"}]).to_csv(
        tmp_path / "model_discovered_candidates.csv", index=False, encoding="utf-8-sig"
    )
    assert read_verification_review_pool(tmp_path)["variable"].tolist() == ["v1"]

    pool = add_to_verification_review_pool(
        tmp_path, ranked, variable="v2", candidate_source="manual_include"
    )
    pool = add_to_verification_review_pool(
        tmp_path, ranked, variable="v3", candidate_source="model_discovery"
    )
    pd.DataFrame([{"variable": "v1", "status": "ok", "model_lift": 0.2}]).to_csv(
        tmp_path / "enhanced_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    write_validation_summary(tmp_path)

    assert list(pool.columns) == COLUMNS
    assert pool["candidate_source"].tolist() == [
        "initial_screening",
        "manual_include",
        "model_discovery",
    ]
    assert {path: path.read_bytes() for path in before} == before
    assert pd.read_csv(recommended_path, encoding="utf-8-sig")["candidate_source"].tolist() == [
        "legacy_initial"
    ] * 3


def test_v7_validation_contract_keeps_signed_lags_and_missing_semantics():
    fields = build_validation_fields(
        _ranked(),
        rolling_corr_scores=pd.DataFrame([{"variable": "v1", "best_lag": -2}]),
        conditional_granger_scores=pd.DataFrame([{"variable": "v1", "best_lag": 4}]),
    ).set_index("variable")
    summary = build_validation_summary(
        pd.DataFrame([{"variable": "zero"}, {"variable": "skipped"}]),
        enhanced_validation_summary=pd.DataFrame(
            [
                {"variable": "zero", "status": "ok", "model_lift": 0.0},
                {"variable": "skipped", "status": "skipped: insufficient rows"},
            ]
        ),
    ).set_index("variable")

    assert fields.loc["v1", "initial_screening_lag"] == -3
    assert fields.loc["v1", "validation_lag"] == -2
    assert fields.loc["v1", "conditional_validation_lag"] == 4
    assert build_validation_summary(
        pd.DataFrame([{"variable": "not_run"}])
    ).loc[0, "validation_status"] == "not_run"
    assert summary.loc["zero", "validation_status"] == "limited"
    assert "enhanced_screening:zero_evidence" in summary.loc["zero", "limiting_factors"]
    assert summary.loc["skipped", "validation_status"] == "not_computed"
