from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chem_ts_corr import web
from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.validation_summary import (
    VALIDATION_FIELDS_COLUMNS,
    build_validation_fields,
    build_validation_fields_from_output_dir,
)


def test_validation_fields_keep_stage_sources_and_signed_lags_separate():
    ranked = pd.DataFrame(
        [
            {"variable": "x", "lag": -3, "model_lift": 0.0, "final_score": 0.9},
            {"variable": "missing", "lag": np.nan, "final_score": 0.8},
        ]
    )
    rolling = pd.DataFrame(
        [{"variable": "x", "best_lag": -2, "rolling_stability": 0.7}]
    )
    granger = pd.DataFrame(
        [{"variable": "x", "best_granger_lag": 5, "status": "ok"}]
    )
    lift = pd.DataFrame(
        [{"variable": "x", "model_lift": 0.0, "status": "ok"}]
    )
    conditional = pd.DataFrame(
        [{"variable": "x", "best_lag": 4, "status": "ok"}]
    )

    fields = build_validation_fields(
        ranked,
        rolling_corr_scores=rolling,
        granger_tests=granger,
        model_lift_scores=lift,
        conditional_granger_scores=conditional,
    ).set_index("variable")

    assert list(fields.reset_index().columns) == VALIDATION_FIELDS_COLUMNS
    assert fields.loc["x", "initial_screening_lag"] == -3
    assert fields.loc["x", "validation_lag"] == -2
    assert fields.loc["x", "conditional_validation_lag"] == 4
    assert fields.loc["x", "screening_model_lift"] == 0.0
    assert fields.loc["x", "validation_model_lift"] == 0.0
    assert pd.isna(fields.loc["missing", "initial_screening_lag"])
    assert pd.isna(fields.loc["missing", "screening_model_lift"])


def test_validation_fields_use_granger_lag_only_when_rolling_validation_is_absent():
    fields = build_validation_fields(
        pd.DataFrame([{"variable": "x", "lag": 1}]),
        granger_tests=pd.DataFrame(
            [{"variable": "x", "best_granger_lag": -7, "status": "ok"}]
        ),
    )

    assert fields.iloc[0]["validation_lag"] == -7


def test_validation_fields_from_output_dir_is_read_only_and_keeps_zero(tmp_path: Path):
    ranked = pd.DataFrame(
        [{"variable": "x", "lag": -1, "model_lift": 0.0, "final_score": 0.9}]
    )
    ranked.to_csv(tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"variable": "x", "best_lag": -2}]).to_csv(
        tmp_path / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"variable": "x", "model_lift": 0.0}]).to_csv(
        tmp_path / "model_lift_scores.csv", index=False, encoding="utf-8-sig"
    )

    before = (tmp_path / "ranked_features.csv").read_bytes()
    fields = build_validation_fields_from_output_dir(tmp_path)

    assert fields.iloc[0]["initial_screening_lag"] == -1
    assert fields.iloc[0]["validation_lag"] == -2
    assert fields.iloc[0]["screening_model_lift"] == 0.0
    assert fields.iloc[0]["validation_model_lift"] == 0.0
    assert (tmp_path / "ranked_features.csv").read_bytes() == before
    assert not (tmp_path / "validation_fields.csv").exists()


def test_result_payload_exposes_v3_fields_without_changing_five_column_summary(
    tmp_path: Path,
):
    pd.DataFrame([{"variable": "x", "lag": -3, "model_lift": 0.0, "final_score": 0.9}]).to_csv(
        tmp_path / "ranked_features.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"variable": "x", "best_lag": -2}]).to_csv(
        tmp_path / "rolling_corr_scores.csv", index=False, encoding="utf-8-sig"
    )
    (tmp_path / "summary.md").write_text("# summary\n", encoding="utf-8")
    config = AnalysisConfig(tmp_path / "input.csv", "time", "target", tmp_path)

    payload = web._build_result_payload("run", tmp_path, config)

    assert list(payload["validationFields"][0]) == VALIDATION_FIELDS_COLUMNS
    assert payload["validationFields"][0]["initial_screening_lag"] == -3
    assert payload["validationFields"][0]["validation_lag"] == -2
    assert payload["validationFields"][0]["screening_model_lift"] == 0.0
    assert list(payload["validationSummary"][0]) == [
        "variable",
        "validation_status",
        "evidence_consistency",
        "supporting_methods",
        "limiting_factors",
    ]
