from __future__ import annotations

from pathlib import Path

import pandas as pd

from chem_ts_corr.config import AnalysisConfig
from chem_ts_corr.service import analyze_numeric_frame

from .four_layer_cases import SyntheticCase


KEY_FIELDS = ["variable", "driver_rank", "final_score", "candidate_grade", "recommended_use", "lag", "risk_flags", "candidate_class"]


def run_case(case: SyntheticCase, output_dir: Path) -> pd.DataFrame:
    metadata = case.metadata
    config = AnalysisConfig(
        input_path=output_dir / "unused.csv", time_column="time", target=case.target, output_dir=output_dir,
        max_lag=int(metadata.get("max_lag", 6)), top_k=10, skip_model_lift=True, skip_rolling_corr=False,
        segment_column=metadata.get("segment_column"), residual_control_columns=metadata.get("residual_control_columns"),
    )
    return analyze_numeric_frame(case.frame, config).ranked_features


def metrics(case: SyntheticCase, ranked: pd.DataFrame) -> dict[str, object]:
    index = ranked.set_index("variable") if not ranked.empty else pd.DataFrame()
    ranks = {v: int(index.loc[v, "driver_rank"]) for v in case.true_drivers if v in index.index}
    false_ranks = {v: int(index.loc[v, "driver_rank"]) for v in case.spurious_variables if v in index.index}
    return {
        "top_1_hit": any(rank <= 1 for rank in ranks.values()), "top_3_recall": sum(rank <= 3 for rank in ranks.values()) / max(1, len(case.true_drivers)),
        "top_5_recall": sum(rank <= 5 for rank in ranks.values()) / max(1, len(case.true_drivers)),
        "true_driver_average_rank": sum(ranks.values()) / len(ranks) if ranks else None,
        "spurious_average_rank": sum(false_ranks.values()) / len(false_ranks) if false_ranks else None,
        "lag_identification_error": {v: abs(int(index.loc[v, "lag"]) - lag) for v, lag in case.lags.items() if v in index.index and lag > 0},
        "noise_high_grade_false_positive_rate": float(index.loc[list(case.spurious_variables.intersection(index.index)), "candidate_grade"].isin(["A", "B"]).mean()) if len(case.spurious_variables.intersection(index.index)) else 0.0,
        "downstream_average_rank": false_ranks.get("x_downstream"),
        "common_driver_average_rank": false_ranks.get("x_common"),
        "proxy_average_rank": false_ranks.get("x2_proxy", false_ranks.get("x_proxy")),
        "noise_top_k_rate": sum(rank <= 5 for name, rank in false_ranks.items() if "noise" in name) / max(1, sum("noise" in name for name in false_ranks)),
    }
