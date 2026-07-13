from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: Path
    time_column: str
    target: str
    output_dir: Path
    encoding: str = "utf-8-sig"
    max_lag: int = 12
    resample_rule: str | None = None
    min_valid_ratio: float = 0.7
    top_k: int = 30
    preprocess_mode: str = "raw"
    detrend_window: int = 24
    segment_column: str | None = None
    segment_mode: str = "all"
    segment_min: float | None = None
    segment_max: float | None = None
    capacity_columns: list[str] | None = None
    residual_control_columns: list[str] | None = None
    force_include_variables: list[str] | None = None
    exclude_control_columns_from_candidates: bool = True
    roles_path: Path | None = None
    random_state: int = 42
    enable_granger: bool = False
    enable_model: bool = False
    granger_maxlag: int | None = None
    max_model_features: int = 300
    max_interpolate_gap_points: int = 5
    interpolate_limit_area: str = "inside"
    max_upload_size_mb: int = 100
    skip_model_lift: bool = False
    skip_rolling_corr: bool = False
    enable_xgb_validation: bool = False
    xgb_top_n: int = 8
    xgb_max_lag: int | None = None
    xgb_whitelist: list[str] = field(default_factory=list)

    def resolved_granger_maxlag(self) -> int:
        if self.granger_maxlag is not None:
            return self.granger_maxlag
        return min(self.max_lag, 12)
