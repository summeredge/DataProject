from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from pathlib import Path


# 预处理模式契约：
# - 新模式（raw/lowpass/lowpass_detrend/lowpass_diff）语义固定，见 docs/contracts.md；
# - 旧模式（detrend/diff/detrend_diff）继续保留兼容，不映射为新模式；
# - lowpass* 已具备 transform_frame() 和 transform_frame_causal() 基础执行能力；
# - 尚未接入 analyze_numeric_frame() / 正式 screening flow；
# - 正式流程仍由 NOT_WIRED_ANALYSIS_PREPROCESS_MODES 拦截。
SUPPORTED_PREPROCESS_MODES = frozenset(
    {"raw", "detrend", "diff", "detrend_diff", "lowpass", "lowpass_detrend", "lowpass_diff"}
)
CONTRACT_PREPROCESS_MODES = frozenset({"raw", "lowpass", "lowpass_detrend", "lowpass_diff"})
# 新模式已具备 transform_frame() / transform_frame_causal() 基础执行能力，
# 但尚未接入正式 analysis/screening 流程。
NOT_WIRED_ANALYSIS_PREPROCESS_MODES = frozenset(
    {"lowpass", "lowpass_detrend", "lowpass_diff"}
)
# 兼容别名：旧测试/旧代码仍可导入旧名称，但新代码不得依赖它表达阶段语义。
NOT_IMPLEMENTED_PREPROCESS_MODES = NOT_WIRED_ANALYSIS_PREPROCESS_MODES


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
    top_k: int = 20
    discovery_candidate_window: int = 10
    max_discovery_candidates: int = 5
    preprocess_mode: str = "raw"
    lowpass_tau_minutes: float = 5.0
    diff_interval_minutes: float | None = None
    detrend_window: int = 24
    segment_column: str | None = None
    segment_mode: str = "all"
    segment_min: float | None = None
    segment_max: float | None = None
    capacity_columns: list[str] | None = None
    residual_control_columns: list[str] | None = None
    force_include_variables: list[str] | None = None
    # Historical configuration fields are accepted for compatibility only.
    # They are intentionally not read by the current preliminary screening.
    manual_closed_loop_variables: list[str] = field(default_factory=list)
    manual_non_closed_loop_variables: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    exclude_windows: list[dict[str, str]] = field(default_factory=list)
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

    def __post_init__(self) -> None:
        if (
            isinstance(self.discovery_candidate_window, bool)
            or not isinstance(self.discovery_candidate_window, Integral)
            or self.discovery_candidate_window < 0
        ):
            raise ValueError("discovery_candidate_window must be a non-negative integer")
        if (
            isinstance(self.max_discovery_candidates, bool)
            or not isinstance(self.max_discovery_candidates, Integral)
            or self.max_discovery_candidates < 0
        ):
            raise ValueError("max_discovery_candidates must be a non-negative integer")
        if not math.isfinite(self.lowpass_tau_minutes) or self.lowpass_tau_minutes <= 0:
            raise ValueError("lowpass_tau_minutes must be a finite value greater than 0")
        if self.diff_interval_minutes is not None and (
            not math.isfinite(self.diff_interval_minutes)
            or self.diff_interval_minutes <= 0
        ):
            raise ValueError(
                "diff_interval_minutes must be a finite value greater than 0; "
                "use None for automatic interval"
            )
        if self.preprocess_mode not in SUPPORTED_PREPROCESS_MODES:
            raise ValueError(f"Unknown preprocess mode: {self.preprocess_mode!r}")

    def resolved_granger_maxlag(self) -> int:
        if self.granger_maxlag is not None:
            return self.granger_maxlag
        return min(self.max_lag, 12)
