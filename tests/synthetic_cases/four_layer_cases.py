from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticCase:
    frame: pd.DataFrame
    target: str
    true_drivers: frozenset[str]
    spurious_variables: frozenset[str]
    lags: dict[str, int]
    directions: dict[str, str]
    metadata: dict[str, object]


def _base(n: int, seed: int, noise: float) -> tuple[np.random.Generator, pd.DatetimeIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng, pd.date_range("2025-01-01", periods=n, freq="5min"), rng.normal(scale=noise, size=n)


def _case(name: str, frame: pd.DataFrame, target: str, drivers=(), spurious=(), lags=None, directions=None, **metadata) -> SyntheticCase:
    return SyntheticCase(frame, target, frozenset(drivers), frozenset(spurious), lags or {}, directions or {}, {"scenario": name, **metadata})


def true_lagged_driver(n: int = 360, noise: float = 0.15, seed: int = 101) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); lag = 4; x = rng.normal(size=n)
    y = np.roll(x, lag) + eps; y[:lag] = eps[:lag]
    return _case("true_lagged_driver", pd.DataFrame({"target": y, "x_driver": x, "noise": rng.normal(size=n)}, index=index), "target", ["x_driver"], ["noise"], {"x_driver": lag}, {"x_driver": "variable_leads_target"}, seed=seed, n=n, noise=noise)


def downstream_response(n: int = 360, noise: float = 0.12, seed: int = 102) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); lag = 3; y = rng.normal(size=n); x = np.roll(y, lag) + eps; x[:lag] = eps[:lag]
    return _case("downstream_response", pd.DataFrame({"target": y, "x_downstream": x, "noise": rng.normal(size=n)}, index=index), "target", [], ["x_downstream", "noise"], {"x_downstream": -lag}, {"x_downstream": "target_leads_variable"}, seed=seed, n=n, noise=noise)


def common_driver(n: int = 420, noise: float = 0.12, seed: int = 103) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); z = rng.normal(size=n); y = np.roll(z, 3) + eps; x = np.roll(z, 3) + rng.normal(scale=noise, size=n); y[:3] = eps[:3]
    return _case("common_driver", pd.DataFrame({"target": y, "z_driver": z, "x_common": x}, index=index), "target", ["z_driver"], ["x_common"], {"z_driver": 3}, {"z_driver": "variable_leads_target"}, seed=seed, n=n, noise=noise, residual_control_columns=["z_driver"])


def collinear_proxy(n: int = 360, noise: float = 0.1, seed: int = 104) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); x1 = rng.normal(size=n); x2 = x1 + rng.normal(scale=0.03, size=n); y = np.roll(x1, 2) + eps; y[:2] = eps[:2]
    return _case("collinear_proxy", pd.DataFrame({"target": y, "x1_driver": x1, "x2_proxy": x2}, index=index), "target", ["x1_driver"], ["x2_proxy"], {"x1_driver": 2}, {"x1_driver": "variable_leads_target"}, seed=seed, n=n, noise=noise)


def nonlinear_stable_driver(n: int = 400, noise: float = 0.08, seed: int = 105) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); x = rng.uniform(-1, 1, size=n); y = np.roll(x ** 2, 2) + eps; y[:2] = eps[:2]
    return _case("nonlinear_stable_driver", pd.DataFrame({"target": y, "x_nonlinear": x, "noise": rng.normal(size=n)}, index=index), "target", ["x_nonlinear"], ["noise"], {"x_nonlinear": 2}, {"x_nonlinear": "variable_leads_target"}, seed=seed, n=n, noise=noise)


def regime_sign_reversal(n: int = 420, noise: float = 0.08, seed: int = 106) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); x = rng.normal(size=n); sign = np.where(np.arange(n) < n // 2, 1.0, -1.0); y = sign * np.roll(x, 2) + eps; y[:2] = eps[:2]
    return _case("regime_sign_reversal", pd.DataFrame({"target": y, "x_reversal": x, "load": sign}, index=index), "target", ["x_reversal"], [], {"x_reversal": 2}, {"x_reversal": "variable_leads_target"}, seed=seed, n=n, noise=noise, segment_column="load")


def outlier_driven_correlation(n: int = 360, noise: float = 0.1, seed: int = 107) -> SyntheticCase:
    rng, index, _ = _base(n, seed, noise); x = rng.normal(size=n); y = rng.normal(size=n); points = np.arange(20, n, 60); y[points] = x[points] * 30
    return _case("outlier_driven_correlation", pd.DataFrame({"target": y, "x_outlier": x, "noise": rng.normal(size=n)}, index=index), "target", [], ["x_outlier", "noise"], {}, {}, seed=seed, n=n, noise=noise, outlier_count=len(points))


def lag_boundary_artifact(n: int = 360, noise: float = 0.1, seed: int = 108) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise); x = rng.normal(size=n); y = np.roll(x, 6) + eps; y[:6] = eps[:6]
    return _case("lag_boundary_artifact", pd.DataFrame({"target": y, "x_boundary": x}, index=index), "target", ["x_boundary"], [], {"x_boundary": 6}, {"x_boundary": "variable_leads_target"}, seed=seed, n=n, noise=noise, max_lag=6)


def noise_only(n: int = 360, noise: float = 1.0, seed: int = 109) -> SyntheticCase:
    rng, index, y = _base(n, seed, noise)
    return _case("noise_only", pd.DataFrame({"target": y, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n), "noise_c": rng.normal(size=n)}, index=index), "target", [], ["noise_a", "noise_b", "noise_c"], {}, {}, seed=seed, n=n, noise=noise)


def mixed_evidence(n: int = 480, noise: float = 0.12, seed: int = 110) -> SyntheticCase:
    base = common_driver(n, noise, seed); rng = np.random.default_rng(seed + 1); x = rng.normal(size=n); y = base.frame["target"].to_numpy() + np.roll(x, 4); y[:4] = base.frame["target"].to_numpy()[:4]
    downstream = np.roll(y, 2) + rng.normal(scale=noise, size=n); proxy = x + rng.normal(scale=0.03, size=n)
    frame = pd.DataFrame({"target": y, "x_driver": x, "x_proxy": proxy, "z_driver": base.frame["z_driver"], "x_common": base.frame["x_common"], "x_downstream": downstream, "noise": rng.normal(size=n)}, index=base.frame.index)
    return _case("mixed_evidence", frame, "target", ["x_driver", "z_driver"], ["x_proxy", "x_common", "x_downstream", "noise"], {"x_driver": 4, "z_driver": 3, "x_downstream": -2}, {"x_driver": "variable_leads_target", "z_driver": "variable_leads_target", "x_downstream": "target_leads_variable"}, seed=seed, n=n, noise=noise, residual_control_columns=["z_driver"])


CASES = {name: globals()[name] for name in ["true_lagged_driver", "downstream_response", "common_driver", "collinear_proxy", "nonlinear_stable_driver", "regime_sign_reversal", "outlier_driven_correlation", "lag_boundary_artifact", "noise_only", "mixed_evidence"]}
