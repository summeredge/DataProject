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
    variable_types: dict[str, frozenset[str]]
    reference_map: dict[str, str]
    metadata: dict[str, object]


def _base(n: int, seed: int, noise: float):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="5min")
    return rng, index, rng.normal(scale=noise, size=n)


def _case(
    name: str,
    frame: pd.DataFrame,
    target: str,
    drivers=(),
    spurious=(),
    lags=None,
    directions=None,
    variable_types=None,
    reference_map=None,
    **metadata,
) -> SyntheticCase:
    types = {
        key: frozenset(values)
        for key, values in (variable_types or {}).items()
    }
    return SyntheticCase(
        frame=frame,
        target=target,
        true_drivers=frozenset(drivers),
        spurious_variables=frozenset(spurious),
        lags=lags or {},
        directions=directions or {},
        variable_types=types,
        reference_map=reference_map or {},
        metadata={"scenario": name, **metadata},
    )


def true_lagged_driver(n=360, noise=0.15, seed=101, lag=4) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x = rng.normal(size=n)
    y = np.roll(x, lag) + eps
    y[:lag] = eps[:lag]
    return _case(
        "true_lagged_driver",
        pd.DataFrame({"target": y, "x_driver": x, "noise": rng.normal(size=n)}, index=index),
        "target",
        ["x_driver"],
        ["noise"],
        {"x_driver": lag},
        {"x_driver": "variable_leads_target"},
        {"noise": {"noise"}, "true_driver": {"x_driver"}},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
    )


def downstream_response(n=360, noise=0.12, seed=102, lag=3) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    y = rng.normal(size=n)
    x = np.roll(y, lag) + eps
    x[:lag] = eps[:lag]
    return _case(
        "downstream_response",
        pd.DataFrame({"target": y, "x_downstream": x, "noise": rng.normal(size=n)}, index=index),
        "target",
        [],
        ["x_downstream", "noise"],
        {"x_downstream": -lag},
        {"x_downstream": "target_leads_variable"},
        {"downstream": {"x_downstream"}, "noise": {"noise"}},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
    )


def common_driver(n=420, noise=0.12, seed=103, lag=3) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    z = rng.normal(size=n)
    y = np.roll(z, lag) + eps
    x = np.roll(z, lag) + rng.normal(scale=noise, size=n)
    y[:lag] = eps[:lag]
    return _case(
        "common_driver",
        pd.DataFrame({"target": y, "z_driver": z, "x_common": x}, index=index),
        "target",
        ["z_driver"],
        ["x_common"],
        {"z_driver": lag},
        {"z_driver": "variable_leads_target"},
        {"true_driver": {"z_driver"}, "common_driver_proxy": {"x_common"}},
        {"x_common": "z_driver"},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
        residual_control_columns=["z_driver"],
    )


def collinear_proxy(n=360, noise=0.1, seed=104, lag=2) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.03, size=n)
    y = np.roll(x1, lag) + eps
    y[:lag] = eps[:lag]
    return _case(
        "collinear_proxy",
        pd.DataFrame({"target": y, "x1_driver": x1, "x2_proxy": x2}, index=index),
        "target",
        ["x1_driver"],
        ["x2_proxy"],
        {"x1_driver": lag},
        {"x1_driver": "variable_leads_target"},
        {"true_driver": {"x1_driver"}, "proxy": {"x2_proxy"}},
        {"x2_proxy": "x1_driver"},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
    )


def nonlinear_stable_driver(n=400, noise=0.08, seed=105, lag=2) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x = rng.uniform(-1, 1, size=n)
    y = np.roll(x**2, lag) + eps
    y[:lag] = eps[:lag]
    noise_names = [f"noise_{index:02d}" for index in range(8)]
    frame = pd.DataFrame(
        {
            "target": y,
            "x_nonlinear": x,
            **{name: rng.normal(size=n) for name in noise_names},
        },
        index=index,
    )
    return _case(
        "nonlinear_stable_driver",
        frame,
        "target",
        ["x_nonlinear"],
        noise_names,
        {"x_nonlinear": lag},
        {"x_nonlinear": "variable_leads_target"},
        {"true_driver": {"x_nonlinear"}, "noise": set(noise_names)},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
        skip_model_lift=False,
    )


def regime_sign_reversal(n=420, noise=0.08, seed=106, lag=2, regimes=2) -> SyntheticCase:
    if regimes < 2:
        raise ValueError("regime_sign_reversal requires at least two regimes")
    rng, index, eps = _base(n, seed, noise)
    x = rng.normal(size=n)
    regime_index = np.minimum(np.arange(n) * regimes // n, regimes - 1)
    load = np.where(regime_index % 2 == 0, -1.0, 1.0)
    y = load * np.roll(x, lag) + eps
    y[:lag] = eps[:lag]
    return _case(
        "regime_sign_reversal",
        pd.DataFrame({"target": y, "x_reversal": x, "load": load}, index=index),
        "target",
        ["x_reversal"],
        [],
        {"x_reversal": lag},
        {"x_reversal": "variable_leads_target"},
        {"true_driver": {"x_reversal"}, "regime": {"load"}},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
        regimes=regimes,
        segment_column="load",
    )


def outlier_driven_correlation(n=360, noise=0.1, seed=107, outlier_count=16) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x = rng.normal(size=n)
    y = eps.copy()
    positions = np.linspace(15, n - 15, outlier_count, dtype=int)
    scale = 45.0
    x[positions] = np.linspace(-12, 12, outlier_count)
    y[positions] = x[positions] * scale
    return _case(
        "outlier_driven_correlation",
        pd.DataFrame({"target": y, "x_outlier": x, "noise": rng.normal(size=n)}, index=index),
        "target",
        [],
        ["x_outlier", "noise"],
        variable_types={"outlier_proxy": {"x_outlier"}, "noise": {"noise"}},
        seed=seed,
        n=n,
        noise=noise,
        outlier_count=outlier_count,
        outlier_indices=positions.tolist(),
        clean_expected_correlation_max=0.15,
    )


def lag_boundary_artifact(n=360, noise=0.1, seed=108, lag=6) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x = rng.normal(size=n)
    y = np.roll(x, lag) + eps
    y[:lag] = eps[:lag]
    return _case(
        "lag_boundary_artifact",
        pd.DataFrame({"target": y, "x_boundary": x}, index=index),
        "target",
        ["x_boundary"],
        [],
        {"x_boundary": lag},
        {"x_boundary": "variable_leads_target"},
        {"boundary_artifact": {"x_boundary"}},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
        max_lag=lag,
    )


def noise_only(n=360, noise=1.0, seed=109, variable_count=30) -> SyntheticCase:
    rng, index, y = _base(n, seed, noise)
    names = [f"noise_{index:02d}" for index in range(variable_count)]
    frame = pd.DataFrame({"target": y, **{name: rng.normal(size=n) for name in names}}, index=index)
    return _case(
        "noise_only",
        frame,
        "target",
        [],
        names,
        variable_types={"noise": set(names)},
        seed=seed,
        n=n,
        noise=noise,
        variable_count=variable_count,
    )


def mixed_evidence(n=480, noise=0.12, seed=110) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    y = np.roll(z, 3) + np.roll(x, 4) + eps
    y[:4] = eps[:4]
    proxy = x + rng.normal(scale=0.03, size=n)
    common = np.roll(z, 3) + rng.normal(scale=noise, size=n)
    downstream = np.roll(y, 2) + rng.normal(scale=noise, size=n)
    frame = pd.DataFrame(
        {
            "target": y,
            "x_driver": x,
            "x_proxy": proxy,
            "z_driver": z,
            "x_common": common,
            "x_downstream": downstream,
            "noise": rng.normal(size=n),
        },
        index=index,
    )
    return _case(
        "mixed_evidence",
        frame,
        "target",
        ["x_driver", "z_driver"],
        ["x_proxy", "x_common", "x_downstream", "noise"],
        {"x_driver": 4, "z_driver": 3, "x_downstream": -2},
        {
            "x_driver": "variable_leads_target",
            "z_driver": "variable_leads_target",
            "x_downstream": "target_leads_variable",
        },
        {
            "true_driver": {"x_driver", "z_driver"},
            "proxy": {"x_proxy"},
            "common_driver_proxy": {"x_common"},
            "downstream": {"x_downstream"},
            "noise": {"noise"},
        },
        {"x_common": "z_driver", "x_proxy": "x_driver"},
        seed=seed,
        n=n,
        noise=noise,
        residual_control_columns=["z_driver"],
        skip_model_lift=False,
    )


def model_incremental_validation(n=420, noise=0.12, seed=111, lag=3) -> SyntheticCase:
    rng, index, eps = _base(n, seed, noise)
    x = rng.normal(size=n)
    proxy = x + rng.normal(scale=0.02, size=n)
    y = np.roll(x, lag) + 0.45 * np.roll(x**2, lag) + eps
    y[:lag] = eps[:lag]
    frame = pd.DataFrame(
        {"target": y, "x_incremental": x, "x_proxy": proxy, "noise": rng.normal(size=n)},
        index=index,
    )
    return _case(
        "model_incremental_validation",
        frame,
        "target",
        ["x_incremental"],
        ["x_proxy", "noise"],
        {"x_incremental": lag},
        {"x_incremental": "variable_leads_target"},
        {"true_driver": {"x_incremental"}, "proxy": {"x_proxy"}, "noise": {"noise"}},
        {"x_proxy": "x_incremental"},
        seed=seed,
        n=n,
        noise=noise,
        lag=lag,
        skip_model_lift=False,
    )


CASES = {
    function.__name__: function
    for function in [
        true_lagged_driver,
        downstream_response,
        common_driver,
        collinear_proxy,
        nonlinear_stable_driver,
        regime_sign_reversal,
        outlier_driven_correlation,
        lag_boundary_artifact,
        noise_only,
        mixed_evidence,
        model_incremental_validation,
    ]
}
