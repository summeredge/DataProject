from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearModel:
    coef_: np.ndarray
    feature_names: tuple[object, ...]
    includes_intercept: bool = True


def fit_linear_model(
    X: pd.DataFrame,
    y: object,
    *,
    ridge_alpha: float | None = None,
) -> LinearModel:
    feature_names = _frame_feature_names(X)
    matrix = X.to_numpy(dtype=float)
    if ridge_alpha is None:
        design = _add_intercept(matrix)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        design = _add_intercept(matrix)
        penalty = float(ridge_alpha) * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        xtx = design.T @ design + penalty
        coef = np.linalg.solve(xtx, design.T @ np.asarray(y, dtype=float))
    return LinearModel(np.asarray(coef), feature_names)


def predict_linear_model(model: LinearModel, X: pd.DataFrame) -> np.ndarray:
    validate_feature_alignment(X, model)
    matrix = X.to_numpy(dtype=float)
    if model.includes_intercept:
        matrix = _add_intercept(matrix)
    return np.matmul(matrix, model.coef_)


def fit_named_matrix_model(
    matrix: np.ndarray,
    y: object,
    feature_names: Sequence[object],
) -> tuple[LinearModel, int]:
    names = tuple(feature_names)
    _validate_named_matrix(matrix, names)
    coef, _residuals, rank, _singular_values = np.linalg.lstsq(matrix, y, rcond=None)
    return LinearModel(np.asarray(coef), names, includes_intercept=False), int(rank)


def predict_named_matrix_model(
    model: LinearModel,
    matrix: np.ndarray,
    feature_names: Sequence[object],
) -> np.ndarray:
    names = tuple(feature_names)
    _validate_named_matrix(matrix, names)
    _validate_feature_names(names, model.feature_names)
    return np.matmul(matrix, model.coef_)


def bind_model_feature_names(model: Any, X: pd.DataFrame) -> None:
    model.feature_names = _frame_feature_names(X)


def predict_tabular_model(model: Any, X: pd.DataFrame) -> object:
    validate_feature_alignment(X, model)
    return model.predict(X)


def validate_feature_alignment(X: pd.DataFrame, model: Any) -> None:
    actual = _frame_feature_names(X)
    expected = getattr(model, "feature_names", None)
    if expected is None:
        raise ValueError("model is missing required feature_names")
    _validate_feature_names(actual, tuple(expected))


def _frame_feature_names(X: pd.DataFrame) -> tuple[object, ...]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with ordered columns")
    if not X.columns.is_unique:
        raise ValueError("X.columns must be unique for feature alignment")
    return tuple(X.columns)


def _validate_named_matrix(matrix: np.ndarray, feature_names: tuple[object, ...]) -> None:
    if matrix.ndim != 2:
        raise ValueError("model matrix must be two-dimensional")
    if matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"model matrix has {matrix.shape[1]} columns but feature_names has "
            f"{len(feature_names)} entries"
        )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique")


def _validate_feature_names(
    actual: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    if actual != expected:
        raise ValueError(
            "feature alignment mismatch: "
            f"X.columns={list(actual)!r}, model.feature_names={list(expected)!r}"
        )


def _add_intercept(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix])
