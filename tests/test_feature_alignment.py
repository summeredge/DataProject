from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.feature_alignment import (
    bind_model_feature_names,
    fit_linear_model,
    fit_tabular_model,
    predict_linear_model,
    predict_tabular_model,
)
from chem_ts_corr.preprocess import FrameScaler


def test_linear_model_requires_exact_ordered_feature_alignment():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 8.0]})
    model = fit_linear_model(train, np.array([1.0, 2.0, 3.0]))

    assert model.feature_names == ("a", "b")
    assert predict_linear_model(model, train).shape == (3,)
    with pytest.raises(ValueError, match="feature alignment mismatch"):
        predict_linear_model(model, train[["b", "a"]])
    with pytest.raises(ValueError, match="feature alignment mismatch"):
        predict_linear_model(model, train[["a"]])


def test_scaler_binds_and_validates_complete_feature_names():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 5.0]})
    scaler = FrameScaler(frame.mean(), frame.std(ddof=0), tuple(frame.columns))

    assert scaler.feature_names == ("a", "b")
    with pytest.raises(ValueError, match="feature alignment mismatch"):
        scaler.transform(frame[["a"]])


def test_tabular_prediction_rejects_mismatch_before_calling_model():
    class RecordingModel:
        called = False

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            self.called = True
            return np.zeros(len(X))

    train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    model = RecordingModel()
    bind_model_feature_names(model, train)

    with pytest.raises(ValueError, match="feature alignment mismatch"):
        predict_tabular_model(model, train[["b", "a"]])

    assert model.called is False


def test_tabular_model_fits_before_binding_ordered_feature_names():
    class RecordingModel:
        fitted = False

        def fit(self, X: pd.DataFrame, y: object) -> None:
            assert not hasattr(self, "feature_names")
            assert list(X.columns) == ["A", "B", "C"]
            self.fitted = True

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))

    train = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}
    )
    original = train.copy(deep=True)
    model = RecordingModel()

    result = fit_tabular_model(model, train, np.array([1.0, 2.0]))

    assert result is model
    assert model.fitted is True
    assert model.feature_names == ("A", "B", "C")
    assert predict_tabular_model(model, train).shape == (2,)
    pd.testing.assert_frame_equal(train, original)


@pytest.mark.parametrize(
    "columns",
    [
        ["A", "B"],
        ["A", "B", "C", "D"],
        ["B", "A", "C"],
        ["A", "B", "D"],
    ],
)
def test_tabular_prediction_requires_exact_columns(columns: list[str]):
    class RecordingModel:
        called = False

        def fit(self, X: pd.DataFrame, y: object) -> None:
            pass

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            self.called = True
            return np.zeros(len(X))

    train = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["A", "B", "C"])
    model = fit_tabular_model(RecordingModel(), train, np.array([1.0]))
    prediction = pd.DataFrame([[1.0] * len(columns)], columns=columns)

    with pytest.raises(
        ValueError,
        match=r"feature alignment mismatch: X.columns=.*model.feature_names=",
    ):
        predict_tabular_model(model, prediction)

    assert model.called is False


def test_tabular_fit_rejects_non_dataframe_and_duplicate_columns_before_fit():
    class RecordingModel:
        called = False

        def fit(self, X: pd.DataFrame, y: object) -> None:
            self.called = True

    model = RecordingModel()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        fit_tabular_model(model, np.ones((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="X.columns must be unique"):
        fit_tabular_model(
            model,
            pd.DataFrame([[1.0, 2.0]], columns=["A", "A"]),
            np.ones(1),
        )

    assert model.called is False


def test_source_has_no_direct_coefficient_matmul_outside_alignment_guard():
    source_root = Path("chem_ts_corr")
    offenders = []
    for path in source_root.glob("*.py"):
        if path.name == "feature_alignment.py":
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "@" in line and "coef" in line:
                offenders.append(f"{path}:{line_number}:{line.strip()}")

    assert offenders == []


def test_source_has_no_business_call_that_binds_features_before_fit():
    offenders = []
    for path in Path("chem_ts_corr").glob("*.py"):
        if path.name == "feature_alignment.py":
            continue
        if "bind_model_feature_names" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))

    assert offenders == []
