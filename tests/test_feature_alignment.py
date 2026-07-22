from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chem_ts_corr.feature_alignment import (
    bind_model_feature_names,
    fit_linear_model,
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
