"""Plaintext reference inference for correctness checks and benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .he_spec import LinearHESpec


def standardized_features(
    x_raw: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64).ravel()
    mean = np.asarray(scaler_mean, dtype=np.float64).ravel()
    scale = np.asarray(scaler_scale, dtype=np.float64).ravel()
    return (x - mean) / scale


def plaintext_logit(
    model: LogisticRegression,
    scaler: StandardScaler,
    x_raw: np.ndarray,
) -> float:
    """Sklearn decision_function for one raw sample (uses the same scaler as training)."""
    x = np.asarray(x_raw, dtype=np.float64).reshape(1, -1)
    xs = scaler.transform(x)
    return float(model.decision_function(xs)[0])


def integer_features_from_raw(
    x_raw: np.ndarray,
    spec: LinearHESpec,
) -> np.ndarray:
    z = standardized_features(
        x_raw,
        np.asarray(spec.scaler_mean),
        np.asarray(spec.scaler_scale),
    )
    s = spec.fixed_point_scale
    return np.round(z * s).astype(np.int64)


def plaintext_logit_from_spec(x_raw: np.ndarray, spec: LinearHESpec) -> float:
    """Reconstruct float logit from the same scaling used for HE export."""
    z = standardized_features(
        x_raw,
        np.asarray(spec.scaler_mean),
        np.asarray(spec.scaler_scale),
    )
    w = np.asarray(spec.sklearn_coef, dtype=np.float64)
    b = spec.sklearn_intercept
    return float(w @ z + b)


def recovered_logit_from_integer_dot(
    dot_int: int | float,
    spec: LinearHESpec,
) -> float:
    """Map decrypted integer aggregate back to approximate logit."""
    s = spec.fixed_point_scale
    return float(dot_int) / (s * s)


def predict_label_from_logit(logit: float, threshold: float = 0.0) -> int:
    return int(logit >= threshold)
