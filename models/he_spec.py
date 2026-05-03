"""
Integer linear model specification for Paillier-style encrypted inference.

Carol evaluates sum_i (w_i * x_i) + b on ciphertexts where x_i are Alice's
integer-encoded features and w_i, b are Carol's public integer weights. With
S denoting ``fixed_point_scale``, weights and bias are scaled so that after
decryption Alice divides by S**2 to recover the real-valued logit (up to
rounding error), then applies the usual threshold for binary classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LinearHESpec:
    """Serializable contract between ML export and homomorphic inference."""

    feature_names: list[str]
    fixed_point_scale: int
    integer_weights: list[int]
    integer_bias: int
    sklearn_coef: list[float]
    sklearn_intercept: float
    scaler_mean: list[float]
    scaler_scale: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "feature_names": self.feature_names,
            "fixed_point_scale": self.fixed_point_scale,
            "integer_weights": self.integer_weights,
            "integer_bias": self.integer_bias,
            "sklearn_coef": self.sklearn_coef,
            "sklearn_intercept": self.sklearn_intercept,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LinearHESpec":
        return LinearHESpec(
            feature_names=list(d["feature_names"]),
            fixed_point_scale=int(d["fixed_point_scale"]),
            integer_weights=[int(x) for x in d["integer_weights"]],
            integer_bias=int(d["integer_bias"]),
            sklearn_coef=[float(x) for x in d["sklearn_coef"]],
            sklearn_intercept=float(d["sklearn_intercept"]),
            scaler_mean=[float(x) for x in d["scaler_mean"]],
            scaler_scale=[float(x) for x in d["scaler_scale"]],
        )


def export_linear_he_spec(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    fixed_point_scale: int,
) -> LinearHESpec:
    """
    Build integer weights/bias for homomorphic dot-product + bias.

    Alice should encode each standardized feature as
    round(z_j * S) with z = (x_raw - mean) / scale from this scaler.
    Decrypted integer D satisfies D / S**2 ≈ w·z + b (sklearn logit).
    """
    if fixed_point_scale < 1:
        raise ValueError("fixed_point_scale must be a positive integer")
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0])
    w_int = np.round(coef * fixed_point_scale).astype(np.int64)
    b_int = int(np.round(intercept * (fixed_point_scale**2)))
    return LinearHESpec(
        feature_names=list(feature_names),
        fixed_point_scale=int(fixed_point_scale),
        integer_weights=w_int.tolist(),
        integer_bias=b_int,
        sklearn_coef=coef.tolist(),
        sklearn_intercept=intercept,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
    )


def load_linear_he_spec(path: str | Path) -> LinearHESpec:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return LinearHESpec.from_dict(json.load(f))


def save_linear_he_spec(spec: LinearHESpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)
