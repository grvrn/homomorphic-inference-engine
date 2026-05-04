"""
Compare plaintext sklearn predictions vs fixed-point integer pipeline (HE surrogate).

Expects ``models/classifier.pkl`` (with ``X_test``, ``y_test``) and
``models/linear_he_spec.json`` from ``python -m models.train``.

Run from repo root:  python -m benchmarks.accuracy_tests
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

from models.he_spec import load_linear_he_spec
from models.inference import (
    integer_features_from_raw,
    plaintext_linear_predict,
    plaintext_logit,
    recovered_prediction_from_integer_dot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def main() -> None:
    bundle_path = MODELS_DIR / "classifier.pkl"
    spec_path = MODELS_DIR / "linear_he_spec.json"
    if not bundle_path.is_file() or not spec_path.is_file():
        raise SystemExit(f"Missing {bundle_path} or {spec_path}; run python -m models.train first.")

    bundle = joblib.load(bundle_path)
    spec = load_linear_he_spec(spec_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    X_test = np.asarray(bundle["X_test"], dtype=np.float64)
    y_test = np.asarray(bundle["y_test"], dtype=np.float64)
    w = np.asarray(spec.integer_weights, dtype=np.int64)

    sk_preds: list[float] = []
    for row in X_test:
        x = np.asarray(row, dtype=np.float64).ravel()
        if isinstance(model, LinearRegression):
            sk_preds.append(plaintext_linear_predict(model, scaler, x))
        elif isinstance(model, LogisticRegression):
            sk_preds.append(plaintext_logit(model, scaler, x))
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    sk_preds_arr = np.asarray(sk_preds, dtype=np.float64)

    int_preds: list[float] = []
    for row in X_test:
        xi = integer_features_from_raw(row, spec)
        dot_int = int(w @ xi + spec.integer_bias)
        int_preds.append(recovered_prediction_from_integer_dot(dot_int, spec))
    int_preds_arr = np.asarray(int_preds, dtype=np.float64)

    mse_plain = float(mean_squared_error(y_test, sk_preds_arr))
    mse_int = float(mean_squared_error(y_test, int_preds_arr))
    mse_sk_vs_int = float(mean_squared_error(sk_preds_arr, int_preds_arr))

    print(f"Task (from bundle): {bundle.get('task', 'unknown')}")
    print(f"MSE(labels, sklearn pred):     {mse_plain:.8f}")
    print(f"MSE(labels, integer pipeline): {mse_int:.8f}")
    print(f"MSE(sklearn, integer pipeline):{mse_sk_vs_int:.8e}")


if __name__ == "__main__":
    main()
