#!/usr/bin/env python3
"""
Train a linear model and export artifacts for Paillier inference.

Default: MNIST digits 0 vs 1, flattened 784 pixels, ``LinearRegression`` on
labels {0, 1} (no sigmoid; Carol returns encrypted linear prediction, Alice
decrypts once — matches Gaurav's preference for linear regression + Paillier).

Also available: ``--synthetic`` tiny tabular demo (legacy / no download).

Run from repo root:  python -m models.train [--synthetic] [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .he_spec import export_linear_he_spec, save_linear_he_spec
from .inference import (
    integer_features_from_raw,
    plaintext_linear_predict,
    plaintext_logit,
    predict_label_from_logit,
    recovered_prediction_from_integer_dot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

SYNTHETIC_FEATURE_NAMES = [
    "age_years",
    "annual_income_kusd",
    "credit_score",
    "months_employed",
]


def _make_synthetic_xy(
    n_samples: int = 800,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = n_samples
    age = rng.integers(22, 70, size=n)
    income_k = rng.integers(20, 250, size=n)
    credit = rng.integers(480, 820, size=n)
    tenure = rng.integers(0, 240, size=n)
    X = np.column_stack([age, income_k, credit, tenure]).astype(np.float64)

    score = (
        2.1 * (income_k / 100.0)
        + 0.008 * credit
        + 0.05 * tenure
        - 0.12 * age
        - 5.0
        + rng.normal(0, 0.8, size=n)
    )
    y = (score > 0).astype(np.int64)
    return X, y


def load_mnist_binary_digits(
    digits: tuple[int, int] = (0, 1),
    *,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """MNIST 784-d features, subset to two digit classes; y in {0, 1} as float for regression."""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all = np.asarray(mnist.data, dtype=np.float64)
    y_raw = np.asarray(mnist.target)
    if y_raw.dtype == object or y_raw.dtype.kind in ("U", "S", "O"):
        y_all = y_raw.astype(np.int64)
    else:
        y_all = y_raw.astype(np.int64)

    d0, d1 = digits
    mask = (y_all == d0) | (y_all == d1)
    X = X_all[mask]
    y_cls = y_all[mask]
    y = (y_cls == d1).astype(np.float64)

    if max_samples is not None and len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y


def train_synthetic() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = _make_synthetic_xy()
    y_cls = y.astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train_s, y_train.astype(np.int64))

    acc = float(model.score(X_test_s, y_test.astype(np.int64)))
    fixed_point_scale = 1_000_000

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": SYNTHETIC_FEATURE_NAMES,
        "fixed_point_scale": fixed_point_scale,
        "test_accuracy": acc,
        "task": "synthetic_logistic",
        "X_test": X_test,
        "y_test": y_test,
        "notes": "Legacy 4-feature synthetic demo; use default MNIST for course deliverable.",
    }
    joblib.dump(bundle, MODELS_DIR / "classifier.pkl")

    he_spec = export_linear_he_spec(
        model=model,
        scaler=scaler,
        feature_names=SYNTHETIC_FEATURE_NAMES,
        fixed_point_scale=fixed_point_scale,
    )
    save_linear_he_spec(he_spec, MODELS_DIR / "linear_he_spec.json")

    meta = {
        "feature_names": SYNTHETIC_FEATURE_NAMES,
        "fixed_point_scale": fixed_point_scale,
        "test_accuracy": acc,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "task": "synthetic_logistic",
    }
    with (MODELS_DIR / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODELS_DIR / 'classifier.pkl'}")
    print(f"Saved {MODELS_DIR / 'linear_he_spec.json'}")
    print(f"Holdout accuracy (plaintext): {acc:.4f}")

    w = np.asarray(he_spec.integer_weights, dtype=np.int64)
    max_abs_err = 0.0
    mismatches = 0
    for row in X_test:
        xi = integer_features_from_raw(row, he_spec)
        dot_int = int(w @ xi + he_spec.integer_bias)
        rec = recovered_prediction_from_integer_dot(dot_int, he_spec)
        sk = plaintext_logit(model, scaler, row)
        max_abs_err = max(max_abs_err, abs(rec - sk))
        if predict_label_from_logit(rec) != predict_label_from_logit(sk):
            mismatches += 1
    print(f"Max |recovered - sklearn score| on holdout: {max_abs_err:.6f}")
    print(f"Label mismatches (integer vs sklearn, holdout): {mismatches} / {len(y_test)}")


def train_mnist(*, quick: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    digits = (0, 1)
    max_samples = 8000 if quick else None
    X, y = load_mnist_binary_digits(digits, max_samples=max_samples, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y >= 0.5).astype(np.int64)
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred_plain = model.predict(X_test_s)
    mse_plain = float(mean_squared_error(y_test, y_pred_plain))
    y_hat_bin = (y_pred_plain >= 0.5).astype(np.int64)
    acc_05 = float(accuracy_score(y_test.astype(np.int64), y_hat_bin))

    fixed_point_scale = 1_000_000
    n_features = X.shape[1]
    feature_names = [f"pix_{i:04d}" for i in range(n_features)]

    # Enough rows for accuracy scripts; keeps ``classifier.pkl`` smaller than full test set.
    max_test_store = 600
    if len(X_test) > max_test_store:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=max_test_store, replace=False)
        X_test_store = X_test[idx]
        y_test_store = y_test[idx]
    else:
        X_test_store = X_test
        y_test_store = y_test

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "fixed_point_scale": fixed_point_scale,
        "test_mse": mse_plain,
        "test_accuracy_threshold_0.5": acc_05,
        "digits": list(digits),
        "task": "mnist_binary_linear_regression",
        "X_test": X_test_store,
        "y_test": y_test_store,
        "notes": "784-d MNIST (digits 0 vs 1); Paillier evaluates linear score, Alice decrypts prediction.",
    }
    joblib.dump(bundle, MODELS_DIR / "classifier.pkl")

    he_spec = export_linear_he_spec(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        fixed_point_scale=fixed_point_scale,
    )
    save_linear_he_spec(he_spec, MODELS_DIR / "linear_he_spec.json")

    meta = {
        "dataset": "mnist_784_openml",
        "digits": list(digits),
        "n_features": n_features,
        "fixed_point_scale": fixed_point_scale,
        "test_mse": mse_plain,
        "test_accuracy_threshold_0.5": acc_05,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "task": "mnist_binary_linear_regression",
        "quick_subset": bool(quick),
    }
    with (MODELS_DIR / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODELS_DIR / 'classifier.pkl'}")
    print(f"Saved {MODELS_DIR / 'linear_he_spec.json'}")
    print(f"Holdout MSE (plaintext linear regression): {mse_plain:.6f}")
    print(f"Holdout accuracy @ 0.5 threshold on predicted score: {acc_05:.4f}")

    w = np.asarray(he_spec.integer_weights, dtype=np.int64)
    max_abs_err = 0.0
    for row in X_test_store:
        xi = integer_features_from_raw(row, he_spec)
        dot_int = int(w @ xi + he_spec.integer_bias)
        rec = recovered_prediction_from_integer_dot(dot_int, he_spec)
        sk = plaintext_linear_predict(model, scaler, row)
        max_abs_err = max(max_abs_err, abs(rec - sk))

    print(f"Max |recovered - sklearn prediction| on stored test rows: {max_abs_err:.6f}")

    int_preds = []
    for row in X_test_store:
        xi = integer_features_from_raw(row, he_spec)
        dot_int = int(w @ xi + he_spec.integer_bias)
        int_preds.append(recovered_prediction_from_integer_dot(dot_int, he_spec))
    mse_int_path = float(mean_squared_error(y_test_store, np.asarray(int_preds)))
    print(f"MSE vs labels using integer-fixed-point pipeline (stored test): {mse_int_path:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear model and export HE linear spec.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use small synthetic tabular data + logistic regression (no MNIST download).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="MNIST only: cap subsample size for faster runs (still 784 features).",
    )
    args = parser.parse_args()
    if args.synthetic:
        train_synthetic()
    else:
        train_mnist(quick=args.quick)


if __name__ == "__main__":
    main()
