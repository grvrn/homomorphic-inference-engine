#!/usr/bin/env python3
"""
Train a binary logistic regression classifier and export:
  - models/classifier.pkl   (sklearn model + scaler + metadata for the team)
  - models/linear_he_spec.json   (integer linear layer for Paillier inference)

Run from repo root:  python -m models.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .he_spec import export_linear_he_spec, save_linear_he_spec
from .inference import (
    integer_features_from_raw,
    plaintext_logit,
    predict_label_from_logit,
    recovered_logit_from_integer_dot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# Tabular synthetic data with interpretable-ish feature names (project narrative).
FEATURE_NAMES = [
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

    # Hidden linear-ish rule + noise → labels (not given to Carol at inference).
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


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = _make_synthetic_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train_s, y_train)

    acc = float(model.score(X_test_s, y_test))
    fixed_point_scale = 1_000_000

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "fixed_point_scale": fixed_point_scale,
        "test_accuracy": acc,
        "notes": "Trained on synthetic tabular data; HE export uses integer-scaled linear layer.",
    }
    joblib.dump(bundle, MODELS_DIR / "classifier.pkl")

    he_spec = export_linear_he_spec(
        model=model,
        scaler=scaler,
        feature_names=FEATURE_NAMES,
        fixed_point_scale=fixed_point_scale,
    )
    save_linear_he_spec(he_spec, MODELS_DIR / "linear_he_spec.json")

    meta = {
        "feature_names": FEATURE_NAMES,
        "fixed_point_scale": fixed_point_scale,
        "test_accuracy": acc,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    with (MODELS_DIR / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {MODELS_DIR / 'classifier.pkl'}")
    print(f"Saved {MODELS_DIR / 'linear_he_spec.json'}")
    print(f"Holdout accuracy (plaintext, standardized): {acc:.4f}")

    # Integer linear layer matches sklearn up to rounding (for Carol's Paillier dot-product).
    w = np.asarray(he_spec.integer_weights, dtype=np.int64)
    max_abs_err = 0.0
    mismatches = 0
    for row in X_test:
        xi = integer_features_from_raw(row, he_spec)
        dot_int = int(w @ xi + he_spec.integer_bias)
        rec = recovered_logit_from_integer_dot(dot_int, he_spec)
        sk = plaintext_logit(model, scaler, row)
        max_abs_err = max(max_abs_err, abs(rec - sk))
        if predict_label_from_logit(rec) != predict_label_from_logit(sk):
            mismatches += 1
    print(f"Max |recovered_logit - sklearn_logit| on holdout: {max_abs_err:.6f}")
    print(f"Label mismatches (integer vs sklearn, holdout): {mismatches} / {len(y_test)}")


if __name__ == "__main__":
    main()
