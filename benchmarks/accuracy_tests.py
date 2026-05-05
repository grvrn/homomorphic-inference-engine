"""
Compare plaintext sklearn predictions vs full Paillier HE inference.

Both are evaluated against the MNIST 0-vs-1 test set from ``classifier.pkl``.
Plaintext runs on all stored test samples; HE runs on a configurable subset
(default 20) because each 784-feature sample takes several seconds to encrypt.

Run from repo root:  python -m benchmarks.accuracy_tests [--he-samples N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from crypto import generate_keypair, encrypt, decrypt
from crypto.crypto import homomorphic_linear_score
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
    parser = argparse.ArgumentParser(
        description="Accuracy: plaintext vs HE on MNIST 0/1 test set",
    )
    parser.add_argument(
        "--he-samples", type=int, default=20,
        help="Number of test samples for HE inference (default: 20)",
    )
    args = parser.parse_args()

    # ---- Load artefacts ----
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
    task = bundle.get("task", "unknown")

    is_logistic = "logistic" in task.lower()
    threshold = 0.0 if is_logistic else 0.5
    digits = tuple(bundle.get("digits", (0, 1)))

    n_total = len(X_test)
    n_he = min(args.he_samples, n_total)

    # Pick a fixed random subset for HE so results are reproducible
    rng = np.random.default_rng(42)
    he_indices = np.sort(rng.choice(n_total, size=n_he, replace=False))

    print(f"Task           : {task}")
    print(f"Digit classes  : {digits[0]} vs {digits[1]}")
    print(f"Threshold      : {threshold}")
    print(f"Test samples   : {n_total} (plaintext all, HE subset {n_he})")
    print()

    # ==================================================================
    # Plaintext predictions (all samples)
    # ==================================================================
    print("Running plaintext predictions on all test samples …")
    sk_scores: list[float] = []
    for row in X_test:
        x = np.asarray(row, dtype=np.float64).ravel()
        if isinstance(model, LinearRegression):
            sk_scores.append(plaintext_linear_predict(model, scaler, x))
        elif isinstance(model, LogisticRegression):
            sk_scores.append(plaintext_logit(model, scaler, x))
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    sk_scores_arr = np.asarray(sk_scores, dtype=np.float64)
    sk_labels = (sk_scores_arr >= threshold).astype(int)

    pt_acc = float(accuracy_score(y_test.astype(int), sk_labels))
    pt_mse = float(mean_squared_error(y_test, sk_scores_arr))

    # ==================================================================
    # HE inference (subset)
    # ==================================================================
    print(f"Running Paillier HE inference on {n_he} samples …")
    keypair = generate_keypair()
    pk = keypair.public_key
    sk = keypair.secret_key

    he_scores: list[float] = []
    t0 = time.perf_counter()
    for i, idx in enumerate(he_indices):
        row = np.asarray(X_test[idx], dtype=np.float64).ravel()

        # Alice: standardise → integer-encode → encrypt
        xi = integer_features_from_raw(row, spec)
        enc_features = [encrypt(int(v), pk) for v in xi]

        # Carol: homomorphic evaluation
        enc_score = homomorphic_linear_score(
            pk, enc_features, spec.integer_weights, spec.integer_bias,
        )

        # Alice: decrypt → recover score
        dec_int = int(round(decrypt(enc_score, sk)))
        he_scores.append(recovered_prediction_from_integer_dot(dec_int, spec))

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i + 1}/{n_he}]  elapsed {elapsed:.1f}s")

    he_elapsed = time.perf_counter() - t0
    he_scores_arr = np.asarray(he_scores, dtype=np.float64)
    he_labels = (he_scores_arr >= threshold).astype(int)

    # Ground truth and plaintext scores for the HE subset
    y_he = y_test[he_indices].astype(int)
    sk_he = sk_scores_arr[he_indices]
    sk_labels_he = (sk_he >= threshold).astype(int)

    he_acc = float(accuracy_score(y_he, he_labels))
    pt_acc_sub = float(accuracy_score(y_he, sk_labels_he))
    he_mse = float(mean_squared_error(y_he, he_scores_arr))
    pt_mse_sub = float(mean_squared_error(y_he, sk_he))

    # Score difference between HE and plaintext on the same samples
    score_diffs = np.abs(he_scores_arr - sk_he)
    max_diff = float(score_diffs.max())
    mean_diff = float(score_diffs.mean())
    label_agree = int(np.sum(he_labels == sk_labels_he))

    # ==================================================================
    # Results
    # ==================================================================
    print(f"\n{'═' * 60}")
    print(f"  ACCURACY RESULTS — MNIST digits {digits[0]} vs {digits[1]}")
    print(f"{'═' * 60}")

    print(f"\n  Plaintext Model (all {n_total} samples)")
    print(f"  {'Accuracy vs true labels':35s} {pt_acc:>10.4f}")
    print(f"  {'MSE vs true labels':35s} {pt_mse:>10.8f}")

    print(f"\n  HE vs Plaintext ({n_he} sample subset, {he_elapsed:.1f}s)")
    print(f"  {'':35s} {'HE':>10s} {'Plaintext':>10s}")
    print("  " + "-" * 57)
    print(f"  {'Accuracy vs true labels':35s} {he_acc:>10.4f} {pt_acc_sub:>10.4f}")
    print(f"  {'MSE vs true labels':35s} {he_mse:>10.8f} {pt_mse_sub:>10.8f}")
    print(f"  {'Label agreement (HE = Plaintext)':35s} {label_agree:>10d} / {n_he}")
    print(f"  {'Mean |HE score − PT score|':35s} {mean_diff:>10.2e}")
    print(f"  {'Max  |HE score − PT score|':35s} {max_diff:>10.2e}")

    if label_agree == n_he:
        print(f"\n  ✓ HE and plaintext agree on all {n_he} predictions")
    else:
        print(f"\n  ⚠ {n_he - label_agree} prediction(s) disagree between HE and plaintext")

    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
