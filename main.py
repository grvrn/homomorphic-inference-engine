from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.latency_tests import benchmark_plaintext_latency, benchmark_he_latency
from src.app import evaluate_encrypted

import numpy as np
from PIL import Image

from crypto import (
    PaillierKeyPair,
    PaillierCiphertext,
    generate_keypair,
    encrypt,
    decrypt,
)
from models.he_spec import LinearHESpec, load_linear_he_spec
from models.inference import (
    integer_features_from_raw,
    plaintext_logit_from_spec,
    recovered_logit_from_integer_dot,
    predict_label_from_logit,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"

def plot_latency(
    plaintext_rows: list[dict[str, float | int]],
    he_rows: list[dict[str, float | int]],
) -> None:
    pt_features = [r["n_features"] for r in plaintext_rows]
    pt_means = [r["mean_ms"] for r in plaintext_rows]
    pt_stds = [r["std_ms"] for r in plaintext_rows]
    he_features = [r["n_features"] for r in he_rows]
    he_means = [r["mean_ms"] for r in he_rows]
    he_stds = [r["std_ms"] for r in he_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        pt_features,
        pt_means,
        yerr=pt_stds,
        fmt="o-",
        color="#4c72b0",
        ecolor="#7fa6d4",
        elinewidth=1.2,
        capsize=4,
        linewidth=2,
        markersize=6,
        label="Plaintext",
    )
    ax.errorbar(
        he_features,
        he_means,
        yerr=he_stds,
        fmt="s--",
        color="#c44e52",
        ecolor="#e09194",
        elinewidth=1.2,
        capsize=4,
        linewidth=2,
        markersize=6,
        label="Homomorphic",
    )

    ax.set_xlabel("Number of Features (d)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Inference Latency: Plaintext vs. Homomorphic Encryption", fontsize=13, pad=10)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.savefig("latency_comparison.png", dpi=150)
    print("Plot saved to latency_comparison.png")


def print_table(label: str, rows: list[dict[str, float | int]]) -> None:
    print(f"\n  {label}")
    print(f"  {'Features':>10}  {'Mean (ms)':>10}  {'Std (ms)':>10}")
    print("  " + "-" * 36)
    for r in rows:
        print(f"  {r['n_features']:>10}  {r['mean_ms']:>10.4f}  {r['std_ms']:>10.4f}")


def print_ratio_table(
    plaintext_rows: list[dict[str, float | int]],
    he_rows: list[dict[str, float | int]],
) -> None:
    print("\n  HE / Plaintext")
    print(f"  {'Features':>10}  {'Ratio':>12}")
    print("  " + "-" * 26)
    for pt, he in zip(plaintext_rows, he_rows):
        ratio = float(he["mean_ms"]) / float(pt["mean_ms"])
        print(f"  {pt['n_features']:>10}  {ratio:>12.2f}x")

def encrypt_features(
    x_int: np.ndarray,
    keypair: PaillierKeyPair,
) -> list[PaillierCiphertext]:
    pk = keypair.public_key
    return [encrypt(int(xi), pk) for xi in x_int.ravel()]


def decrypt_score(
    encrypted_score: PaillierCiphertext,
    keypair: PaillierKeyPair,
) -> int:
    return decrypt(encrypted_score, keypair.secret_key)

def he_inference(
    x: np.ndarray,
    spec: LinearHESpec,
    *,
    keypair: PaillierKeyPair | None = None,
) -> dict:
    """
    Run homomorphic inference on a raw feature vector.

    Parameters
    ----------
    x : np.ndarray
        Raw feature vector (same shape the model was trained on).
    spec : LinearHESpec
        Integer weights, bias, and scaler parameters exported from training.
    keypair : PaillierKeyPair, optional
        Paillier key pair. A fresh one is generated if not supplied.

    Returns
    -------
    dict with keys:
        encrypted_score : PaillierCiphertext – the raw encrypted result
        decrypted_int   : int   – decrypted integer aggregate
        logit           : float – recovered real-valued logit (D / S²)
        label           : int   – predicted class (0 or 1)
    """
    if keypair is None:
        keypair = generate_keypair()

    # Alice: standardise → integer-encode → encrypt
    x_int = integer_features_from_raw(x, spec)
    enc_features = encrypt_features(x_int, keypair)

    # Carol: evaluate linear model on ciphertexts
    enc_score = evaluate_encrypted(enc_features, spec)

    # Alice: decrypt → recover logit → classify
    decrypted_int = decrypt_score(enc_score, keypair)
    logit = recovered_logit_from_integer_dot(decrypted_int, spec)
    label = predict_label_from_logit(logit)

    return {
        "encrypted_score": enc_score,
        "decrypted_int": decrypted_int,
        "logit": logit,
        "label": label,
    }

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image file and preprocess it for the MNIST model.

    Steps:
        1. Open image and convert to grayscale ("L" mode)
        2. Resize to 28×28 using antialiased resampling
        3. Invert so digit is white-on-black (MNIST convention)
        4. Convert to float64 numpy array (pixel values 0–255)
        5. Flatten to a 784-element vector

    Returns
    -------
    np.ndarray of shape (784,) with dtype float64.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)  # shape (28, 28), values 0–255

    # MNIST convention: 0 = black background, 255 = white digit stroke.
    # Most user images are black-on-white (inverted). Detect and flip:
    # if the border (background) is bright, the image needs inverting.
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if border.mean() > 128:
        pixels = 255.0 - pixels

    return pixels.ravel()  # shape (784,)


def predict_mnist_image(image_path: str) -> None:
    """
    Full MNIST HE inference pipeline for a single image.

    1. Load the trained LinearHESpec and training metadata
    2. Detect model type (linear regression vs logistic regression)
    3. Preprocess the input image to a 784-d pixel vector
    4. Run homomorphic inference: encrypt → evaluate → decrypt
    5. Apply the appropriate threshold to classify digit 0 vs 1
    """
    # ---- Load model artefacts ----
    spec_path = MODELS_DIR / "linear_he_spec.json"
    meta_path = MODELS_DIR / "training_meta.json"

    if not spec_path.exists():
        print(f"Error: HE spec not found at {spec_path}")
        print("Run 'python -m models.train' first to train the model.")
        sys.exit(1)

    spec = load_linear_he_spec(spec_path)

    # ---- Detect model type from training metadata ----
    task = "mnist_binary_linear_regression"  # default
    digits = (0, 1)
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        task = meta.get("task", task)
        digits = tuple(meta.get("digits", digits))

    is_logistic = "logistic" in task.lower()
    # Linear regression scores land near 0.0 (digit 0) and 1.0 (digit 1) → threshold 0.5
    # Logistic regression logits are centered at 0.0 → threshold 0.0
    threshold = 0.0 if is_logistic else 0.5
    model_type = "Logistic Regression" if is_logistic else "Linear Regression"

    print(f"Model type   : {model_type}")
    print(f"Task         : {task}")
    print(f"Digit classes: {digits[0]} vs {digits[1]}")
    print(f"Threshold    : {threshold}")
    print()

    # ---- Preprocess image ----
    print(f"Loading image: {image_path}")
    x_raw = load_and_preprocess_image(image_path)
    n_expected = len(spec.feature_names)
    if x_raw.size != n_expected:
        print(
            f"Error: image flattened to {x_raw.size} pixels, "
            f"but model expects {n_expected} features."
        )
        sys.exit(1)
    print(f"Image shape  : 28×28 → {x_raw.size} features")
    print()

    # ---- Generate Paillier keys ----
    print("Generating Paillier key pair ...")
    keypair = generate_keypair()

    # ---- Run HE inference ----
    print("Encrypting features ...")
    x_int = integer_features_from_raw(x_raw, spec)
    enc_features = encrypt_features(x_int, keypair)
    print(f"Encrypted {len(enc_features)} feature ciphertexts")

    print("Evaluating encrypted linear model ...")
    enc_score = evaluate_encrypted(enc_features, spec)

    print("Decrypting result ...")
    decrypted_int = decrypt_score(enc_score, keypair)
    he_score = recovered_logit_from_integer_dot(decrypted_int, spec)

    # ---- Plaintext reference score ----
    pt_score = plaintext_logit_from_spec(x_raw, spec)

    # ---- Classify ----
    he_label = int(he_score >= threshold)
    pt_label = int(pt_score >= threshold)
    he_digit = digits[1] if he_label == 1 else digits[0]
    pt_digit = digits[1] if pt_label == 1 else digits[0]

    print()
    print("═" * 48)
    print(f"  {'':20s} {'HE':>12s} {'Plaintext':>12s}")
    print("  " + "-" * 46)
    print(f"  {'Raw score':20s} {he_score:>12.6f} {pt_score:>12.6f}")
    print(f"  {'|HE − Plaintext|':20s} {abs(he_score - pt_score):>12.6e}")
    print(f"  {'Threshold':20s} {threshold:>12.1f} {threshold:>12.1f}")
    print(f"  {'Predicted digit':20s} {he_digit:>12} {pt_digit:>12}")
    if he_digit != pt_digit:
        print("Predictions do not match")
    else:
        print("Predictions match")
    print("═" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Homomorphic Inference Engine CLI")
    parser.add_argument(
        "--test",
        choices=["latency", "accuracy"],
        help="Run a benchmark test",
    )
    parser.add_argument(
        "--predict",
        metavar="IMAGE",
        help="Path to a handwritten digit image (0 or 1) for HE inference",
    )
    parser.add_argument(
        "--he-samples",
        type=int,
        default=20,
        help="Number of HE samples for accuracy test (default: 20)",
    )
    args = parser.parse_args()

    if args.test is None and args.predict is None:
        parser.print_help()
        sys.exit(0)

    if args.predict is not None:
        predict_mnist_image(args.predict)
        return

    if args.test == "latency":
        print("Running plaintext latency benchmark")
        pt_rows = benchmark_plaintext_latency()
        print_table("Plaintext", pt_rows)

        print("\nRunning HE latency benchmark")
        keypair = generate_keypair()
        he_rows = benchmark_he_latency(
            he_inference=lambda spec, x: he_inference(x, spec, keypair=keypair)
        )
        print_table("Homomorphic", he_rows)
        print_ratio_table(pt_rows, he_rows)
        plot_latency(pt_rows, he_rows)

    elif args.test == "accuracy":
        from benchmarks.accuracy_tests import main as run_accuracy_tests
        # Forward the --he-samples argument
        sys.argv = ["accuracy_tests", "--he-samples", str(args.he_samples)]
        run_accuracy_tests()


if __name__ == "__main__":
    main()
