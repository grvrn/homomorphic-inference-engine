"""
Homomorphic inference pipeline.

Implements the full encrypt → evaluate → decrypt flow for a linear model
using Paillier-style homomorphic encryption.

Roles (following the Alice / Carol convention from ``he_spec.py``):
  • Alice  – data owner: standardises raw features, encrypts integer-encoded
             feature vector, sends ciphertexts to Carol, decrypts the result.
  • Carol  – model owner: holds public integer weights/bias from the
             ``LinearHESpec``, evaluates the linear model on ciphertexts using
             ``scalar_mul`` and ``add_encrypted``.
"""

from __future__ import annotations

import numpy as np

from crypto import (
    PaillierKeyPair,
    PaillierCiphertext,
    generate_keypair,
    encrypt,
    decrypt,
    add_encrypted,
    scalar_mul,
)
from models.he_spec import LinearHESpec
from models.inference import (
    integer_features_from_raw,
    recovered_logit_from_integer_dot,
    predict_label_from_logit,
)


# ---------------------------------------------------------------------------
# Alice-side helpers
# ---------------------------------------------------------------------------

def encrypt_features(
    x_int: np.ndarray,
    keypair: PaillierKeyPair,
) -> list[PaillierCiphertext]:
    """Encrypt each integer-encoded feature element-wise under the public key."""
    pk = keypair.public_key
    return [encrypt(int(xi), pk) for xi in x_int.ravel()]


def decrypt_score(
    encrypted_score: PaillierCiphertext,
    keypair: PaillierKeyPair,
) -> int:
    """Decrypt the aggregated ciphertext to obtain the integer dot-product + bias."""
    return decrypt(encrypted_score, keypair.secret_key)


# ---------------------------------------------------------------------------
# Carol-side evaluation (model owner, never sees plaintext)
# ---------------------------------------------------------------------------

def evaluate_encrypted(
    encrypted_features: list[PaillierCiphertext],
    spec: LinearHESpec,
) -> PaillierCiphertext:
    """
    Evaluate the linear model on encrypted feature vector (Carol's side).

    Computes  sum_i(w_i · Enc(x_i)) + Enc(0) ⊕ bias  homomorphically:
      1. For each feature, scalar-multiply the ciphertext by the integer weight.
      2. Sum all resulting ciphertexts via homomorphic addition.
      3. Add the encrypted bias term.

    Returns a single ciphertext whose decryption equals
    sum(w_i * x_i) + bias  (in fixed-point integer space).
    """
    weights = spec.integer_weights
    if len(encrypted_features) != len(weights):
        raise ValueError(
            f"Feature count mismatch: got {len(encrypted_features)} "
            f"ciphertexts but spec has {len(weights)} weights"
        )

    pk = encrypted_features[0].public_key

    # w_0 * Enc(x_0)
    result = scalar_mul(encrypted_features[0], weights[0])

    # Accumulate w_i * Enc(x_i) for i >= 1
    for ct, w in zip(encrypted_features[1:], weights[1:]):
        result = add_encrypted(result, scalar_mul(ct, w))

    # Add bias: Enc(sum) ⊕ Enc(bias) = Enc(sum + bias)
    enc_bias = encrypt(spec.integer_bias, pk)
    result = add_encrypted(result, enc_bias)

    return result


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

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
