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

from crypto import PaillierCiphertext
from crypto.crypto import homomorphic_linear_score
from models.he_spec import LinearHESpec


# ---------------------------------------------------------------------------
# Carol-side evaluation (model owner, never sees plaintext)
# ---------------------------------------------------------------------------

def evaluate_encrypted(
    encrypted_features: list[PaillierCiphertext],
    spec: LinearHESpec,
) -> PaillierCiphertext:
    """
    Evaluate the linear model on encrypted feature vector (Carol's side).

    Computes  sum_i(w_i · Enc(x_i)) + Enc(bias)  homomorphically via
    the canonical ``homomorphic_linear_score`` in the crypto layer.

    Returns a single ciphertext whose decryption equals
    sum(w_i * x_i) + bias  (in fixed-point integer space).
    """
    if len(encrypted_features) != len(spec.integer_weights):
        raise ValueError(
            f"Feature count mismatch: got {len(encrypted_features)} "
            f"ciphertexts but spec has {len(spec.integer_weights)} weights"
        )

    pk = encrypted_features[0].public_key

    return homomorphic_linear_score(
        public_key=pk,
        encrypted_features=encrypted_features,
        integer_weights=spec.integer_weights,
        integer_bias=spec.integer_bias,
    )
