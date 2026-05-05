from __future__ import annotations

from collections.abc import Sequence

from phe.paillier import EncryptedNumber

PaillierCiphertext = EncryptedNumber


def encrypt(value: int, public_key):
    return public_key.encrypt(int(value))


def decrypt(encrypted_value, private_key) -> int:
    if hasattr(private_key, "private_key"):
        private_key = private_key.private_key
    return int(private_key.decrypt(encrypted_value))


def add_encrypted(left, right):
    return left + right


def scalar_mul(encrypted_value, scalar: int):
    return encrypted_value * int(scalar)


def encrypt_features(public_key, features: Sequence[int]):
    return [encrypt(x, public_key) for x in features]


def encrypt_value(public_key, value: int):
    return encrypt(value, public_key)


def decrypt_score(private_key, encrypted_score) -> int:
    return decrypt(encrypted_score, private_key)


def homomorphic_linear_score(
    public_key,
    encrypted_features,
    integer_weights: Sequence[int],
    integer_bias: int,
):
    encrypted_score = encrypt_value(public_key, integer_bias)
    for encrypted_x, weight in zip(encrypted_features, integer_weights):
        encrypted_score = add_encrypted(encrypted_score, scalar_mul(encrypted_x, weight))
    return encrypted_score
