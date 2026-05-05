"""
Latency / scaling helpers: one linear model (``LinearHESpec``) per feature count.

Plaintext and homomorphic paths should use the **same** spec for a given ``d``:
weights and bias length follow ``d``; Alice's vector must be ``shape == (d,)``.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from models.benchmark_linear import DEFAULT_FEATURE_SIZES, train_linear_he_spec_for_dim
from models.he_spec import LinearHESpec
from models.inference import plaintext_logit_from_spec

_spec_cache: dict[int, LinearHESpec] = {}


def spec_for_feature_size(n_features: int, *, refresh: bool = False) -> LinearHESpec:
    """Return (and memoize) a trained ``LinearHESpec`` with ``n_features`` columns."""
    if refresh:
        _spec_cache.pop(n_features, None)
    if n_features not in _spec_cache:
        _spec_cache[n_features] = train_linear_he_spec_for_dim(n_features)
    return _spec_cache[n_features]


def random_raw_sample(n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Raw feature vector for timing (same shape Carol / Alice would use)."""
    return rng.standard_normal(n_features).astype(np.float64)


def compute_plaintext(x: np.ndarray, feature_size: int | None = None) -> float:
    """
    Linear logit w·z + b using the benchmark spec for this dimension.

    If ``feature_size`` is omitted, it is ``x.size`` (or ``x.shape[-1]`` for one row).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    d = int(feature_size if feature_size is not None else x.size)
    if x.size != d:
        raise ValueError(f"x has length {x.size}, expected {d} for this benchmark")
    spec = spec_for_feature_size(d)
    if len(spec.integer_weights) != d:
        raise ValueError("spec / feature_size mismatch")
    return plaintext_logit_from_spec(x, spec)


def benchmark_plaintext_latency(
    feature_sizes: tuple[int, ...] = DEFAULT_FEATURE_SIZES,
    *,
    trials_per_size: int = 200,
    warmup: int = 20,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """
    Time ``compute_plaintext`` for each ``d`` in ``feature_sizes``.

    Returns rows with keys: ``n_features``, ``mean_ms``, ``std_ms`` (per call).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for d in feature_sizes:
        spec_for_feature_size(d)  # train/cache once per d
        for _ in range(warmup):
            compute_plaintext(random_raw_sample(d, rng), d)

        times: list[float] = []
        for _ in range(trials_per_size):
            x = random_raw_sample(d, rng)
            t0 = time.perf_counter()
            compute_plaintext(x, d)
            times.append((time.perf_counter() - t0) * 1000.0)

        a = np.asarray(times, dtype=np.float64)
        rows.append(
            {
                "n_features": d,
                "mean_ms": float(a.mean()),
                "std_ms": float(a.std(ddof=0)),
            }
        )
    return rows


def benchmark_he_latency(
    feature_sizes: tuple[int, ...] = DEFAULT_FEATURE_SIZES,
    *,
    he_inference: Callable[[LinearHESpec, np.ndarray], None] | None = None,
) -> list[dict[str, float | int]]:
    """
    Hook for zirui: pass ``he_inference(spec, x_raw)`` that runs full encrypt→eval→decrypt.

    Until wired, this is a no-op documenting the intended interface.
    """
    if he_inference is None:
        return []
    rng = np.random.default_rng(1)
    rows: list[dict[str, float | int]] = []
    for d in feature_sizes:
        spec = spec_for_feature_size(d)
        times: list[float] = []
        for _ in range(3):
            x = random_raw_sample(d, rng)
            t0 = time.perf_counter()
            he_inference(spec, x)
            times.append((time.perf_counter() - t0) * 1000.0)

        a = np.asarray(times, dtype=np.float64)
        rows.append(
            {
                "n_features": d,
                "mean_ms": float(a.mean()),
                "std_ms": float(a.std(ddof=0)),
            }
        )
    return rows


if __name__ == "__main__":
    for row in benchmark_plaintext_latency(trials_per_size=50, warmup=5):
        print(row)
