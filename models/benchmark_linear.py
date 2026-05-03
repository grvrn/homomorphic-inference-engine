"""
Train logistic regression + export ``LinearHESpec`` for an arbitrary feature count.

Used when sweeping feature dimension for latency / scaling experiments: each ``d``
gets its own weights, bias, and scaler (length ``d``). Do not slice or pad one
model across different ``d``; train (or load) a separate spec per size.
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .he_spec import LinearHESpec, export_linear_he_spec

DEFAULT_FEATURE_SIZES = (4, 8, 16, 32, 64, 128)


def train_linear_he_spec_for_dim(
    n_features: int,
    *,
    n_samples: int = 4000,
    fixed_point_scale: int = 1_000_000,
    random_state: int | None = None,
    max_iter: int = 1000,
) -> LinearHESpec:
    """
    Synthetic binary data with ``n_features`` columns, then LR + scaler export.

    ``random_state`` is fixed per call site (e.g. ``1000 + n_features``) so
    sweeps over ``FEATURE_SIZES`` are reproducible.
    """
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    rs = random_state if random_state is not None else 10_000 + n_features
    n_informative = min(max(2, n_features // 2), n_features)
    n_redundant = max(0, n_features - n_informative)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        random_state=rs,
    )
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=rs, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    model.fit(X_train_s, y_train)

    names = [f"f{i}" for i in range(n_features)]
    return export_linear_he_spec(
        model=model,
        scaler=scaler,
        feature_names=names,
        fixed_point_scale=fixed_point_scale,
    )
