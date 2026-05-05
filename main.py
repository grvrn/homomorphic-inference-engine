from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.latency_tests import benchmark_plaintext_latency, benchmark_he_latency
from crypto import generate_keypair
from src.app import he_inference


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Homomorphic Inference Engine CLI")
    parser.add_argument(
        "--test",
        choices=["latency"],
        help="Run a benchmark test",
    )
    args = parser.parse_args()

    if args.test is None:
        parser.print_help()
        sys.exit(0)

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


if __name__ == "__main__":
    main()
