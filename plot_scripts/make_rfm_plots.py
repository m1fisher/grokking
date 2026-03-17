#!/usr/bin/env python3
"""Plot RFM test accuracy per prime across noise levels."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SWEEP_DIR = "results/sweep"
PRIMES = [11, 31, 59, 97]
NOISES = [0.0, 0.1, 0.25, 0.5]

NOISE_COLORS = {
    0.0:  "#27ae60",
    0.1:  "#2980b9",
    0.25: "#e67e22",
    0.5:  "#c0392b",
}


def load_history(p, noise):
    tag = f"rfm_p{p}_n{noise}"
    hist_path = os.path.join(SWEEP_DIR, tag, "history.json")
    if os.path.exists(hist_path):
        return json.load(open(hist_path))
    # Parse run.log for in-progress runs
    log_path = os.path.join(SWEEP_DIR, tag, "run.log")
    if os.path.exists(log_path):
        h = {"epoch": [], "test_acc": []}
        with open(log_path) as f:
            for line in f:
                if line.startswith("iter"):
                    parts = line.split()
                    try:
                        h["epoch"].append(int(parts[1]))
                        h["test_acc"].append(float(parts[5].split("=")[1]))
                    except (IndexError, ValueError):
                        continue
        if h["epoch"]:
            return h
    return None


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for idx, p in enumerate(PRIMES):
        ax = axes[idx // 2][idx % 2]

        for noise in NOISES:
            h = load_history(p, noise)
            if h is None or not h["epoch"]:
                continue
            iters = np.array(h["epoch"])
            ax.plot(iters, h["test_acc"], color=NOISE_COLORS[noise],
                    linewidth=2,
                    label=f"$\\xi = {noise*100:.0f}\\%$")

        ax.set_title(f"$p = {p}$", fontweight="bold")
        ax.set_xlabel("RFM Iteration")
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(loc="center right", framealpha=0.9)

    axes[0][0].set_ylabel("Test Accuracy")
    axes[1][0].set_ylabel("Test Accuracy")

    fig.suptitle(
        "RFM (Gaussian Kernel) — Test Accuracy by Prime and Noise Level"
        "\n(bandwidth = 2.5, ridge = 0, 200 iterations)",
        fontsize=15, y=1.03,
    )
    fig.tight_layout()
    out = "results/rfm_noise_comparison.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
