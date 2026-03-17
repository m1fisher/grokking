#!/usr/bin/env python3
"""Plot MLP train/test accuracy per prime and activation across noise levels."""

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
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SWEEP_DIR = os.environ.get("SWEEP_DIR", "results/sweep")
PRIMES = [11, 31, 59, 97]
NOISES = [0.0, 0.1, 0.25, 0.5]
ACTIVATIONS = ["relu", "silu", "quadratic"]
ACT_LABELS = {"relu": "ReLU", "silu": "SiLU", "quadratic": r"Quadratic ($x^2$)"}

NOISE_COLORS = {
    0.0:  "#27ae60",
    0.1:  "#2980b9",
    0.25: "#e67e22",
    0.5:  "#c0392b",
}


def load_history(act, p, noise):
    tag = f"mlp_{act}_p{p}_n{noise}"
    path = os.path.join(SWEEP_DIR, tag, "history.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def make_plot(act):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for idx, p in enumerate(PRIMES):
        ax = axes[idx // 2][idx % 2]

        for noise in NOISES:
            h = load_history(act, p, noise)
            if h is None:
                continue
            epochs = np.array(h["epoch"])
            c = NOISE_COLORS[noise]
            label = f"$\\xi = {noise*100:.0f}\\%$"

            ax.plot(epochs, h["train_acc"], color=c, linestyle="--",
                    linewidth=1.5, alpha=0.6)
            ax.plot(epochs, h["test_acc"], color=c, linestyle="-",
                    linewidth=2, label=label)

        ax.set_title(f"$p = {p}$", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(loc="center right", framealpha=0.9)

    axes[0][0].set_ylabel("Accuracy")
    axes[1][0].set_ylabel("Accuracy")

    fig.suptitle(
        f"{ACT_LABELS[act]} MLP — Train (dashed) / Test (solid) Accuracy"
        "\n(SGD, lr = 50, batch = 128, no weight decay)",
        fontsize=15, y=1.03,
    )
    fig.tight_layout()
    out_dir = os.path.dirname(SWEEP_DIR) or "results"
    out = os.path.join(out_dir, f"mlp_{act}_sweep.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    for act in ACTIVATIONS:
        make_plot(act)
