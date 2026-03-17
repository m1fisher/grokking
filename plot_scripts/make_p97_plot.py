#!/usr/bin/env python3
"""Plot p=97 ReLU MLP train/test accuracy, one subplot per noise level, log-scale epoch."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP_DIR = "results/sweep"
NOISES = [0.0, 0.1, 0.25, 0.5]

COLORS = {
    "train": "#e74c3c",
    "test":  "#2ecc71",
}


def load_history(act, p, noise):
    tag = f"mlp_{act}_p{p}_n{noise}"
    path = os.path.join(SWEEP_DIR, tag, "history.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def main():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

    for i, noise in enumerate(NOISES):
        ax = axes[i]
        h = load_history("quadratic", 59, noise)
        if h is None:
            continue

        epochs = np.array(h["epoch"])

        ax.plot(epochs, h["train_acc"], color=COLORS["train"],
                linewidth=1.5, label="Train")
        ax.plot(epochs, h["test_acc"], color=COLORS["test"],
                linewidth=1.5, label="Test")

        # ax.set_xscale("log")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(f"$\\xi = {noise*100:.0f}\\%$", fontsize=13, fontweight="bold")
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_ylim(-0.05, 1.08)

        if i == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend(fontsize=10, loc="center right")

    fig.suptitle(
        r"Quadratic MLP on $\mathbb{Z}_{59}$ Addition — Effect of Label Noise"
        "\n(SGD, lr=50, batch=128, no weight decay)",
        fontsize=14, y=1.05,
    )
    fig.tight_layout()
    out = "results/p59_quadratic_noise_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
