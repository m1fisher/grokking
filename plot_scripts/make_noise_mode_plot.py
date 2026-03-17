#!/usr/bin/env python3
"""Plot noise mode comparison: SGD fixed, SGD stratified, GD per_epoch."""

import json
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


def main():
    runs = [
        ("results/sweep/mlp_relu_p97_n0.1/history.json",
         "SGD, fixed noise"),
        ("results/mlp_p97_noise_stratified/history.json",
         "SGD, stratified noise"),
        ("results/mlp_relu_p97_fullbatch_noise0.1/history.json",
         "Full-batch GD, fixed noise"),
        ("results/mlp_p97_noise_per_epoch_long/history.json",
         "Full-batch GD, per-epoch noise"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True, sharex=False)

    for i, (path, label) in enumerate(runs):
        ax = axes[i // 2][i % 2]
        h = json.load(open(path))
        epochs = np.array(h["epoch"])

        ax.plot(epochs, h["train_acc"], color="#c0392b",
                linewidth=2, label="Train")
        ax.plot(epochs, h["test_acc"], color="#27ae60",
                linewidth=2, label="Test")

        ax.set_xlabel("Epoch")
        ax.set_title(label, fontweight="bold")
        ax.axhline(y=1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.3)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(loc="center right", framealpha=0.9)

    axes[0][0].set_ylabel("Accuracy")
    axes[1][0].set_ylabel("Accuracy")

    fig.suptitle(
        r"ReLU MLP, $p = 97$, 10% label noise — Noise Mode Comparison"
        "\n(lr = 50, no weight decay)",
        fontsize=15, y=1.06,
    )
    fig.tight_layout()
    out = "results/p97_noise_modes.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
