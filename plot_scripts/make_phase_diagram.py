#!/usr/bin/env python3
"""Generate phase diagrams from MLP and RFM sweep results.

Row 1: MLP (ReLU, SiLU, Quadratic)
Row 2: RFM (Gaussian Kernel)

Uses scipy interpolation to create smooth continuous phase regions.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import RegularGridInterpolator

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
})

SWEEP_DIR = "results/sweep"
PRIMES = [11, 31, 59, 97]
NOISES = [0.0, 0.1, 0.25, 0.5]
ACTIVATIONS = ["relu", "silu", "quadratic"]
ACT_LABELS = {"relu": "ReLU", "silu": "SiLU (Swish)", "quadratic": r"Quadratic ($x^2$)"}

# Phase classification
TEST_HIGH = 0.90
TEST_PARTIAL_LOW = 0.10
TRAIN_HIGH = 0.90

PHASE_ID = {
    "Confusion":              0,
    "Memorization":           1,
    "Partial Generalization": 2,
    "Coexistence":            3,
    "Partial Inversion":      4,
    "Full Inversion":         5,
}

PHASE_CMAP_COLORS = [
    "#bdc3c7",  # Confusion - light gray
    "#e74c3c",  # Memorization - red
    "#f39c12",  # Partial Generalization - orange
    "#27ae60",  # Coexistence - green
    "#2ecc71",  # Partial Inversion - light green
    "#a3e4d7",  # Full Inversion - pale green
]


def classify_phase(train_acc, test_acc, noise, p=97):
    xi = noise
    partial_thresh = max(TEST_PARTIAL_LOW, 2.0 / p)
    if test_acc >= TEST_HIGH:
        if train_acc >= TRAIN_HIGH:
            return "Coexistence"
        elif train_acc >= (1.05 - xi):
            return "Partial Inversion"
        else:
            return "Full Inversion"
    elif test_acc >= partial_thresh:
        return "Partial Generalization"
    elif train_acc >= TRAIN_HIGH:
        return "Memorization"
    else:
        return "Confusion"


def load_results():
    results = {}
    # MLP results
    for act in ACTIVATIONS:
        for p in PRIMES:
            for noise in NOISES:
                tag = f"mlp_{act}_p{p}_n{noise}"
                path = os.path.join(SWEEP_DIR, tag, "history.json")
                if os.path.exists(path):
                    h = json.load(open(path))
                    results[("mlp", act, p, noise)] = {
                        "train_acc": h["train_acc"][-1],
                        "test_acc": h["test_acc"][-1],
                    }
    # RFM results
    for p in PRIMES:
        for noise in NOISES:
            tag = f"rfm_p{p}_n{noise}"
            hist_path = os.path.join(SWEEP_DIR, tag, "history.json")
            log_path = os.path.join(SWEEP_DIR, tag, "run.log")
            if os.path.exists(hist_path):
                h = json.load(open(hist_path))
                results[("rfm", None, p, noise)] = {
                    "train_acc": h.get("train_acc", [1.0])[-1],
                    "test_acc": h["test_acc"][-1],
                }
            elif os.path.exists(log_path):
                # Parse in-progress run.log
                train_acc, test_acc = 1.0, 0.0
                with open(log_path) as f:
                    for line in f:
                        if line.startswith("iter"):
                            parts = line.split()
                            try:
                                train_acc = float(parts[4].split("=")[1])
                                test_acc = float(parts[5].split("=")[1])
                            except (IndexError, ValueError):
                                continue
                results[("rfm", None, p, noise)] = {
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                }
    return results


def _plot_phase_panel(ax, grid_data, cmap, norm, prime_coords, noise_coords,
                      fine_pp, fine_nn, fine_p, fine_n, all_phases):
    """Fill one axes with interpolated phase colors."""
    interp = RegularGridInterpolator(
        (prime_coords, noise_coords), grid_data,
        method="nearest", bounds_error=False, fill_value=None,
    )
    fine_grid = interp((fine_pp, fine_nn))
    ax.pcolormesh(
        fine_n, fine_p, fine_grid,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )


def make_phase_diagram(results, out_path="results/phase_diagram.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={"wspace": 0.12, "hspace": 0.35})

    cmap = ListedColormap(PHASE_CMAP_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    all_phases = set()

    prime_coords = np.arange(len(PRIMES), dtype=float)
    noise_coords = np.arange(len(NOISES), dtype=float)

    fine_p = np.linspace(-0.5, len(PRIMES) - 0.5, 400)
    fine_n = np.linspace(-0.5, len(NOISES) - 0.5, 400)
    fine_pp, fine_nn = np.meshgrid(fine_p, fine_n, indexing="ij")

    # --- Row 0: MLP activations ---
    for ax_i, act in enumerate(ACTIVATIONS):
        ax = axes[0][ax_i]

        grid = np.full((len(PRIMES), len(NOISES)), np.nan)
        for i, p in enumerate(PRIMES):
            for j, noise in enumerate(NOISES):
                key = ("mlp", act, p, noise)
                if key not in results:
                    continue
                r = results[key]
                phase = classify_phase(r["train_acc"], r["test_acc"], noise, p)
                all_phases.add(phase)
                grid[i, j] = PHASE_ID[phase]

        _plot_phase_panel(ax, grid, cmap, norm, prime_coords, noise_coords,
                          fine_pp, fine_nn, fine_p, fine_n, all_phases)

        ax.set_title(f"MLP — {ACT_LABELS[act]}", fontweight="bold", pad=10)
        ax.set_xticks(range(len(NOISES)))
        ax.set_xticklabels([f"${n*100:.0f}\\%$" for n in NOISES])
        ax.set_xlabel(r"Label noise ($\xi$)")

        if ax_i == 0:
            ax.set_yticks(range(len(PRIMES)))
            ax.set_yticklabels([f"$p = {p}$" for p in PRIMES])
            ax.set_ylabel("Prime modulus")
        else:
            ax.set_yticks(range(len(PRIMES)))
            ax.set_yticklabels([])

        ax.set_xlim(-0.5, len(NOISES) - 0.5)
        ax.set_ylim(-0.5, len(PRIMES) - 0.5)
        ax.tick_params(length=0)

    # --- Row 1: RFM (single panel, centered) ---
    # Hide the two flanking axes
    axes[1][0].set_visible(False)
    axes[1][2].set_visible(False)

    ax_rfm = axes[1][1]

    grid = np.full((len(PRIMES), len(NOISES)), np.nan)
    for i, p in enumerate(PRIMES):
        for j, noise in enumerate(NOISES):
            key = ("rfm", None, p, noise)
            if key not in results:
                continue
            r = results[key]
            phase = classify_phase(r["train_acc"], r["test_acc"], noise, p)
            all_phases.add(phase)
            grid[i, j] = PHASE_ID[phase]

    _plot_phase_panel(ax_rfm, grid, cmap, norm, prime_coords, noise_coords,
                      fine_pp, fine_nn, fine_p, fine_n, all_phases)

    ax_rfm.set_title("RFM — Gaussian Kernel", fontweight="bold", pad=10)
    ax_rfm.set_xticks(range(len(NOISES)))
    ax_rfm.set_xticklabels([f"${n*100:.0f}\\%$" for n in NOISES])
    ax_rfm.set_xlabel(r"Label noise ($\xi$)")
    ax_rfm.set_yticks(range(len(PRIMES)))
    ax_rfm.set_yticklabels([f"$p = {p}$" for p in PRIMES])
    ax_rfm.set_ylabel("Prime modulus")
    ax_rfm.set_xlim(-0.5, len(NOISES) - 0.5)
    ax_rfm.set_ylim(-0.5, len(PRIMES) - 0.5)
    ax_rfm.tick_params(length=0)

    # Legend
    legend_order = [
        "Coexistence", "Partial Inversion", "Full Inversion",
        "Partial Generalization", "Memorization", "Confusion",
    ]
    patches = [mpatches.Patch(
        facecolor=PHASE_CMAP_COLORS[PHASE_ID[l]], edgecolor="gray",
        linewidth=0.5, label=l)
        for l in legend_order if l in all_phases]

    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=11, frameon=True, edgecolor="gray",
               bbox_to_anchor=(0.5, -0.04), handlelength=1.8, handleheight=1.4)

    fig.suptitle(
        r"Phase Diagrams — Modular Addition ($\eta = 50$, no weight decay)",
        fontsize=15, y=1.01,
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    results = load_results()
    make_phase_diagram(results)
