"""Fourier analysis utilities: IPR, Gini coefficient, weight extraction."""

import numpy as np


def ipr(x: np.ndarray, r: float = 2.0) -> float:
    """Inverse participation ratio of a 1-d array."""
    norm = np.sqrt((x**2).sum())
    if norm == 0:
        return 0.0
    return np.power(x / norm, 2 * r).sum()


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a 1-d array."""
    mu = x.mean()
    if mu == 0:
        return 0.0
    return np.abs(np.expand_dims(x, 0) - np.expand_dims(x, 1)).mean() / (2 * mu)


def extract_weights(model) -> dict:
    """Extract U, V, W weight matrices as numpy arrays.

    fc1.weight has shape (N, 2p) -> U = [:, :p], V = [:, p:]
    fc2.weight has shape (p, N) -> W
    """
    w1 = model.fc1.weight.detach().cpu().numpy()
    p = w1.shape[1] // 2
    return {
        "U": w1[:, :p].copy(),
        "V": w1[:, p:].copy(),
        "W": model.fc2.weight.detach().cpu().numpy().copy(),
    }


def fft_iprs_and_ginis(weights: dict) -> dict:
    """Compute per-row/column FFT IPRs and Gini coefficients for U, V, W."""
    results = {}
    N = weights["U"].shape[0]
    for name, mat, axis in [("U", weights["U"], 1), ("V", weights["V"], 1), ("W", weights["W"], 0)]:
        vecs = mat if axis == 1 else mat.T  # iterate over rows for U/V, columns for W
        iprs = np.array([ipr(np.abs(np.fft.rfft(vecs[k]))) for k in range(N)])
        ginis = np.array([gini(np.abs(np.fft.rfft(vecs[k]))) for k in range(N)])
        results[f"{name}_ipr"] = iprs
        results[f"{name}_gini"] = ginis
    return results


def compute_norms(weights: dict) -> dict:
    """Frobenius norms squared of U, V, W."""
    return {k: (v**2).sum() for k, v in weights.items()}
