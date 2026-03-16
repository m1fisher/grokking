"""Recursive Feature Machine (RFM) with Gaussian kernel for grokking experiments.

Implements the RFM algorithm from "Emergence in non-neural models: grokking modular
arithmetic via average gradient outer product" (arXiv:2407.20199).

The RFM iteratively:
1. Solves kernel ridge regression with current metric M
2. Computes the Average Gradient Outer Product (AGOP)
3. Updates M = sqrt(AGOP)
"""

import numpy as np
import scipy.linalg
import torch


def _euclidean_distances_M(X, Y, M):
    """Squared Mahalanobis distances: ||x - y||_M^2 = (x-y)^T M (x-y)."""
    XM = X @ M
    X_norm = (XM * X).sum(dim=1, keepdim=True)
    if X is Y:
        Y_norm = X_norm
    else:
        YM = Y @ M
        Y_norm = (YM * Y).sum(dim=1, keepdim=True)
    # ||x-y||_M^2 = x^T M x - 2 x^T M y + y^T M y
    dist = X_norm - 2 * (X @ M @ Y.T) + Y_norm.T
    return dist


def gaussian_kernel(X, Y, bandwidth, M):
    """Gaussian kernel K(x,y) = exp(-||x-y||_M^2 / (2 * bandwidth^2))."""
    dist = _euclidean_distances_M(X, Y, M)
    dist.clamp_(min=0)
    return torch.exp(-dist / (2 * bandwidth**2))


def _compute_agop(X_tr, K_train, sol, M, bandwidth, centering=True):
    """Compute the Average Gradient Outer Product for the Gaussian kernel.

    The gradient of K(x_i, x_j) w.r.t. M is used to form the AGOP,
    which captures which input features matter for prediction.

    Args:
        sol: shape (c, n) — kernel regression coefficients (classes x samples).
    """
    n, d = X_tr.shape
    c = sol.shape[0]

    # Following reference: a1 = sol.T -> (n, c), a2 = sol -> (c, n)
    a1 = sol.T  # (n, c)
    XM = X_tr @ M  # (n, d)

    # step1: sum over training points weighted by alpha and kernel
    a1_r = a1.reshape(n, c, 1)
    XM_r = XM.reshape(n, 1, d)
    step1 = (a1_r @ XM_r).reshape(-1, c * d)  # (n, c*d)
    step2 = (K_train.T @ step1).reshape(-1, c, d)  # (n, c, d)

    # step3: second term
    step3_coeff = (sol @ K_train).T  # (c,n) @ (n,n) -> (c,n) -> .T -> (n, c)
    step3 = step3_coeff.reshape(n, c, 1) @ XM.reshape(n, 1, d)  # (n, c, d)

    G = (step2 - step3) * (-1.0 / bandwidth**2)  # (n, c, d)

    if centering:
        G = G - G.mean(dim=0)

    # AGOP = (1/n) sum_i G_i^T G_i
    # Reshape G from (n, c, d) to (n*c, d) and do a single matmul
    G_flat = G.reshape(-1, d)
    agop = (G_flat.T @ G_flat) / n

    # M_new = sqrt(AGOP)
    agop_np = agop.numpy()
    M_new = np.real(scipy.linalg.sqrtm(agop_np))
    return torch.from_numpy(M_new)


def rfm_solve(X_tr, Y_tr_oh, M, bandwidth, ridge):
    """Solve kernel ridge regression: alpha = (K + ridge*I)^-1 Y.

    Returns sol with shape (n, c) and K_train (n, n).
    """
    K_train = gaussian_kernel(X_tr, X_tr, bandwidth, M)
    K_np = K_train.numpy() + ridge * np.eye(len(K_train))
    sol = np.linalg.solve(K_np, Y_tr_oh.numpy())
    return torch.from_numpy(sol), K_train


def rfm_eval(sol, K, Y_oh):
    """Evaluate predictions. sol is (n, c), K is (n_train, n_eval)."""
    preds = K.T @ sol
    loss = (preds - Y_oh).pow(2).mean().item()
    acc = (preds.argmax(dim=1) == Y_oh.argmax(dim=1)).float().mean().item()
    return acc, loss


def train_rfm(
    dataset: dict,
    iters: int = 50,
    bandwidth: float = 2.5,
    ridge: float = 0.0,
    early_stop_patience: int = 0,
    on_log=None,
) -> dict:
    """Run the RFM algorithm.

    Args:
        dataset: Dict with X_train, Y_train, X_test, Y_test, p.
        iters: Number of RFM iterations.
        bandwidth: Gaussian kernel bandwidth.
        ridge: Ridge regression regularization.
        early_stop_patience: Stop after N consecutive iters with test_acc=1.0.
        on_log: Optional callback(M, iteration).

    Returns:
        History dict with per-iteration metrics.
    """
    p = dataset["p"]
    # RFM uses float64 for numerical stability in kernel solves
    X_tr = dataset["X_train"].flatten(1).double()
    X_te = dataset["X_test"].flatten(1).double()
    Y_tr_oh = torch.nn.functional.one_hot(dataset["Y_train"], p).double()
    Y_te_oh = torch.nn.functional.one_hot(dataset["Y_test"], p).double()

    d = X_tr.shape[1]
    M = torch.eye(d, dtype=torch.float64)

    history = {
        "iter": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    perfect_streak = 0

    for it in range(iters):
        sol, K_train = rfm_solve(X_tr, Y_tr_oh, M, bandwidth, ridge)

        train_acc, train_loss = rfm_eval(sol, K_train, Y_tr_oh)

        K_test = gaussian_kernel(X_tr, X_te, bandwidth, M)
        test_acc, test_loss = rfm_eval(sol, K_test, Y_te_oh)

        history["iter"].append(it)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"iter {it:>4d}  "
            f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )

        if on_log is not None:
            on_log(M, it)

        if early_stop_patience > 0:
            if test_acc >= 1.0:
                perfect_streak += 1
                if perfect_streak >= early_stop_patience:
                    print(f"Early stopping: test_acc=1.0 for {perfect_streak} iters")
                    return history
            else:
                perfect_streak = 0

        # Update M via AGOP (sol.T gives (c, n) as expected by _compute_agop)
        M = _compute_agop(X_tr, K_train, sol.T, M, bandwidth, centering=True)

    return history
