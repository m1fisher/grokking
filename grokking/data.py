"""Dataset generation for modular arithmetic grokking experiments."""

import random

import torch
import torch.nn.functional as F


def make_dataset(
    p: int,
    data_frac: float,
    noise_level: float = 0.0,
    operation: str = "addition",
    pair_seed: int = 420,
    noise_seed: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Generate a modular arithmetic dataset with optional label noise.

    Args:
        p: Prime modulus.
        data_frac: Fraction of all p^2 pairs used for training.
        noise_level: Fraction of training labels to corrupt.
        operation: "addition" or "multiplication".
        pair_seed: Seed for the deterministic train/test split shuffle.
        noise_seed: Seed for generating noisy labels.
        device: Torch device.
        dtype: Torch float dtype for inputs.

    Returns:
        Dictionary with X_train, Y_train, X_test, Y_test, X_all, Y_all, p.
    """
    if operation not in ("addition", "multiplication"):
        raise ValueError(f"Unknown operation: {operation}")

    pairs = [(i, j) for i in range(p) for j in range(p)]
    X_all = torch.tensor(pairs)
    if operation == "addition":
        Y_all = (X_all[:, 0] + X_all[:, 1]) % p
    else:
        Y_all = (X_all[:, 0] * X_all[:, 1]) % p

    # One-hot encode: concat one_hot(i) and one_hot(j) -> 2p dims
    X_all_oh = F.one_hot(X_all, num_classes=p).to(dtype=dtype, device=device)
    Y_all = Y_all.to(dtype=torch.long, device=device)

    # Deterministic shuffle for train/test split
    random.seed(pair_seed)
    order = list(range(len(pairs)))
    random.shuffle(order)
    perm = torch.tensor(order, device=device)

    total = len(pairs)
    train_size = int(data_frac * total)

    X_shuffled = X_all_oh[perm]
    Y_shuffled = Y_all[perm]

    # Apply label noise to training set
    Y_noisy = Y_shuffled.clone()
    if noise_level > 0:
        n_noise = int(noise_level * train_size)
        torch.manual_seed(noise_seed)
        Y_noisy[:n_noise] = torch.randint(0, p, (n_noise,), device=device)

    return {
        "X_train": X_shuffled[:train_size],
        "Y_train": Y_noisy[:train_size],
        "X_test": X_shuffled[train_size:],
        "Y_test": Y_noisy[train_size:],
        "X_all": X_all_oh,
        "Y_all": Y_all,
        "p": p,
    }
