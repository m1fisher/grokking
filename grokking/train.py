"""Training loop and evaluation for grokking experiments."""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_loss(scores: torch.Tensor, Y: torch.Tensor, p: int, loss_fn: str) -> torch.Tensor:
    if loss_fn == "mse":
        targets = F.one_hot(Y, num_classes=p).to(dtype=scores.dtype, device=scores.device)
        return F.mse_loss(scores, targets)
    else:
        return F.cross_entropy(scores, Y)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    model.eval()
    preds = model(X).argmax(dim=1)
    return (preds == Y).float().mean().item()


@torch.no_grad()
def eval_loss(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, p: int, loss_fn: str) -> float:
    model.eval()
    return _compute_loss(model(X), Y, p, loss_fn).item()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    p: int,
    loss_fn: str,
) -> float:
    """One full-batch gradient step. Returns training loss."""
    model.train()
    optimizer.zero_grad()
    loss = _compute_loss(model(X_train), Y_train, p, loss_fn)
    loss.backward()
    optimizer.step()
    return loss.item()


def make_log_epochs(epochs: int, log_every: int) -> list[int]:
    log_epochs = list(range(log_every, epochs + 1, log_every))
    if 1 not in log_epochs:
        log_epochs.insert(0, 1)
    if epochs not in log_epochs:
        log_epochs.append(epochs)
    return sorted(log_epochs)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: dict,
    epochs: int,
    log_every: int = 10,
    loss_fn: str = "mse",
    device: torch.device = torch.device("cpu"),
    on_log: Optional[Callable[[nn.Module, int], None]] = None,
) -> dict:
    """Full training loop.

    Args:
        loss_fn: "mse" or "ce" (cross-entropy).
        on_log: Optional callback(model, epoch) called at each log point.

    Returns a history dict with per-logged-epoch metrics.
    """
    p = dataset["p"]
    X_train = dataset["X_train"].to(device)
    Y_train = dataset["Y_train"].to(device)
    X_test = dataset["X_test"].to(device)
    Y_test = dataset["Y_test"].to(device)

    model.to(device)
    log_set = set(make_log_epochs(epochs, log_every))

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, optimizer, X_train, Y_train, p, loss_fn)

        if epoch in log_set:
            test_loss_val = eval_loss(model, X_test, Y_test, p, loss_fn)
            train_acc = accuracy(model, X_train, Y_train)
            test_acc = accuracy(model, X_test, Y_test)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss_val)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            if on_log is not None:
                on_log(model, epoch)

            print(
                f"epoch {epoch:>5d}  "
                f"train_loss={train_loss:.6f}  test_loss={test_loss_val:.6f}  "
                f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
            )

    return history
