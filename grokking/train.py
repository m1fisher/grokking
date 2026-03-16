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
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    p: int,
    loss_fn: str,
) -> float:
    """One gradient step on a batch. Returns batch loss."""
    model.train()
    optimizer.zero_grad()
    loss = _compute_loss(model(X_batch), Y_batch, p, loss_fn)
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
    batch_size: int = 0,
    early_stop_patience: int = 0,
    device: torch.device = torch.device("cpu"),
    on_log: Optional[Callable[[nn.Module, int], None]] = None,
) -> dict:
    """Full training loop.

    Args:
        loss_fn: "mse" or "ce" (cross-entropy).
        batch_size: Minibatch size. 0 means full-batch.
        early_stop_patience: Stop after this many consecutive logged epochs
            with test_acc == 1.0. 0 disables early stopping.
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
    n_train = X_train.shape[0]
    full_batch = batch_size <= 0 or batch_size >= n_train
    perfect_streak = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        if full_batch:
            train_loss = train_step(model, optimizer, X_train, Y_train, p, loss_fn)
        else:
            # Shuffle and iterate over minibatches
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                batch_loss = train_step(model, optimizer, X_train[idx], Y_train[idx], p, loss_fn)
                epoch_loss += batch_loss
                n_batches += 1
            train_loss = epoch_loss / n_batches

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

            if early_stop_patience > 0:
                if test_acc >= 1.0:
                    perfect_streak += 1
                    if perfect_streak >= early_stop_patience:
                        print(f"Early stopping: test_acc=1.0 for {perfect_streak} log points")
                        return history
                else:
                    perfect_streak = 0

    return history
