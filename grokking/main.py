"""Main entry point for the grokking modular arithmetic experiment."""

import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grokking.data import make_dataset
from grokking.model import MLP
from grokking.train import train
from grokking.rfm import train_rfm
from grokking.analysis import extract_weights, fft_iprs_and_ginis, compute_norms


def parse_args():
    parser = argparse.ArgumentParser(description="Grokking modular arithmetic")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "rfm"],
                        help="Model type: mlp or rfm (Recursive Feature Machine)")
    parser.add_argument("--p", type=int, default=97, help="Prime modulus")
    parser.add_argument("--hidden", type=int, default=500, help="Hidden layer width (mlp only)")
    parser.add_argument("--data-frac", type=float, default=0.5, help="Training data fraction")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Label noise fraction")
    parser.add_argument("--operation", type=str, default="addition", choices=["addition", "multiplication"])
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=50.0, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "sgd"],
                        help="Optimizer")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "ce"], help="Loss function")
    parser.add_argument("--depth", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu", "tanh", "quadratic"],
                        help="Activation function")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size (0 = full-batch)")
    parser.add_argument("--early-stop", type=int, default=0,
                        help="Stop after N consecutive log points with test_acc=1.0 (0 = off)")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N epochs")
    # RFM-specific
    parser.add_argument("--rfm-iters", type=int, default=50, help="RFM iterations (rfm only)")
    parser.add_argument("--bandwidth", type=float, default=2.5, help="Gaussian kernel bandwidth (rfm only)")
    parser.add_argument("--ridge", type=float, default=0.0, help="Ridge regularization (rfm only)")
    parser.add_argument("--seed", type=int, default=1, help="Model init seed")
    parser.add_argument("--pair-seed", type=int, default=420, help="Train/test split seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_plots(history: dict, weight_snapshots: list, args, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.array(history["epoch"])

    # --- Losses & Accuracies ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].set_title("Train / Test Loss")
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["test_loss"], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].set_title("Train / Test Loss (log)")
    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["test_loss"], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].legend()

    axes[2].set_title("Train / Test Accuracy")
    axes[2].plot(epochs, history["train_acc"], label="train")
    axes[2].plot(epochs, history["test_acc"], label="test")
    axes[2].axhline(y=1.0, color="black", linestyle="dashed")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("accuracy")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses_accuracies.png"), dpi=150)
    plt.close(fig)

    # --- Weight norms ---
    u_norms = [s["norms"]["U"] for s in weight_snapshots]
    v_norms = [s["norms"]["V"] for s in weight_snapshots]
    w_norms = [s["norms"]["W"] for s in weight_snapshots]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title("Weight Norms (Frobenius)")
    ax1.plot(epochs, u_norms, label="U")
    ax1.plot(epochs, v_norms, label="V")
    ax1.plot(epochs, w_norms, label="W")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("||W||^2_F")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["test_loss"], color="grey", alpha=0.5, linestyle="dashed", label="test loss")
    ax2.set_ylabel("loss")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_norms.png"), dpi=150)
    plt.close(fig)

    # --- IPRs & Gini ---
    avg_iprs = {k: [s["fft"][f"{k}_ipr"].mean() for s in weight_snapshots] for k in "UVW"}
    avg_ginis = {k: [s["fft"][f"{k}_gini"].mean() for s in weight_snapshots] for k in "UVW"}
    chi = 0.5
    ipr_counts = {k: [(s["fft"][f"{k}_ipr"] > chi).sum() for s in weight_snapshots] for k in "UVW"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].set_title("Average IPRs")
    for k in "UVW":
        axes[0].plot(epochs, avg_iprs[k], label=k)
    axes[0].axhline(y=1.0, color="black", linestyle="dashed", label="pure freq")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("avg IPR")
    axes[0].legend()

    axes[1].set_title(f"Rows/columns with IPR > {chi}")
    for k in "UVW":
        axes[1].plot(epochs, ipr_counts[k], label=k)
    axes[1].axhline(y=args.hidden, color="black", linestyle="dashed", label="all")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("# rows/columns")
    axes[1].legend()

    axes[2].set_title("Average Gini Coefficients")
    for k in "UVW":
        axes[2].plot(epochs, avg_ginis[k], label=k)
    axes[2].axhline(y=1.0, color="black", linestyle="dashed", label="pure freq")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("avg Gini coef")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ipr_gini.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


def setup_logging(out_dir: str):
    """Log to both stdout and out_dir/run.log."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    # Root logger writes to file + stdout
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    # Redirect print() to logger
    import builtins
    _orig_print = builtins.print
    def _log_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logging.info(msg)
    builtins.print = _log_print


def run_mlp(args, dataset, device):
    torch.manual_seed(args.seed)
    model = MLP(
        2 * args.p, args.hidden, args.p,
        depth=args.depth, activation=args.activation, dropout=args.dropout,
    )
    model.to(device)
    print(f"Model: MLP depth={args.depth}, activation={args.activation}, "
          f"{sum(p.numel() for p in model.parameters())} parameters")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.wd,
            betas=(0.9, 0.98), eps=1e-8,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd,
        )

    weight_snapshots = []

    def on_log(model, epoch):
        w = extract_weights(model)
        weight_snapshots.append({
            "norms": compute_norms(w),
            "fft": fft_iprs_and_ginis(w),
        })

    history = train(
        model, optimizer, dataset,
        epochs=args.epochs, log_every=args.log_every,
        loss_fn=args.loss, batch_size=args.batch_size,
        early_stop_patience=args.early_stop,
        device=device, on_log=on_log,
    )
    return history, weight_snapshots


def run_rfm(args, dataset):
    print(f"Model: RFM (Gaussian kernel, bandwidth={args.bandwidth}, ridge={args.ridge})")
    # RFM needs CPU float64 data
    cpu_dataset = make_dataset(
        p=args.p, data_frac=args.data_frac, noise_level=args.noise_level,
        operation=args.operation, pair_seed=args.pair_seed,
        device=torch.device("cpu"),
    )
    history = train_rfm(
        cpu_dataset, iters=args.rfm_iters, bandwidth=args.bandwidth,
        ridge=args.ridge, early_stop_patience=args.early_stop,
    )
    history["epoch"] = history.pop("iter")
    return history, []


def save_basic_plots(history: dict, out_dir: str):
    """Loss/accuracy plots (no weight analysis)."""
    os.makedirs(out_dir, exist_ok=True)
    x = np.array(history["epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].set_title("Train / Test Loss")
    axes[0].plot(x, history["train_loss"], label="train")
    axes[0].plot(x, history["test_loss"], label="test")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].set_title("Train / Test Loss (log)")
    axes[1].plot(x, history["train_loss"], label="train")
    axes[1].plot(x, history["test_loss"], label="test")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].legend()

    axes[2].set_title("Train / Test Accuracy")
    axes[2].plot(x, history["train_acc"], label="train")
    axes[2].plot(x, history["test_acc"], label="test")
    axes[2].axhline(y=1.0, color="black", linestyle="dashed")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("accuracy")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses_accuracies.png"), dpi=150)
    plt.close(fig)
    print(f"Plots saved to {out_dir}/")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Save full command, git hash, and args
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
    with open(os.path.join(args.out_dir, "command.txt"), "w") as f:
        f.write(f"git: {git_hash}\n")
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    setup_logging(args.out_dir)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    dataset = make_dataset(
        p=args.p,
        data_frac=args.data_frac,
        noise_level=args.noise_level,
        operation=args.operation,
        pair_seed=args.pair_seed,
        device=device,
    )
    print(
        f"Dataset: p={args.p}, operation={args.operation}, "
        f"train={dataset['X_train'].shape[0]}, test={dataset['X_test'].shape[0]}"
    )

    history, weight_snapshots = run_mlp(args, dataset, device) if args.model == "mlp" \
        else run_rfm(args, dataset)

    # Save scalar history
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {args.out_dir}/history.json")

    if not args.no_plots and weight_snapshots:
        save_plots(history, weight_snapshots, args, args.out_dir)
    elif not args.no_plots:
        save_basic_plots(history, args.out_dir)

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss:  {history['test_loss'][-1]:.6f}")
    print(f"Final train acc:  {history['train_acc'][-1]:.4f}")
    print(f"Final test acc:   {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
