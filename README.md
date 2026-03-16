# Grokking Modular Arithmetic

Python code for studying the grokking phenomenon on modular arithmetic tasks. Supports MLP, RFM (Recursive Feature Machine), and Transformer models.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e "."
uv pip install scipy  # required for RFM
```

## Usage

```bash
python -m grokking.main [OPTIONS]
```

### Models

**MLP** (default): Fully-connected network with configurable activation and depth.
```bash
python -m grokking.main --model mlp --activation relu --depth 1 --hidden 500
```

**RFM**: Recursive Feature Machine with Gaussian kernel. Iteratively learns a metric via the Average Gradient Outer Product (AGOP).
```bash
python -m grokking.main --model rfm --rfm-iters 200 --bandwidth 2.5 --ridge 0
```

**Transformer**: Small decoder-only transformer (pre-norm, GELU).
```bash
python -m grokking.main --model transformer --loss ce --lr 1e-3 --optimizer adam
```

**Transformer-Original**: Intended to reproduce the Power et al. (2022) architecture (post-norm, ReLU, sinusoidal PE, no bias).
```bash
python -m grokking.main --model transformer-original --loss ce --lr 1e-3 --optimizer adam
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--p` | 97 | Prime modulus |
| `--operation` | addition | `addition`, `multiplication`, or `division` |
| `--data-frac` | 0.5 | Fraction of all p^2 pairs used for training |
| `--noise-level` | 0.0 | Fraction of training labels to corrupt |
| `--noise-mode` | fixed | `fixed`, `per_epoch` (re-corrupt each epoch), or `stratified` (balanced noise per batch) |
| `--activation` | relu | `relu`, `gelu`, `silu`, `tanh`, or `quadratic` |
| `--depth` | 1 | Number of hidden layers (MLP) |
| `--hidden` | 500 | Hidden layer width (MLP) |
| `--optimizer` | sgd | `sgd`, `adam`, or `adamw` |
| `--lr` | 50.0 | Learning rate |
| `--wd` | 0.0 | Weight decay |
| `--loss` | mse | `mse` or `ce` (cross-entropy) |
| `--batch-size` | 128 | Minibatch size (0 = full-batch) |
| `--epochs` | 1000 | Training epochs (MLP/Transformer) |
| `--rfm-iters` | 50 | RFM iterations |
| `--bandwidth` | 2.5 | Gaussian kernel bandwidth (RFM) |
| `--ridge` | 0.0 | Ridge regularization (RFM) |
| `--early-stop` | 0 | Stop after N consecutive log points with test_acc=1.0 (0 = off) |
| `--log-every` | 10 | Log metrics every N epochs |
| `--warmup-steps` | 10 | Linear LR warmup steps (Transformer) |
| `--seed` | 1 | Model init seed |
| `--pair-seed` | 420 | Train/test split seed |
| `--device` | auto | `cpu`, `cuda`, `cuda:N`, or `auto` |
| `--out-dir` | results | Output directory |
| `--no-plots` | false | Skip plot generation |

### Outputs

Each run saves to `--out-dir`:
- `command.txt` — git hash and full CLI command
- `args.json` — all arguments (including defaults)
- `run.log` — full training log
- `history.json` — per-epoch metrics
- `losses_accuracies.png` — loss and accuracy curves
- `weight_norms.png` — weight norm evolution (MLP only)
- `ipr_gini.png` — IPR and Gini coefficient analysis (MLP only)

## Project structure

```
grokking/
    __init__.py
    main.py                  # CLI entry point
    data.py                  # Dataset generation (modular arithmetic, label noise)
    model.py                 # MLP with configurable activation/depth
    train.py                 # Training loop (full-batch, minibatch, noise modes)
    rfm.py                   # Recursive Feature Machine (Gaussian kernel + AGOP)
    analysis.py              # FFT-based IPR, Gini coefficients, weight norms
    hooks.py                 # Forward/backward activation hooks
    transformer.py           # Pre-norm decoder-only transformer
    transformer_original.py  # Power et al. (2022) reproduction (post-norm)
run_sweep.py                 # Ablation sweep script
pyproject.toml
```

## References
We are grateful for the following reference papers and repositories:

[1] Doshi et al. "To grok or not to grok: Disentangling generalization and memorization on corrupted algorithmic datasets". arXiv:2310.13061
https://github.com/d-doshi/Grokking

[2] Mallinar et al. "Emergence in non-neural models: grokking modular arithmetic via average gradient outer product". arXiv:2407.20199
https://github.com/nmallinar/rfm-grokking

[3] Power et. al. "Grokking: Generalization Beyond Overfitting on Small Algorithic Datsets". arXiv:220102177
https://github.com/openai/grok
