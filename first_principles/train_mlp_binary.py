#!/usr/bin/env python3
"""
train_mlp_binary.py

Binary classifier (PyTorch MLP) for 30-day trajectories.
- Loads tensors from ./tensor_data (or a custom directory).
- Configurable MLP layers and hyperparameters.
- Tracks and prints Accuracy, Sensitivity (Recall for class=1), Specificity (Recall for class=0).
- Evaluates train/val/test every epoch; saves best model (.pth) by validation loss.
- Plots Loss/Accuracy/Sensitivity/Specificity vs epochs (train/val/test curves).

Usage (examples)
---------------
python train_mlp_binary.py
python train_mlp_binary.py --data-dir tensor_data --layers 128,64 --epochs 50 --batch-size 128 --lr 1e-3 --dropout 0.2
python train_mlp_binary.py --activation gelu --weight-decay 1e-4 --standardize 1

Outputs
-------
- {out_dir}/mlp_best.pth        (best-by-val-loss state_dict)
- {out_dir}/metrics.npy         (per-epoch metrics dict)
- {out_dir}/loss_curve.png, acc_curve.png, sens_curve.png, spec_curve.png
- Console logs with per-epoch metrics

Notes
-----
- Positive class = 1 (REAL sampled), Negative class = 0 (FAKE Monte Carlo).
- Sensitivity = TPR = TP / (TP + FN)
- Specificity = TNR = TN / (TN + FP)
"""

import os
import json
import argparse
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# --------------- Utilities ---------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_tensors(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor]:
    """Load X_train.pt, y_train.pt, X_val.pt, y_val.pt, X_test.pt, y_test.pt."""
    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    y_train = torch.load(os.path.join(data_dir, "y_train.pt"))
    X_val   = torch.load(os.path.join(data_dir, "X_val.pt"))
    y_val   = torch.load(os.path.join(data_dir, "y_val.pt"))
    X_test  = torch.load(os.path.join(data_dir, "X_test.pt"))
    y_test  = torch.load(os.path.join(data_dir, "y_test.pt"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def standardize(train: torch.Tensor, *others: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Z-score by training statistics: x' = (x - mean) / std (with epsilon).
    Applies to train and all others using the same mean/std.
    """
    eps = 1e-8
    mean = train.mean(dim=0, keepdim=True)
    std  = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    out = [(train - mean) / std]
    for t in others:
        out.append((t - mean) / std)
    return tuple(out)


def make_dataloaders(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xva: torch.Tensor, yva: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    batch_size: int, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(Xtr, ytr)
    val_ds   = TensorDataset(Xva, yva)
    test_ds  = TensorDataset(Xte, yte)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl


# --------------- Model ---------------

class MLP(nn.Module):
    """
    Flexible MLP for binary classification (outputs 1 logit).
    Example: input_dim=30, layers=[128,64], dropout=0.1, activation='relu'
    """
    def __init__(self, input_dim: int, layers: List[int], dropout: float, activation: str, batchnorm: bool):
        super().__init__()
        act = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu":  nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }.get(activation.lower(), nn.ReLU)

        mods = []
        prev = input_dim
        for h in layers:
            mods.append(nn.Linear(prev, h))
            if batchnorm:
                mods.append(nn.BatchNorm1d(h))
            mods.append(act())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            prev = h
        # final 1-logit head for BCEWithLogitsLoss
        mods.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape [B]


# --------------- Metrics ---------------

@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    logits -> probs via sigmoid; threshold to get preds.
    Returns accuracy, sensitivity (TPR), specificity (TNR).
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()

    y = y.long()
    TP = ((preds == 1) & (y == 1)).sum().item()
    TN = ((preds == 0) & (y == 0)).sum().item()
    FP = ((preds == 1) & (y == 0)).sum().item()
    FN = ((preds == 0) & (y == 1)).sum().item()

    acc  = (TP + TN) / max(1, (TP + TN + FP + FN))
    sens = TP / max(1, (TP + FN))  # recall for positive class
    spec = TN / max(1, (TN + FP))  # recall for negative class

    return {"acc": acc, "sens": sens, "spec": spec}


def eval_split(model: nn.Module, dl: DataLoader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            all_logits.append(logits.cpu())
            all_y.append(yb.cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    metrics = compute_metrics_from_logits(logits_cat, y_cat)
    metrics["loss"] = total_loss / len(y_cat)
    return metrics


# --------------- Training loop ---------------

def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int = 50,
    out_dir: str = "mlp_runs"
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [],  "val_acc": [],  "test_acc": [],
        "train_sens": [], "val_sens": [], "test_sens": [],
        "train_spec": [], "val_spec": [], "test_spec": [],
    }

    best_val = float("inf")
    best_path = os.path.join(out_dir, "mlp_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for i, (xb, yb) in enumerate(train_dl, start=1):
            xb = xb.to(device)
            yb = yb.to(device).float()

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            seen += xb.size(0)

            # Regular console updates during training
            if i % print_every == 0:
                print(f"Epoch {epoch:03d} | Batch {i:04d} | Train Loss (avg): {running/seen:.6f}")

        # Evaluate full splits each epoch
        train_metrics = eval_split(model, train_dl, device, loss_fn)
        val_metrics   = eval_split(model, val_dl,   device, loss_fn)
        test_metrics  = eval_split(model, test_dl,  device, loss_fn)

        # Log/print epoch summary
        print(
            f"[Epoch {epoch:03d}] "
            f"train: loss={train_metrics['loss']:.6f} acc={train_metrics['acc']:.4f} sens={train_metrics['sens']:.4f} spec={train_metrics['spec']:.4f} | "
            f"val: loss={val_metrics['loss']:.6f} acc={val_metrics['acc']:.4f} sens={val_metrics['sens']:.4f} spec={val_metrics['spec']:.4f} | "
            f"test: loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.4f} sens={test_metrics['sens']:.4f} spec={test_metrics['spec']:.4f}"
        )

        # Save best by validation loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(),
                        "best_val_loss": best_val}, best_path)
            print(f"  -> Saved new best model to {best_path}")

        # Append to history
        for split, metr in zip(
            ["train", "val", "test"],
            [train_metrics, val_metrics, test_metrics]
        ):
            history[f"{split}_loss"].append(metr["loss"])
            history[f"{split}_acc"].append(metr["acc"])
            history[f"{split}_sens"].append(metr["sens"])
            history[f"{split}_spec"].append(metr["spec"])

    # Save metrics to numpy (for quick reload)
    np.save(os.path.join(out_dir, "metrics.npy"), history, allow_pickle=True)
    return history


# --------------- Plotting ---------------

def plot_curves(history: Dict[str, List[float]], out_dir: str) -> None:
    """
    Make one plot per metric: Loss, Accuracy, Sensitivity, Specificity.
    Each plot shows train/val/test curves with legend.
    """
    ensure_dir(out_dir)

    def _plot(metric_key: str, title: str, fname: str):
        plt.figure()
        plt.plot(history[f"train_{metric_key}"], label="train")
        plt.plot(history[f"val_{metric_key}"],   label="val")
        plt.plot(history[f"test_{metric_key}"],  label="test")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    _plot("loss", "Loss", "loss_curve.png")
    _plot("acc",  "Accuracy", "acc_curve.png")
    _plot("sens", "Sensitivity (TPR)", "sens_curve.png")
    _plot("spec", "Specificity (TNR)", "spec_curve.png")


# --------------- Main ---------------

def build_model(input_dim: int, layers_str: str, dropout: float, activation: str, batchnorm: int) -> nn.Module:
    layers = []
    if layers_str.strip():
        layers = [int(x) for x in layers_str.split(",") if x.strip()]
    return MLP(
        input_dim=input_dim,
        layers=layers if layers else [128, 64],  # sensible default
        dropout=dropout,
        activation=activation,
        batchnorm=bool(batchnorm),
    )


def main():
    parser = argparse.ArgumentParser(description="Train/validate/test a configurable MLP on binary trajectories.")
    parser.add_argument("--data-dir",   default="tensor_data", help="Folder with X_*.pt, y_*.pt")
    parser.add_argument("--out-dir",    default="mlp_runs", help="Output folder for model & plots")
    parser.add_argument("--layers",     default="128,64", help="Comma-separated hidden sizes, e.g. '256,128,64'")
    parser.add_argument("--dropout",    type=float, default=0.10, help="Dropout probability (each hidden layer)")
    parser.add_argument("--activation", default="relu", choices=["relu","gelu","tanh","elu","leaky_relu"])
    parser.add_argument("--batchnorm",  type=int, default=1, help="1 to enable BatchNorm, 0 to disable")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--standardize",  type=int, default=1, help="Z-score features using train stats (1=yes, 0=no)")
    parser.add_argument("--seed",       type=int, default=123)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=50, help="Print training loss every N batches")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load tensors
    Xtr, ytr, Xva, yva, Xte, yte = load_tensors(args.data_dir)

    # Optional standardization (recommended)
    if args.standardize:
        Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    input_dim = Xtr.shape[1]

    # Build dataloaders
    train_dl, val_dl, test_dl = make_dataloaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=args.batch_size)

    # Build model
    model = build_model(input_dim, args.layers, args.dropout, args.activation, args.batchnorm)
    device = torch.device(args.device)
    model.to(device)

    # Save run config
    with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Train
    print(f"Starting training on device={device} | input_dim={input_dim} | layers={args.layers}")
    history = train_model(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        print_every=args.print_every,
        out_dir=args.out_dir,
    )

    # Final test metrics using best checkpoint
    best_ckpt = torch.load(os.path.join(args.out_dir, "mlp_best.pth"), map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    final_test = eval_split(model, test_dl, device, nn.BCEWithLogitsLoss())
    print("\n=== FINAL TEST METRICS (best-by-val-loss) ===")
    print(f"Test Loss: {final_test['loss']:.6f}")
    print(f"Accuracy : {final_test['acc']:.4f}")
    print(f"Sensitivity (TPR): {final_test['sens']:.4f}")
    print(f"Specificity (TNR): {final_test['spec']:.4f}")

    # Plots
    plot_curves(history, args.out_dir)
    print(f"\nSaved model/plots to '{args.out_dir}'. Done.")


if __name__ == "__main__":
    main()
