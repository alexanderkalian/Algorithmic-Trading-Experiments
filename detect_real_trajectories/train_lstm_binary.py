#!/usr/bin/env python3
"""
train_lstm_binary.py

Binary classifier (PyTorch LSTM -> FC) for 30-day trajectories.
- Loads tensors from ./tensor_data (or custom --data-dir).
- Matches the MLP script outputs and metrics exactly, but writes to lstm_runs by default.
- Tracks Accuracy, Sensitivity (TPR), Specificity (TNR) + Loss across train/val/test each epoch.
- Saves best model (by val loss) as lstm_best.pth.
- Plots Loss/Accuracy/Sensitivity/Specificity curves (train/val/test).

Outputs
-------
- {out_dir}/lstm_best.pth
- {out_dir}/metrics.npy
- {out_dir}/loss_curve.png, acc_curve.png, sens_curve.png, spec_curve.png
- {out_dir}/hparams.json
"""

import os
import json
import argparse
from typing import Tuple, Dict, List

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
    Z-score by training statistics: x' = (x - mean) / std (eps).
    If input is 3D (N,L,F), standardizes featurewise; if 2D (N,D), standardizes per feature.
    """
    eps = 1e-8
    if train.ndim == 3:
        N, L, F = train.shape
        flat_tr = train.reshape(N*L, F)
        mean = flat_tr.mean(dim=0, keepdim=True)
        std  = flat_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        out = [(train - mean.view(1,1,F)) / std.view(1,1,F)]
        for t in others:
            out.append((t - mean.view(1,1,F)) / std.view(1,1,F))
        return tuple(out)
    else:
        mean = train.mean(dim=0, keepdim=True)
        std  = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        out = [(train - mean) / std]
        for t in others:
            out.append((t - mean) / std)
        return tuple(out)


def make_sequence_loaders(
    Xtr: torch.Tensor, ytr: torch.Tensor,
    Xva: torch.Tensor, yva: torch.Tensor,
    Xte: torch.Tensor, yte: torch.Tensor,
    batch_size: int, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders for sequence tensors: X: (N,L,F), y: (N,) or (N,1).
    """
    def _y1d(y):
        y = y.float()
        if y.ndim == 2 and y.size(1) == 1:
            y = y[:, 0]
        return y
    ytr = _y1d(ytr); yva = _y1d(yva); yte = _y1d(yte)

    train_ds = TensorDataset(Xtr.float(), ytr)
    val_ds   = TensorDataset(Xva.float(), yva)
    test_ds  = TensorDataset(Xte.float(), yte)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl


# --------------- Model ---------------

class LSTM2FC(nn.Module):
    """
    Exactly 2-layer LSTM (no bidirectional) -> MLP head -> 1 logit.
    Uses LeakyReLU activations in the head.
    """
    def __init__(self, input_dim: int, hidden: int = 96, dropout: float = 0.0, negative_slope: float = 0.01):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, _ = self.lstm(x)         # (B, L, H)
        last = out[:, -1, :]          # (B, H)
        return self.fc(last).squeeze(-1)  # (B,)


# --------------- Metrics ---------------

@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    y = y.long()

    TP = ((preds == 1) & (y == 1)).sum().item()
    TN = ((preds == 0) & (y == 0)).sum().item()
    FP = ((preds == 1) & (y == 0)).sum().item()
    FN = ((preds == 0) & (y == 1)).sum().item()

    acc  = (TP + TN) / max(1, (TP + TN + FP + FN))
    sens = TP / max(1, (TP + FN))  # recall+
    spec = TN / max(1, (TN + FP))  # recall-
    return {"acc": acc, "sens": sens, "spec": spec}


def eval_split(model: nn.Module, dl: DataLoader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_y = [], []
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
    print_every: int,
    out_dir: str,
    grad_clip_norm: float = 1.0  # <<< gradient clipping (global norm)
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
    best_path = os.path.join(out_dir, "lstm_best.pth")

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

            # ---- Gradient clipping (global norm) ----
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            running += loss.item() * xb.size(0)
            seen += xb.size(0)

            if i % print_every == 0:
                print(f"Epoch {epoch:03d} | Batch {i:04d} | Train Loss (avg): {running/seen:.6f}")

        # Evaluate all splits each epoch
        train_metrics = eval_split(model, train_dl, device, loss_fn)
        val_metrics   = eval_split(model, val_dl,   device, loss_fn)
        test_metrics  = eval_split(model, test_dl,  device, loss_fn)

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

        # Log history
        for split, metr in zip(["train","val","test"], [train_metrics, val_metrics, test_metrics]):
            history[f"{split}_loss"].append(metr["loss"])
            history[f"{split}_acc"].append(metr["acc"])
            history[f"{split}_sens"].append(metr["sens"])
            history[f"{split}_spec"].append(metr["spec"])

    # Save metrics
    np.save(os.path.join(out_dir, "metrics.npy"), history, allow_pickle=True)
    return history


# --------------- Plotting ---------------

def plot_curves(history: Dict[str, List[float]], out_dir: str) -> None:
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


# --------------- Reshaping helper ---------------

def to_sequence(X: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Ensure X is (N, L, F). If it's (N, D), reshape to (N, seq_len, D//seq_len).
    """
    if X.ndim == 3:
        return X
    if X.ndim == 2:
        N, D = X.shape
        assert D % seq_len == 0, f"Input dim {D} not divisible by seq_len={seq_len}"
        F = D // seq_len
        return X.view(N, seq_len, F)
    raise ValueError("X must be 2D or 3D tensor")


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="Train/validate/test a 2-layer LSTM on binary trajectories.")
    parser.add_argument("--data-dir",     default="tensor_data", help="Folder with X_*.pt, y_*.pt")
    parser.add_argument("--out-dir",      default="lstm_runs",   help="Output folder for model & plots")
    parser.add_argument("--epochs",       type=int, default=100)
    parser.add_argument("--batch-size",   type=int, default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len",      type=int, default=30, help="Used if X are flattened (N, L*F)")
    parser.add_argument("--hidden",       type=int, default=96)
    parser.add_argument("--dropout",      type=float, default=0.0)
    parser.add_argument("--standardize",  type=int, default=1, help="Z-score features using train stats (1=yes, 0=no)")
    parser.add_argument("--seed",         type=int, default=123)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every",  type=int, default=50)
    # optional: expose clip + leaky slope if you want to tweak via CLI later
    parser.add_argument("--grad-clip",    type=float, default=1.0, help="Global norm gradient clipping")
    parser.add_argument("--leaky-slope",  type=float, default=0.01, help="LeakyReLU negative slope in FC head")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load tensors
    Xtr, ytr, Xva, yva, Xte, yte = load_tensors(args.data_dir)

    # Reshape to sequences if needed
    Xtr = to_sequence(Xtr, args.seq_len)
    Xva = to_sequence(Xva, args.seq_len)
    Xte = to_sequence(Xte, args.seq_len)

    # Optional standardization
    if args.standardize:
        Xtr, Xva, Xte = standardize(Xtr, Xva, Xte)

    # Dataloaders
    train_dl, val_dl, test_dl = make_sequence_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=args.batch_size)

    # Model
    input_dim = Xtr.shape[2]  # F
    model = LSTM2FC(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout, negative_slope=args.leaky_slope)
    device = torch.device(args.device)
    model.to(device)

    # Save run config
    with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Train
    print(f"Starting LSTM training on device={device} | input_dim(F)={input_dim} | seq_len={args.seq_len} | hidden={args.hidden}")
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
        grad_clip_norm=args.grad_clip,  # pass clip setting
    )

    # Final test metrics using best checkpoint
    best_ckpt = torch.load(os.path.join(args.out_dir, "lstm_best.pth"), map_location=device)
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
