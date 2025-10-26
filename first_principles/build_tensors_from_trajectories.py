#!/usr/bin/env python3
"""
build_tensors_from_trajectories.py

What it does
------------
1) Finds all CSV files in ./monte_carlo_data (fake) and ./sampled_data (real).
2) Loads each normalised 30-day trajectory (columns: day, price).
3) Rejects a trajectory if ANY price is > upper_bound (default 5.0) or < lower_bound (default -5.0).
4) Aggregates all accepted 30-day trajectories into:
      X: shape (N, steps)  -- each row is the price path
      y: shape (N,)        -- 0=fake (MC), 1=real (sampled)
5) Balances classes and splits into 80% train / 10% val / 10% test
   (class balance enforced within each split; excess from larger class discarded).
6) Saves PyTorch tensors to ./tensor_data:
      X_train.pt, y_train.pt, X_val.pt, y_val.pt, X_test.pt, y_test.pt
   plus a small metadata JSON.

Usage
-----
python build_tensors_from_trajectories.py
# or with options:
python build_tensors_from_trajectories.py \
    --mc-dir monte_carlo_data --real-dir sampled_data --out-dir tensor_data \
    --steps 30 --lower -5.0 --upper 5.0 --seed 123

Assumptions
-----------
• Each input CSV has at least a 'price' column (float) and ideally a 'day' column (1..steps).
• Trajectories are already normalised to start at ~1.0.
• This script enforces exact length == steps (default 30). Others are skipped.
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


# ------------------------- Helpers -------------------------

def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


def list_csvs(folder: str) -> List[str]:
    """List .csv files in a folder (non-recursive)."""
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def load_price_vector(csv_path: str,
                      steps: int,
                      lower_bound: float,
                      upper_bound: float) -> np.ndarray | None:
    """
    Load a CSV and return a clean 1D price vector of length `steps`.
    Reject if:
      - cannot read / missing 'price' column
      - contains NaN or inf
      - length != steps
      - any price < lower_bound or > upper_bound
    Returns:
      np.ndarray of shape (steps,) or None if rejected.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    # Prefer a column specifically named 'price'; else try last numeric column.
    if "price" in df.columns:
        price = df["price"].astype(float)
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            print(f"[WARN] No numeric columns in {os.path.basename(csv_path)}; skipping.")
            return None
        price = df[numeric_cols[-1]].astype(float)

    # Clean & basic checks
    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    if len(price) != steps:
        print(f"[SKIP] {os.path.basename(csv_path)} length={len(price)} != steps={steps}")
        return None

    # Boundary filter: reject if any price out of bounds
    if (price > upper_bound).any() or (price < lower_bound).any():
        print(f"[REJECT] {os.path.basename(csv_path)} out-of-bounds price detected.")
        return None

    return price.to_numpy()


def build_features_and_labels(mc_files: List[str],
                              real_files: List[str],
                              steps: int,
                              lower_bound: float,
                              upper_bound: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all files and return (X, y).
      X: (N, steps) float32
      y: (N,) int64  where 0=fake (MC), 1=real (sampled)
    """
    X_list = []
    y_list = []

    print(f"[INFO] Loading fake (MC) trajectories from {len(mc_files)} files …")
    for fp in mc_files:
        vec = load_price_vector(fp, steps, lower_bound, upper_bound)
        if vec is None:
            continue
        X_list.append(vec)
        y_list.append(0)

    print(f"[INFO] Loading real (sampled) trajectories from {len(real_files)} files …")
    for fp in real_files:
        vec = load_price_vector(fp, steps, lower_bound, upper_bound)
        if vec is None:
            continue
        X_list.append(vec)
        y_list.append(1)

    if not X_list:
        print("[ERROR] No valid trajectories loaded.")
        sys.exit(1)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    print(f"[INFO] Total accepted: N={len(y)} (features: {X.shape})")
    return X, y


def balanced_split(X: np.ndarray,
                   y: np.ndarray,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   seed: int = 123) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                             Tuple[np.ndarray, np.ndarray],
                                             Tuple[np.ndarray, np.ndarray]]:
    """
    Make class-balanced train/val/test splits by:
      1) Separate indices for each class.
      2) Determine c = min(count0, count1).
      3) Take only c from each class, shuffle.
      4) Split each class into train/val/test by the given ratios.
      5) Combine class-wise splits; shuffle within each split.

    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    rng = np.random.default_rng(seed)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    n0, n1 = len(idx0), len(idx1)

    if n0 == 0 or n1 == 0:
        print("[ERROR] One of the classes is empty after filtering; cannot balance.")
        sys.exit(1)

    c = min(n0, n1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    idx0 = idx0[:c]
    idx1 = idx1[:c]

    # compute per-class split sizes
    n = c
    n_train = int(np.floor(train_ratio * n))
    n_val   = int(np.floor(val_ratio   * n))
    n_test  = n - n_train - n_val  # ensure all used

    # slice helper
    def split_indices(idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return idxs[:n_train], idxs[n_train:n_train+n_val], idxs[n_train+n_val:n_train+n_val+n_test]

    tr0, va0, te0 = split_indices(idx0)
    tr1, va1, te1 = split_indices(idx1)

    # combine per-split
    train_idx = np.concatenate([tr0, tr1])
    val_idx   = np.concatenate([va0, va1])
    test_idx  = np.concatenate([te0, te1])

    # shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"[INFO] Balanced counts per class kept: {c} each  (total used: {2*c})")
    print(f"[INFO] Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    # Sanity: class balance within each split
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        n0s, n1s = np.sum(yy == 0), np.sum(yy == 1)
        print(f"  - {name}: class0={n0s}, class1={n1s}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_tensors(out_dir: str,
                 train: Tuple[np.ndarray, np.ndarray],
                 val:   Tuple[np.ndarray, np.ndarray],
                 test:  Tuple[np.ndarray, np.ndarray],
                 steps: int,
                 seed:  int) -> None:
    """
    Save tensors to .pt files and a small metadata JSON.
    """
    ensure_dir(out_dir)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = train, val, test

    # Convert to torch tensors
    tXtr = torch.from_numpy(Xtr)  # float32
    tytr = torch.from_numpy(ytr)  # int64
    tXva = torch.from_numpy(Xva)
    tyva = torch.from_numpy(yva)
    tXte = torch.from_numpy(Xte)
    tyte = torch.from_numpy(yte)

    torch.save(tXtr, os.path.join(out_dir, "X_train.pt"))
    torch.save(tytr, os.path.join(out_dir, "y_train.pt"))
    torch.save(tXva, os.path.join(out_dir, "X_val.pt"))
    torch.save(tyva, os.path.join(out_dir, "y_val.pt"))
    torch.save(tXte, os.path.join(out_dir, "X_test.pt"))
    torch.save(tyte, os.path.join(out_dir, "y_test.pt"))

    meta = {
        "steps": steps,
        "feature": "normalised_price_path",
        "dtype_X": "float32",
        "dtype_y": "int64",
        "files": {
            "X_train": "X_train.pt", "y_train": "y_train.pt",
            "X_val":   "X_val.pt",   "y_val":   "y_val.pt",
            "X_test":  "X_test.pt",  "y_test":  "y_test.pt",
        },
        "sizes": {
            "train": len(ytr),
            "val":   len(yva),
            "test":  len(yte),
        },
        "class_mapping": { "0": "fake_monte_carlo", "1": "real_sampled" },
        "seed": seed,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Saved tensors and metadata to '{out_dir}'.")


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced PyTorch tensors from 30-day trajectories (fake vs real)."
    )
    parser.add_argument("--mc-dir",   default="monte_carlo_data", help="Folder with fake (Monte Carlo) CSVs.")
    parser.add_argument("--real-dir", default="sampled_data",     help="Folder with real (sampled) CSVs.")
    parser.add_argument("--out-dir",  default="tensor_data",      help="Output folder for tensors.")
    parser.add_argument("--steps", type=int, default=30, help="Expected trajectory length (default: 30).")
    parser.add_argument("--lower", type=float, default=-5.0, help="Reject if price < lower bound (default: -5).")
    parser.add_argument("--upper", type=float, default=5.0,  help="Reject if price > upper bound (default: 5).")
    parser.add_argument("--seed",  type=int,   default=123,  help="Random seed for shuffling/splits.")
    args = parser.parse_args()

    # Gather files
    mc_files   = list_csvs(args.mc_dir)
    real_files = list_csvs(args.real_dir)

    if not mc_files:
        print(f"[WARN] No CSVs found in '{args.mc_dir}'.")
    if not real_files:
        print(f"[WARN] No CSVs found in '{args.real_dir}'.")

    if not mc_files and not real_files:
        print("[ERROR] No input CSVs found.")
        sys.exit(1)

    print(f"[INFO] Files: fake={len(mc_files)}, real={len(real_files)}")
    print(f"[INFO] Bounds: ({args.lower}, {args.upper}), steps={args.steps}, seed={args.seed}")

    # Build dataset
    X, y = build_features_and_labels(mc_files, real_files, args.steps, args.lower, args.upper)

    # Balanced split
    train, val, test = balanced_split(X, y, 0.8, 0.1, 0.1, seed=args.seed)

    # Save tensors
    save_tensors(args.out_dir, train, val, test, steps=args.steps, seed=args.seed)

    print("[DONE] All set.")

if __name__ == "__main__":
    main()
