#!/usr/bin/env python3
"""
monte_carlo_from_histograms.py

What’s new vs v1
----------------
• Adaptive histogram bins: k = clamp(round(sqrt(n_returns)), 5, 20).
  - Non-parametric; no normality assumption. Fewer empty bins on short series.
• Generates TWO independent Monte Carlo paths per input file:
  <base>_mc1.csv and <base>_mc2.csv.

Pipeline
--------
1) Find all CSVs in ./data (configurable).
2) Load prices, compute daily % changes (decimals).
3) Fit adaptive-bin histogram to returns.
4) Use scipy.stats.rv_histogram to sample 30 returns (uniform within bins).
5) Build 30-day random-walk from start=1.0.
6) Repeat step 4–5 twice -> two different paths.
7) Save to ./monte_carlo_data (auto-created), each with columns: day, price.
8) Print continuous progress and a final summary.

Usage
-----
python monte_carlo_from_histograms.py \
    --data-dir data --out-dir monte_carlo_data --steps 30 --seed 123

Notes
-----
• We allow short histories. Minimum needed: ≥6 prices (5 returns).
• If histogram degenerates (e.g., constant returns), the file is skipped.
• Price column detection is robust: will try common names, else last numeric col.
"""

import os
import sys
import glob
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import rv_histogram


# ------------------------- Config -------------------------

PREFERRED_PRICE_COLUMNS = [
    "close_usd", "close", "price", "adj_close", "Adj Close", "Close",
    "close_price", "last", "Last"
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------- Loading & returns -------------------------

def _guess_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try to locate a sensible 'price' column.
    1) Use preferred names (case-insensitive).
    2) Fallback to last numeric column.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for name in PREFERRED_PRICE_COLUMNS:
        if name in df.columns:
            return df[name].astype(float)
        if name.lower() in cols_lower:
            return df[cols_lower[name.lower()]].astype(float)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return df[numeric_cols[-1]].astype(float)
    return None


def load_daily_returns(csv_path: str, min_prices: int = 6) -> Optional[np.ndarray]:
    """
    Load CSV, extract price series, compute daily returns as decimals.
    Require at least min_prices points (default 6 -> ≥5 returns).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    price = _guess_price_series(df)
    if price is None:
        print(f"[WARN] No suitable price column in {os.path.basename(csv_path)}; skipping.")
        return None

    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    price = price[price > 0]

    if len(price) < min_prices:
        print(f"[WARN] Too few price points (got {len(price)} < {min_prices}) in {os.path.basename(csv_path)}; skipping.")
        return None

    returns = price.pct_change().dropna().to_numpy()
    if returns.size < 1:
        print(f"[WARN] No returns computed in {os.path.basename(csv_path)}; skipping.")
        return None

    return returns


# ------------------------- Histogram & sampling -------------------------

def adaptive_bins(n_returns: int, k_min: int = 5, k_max: int = 20) -> int:
    """
    Choose number of bins adaptively (non-parametric):
      k = clamp(round(sqrt(n)), k_min, k_max)
    """
    k = int(round(np.sqrt(max(1, n_returns))))
    return int(max(k_min, min(k_max, k)))


def fit_histogram(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit numpy histogram with adaptive number of bins.
    Returns (hist_counts, bin_edges).
    """
    k = adaptive_bins(len(returns))
    hist, edges = np.histogram(returns, bins=k, density=False)
    return hist, edges


def make_hist_sampler(hist: np.ndarray, edges: np.ndarray) -> rv_histogram:
    """
    Wrap (hist, edges) in a scipy.stats.rv_histogram sampler.
    Sampling is uniform within each bin -> satisfies “exact % in bin range”.
    """
    return rv_histogram((hist, edges))


# ------------------------- Monte Carlo -------------------------

def simulate_path(start_value: float, sampled_returns: np.ndarray) -> np.ndarray:
    """
    price_t = price_{t-1} * (1 + r_t)
    """
    growth = np.cumprod(1.0 + sampled_returns.astype(float))
    return start_value * growth


# ------------------------- Naming helpers -------------------------

def derive_base_name(csv_path: str) -> str:
    """
    For '0001_bitcoin_usd_prices_365d.csv' -> 'bitcoin'.
    Otherwise returns the stem.
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return stem


# ------------------------- Per-file workflow -------------------------

def process_one_file(csv_path: str,
                     out_dir: str,
                     steps: int,
                     rng: np.random.Generator) -> bool:
    """
    End-to-end for one input:
      - load returns
      - adaptive histogram -> rv_histogram
      - sample two *independent* length-`steps` return series
      - simulate two paths from 1.0
      - save as <base>_mc1.csv and <base>_mc2.csv
    """
    returns = load_daily_returns(csv_path)
    if returns is None:
        return False

    mean_r = float(np.mean(returns))
    std_r  = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    k_bins = adaptive_bins(len(returns))
    print(f"  {os.path.basename(csv_path)}: {returns.size} returns | mean={mean_r:.6f} std={std_r:.6f} | bins={k_bins}")

    hist, edges = fit_histogram(returns)
    if np.sum(hist) == 0 or np.all(hist == 0):
        print("  [WARN] Degenerate histogram; skipping.")
        return False

    rv = make_hist_sampler(hist, edges)

    # Two independent samples -> two paths
    sampled1 = rv.rvs(size=steps, random_state=rng)
    sampled2 = rv.rvs(size=steps, random_state=rng)

    path1 = simulate_path(1.0, sampled1)
    path2 = simulate_path(1.0, sampled2)

    # Prepare outputs (day 1..steps)
    day = np.arange(1, steps + 1, dtype=int)
    out1 = pd.DataFrame({"day": day, "price": path1})
    out2 = pd.DataFrame({"day": day, "price": path2})

    ensure_dir(out_dir)
    base = derive_base_name(csv_path)
    out_path1 = os.path.join(out_dir, f"{base}_mc1.csv")
    out_path2 = os.path.join(out_dir, f"{base}_mc2.csv")

    out1.to_csv(out_path1, index=False)
    out2.to_csv(out_path2, index=False)

    print(f"  Saved: {os.path.basename(out_path1)} (final={path1[-1]:.6f}), "
          f"{os.path.basename(out_path2)} (final={path2[-1]:.6f})")
    return True


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo from empirical histograms (adaptive bins, two paths/file).")
    parser.add_argument("--data-dir", default="data", help="Input folder with CSVs (default: ./data)")
    parser.add_argument("--out-dir", default="monte_carlo_data", help="Output folder (default: ./monte_carlo_data)")
    parser.add_argument("--steps", type=int, default=30, help="Monte Carlo horizon in days (default: 30)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility (default: 123)")
    args = parser.parse_args()

    in_dir  = args.data_dir
    out_dir = args.out_dir
    steps   = args.steps
    rng     = np.random.default_rng(args.seed)

    ensure_dir(out_dir)

    files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not files:
        print(f"[INFO] No CSV files found in '{in_dir}'.")
        sys.exit(0)

    print(f"[INFO] Found {len(files)} CSV(s). Output → '{out_dir}'. steps={steps}, seed={args.seed}\n")

    saved = skipped = 0
    for i, csv_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing {os.path.basename(csv_path)} …")
        ok = process_one_file(csv_path, out_dir, steps, rng)
        saved += int(ok)
        skipped += int(not ok)

    print(f"\n[SUMMARY] Done. Saved: {saved} | Skipped: {skipped} | Total: {len(files)}")


if __name__ == "__main__":
    main()
