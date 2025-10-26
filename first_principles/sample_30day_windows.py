#!/usr/bin/env python3
"""
sample_30day_windows.py

What it does
------------
1) Finds all CSV files in ./data (configurable).
2) For each file, randomly selects TWO independent 30-day windows of prices.
3) Normalises each 30-day window so day 1 = 1.0 while preserving % changes.
4) Saves both windows to ./sampled_data as <base>_sample1.csv and <base>_sample2.csv
   with columns: day, price (day spans 1..30).

Usage
-----
python sample_30day_windows.py
# or with options:
python sample_30day_windows.py --data-dir data --out-dir sampled_data --steps 30 --seed 123

Notes
-----
• The script tries to detect a sensible price column:
    preferred names: close_usd, close, price, adj_close, Close, etc.
  If not found, it falls back to the last numeric column.
• If a timestamp column ('timestamp_utc' or 'date') exists, it will sort ascending.
• Requires at least `steps` valid price points; otherwise the file is skipped.
"""

import os
import sys
import glob
import argparse
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------- Configuration -------------------------

PREFERRED_PRICE_COLUMNS = [
    "close_usd", "close", "price", "adj_close", "Adj Close", "Close",
    "close_price", "last", "Last"
]
POSSIBLE_TIME_COLUMNS = ["timestamp_utc", "date", "Date", "timestamp", "Timestamp"]


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ------------------------- CSV loading helpers -------------------------

def _maybe_sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a recognizable time column exists, parse it and sort ascending to ensure
    chronological order before slicing 30-day windows.
    """
    for col in POSSIBLE_TIME_COLUMNS:
        if col in df.columns:
            try:
                df = df.copy()
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                if df[col].notna().any():
                    df = df.sort_values(col).reset_index(drop=True)
            except Exception:
                # If parsing fails, just leave as-is
                pass
            break
    return df


def _guess_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Try to locate a sensible 'price' column.
      1) Use preferred names (case-insensitive).
      2) Fallback to last numeric column.
    Returns a float Series or None if nothing appropriate is found.
    """
    # First, try preferred names with case handling
    lower_map = {c.lower(): c for c in df.columns}
    for name in PREFERRED_PRICE_COLUMNS:
        if name in df.columns:
            return df[name].astype(float)
        if name.lower() in lower_map:
            return df[lower_map[name.lower()]].astype(float)

    # Fallback: last numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return df[numeric_cols[-1]].astype(float)
    return None


def load_price_series(csv_path: str, steps: int) -> Optional[pd.Series]:
    """
    Load a CSV, ensure reasonable ordering, pick a price column,
    and return a clean Series (no NaNs/inf, strictly positive).
    Requires at least `steps` usable data points; else returns None.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}")
        return None

    df = _maybe_sort_by_time(df)

    price = _guess_price_series(df)
    if price is None:
        print(f"[WARN] No suitable price column in {os.path.basename(csv_path)}; skipping.")
        return None

    # Clean: drop NaNs/inf and non-positive values (avoid division issues in normalisation)
    price = price.replace([np.inf, -np.inf], np.nan).dropna()
    price = price[price > 0]

    if len(price) < steps:
        print(f"[WARN] Not enough prices (need ≥{steps}, got {len(price)}) in {os.path.basename(csv_path)}; skipping.")
        return None

    # Reset index so slicing by position is straightforward
    return price.reset_index(drop=True)


# ------------------------- Sampling windows -------------------------

def choose_window(price: pd.Series, steps: int, rng: np.random.Generator) -> pd.Series:
    """
    Choose a random contiguous window of length `steps` from the 1D price Series.
    """
    max_start = len(price) - steps
    start = int(rng.integers(0, max_start + 1))
    end = start + steps
    return price.iloc[start:end]


def normalise_to_one(window: pd.Series) -> pd.Series:
    """
    Scale the window so the first value becomes 1.0 and % changes are preserved:
        norm_i = price_i / price_0
    """
    first = float(window.iloc[0])
    return window.astype(float) / first


# ------------------------- Naming helpers -------------------------

def derive_base_name(csv_path: str) -> str:
    """
    For '0001_bitcoin_usd_prices_365d.csv' -> 'bitcoin'.
    Otherwise returns the stem (filename without extension).
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return stem


# ------------------------- Per-file workflow -------------------------

def process_file(csv_path: str,
                 out_dir: str,
                 steps: int,
                 rng: np.random.Generator) -> bool:
    """
    For a single input CSV:
      - load price series
      - sample TWO independent 30-day windows
      - normalise each to start at 1.0
      - save as <base>_sample1.csv and <base>_sample2.csv with columns day,price
    """
    price = load_price_series(csv_path, steps)
    if price is None:
        return False

    # Sample two independent windows
    w1 = choose_window(price, steps, rng)
    w2 = choose_window(price, steps, rng)

    n1 = normalise_to_one(w1)
    n2 = normalise_to_one(w2)

    day = np.arange(1, steps + 1, dtype=int)
    df1 = pd.DataFrame({"day": day, "price": n1.values})
    df2 = pd.DataFrame({"day": day, "price": n2.values})

    ensure_dir(out_dir)
    base = derive_base_name(csv_path)
    out1 = os.path.join(out_dir, f"{base}_sample1.csv")
    out2 = os.path.join(out_dir, f"{base}_sample2.csv")

    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)

    print(f"  Saved {os.path.basename(out1)} and {os.path.basename(out2)}")
    return True


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample two 30-day normalised windows per CSV (Spyder-friendly)."
    )
    parser.add_argument("--data-dir", default="data", help="Input folder with CSVs (default: ./data)")
    parser.add_argument("--out-dir", default="sampled_data", help="Output folder (default: ./sampled_data)")
    parser.add_argument("--steps", type=int, default=30, help="Window length in days (default: 30)")
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
        ok = process_file(csv_path, out_dir, steps, rng)
        saved += int(ok)
        skipped += int(not ok)

    print(f"\n[SUMMARY] Done. Saved: {saved} | Skipped: {skipped} | Total: {len(files)}")


if __name__ == "__main__":
    main()
