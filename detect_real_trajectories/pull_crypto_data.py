#!/usr/bin/env python3
"""
pull_crypto_data.py
- Works in Spyder (no asyncio).
- Downloads LAST 365 DAYS of DAILY USD prices (close) for top N coins.
- Saves one CSV per coin in ./data and a manifest at ./data/_manifest.csv.

Examples (Anaconda Prompt):
  python pull_crypto_data.py --limit 1000 --workers 8 --rpm 60
  python pull_crypto_data.py --limit 200 --overwrite --save-vol

Optional env:
  COINGECKO_API_KEY  -> sent as 'x-cg-demo-api-key'
"""

import os
import sys
import csv
import time
import math
import threading
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_BASE = "https://api.coingecko.com/api/v3"

# --------------------- Rate limiter (thread-safe) ---------------------

class RateLimiter:
    """Global RPM limiter for thread pool."""
    def __init__(self, rpm: int):
        self.rps = max(1e-9, rpm / 60.0)
        self.lock = threading.Lock()
        self.next_time = time.perf_counter()

    def acquire(self):
        with self.lock:
            now = time.perf_counter()
            # Ensure at most rps requests per second
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.perf_counter()
            # Schedule next slot
            self.next_time = now + 1.0 / self.rps

def _headers() -> Dict[str, str]:
    h = {}
    k = os.getenv("COINGECKO_API_KEY")
    if k:
        h["x-cg-demo-api-key"] = k
    return h

def http_get(url: str, params: Dict, limiter: RateLimiter, retries: int = 6, base_sleep: float = 1.0):
    for i in range(retries):
        limiter.acquire()
        r = requests.get(url, params=params, headers=_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_sleep * (2 ** i))
            continue
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise RuntimeError(f"HTTP {r.status_code} {url} -> {msg}")
    raise RuntimeError("Failed after retries (likely rate-limited).")

# --------------------- Fetchers ---------------------

def get_top_coins(limit: int, limiter: RateLimiter) -> List[Dict]:
    """Return list of {rank, id, symbol, name} for top 'limit' by market cap."""
    out = []
    per_page = 250
    pages = (limit + per_page - 1) // per_page
    for page in range(1, pages + 1):
        data = http_get(
            f"{API_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "price_change_percentage": "24h",
            },
            limiter=limiter,
        )
        if not data:
            break
        out.extend(data)
        if len(out) >= limit:
            break
    shaped = []
    for i, c in enumerate(out[:limit], start=1):
        shaped.append({"rank": i, "id": c.get("id"),
                       "symbol": (c.get("symbol") or "").lower(),
                       "name": c.get("name")})
    return shaped

def fetch_daily_prices_365(coin_id: str, limiter: RateLimiter, save_vol: bool = False) -> List[List]:
    """
    Uses /coins/{id}/market_chart?vs_currency=usd&days=365
    Returns rows: [iso_utc, close_usd] or [iso_utc, close_usd, volume_24h]
    """
    data = http_get(
        f"{API_BASE}/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": "365"},  # daily points for >90d window
        limiter=limiter,
    )
    prices = data.get("prices", [])
    vols_map = {}
    if save_vol:
        vols_map = {t: v for t, v in data.get("total_volumes", [])}
    rows = []
    for t_ms, p in prices:
        iso = datetime.utcfromtimestamp(t_ms / 1000).isoformat() + "Z"
        if save_vol:
            rows.append([iso, p, vols_map.get(t_ms, "")])
        else:
            rows.append([iso, p])
    return rows

# --------------------- I/O helpers ---------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, header: List[str], rows: List[List]):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def append_manifest(manifest_path: str, row: Dict, header: List[str]):
    exists = os.path.exists(manifest_path)
    ensure_dir(os.path.dirname(os.path.abspath(manifest_path)))
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

# --------------------- Worker ---------------------

def process_coin(coin: Dict, outdir: str, overwrite: bool, limiter: RateLimiter, manifest_path: str, save_vol: bool):
    rank, coin_id, symbol, name = coin["rank"], coin["id"], coin["symbol"], coin["name"]
    suffix = "prices_vol" if save_vol else "prices"
    out_file = os.path.join(outdir, f"{rank:04d}_{coin_id}_usd_{suffix}_365d.csv")
    header = ["timestamp_utc", "close_usd"] if not save_vol else ["timestamp_utc", "close_usd", "volume_24h"]

    if os.path.exists(out_file) and not overwrite:
        append_manifest(manifest_path, {
            "rank": rank, "id": coin_id, "symbol": symbol, "name": name,
            "filepath": out_file, "rows": "", "status": "skipped_exists", "error": ""
        }, M_HEADER)
        return ("skip", coin_id)

    try:
        rows = fetch_daily_prices_365(coin_id, limiter, save_vol=save_vol)
        write_csv(out_file, header, rows)
        append_manifest(manifest_path, {
            "rank": rank, "id": coin_id, "symbol": symbol, "name": name,
            "filepath": out_file, "rows": len(rows), "status": "ok", "error": ""
        }, M_HEADER)
        return ("ok", coin_id, len(rows))
    except Exception as e:
        append_manifest(manifest_path, {
            "rank": rank, "id": coin_id, "symbol": symbol, "name": name,
            "filepath": out_file, "rows": 0, "status": "error", "error": str(e)[:500]
        }, M_HEADER)
        return ("error", coin_id, str(e))

# --------------------- Main ---------------------

M_HEADER = ["rank","id","symbol","name","filepath","rows","status","error"]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CoinGecko top-N daily (365d) USD prices to CSVs (Spyder-safe).")
    ap.add_argument("--limit", type=int, default=1000, help="How many top coins (default 1000).")
    ap.add_argument("--outdir", default="data", help="Output folder.")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool size.")
    ap.add_argument("--rpm", type=int, default=60, help="Global requests per minute cap.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs.")
    ap.add_argument("--save-vol", action="store_true", help="Also save 24h volume per day.")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    manifest_path = os.path.join(args.outdir, "_manifest.csv")
    limiter = RateLimiter(args.rpm)

    print(f"Fetching top {args.limit} by market cap (rpm cap {args.rpm}) …")
    try:
        top = get_top_coins(args.limit, limiter)
    except Exception as e:
        print(f"Failed to fetch top coins: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(top)
    print(f"Got {total} coins. Downloading 365d daily series with {args.workers} workers …")

    ok = err = skip = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_coin, c, args.outdir, args.overwrite, limiter, manifest_path, args.save_vol)
                   for c in top]
        for i, fut in enumerate(as_completed(futures), start=1):
            status = fut.result()
            if status[0] == "ok":
                ok += 1
                coin_id, rows = status[1], status[2]
                print(f"[{i}/{total}] {coin_id:<20} -> {rows} rows")
            elif status[0] == "skip":
                skip += 1
                coin_id = status[1]
                print(f"[{i}/{total}] {coin_id:<20} -> skipped (exists)")
            else:
                err += 1
                coin_id, msg = status[1], status[2]
                print(f"[{i}/{total}] {coin_id:<20} !! ERROR: {msg}", file=sys.stderr)

    print(f"Done. ok={ok}, skipped={skip}, errors={err}. Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
