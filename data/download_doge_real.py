"""
download_doge_real.py
Download real DOGE-USD 5-minute OHLCV from the Coinbase Exchange public
REST API and write files matching the BTC/ETH/XRP/LTC CSV schema.

Endpoint: GET https://api.exchange.coinbase.com/products/DOGE-USD/candles
  params: start, end (ISO-8601 UTC), granularity=300 (seconds)
  returns: [[unix_ts, low, high, open, close, volume], ...] newest first
  limit: 300 candles per request

Usage:
    python download_doge_real.py
"""

import json
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import pandas as pd

BASE = "https://api.exchange.coinbase.com/products/DOGE-USD/candles"
GRAN = 300          # 5 minutes in seconds
CHUNK = 300         # max candles per request
SLEEP = 0.35        # seconds between requests (stay under rate limit)

PERIODS = {
    "P1": ("2022-11-01T00:00:00Z", "2022-11-30T23:55:00Z", "20221101"),
    "P2": ("2023-10-01T00:00:00Z", "2023-10-31T23:55:00Z", "20231001"),
    "P3": ("2024-03-01T00:00:00Z", "2024-03-31T23:55:00Z", "20240301"),
}

OUTPUT_COLS = [
    "timestamp", "low", "high", "open", "close",
    "volume", "vwap", "turnover", "homeNotional", "foreignNotional", "trades",
]


def fetch_chunk(start_ts: int, end_ts: int) -> list:
    """Fetch one chunk [start_ts, end_ts] (unix seconds). Returns list of rows."""
    start_iso = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    end_iso = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    url = f"{BASE}?start={start_iso}&end={end_iso}&granularity={GRAN}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        print(f"  [warn] fetch failed ({exc}), retrying after 2 s …")
        time.sleep(2.0)
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())


def download_period(start_iso: str, end_iso: str, label: str) -> pd.DataFrame:
    """Download the full period in CHUNK-sized requests; return tidy DataFrame."""
    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

    start_u = int(start_dt.timestamp())
    end_u = int(end_dt.timestamp())

    all_rows: list = []
    cursor = start_u
    total_chunks = ((end_u - start_u) // GRAN // CHUNK) + 1

    print(f"  Downloading {label} ({total_chunks} requests) …", flush=True)
    chunk_idx = 0
    while cursor <= end_u:
        chunk_end = min(cursor + CHUNK * GRAN - GRAN, end_u)
        rows = fetch_chunk(cursor, chunk_end)
        all_rows.extend(rows)
        cursor = chunk_end + GRAN
        chunk_idx += 1
        if chunk_idx % 5 == 0:
            print(f"    {chunk_idx}/{total_chunks} chunks done", flush=True)
        time.sleep(SLEEP)

    if not all_rows:
        raise RuntimeError(f"No data returned for {label}")

    # Build DataFrame from [unix_ts, low, high, open, close, volume]
    df = pd.DataFrame(all_rows, columns=["_ts", "low", "high", "open", "close", "volume"])
    df["_ts"] = df["_ts"].astype(int)
    df = df.drop_duplicates("_ts").sort_values("_ts").reset_index(drop=True)

    # Filter to the requested window
    df = df[(df["_ts"] >= start_u) & (df["_ts"] <= end_u)].reset_index(drop=True)

    # Fill any missing 5-min slots with forward-fill (exchange gaps)
    all_ts = range(start_u, end_u + GRAN, GRAN)
    complete = pd.DataFrame({"_ts": list(all_ts)})
    df = complete.merge(df, on="_ts", how="left")
    df = df.ffill().bfill()

    # Timestamp string matching existing CSVs
    df["timestamp"] = df["_ts"].apply(
        lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S+00:00"
        )
    )

    # Derived columns
    df["vwap"] = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).round(6)
    df["turnover"] = (df["volume"] * df["vwap"]).round(8)
    df["homeNotional"] = df["volume"].round(8)
    df["foreignNotional"] = df["turnover"]
    df["trades"] = ""

    return df[OUTPUT_COLS]


def main():
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for period, (start_iso, end_iso, date_tag) in PERIODS.items():
        fname = f"DOGE_5m_{period}_{date_tag}.csv"
        fpath = os.path.join(out_dir, fname)

        print(f"\n{'='*55}")
        print(f"Period {period}  ({start_iso} → {end_iso})")
        df = download_period(start_iso, end_iso, period)

        df.to_csv(fpath, index=False)
        c = df["close"].astype(float)
        print(
            f"  Saved {len(df)} bars → {fname}\n"
            f"  close {c.iloc[0]:.5f} → {c.iloc[-1]:.5f},  "
            f"range [{c.min():.5f}, {c.max():.5f}],  "
            f"vol_mean={df['volume'].astype(float).mean():.0f}"
        )

    print("\nDone — real DOGE files written.")


if __name__ == "__main__":
    main()
