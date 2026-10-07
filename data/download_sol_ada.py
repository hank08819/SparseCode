"""
download_sol_ada.py  — Download SOL-USD and ADA-USD 5-minute OHLCV
from Coinbase Exchange public REST API.
Matches the CSV schema of the existing BTC/ETH/XRP/LTC/DOGE files.

Usage:
    cd /Users/henry_han/LHP/BCXH2026/data
    python download_sol_ada.py
"""
import json, time, urllib.request
from datetime import datetime, timezone
import pandas as pd

GRAN   = 300   # 5-min in seconds
CHUNK  = 300   # candles per request
SLEEP  = 0.40

ASSETS = ["SOL", "ADA"]
PERIODS = {
    "P1": ("2022-11-01T00:00:00Z", "2022-11-30T23:55:00Z", "20221101"),
    "P2": ("2023-10-01T00:00:00Z", "2023-10-31T23:55:00Z", "20231001"),
    "P3": ("2024-03-01T00:00:00Z", "2024-03-31T23:55:00Z", "20240301"),
}
OUTPUT_COLS = [
    "timestamp","low","high","open","close","volume",
    "vwap","turnover","homeNotional","foreignNotional","trades",
]


def fetch_chunk(url):
    req = urllib.request.Request(
        url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  warn: {e}, retrying...")
        time.sleep(2)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())


def download(asset, start_iso, end_iso, label):
    base = f"https://api.exchange.coinbase.com/products/{asset}-USD/candles"
    su = int(datetime.fromisoformat(start_iso.replace("Z","+00:00")).timestamp())
    eu = int(datetime.fromisoformat(end_iso.replace("Z","+00:00")).timestamp())
    rows, cursor = [], su
    n_chunks = (eu - su) // GRAN // CHUNK + 1
    print(f"  {asset} {label} ({n_chunks} requests)...", end=" ", flush=True)
    i = 0
    while cursor <= eu:
        ce = min(cursor + CHUNK*GRAN - GRAN, eu)
        s = datetime.fromtimestamp(cursor, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        e = datetime.fromtimestamp(ce,     tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.extend(fetch_chunk(f"{base}?start={s}&end={e}&granularity={GRAN}"))
        cursor = ce + GRAN; i += 1
        time.sleep(SLEEP)
    print(f"got {len(rows)} raw rows", flush=True)

    df = pd.DataFrame(rows, columns=["_ts","low","high","open","close","volume"])
    df["_ts"] = df["_ts"].astype(int)
    df = df.drop_duplicates("_ts").sort_values("_ts").reset_index(drop=True)
    df = df[(df["_ts"]>=su)&(df["_ts"]<=eu)].reset_index(drop=True)
    all_ts = range(su, eu+GRAN, GRAN)
    df = pd.DataFrame({"_ts":list(all_ts)}).merge(df, on="_ts", how="left").ffill().bfill()
    df["timestamp"] = df["_ts"].apply(
        lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00"))
    df["vwap"]          = ((df["open"]+df["high"]+df["low"]+df["close"])/4).round(6)
    df["turnover"]      = (df["volume"]*df["vwap"]).round(8)
    df["homeNotional"]  = df["volume"].round(8)
    df["foreignNotional"] = df["turnover"]
    df["trades"]        = ""
    return df[OUTPUT_COLS]


def main():
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    for asset in ASSETS:
        for period, (s, e, tag) in PERIODS.items():
            fname = f"{asset}_5m_{period}_{tag}.csv"
            fpath = os.path.join(out, fname)
            df = download(asset, s, e, period)
            df.to_csv(fpath, index=False)
            c = df["close"].astype(float)
            print(f"  Saved {len(df)} bars → {fname}  "
                  f"close {c.iloc[0]:.4f}→{c.iloc[-1]:.4f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
