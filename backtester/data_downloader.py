"""
data_downloader.py
Downloads full BTCUSDT 1-minute historical data from Binance
Starts from September 2019 to present
Saves to data/ folder as CSV files by month
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("/root/btc-backtest/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
LIMIT    = 1000  # max per request

# Start from Sept 2019
START_TS = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)


def ts_to_str(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def download_chunk(start_ms, end_ms=None):
    params = {
        "symbol":    SYMBOL,
        "interval":  INTERVAL,
        "startTime": start_ms,
        "limit":     LIMIT,
    }
    if end_ms:
        params["endTime"] = end_ms

    for attempt in range(5):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  Retry {attempt+1}/5: {e}")
            time.sleep(2 ** attempt)
    return []


def download_all():
    print("="*60)
    print("  BTC BACKTEST DATA DOWNLOADER")
    print(f"  Symbol: {SYMBOL} | Interval: {INTERVAL}")
    print(f"  From: {ts_to_str(START_TS)}")
    print(f"  To:   Now")
    print("="*60)

    now_ms      = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_ms  = START_TS
    all_candles = []
    total       = 0
    current_month = None

    output_file = None
    writer      = None

    while current_ms < now_ms:
        chunk = download_chunk(current_ms)
        if not chunk:
            print("Empty chunk — stopping")
            break

        rows = []
        for c in chunk:
            rows.append({
                "ts":         int(c[0]),
                "open":       float(c[1]),
                "high":       float(c[2]),
                "low":        float(c[3]),
                "close":      float(c[4]),
                "volume":     float(c[5]),
                "trades":     int(c[8]),
                "taker_buy":  float(c[9]),
            })

        df        = pd.DataFrame(rows)
        last_ts   = chunk[-1][0]
        last_str  = ts_to_str(last_ts)
        month_key = last_str[:7]  # "2024-03"

        # Save by month
        if month_key != current_month:
            if output_file:
                output_file.close()
                print(f"  Saved month: {current_month}")
            current_month = month_key
            path = DATA_DIR / f"BTCUSDT_1m_{month_key}.csv"
            output_file = open(path, "a")
            # Write header only if file is new
            if path.stat().st_size == 0:
                output_file.write("ts,open,high,low,close,volume,trades,taker_buy\n")

        for row in rows:
            output_file.write(
                f"{row['ts']},{row['open']},{row['high']},{row['low']},"
                f"{row['close']},{row['volume']},{row['trades']},{row['taker_buy']}\n"
            )

        total      += len(rows)
        current_ms  = last_ts + 60_000  # next minute

        print(f"  Downloaded to {last_str} | Total: {total:,} candles", end="\r")
        time.sleep(0.1)  # be gentle with API

    if output_file:
        output_file.close()

    print(f"\n\nDone! {total:,} candles saved to {DATA_DIR}")
    print(f"Files: {len(list(DATA_DIR.glob('*.csv')))} months")


def get_data_info():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        return None
    total = 0
    for f in files:
        with open(f) as fp:
            total += sum(1 for _ in fp) - 1  # minus header
    first = files[0].stem.split("_")[-1]
    last  = files[-1].stem.split("_")[-1]
    return {"files": len(files), "candles": total, "from": first, "to": last}


if __name__ == "__main__":
    info = get_data_info()
    if info:
        print(f"Existing data: {info['candles']:,} candles | {info['from']} → {info['to']}")
        ans = input("Continue downloading (update)? [y/n]: ")
        if ans.lower() != "y":
            exit()

        # Find last timestamp and resume
        last_file = sorted(DATA_DIR.glob("*.csv"))[-1]
        df = pd.read_csv(last_file)
        if not df.empty:
            last_ts = int(df["ts"].iloc[-1])
            print(f"Resuming from {ts_to_str(last_ts)}")
            START_TS_OVERRIDE = last_ts + 60_000

            # Monkey-patch
            import data_downloader as self_module
            self_module.START_TS = START_TS_OVERRIDE

    download_all()