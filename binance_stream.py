# ============================================================
#  binance_stream.py — Live data from Binance WebSocket
#  Pulls: 5-min klines, order book depth, aggr. trades
# ============================================================

import json
import time
import threading
import websocket
import requests
import numpy as np
from collections import deque
from datetime import datetime
from config import (
    BINANCE_API_KEY, TRADING_PAIR, TIMEFRAME,
    ORDER_BOOK_DEPTH, LOOKBACK_CANDLES
)


class BinanceStream:
    """
    Maintains a live, always-updated view of:
      - Last N completed 5-min candles
      - Current live (incomplete) candle
      - Full order book snapshot (top 20 levels)
      - Buy/Sell pressure from aggr. trade stream
    """

    BASE_REST = "https://api.binance.com"

    def __init__(self):
        # Candle storage — deque auto-drops oldest when full
        self.candles = deque(maxlen=LOOKBACK_CANDLES + 1)

        # Order book
        self.bids = {}   # price -> quantity
        self.asks = {}

        # Live candle (current incomplete candle)
        self.live_candle = None

        # Aggr. trade pressure (rolling 60s window)
        self.buy_volume_60s  = deque(maxlen=12)   # 12 x 5sec buckets
        self.sell_volume_60s = deque(maxlen=12)
        self._current_buy    = 0.0
        self._current_sell   = 0.0
        self._bucket_ts      = time.time()

        # Liquidation events (futures)
        self.recent_liquidations = deque(maxlen=20)

        # State flags
        self.is_ready  = False   # True once we have full candle history
        self._ws_kline = None
        self._ws_depth = None
        self._ws_trade = None
        self._lock     = threading.Lock()

    # ----------------------------------------------------------
    # Bootstrap — fetch historical candles via REST first
    # ----------------------------------------------------------
    def bootstrap(self):
        """Fetch historical candles so agent has context immediately."""
        print(f"[Stream] Bootstrapping {LOOKBACK_CANDLES} candles for {TRADING_PAIR}...")
        url = f"{self.BASE_REST}/api/v3/klines"
        params = {
            "symbol": TRADING_PAIR,
            "interval": TIMEFRAME,
            "limit": LOOKBACK_CANDLES + 1
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        for k in raw:
            candle = self._parse_rest_candle(k)
            self.candles.append(candle)

        print(f"[Stream] Loaded {len(self.candles)} historical candles ✓")

        # Also fetch initial order book snapshot
        self._fetch_order_book_snapshot()
        print(f"[Stream] Order book loaded ✓")

    def _parse_rest_candle(self, k):
        return {
            "ts":     int(k[0]),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
            "trades": int(k[8]),
            "taker_buy_vol": float(k[9]),   # buy-side volume
        }

    def _fetch_order_book_snapshot(self):
        url = f"{self.BASE_REST}/api/v3/depth"
        params = {"symbol": TRADING_PAIR, "limit": ORDER_BOOK_DEPTH}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        with self._lock:
            self.bids = {float(p): float(q) for p, q in data["bids"]}
            self.asks = {float(p): float(q) for p, q in data["asks"]}

    # ----------------------------------------------------------
    # WebSocket — Klines (candle stream)
    # ----------------------------------------------------------
    def _on_kline_message(self, ws, message):
        data = json.loads(message)
        k = data["k"]
        candle = {
            "ts":     int(k["t"]),
            "open":   float(k["o"]),
            "high":   float(k["h"]),
            "low":    float(k["l"]),
            "close":  float(k["c"]),
            "volume": float(k["v"]),
            "trades": int(k["n"]),
            "taker_buy_vol": float(k["V"]),
        }
        with self._lock:
            self.live_candle = candle
            if k["x"]:  # candle closed
                self.candles.append(candle)
                if len(self.candles) >= LOOKBACK_CANDLES:
                    self.is_ready = True

    # ----------------------------------------------------------
    # WebSocket — Depth (order book diff stream)
    # ----------------------------------------------------------
    def _on_depth_message(self, ws, message):
        data = json.loads(message)
        with self._lock:
            for price, qty in data.get("b", []):
                p, q = float(price), float(qty)
                if q == 0:
                    self.bids.pop(p, None)
                else:
                    self.bids[p] = q
            for price, qty in data.get("a", []):
                p, q = float(price), float(qty)
                if q == 0:
                    self.asks.pop(p, None)
                else:
                    self.asks[p] = q

    # ----------------------------------------------------------
    # WebSocket — Aggr. Trades (buy/sell pressure)
    # ----------------------------------------------------------
    def _on_trade_message(self, ws, message):
        data = json.loads(message)
        qty = float(data["q"])
        is_buyer_maker = data["m"]   # True = sell trade hit the book
        now = time.time()

        # Bucket every 5 seconds
        if now - self._bucket_ts >= 5.0:
            self.buy_volume_60s.append(self._current_buy)
            self.sell_volume_60s.append(self._current_sell)
            self._current_buy  = 0.0
            self._current_sell = 0.0
            self._bucket_ts    = now

        if is_buyer_maker:
            self._current_sell += qty
        else:
            self._current_buy  += qty

    # ----------------------------------------------------------
    # Launch all WebSocket connections in background threads
    # ----------------------------------------------------------
    def start(self):
        self.bootstrap()

        base_ws = "wss://stream.binance.com:9443/ws"
        pair    = TRADING_PAIR.lower()

        def _run_ws(url, on_msg, name):
            def on_error(ws, err):
                print(f"[{name}] WS Error: {err}")
            def on_close(ws, *args):
                print(f"[{name}] WS Closed — reconnecting in 5s...")
                time.sleep(5)
                _run_ws(url, on_msg, name)

            ws = websocket.WebSocketApp(
                url,
                on_message=on_msg,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=20)

        streams = [
            (f"{base_ws}/{pair}@kline_{TIMEFRAME}", self._on_kline_message, "Kline"),
            (f"{base_ws}/{pair}@depth@100ms",       self._on_depth_message, "Depth"),
            (f"{base_ws}/{pair}@aggTrade",           self._on_trade_message, "Trade"),
        ]
        for url, handler, name in streams:
            t = threading.Thread(target=_run_ws, args=(url, handler, name), daemon=True)
            t.start()

        # Wait until we have enough candles
        print("[Stream] Waiting for live data to sync...")
        while not self.is_ready:
            time.sleep(1)
        print("[Stream] Stream is live and ready ✓")

    # ----------------------------------------------------------
    # Public accessors (thread-safe)
    # ----------------------------------------------------------
    def get_candles(self):
        """Returns list of closed candles (dicts)."""
        with self._lock:
            return list(self.candles)

    def get_order_book_features(self):
        """
        Returns computed order book signals:
          - bid_ask_spread
          - book_imbalance   (bid volume vs ask volume)
          - top_bid, top_ask
          - bid_wall, ask_wall (largest single level)
        """
        with self._lock:
            if not self.bids or not self.asks:
                return None

            top_bid = max(self.bids.keys())
            top_ask = min(self.asks.keys())
            spread  = top_ask - top_bid

            # Imbalance: positive = more buy pressure
            bid_vol = sum(self.bids.values())
            ask_vol = sum(self.asks.values())
            total   = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

            # Walls — biggest single price level
            bid_wall = max(self.bids.values()) if self.bids else 0
            ask_wall = max(self.asks.values()) if self.asks else 0

            return {
                "spread":     spread,
                "imbalance":  imbalance,
                "top_bid":    top_bid,
                "top_ask":    top_ask,
                "bid_wall":   bid_wall,
                "ask_wall":   ask_wall,
                "bid_vol":    bid_vol,
                "ask_vol":    ask_vol,
            }

    def get_trade_pressure(self):
        """Returns rolling 60s buy/sell ratio."""
        buy  = sum(self.buy_volume_60s)  + self._current_buy
        sell = sum(self.sell_volume_60s) + self._current_sell
        total = buy + sell
        if total == 0:
            return 0.5
        return buy / total   # > 0.5 = buy pressure dominant