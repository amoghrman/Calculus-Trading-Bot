# ============================================================
#  feature_engine.py — Converts raw market data into
#  rich numerical features for the RL agent
#
#  Layers:
#    1. Classical indicators  (RSI, MACD, Bollinger)
#    2. Calculus features     (velocity, acceleration, jerk)
#    3. Order book signals    (imbalance, spread, walls)
#    4. Statistical features  (Hurst, entropy, z-score)
#    5. Trade pressure        (buy/sell flow ratio)
# ============================================================

import numpy as np
from collections import deque
from config import FEATURE_WINDOW, LOOKBACK_CANDLES


class FeatureEngine:

    def __init__(self):
        self.feature_names = []   # Populated on first call

    # ----------------------------------------------------------
    # Main entry — returns flat numpy array for the agent
    # ----------------------------------------------------------
    def compute(self, candles: list, order_book: dict, trade_pressure: float) -> np.ndarray:
        """
        Args:
            candles:        List of candle dicts (at least 60)
            order_book:     Dict from BinanceStream.get_order_book_features()
            trade_pressure: Float 0-1 from BinanceStream.get_trade_pressure()

        Returns:
            np.ndarray of shape (N_FEATURES,) normalized to [-1, 1]
        """
        if len(candles) < FEATURE_WINDOW + 5:
            return None

        closes  = np.array([c["close"]  for c in candles])
        highs   = np.array([c["high"]   for c in candles])
        lows    = np.array([c["low"]    for c in candles])
        volumes = np.array([c["volume"] for c in candles])
        taker   = np.array([c["taker_buy_vol"] for c in candles])

        features = []

        # ---- 1. Classical Indicators -------------------------
        features += self._rsi(closes, FEATURE_WINDOW)
        features += self._macd(closes)
        features += self._bollinger(closes, FEATURE_WINDOW)
        features += self._atr(highs, lows, closes, FEATURE_WINDOW)
        features += self._vwap_deviation(closes, volumes)
        features += self._taker_buy_ratio(taker, volumes)

        # ---- 2. Calculus Features ----------------------------
        features += self._calculus_features(closes)
        features += self._calculus_features(volumes, prefix="vol")

        # ---- 3. Statistical Features -------------------------
        features += self._hurst_exponent(closes)
        features += self._sample_entropy(closes)
        features += self._zscore(closes, 20)
        features += self._rolling_correlation(closes, volumes, 20)

        # ---- 4. Order Book Features -------------------------
        if order_book:
            features += self._order_book_features(order_book, closes[-1])
        else:
            features += [0.0] * 6

        # ---- 5. Trade Pressure ------------------------------
        features.append(trade_pressure * 2 - 1)   # normalize to [-1, 1]

        # ---- 6. Time Features (cyclical) --------------------
        features += self._time_features(candles[-1]["ts"])

        arr = np.array(features, dtype=np.float32)
        arr = np.clip(arr, -10, 10)   # clip before returning
        return arr

    # ----------------------------------------------------------
    # 1. Classical Indicators
    # ----------------------------------------------------------
    def _rsi(self, closes, period=14):
        delta  = np.diff(closes)
        gain   = np.where(delta > 0, delta, 0)
        loss   = np.where(delta < 0, -delta, 0)
        avg_g  = np.mean(gain[-period:])
        avg_l  = np.mean(loss[-period:])
        if avg_l == 0:
            rsi = 100.0
        else:
            rs  = avg_g / avg_l
            rsi = 100 - (100 / (1 + rs))
        # Return normalized: 0-100 → -1 to 1
        return [(rsi / 50) - 1]

    def _macd(self, closes):
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd  = ema12 - ema26
        signal = self._ema_scalar(
            [self._ema(closes[:i], 12) - self._ema(closes[:i], 26)
             for i in range(26, len(closes)+1)], 9
        )
        hist  = macd - signal
        price = closes[-1]
        return [
            np.clip(macd / price, -1, 1),
            np.clip(signal / price, -1, 1),
            np.clip(hist / price, -1, 1)
        ]

    def _bollinger(self, closes, period=20):
        sma  = np.mean(closes[-period:])
        std  = np.std(closes[-period:])
        if std == 0:
            return [0.0, 0.0, 0.0]
        upper = sma + 2 * std
        lower = sma - 2 * std
        price = closes[-1]
        # Position within bands: -1 (at lower) to +1 (at upper)
        position = (price - sma) / (2 * std)
        bandwidth = (upper - lower) / sma
        return [
            np.clip(position, -2, 2),
            np.clip(bandwidth, 0, 0.2) / 0.1 - 1,
            np.clip((price - lower) / (upper - lower) * 2 - 1, -1, 1)
        ]

    def _atr(self, highs, lows, closes, period=14):
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period-1:-1]),
                np.abs(lows[-period:]  - closes[-period-1:-1])
            )
        )
        atr = np.mean(tr)
        # Normalize by price
        return [np.clip(atr / closes[-1], 0, 0.05) / 0.025 - 1]

    def _vwap_deviation(self, closes, volumes, period=20):
        v = volumes[-period:]
        c = closes[-period:]
        vwap = np.sum(c * v) / np.sum(v) if np.sum(v) > 0 else c[-1]
        deviation = (closes[-1] - vwap) / vwap
        return [np.clip(deviation, -0.05, 0.05) / 0.025]

    def _taker_buy_ratio(self, taker, volumes, period=10):
        t = taker[-period:]
        v = volumes[-period:]
        total = np.sum(v)
        if total == 0:
            return [0.0]
        ratio = np.sum(t) / total
        return [ratio * 2 - 1]   # 0-1 → -1 to 1

    # ----------------------------------------------------------
    # 2. Calculus Features — Velocity, Acceleration, Jerk
    # ----------------------------------------------------------
    def _calculus_features(self, series, prefix="price"):
        """
        Compute 1st, 2nd, 3rd derivatives of the series.
        These tell the agent HOW FAST and in WHICH DIRECTION
        the market is moving — and whether momentum is growing or dying.
        """
        s = series[-30:]   # Use last 30 points
        if len(s) < 4:
            return [0.0] * 6

        # Normalize series first
        scale = np.std(s)
        if scale == 0:
            return [0.0] * 6
        s_norm = (s - np.mean(s)) / scale

        velocity     = np.gradient(s_norm)
        acceleration = np.gradient(velocity)
        jerk         = np.gradient(acceleration)

        return [
            np.clip(velocity[-1],     -3, 3),
            np.clip(acceleration[-1], -3, 3),
            np.clip(jerk[-1],         -3, 3),
            np.clip(np.mean(velocity[-5:]),     -3, 3),   # 5-step avg velocity
            np.clip(np.mean(acceleration[-5:]), -3, 3),   # 5-step avg accel
            np.clip(np.std(jerk[-10:]),          0, 3),   # Jerk volatility
        ]

    # ----------------------------------------------------------
    # 3. Statistical Features
    # ----------------------------------------------------------
    def _hurst_exponent(self, closes, min_period=5):
        """
        Hurst > 0.5 = trending, < 0.5 = mean-reverting, = 0.5 = random walk.
        Tells agent which strategy to apply.
        """
        try:
            s = closes[-50:]
            lags = range(2, 20)
            tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            h = poly[0]
            return [np.clip(h, 0, 1) * 2 - 1]   # 0-1 → -1 to 1
        except Exception:
            return [0.0]

    def _sample_entropy(self, closes, m=2):
        """
        Low entropy = predictable. High entropy = chaotic.
        """
        try:
            s = closes[-30:]
            s = (s - np.mean(s)) / (np.std(s) + 1e-8)
            r = 0.2
            N = len(s)
            templates_m   = [s[i:i+m]   for i in range(N - m)]
            templates_m1  = [s[i:i+m+1] for i in range(N - m)]

            def count_matches(templates, r):
                count = 0
                for i in range(len(templates)):
                    for j in range(len(templates)):
                        if i != j and np.max(np.abs(templates[i] - templates[j])) < r:
                            count += 1
                return count

            Cm  = count_matches(templates_m, r)
            Cm1 = count_matches(templates_m1, r)
            if Cm == 0:
                return [0.0]
            entropy = -np.log(Cm1 / Cm)
            return [np.clip(entropy, 0, 5) / 2.5 - 1]
        except Exception:
            return [0.0]

    def _zscore(self, closes, period=20):
        """How far is current price from its recent mean (in std devs)."""
        s = closes[-period:]
        mean = np.mean(s)
        std  = np.std(s)
        if std == 0:
            return [0.0]
        z = (closes[-1] - mean) / std
        return [np.clip(z, -3, 3)]

    def _rolling_correlation(self, closes, volumes, period=20):
        """Price-volume correlation. Trend strength signal."""
        try:
            c = closes[-period:]
            v = volumes[-period:]
            corr = np.corrcoef(c, v)[0, 1]
            return [np.clip(corr, -1, 1)]
        except Exception:
            return [0.0]

    # ----------------------------------------------------------
    # 4. Order Book Features
    # ----------------------------------------------------------
    def _order_book_features(self, ob: dict, price: float):
        spread_pct   = ob["spread"] / price
        imbalance    = ob["imbalance"]
        bid_wall_pct = ob["bid_wall"] / (ob["bid_vol"] + 1e-8)
        ask_wall_pct = ob["ask_wall"] / (ob["ask_vol"] + 1e-8)
        vol_ratio    = ob["bid_vol"] / (ob["ask_vol"] + 1e-8)

        return [
            np.clip(spread_pct / 0.001, -2, 2),      # Normalized spread
            np.clip(imbalance, -1, 1),                 # Bid/ask imbalance
            np.clip(bid_wall_pct, 0, 1) * 2 - 1,      # Bid wall dominance
            np.clip(ask_wall_pct, 0, 1) * 2 - 1,      # Ask wall dominance
            np.clip(np.log(vol_ratio + 1e-8), -3, 3), # Log vol ratio
            np.clip(spread_pct * 1000, 0, 2) - 1,     # Spread signal
        ]

    # ----------------------------------------------------------
    # 5. Time Features (cyclical encoding so agent knows time of day)
    # ----------------------------------------------------------
    def _time_features(self, ts_ms: int):
        """
        Encode hour-of-day and day-of-week cyclically.
        Markets behave differently at different times.
        """
        from datetime import datetime
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        hour = dt.hour + dt.minute / 60
        dow  = dt.weekday()
        return [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow  / 7),
            np.cos(2 * np.pi * dow  / 7),
        ]

    # ----------------------------------------------------------
    # Helper: EMA
    # ----------------------------------------------------------
    def _ema(self, series, period):
        k   = 2 / (period + 1)
        ema = series[0]
        for price in series[1:]:
            ema = price * k + ema * (1 - k)
        return ema

    def _ema_scalar(self, values, period):
        if not values:
            return 0.0
        k   = 2 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    # ----------------------------------------------------------
    # Utility: how many features does this produce?
    # ----------------------------------------------------------
    @property
    def n_features(self):
        return (
            1 +   # RSI
            3 +   # MACD
            3 +   # Bollinger
            1 +   # ATR
            1 +   # VWAP deviation
            1 +   # Taker buy ratio
            6 +   # Calculus (price)
            6 +   # Calculus (volume)
            1 +   # Hurst
            1 +   # Entropy
            1 +   # Z-score
            1 +   # Correlation
            6 +   # Order book
            1 +   # Trade pressure
            4     # Time features
        )        # = 37 features total