#!/usr/bin/env python3
# ============================================================
#  feature_engine.py — 42 Market Features (Backtest Version)
#  Identical logic to live bot, works on historical arrays
# ============================================================

import numpy as np
from config import LOOKBACK, FEATURE_WINDOW


def _safe(arr, default=0.0):
    v = float(arr) if np.isscalar(arr) else float(arr[-1])
    return default if (np.isnan(v) or np.isinf(v)) else v


class FeatureEngine:

    def compute_from_candles(self, candles: list) -> np.ndarray:
        """
        candles: list of dicts with keys:
          open, high, low, close, volume, taker_buy_base, trades
        Returns numpy array shape (42,) dtype float32
        """
        if len(candles) < LOOKBACK:
            return np.zeros(42, dtype=np.float32)

        c = candles[-LOOKBACK:]
        opens  = np.array([x["open"]  for x in c], dtype=float)
        highs  = np.array([x["high"]  for x in c], dtype=float)
        lows   = np.array([x["low"]   for x in c], dtype=float)
        closes = np.array([x["close"] for x in c], dtype=float)
        vols   = np.array([x["volume"] for x in c], dtype=float)
        takers = np.array([x["taker_buy_base"] for x in c], dtype=float)

        feats = np.zeros(42, dtype=np.float32)

        try:
            # ── Classical (10) ──────────────────────────────
            # RSI
            w = FEATURE_WINDOW
            delta = np.diff(closes)
            gain  = np.where(delta > 0, delta, 0)
            loss  = np.where(delta < 0, -delta, 0)
            avg_g = np.mean(gain[-w:]) + 1e-10
            avg_l = np.mean(loss[-w:]) + 1e-10
            rsi   = 100 - 100 / (1 + avg_g / avg_l)
            feats[0] = np.clip((rsi / 50) - 1, -3, 3)

            # MACD
            ema12 = _ema(closes, 12)
            ema26 = _ema(closes, 26)
            macd  = ema12 - ema26
            sig   = _ema_of(macd, 9)
            hist  = macd - sig
            p     = closes[-1] + 1e-10
            feats[1] = np.clip(macd[-1] / p * 100, -5, 5)
            feats[2] = np.clip(sig[-1]  / p * 100, -5, 5)
            feats[3] = np.clip(hist[-1] / p * 100, -5, 5)

            # Bollinger
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:]) + 1e-10
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            feats[4] = np.clip((closes[-1] - sma20) / (2 * std20), -3, 3)
            feats[5] = np.clip((upper - lower) / sma20 * 10, 0, 5)
            feats[6] = np.clip((closes[-1] - lower) / (upper - lower + 1e-10), -1, 2)

            # ATR
            tr = np.maximum(highs - lows,
                 np.maximum(np.abs(highs[1:] - closes[:-1]),
                            np.abs(lows[1:]  - closes[:-1])))
            feats[7] = np.clip(np.mean(tr[-w:]) / p * 100, 0, 5)

            # VWAP deviation
            vwap = np.sum(closes[-20:] * vols[-20:]) / (np.sum(vols[-20:]) + 1e-10)
            feats[8] = np.clip((closes[-1] - vwap) / vwap * 100, -5, 5)

            # Taker ratio
            taker_ratio = np.sum(takers[-10:]) / (np.sum(vols[-10:]) + 1e-10)
            feats[9] = np.clip(taker_ratio * 2 - 1, -1, 1)

            # ── Calculus Price (6) ───────────────────────────
            s = _norm(closes[-30:])
            vel = np.gradient(s)
            acc = np.gradient(vel)
            jrk = np.gradient(acc)
            feats[10] = np.clip(vel[-1] * 10, -5, 5)
            feats[11] = np.clip(acc[-1] * 10, -5, 5)
            feats[12] = np.clip(jrk[-1] * 10, -5, 5)
            feats[13] = np.clip(np.mean(vel[-5:]) * 10, -5, 5)
            feats[14] = np.clip(np.mean(acc[-5:]) * 10, -5, 5)
            feats[15] = np.clip(np.std(jrk[-10:]) * 10, 0, 5)

            # ── Calculus Volume (6) ──────────────────────────
            sv  = _norm(vols[-30:])
            vv  = np.gradient(sv)
            va  = np.gradient(vv)
            vj  = np.gradient(va)
            feats[16] = np.clip(vv[-1] * 10, -5, 5)
            feats[17] = np.clip(va[-1] * 10, -5, 5)
            feats[18] = np.clip(vj[-1] * 10, -5, 5)
            feats[19] = np.clip(np.mean(vv[-5:]) * 10, -5, 5)
            feats[20] = np.clip(np.mean(va[-5:]) * 10, -5, 5)
            feats[21] = np.clip(np.std(vj[-10:]) * 10, 0, 5)

            # ── Statistical (4) ──────────────────────────────
            # Hurst
            feats[22] = np.clip(_hurst(closes[-50:]), -1, 1)

            # Sample entropy
            feats[23] = np.clip(_sample_entropy(closes[-30:]), -3, 3)

            # Z-score
            mu  = np.mean(closes[-20:])
            std = np.std(closes[-20:]) + 1e-10
            feats[24] = np.clip((closes[-1] - mu) / std, -3, 3)

            # Price-volume correlation
            feats[25] = np.clip(np.corrcoef(closes[-20:], vols[-20:])[0, 1], -1, 1)

            # ── Order book placeholder (6) ───────────────────
            # In backtest we don't have live order book
            # Use approximate proxies from candle data
            spread_pct = (highs[-1] - lows[-1]) / closes[-1]
            feats[26] = np.clip(spread_pct * 1000, 0, 5)   # spread
            feats[27] = feats[9]                             # imbalance ≈ taker ratio
            feats[28] = np.clip(highs[-1] / closes[-1] - 1, 0, 1) * 10
            feats[29] = np.clip(1 - lows[-1] / closes[-1], 0, 1) * 10
            feats[30] = np.clip(np.log(vols[-1] / (np.mean(vols[-20:]) + 1e-10)), -3, 3)
            feats[31] = feats[26] * 2

            # ── Trade pressure (1) ───────────────────────────
            feats[32] = feats[9]   # same as taker ratio

            # ── Time features (4) ────────────────────────────
            # Estimated from open_time if available, else zeros
            feats[33] = 0.0
            feats[34] = 1.0
            feats[35] = 0.0
            feats[36] = 1.0

            # ── Agent state filled in by env (5) ─────────────
            # feats[37..41] = position, balance_ratio, steps_held,
            #                 unrealized_pct, consec_loss
            # These are set by trading_env.py after feature compute

        except Exception:
            pass

        feats = np.clip(feats, -10, 10)
        return feats.astype(np.float32)


def compute_with_time(candles, hour, dow):
    """Helper: compute features and fill in time features."""
    fe    = FeatureEngine()
    feats = fe.compute_from_candles(candles)
    import math
    feats[33] = math.sin(2 * math.pi * hour / 24)
    feats[34] = math.cos(2 * math.pi * hour / 24)
    feats[35] = math.sin(2 * math.pi * dow  / 7)
    feats[36] = math.cos(2 * math.pi * dow  / 7)
    return feats


# ── Helpers ──────────────────────────────────────────────────

def _norm(arr):
    mu  = np.mean(arr)
    std = np.std(arr) + 1e-10
    return (arr - mu) / std


def _ema(arr, span):
    alpha = 2 / (span + 1)
    ema   = np.zeros_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _ema_of(arr, span):
    return _ema(arr, span)


def _hurst(ts):
    if len(ts) < 20:
        return 0.0
    try:
        lags   = range(2, min(20, len(ts) // 2))
        tau    = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly   = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        hurst  = poly[0]
        return float(np.clip((hurst - 0.5) * 2, -1, 1))
    except Exception:
        return 0.0


def _sample_entropy(ts):
    if len(ts) < 10:
        return 0.0
    try:
        m   = 2
        r   = 0.2 * np.std(ts)
        N   = len(ts)
        def _count(m_val):
            count = 0
            for i in range(N - m_val):
                template = ts[i:i + m_val]
                for j in range(i + 1, N - m_val):
                    if np.max(np.abs(ts[j:j + m_val] - template)) < r:
                        count += 1
            return count
        Cm   = _count(m)
        Cm1  = _count(m + 1)
        if Cm == 0 or Cm1 == 0:
            return 0.0
        ent = -np.log(Cm1 / Cm)
        return float(np.clip(ent, -3, 3))
    except Exception:
        return 0.0