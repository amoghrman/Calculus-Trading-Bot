#!/usr/bin/env python3
# ============================================================
#  journal.py — Trade Journal (identical to live bot)
# ============================================================

import json
import os
from datetime import datetime, timezone
from config import LOG_DIR


class TradeJournal:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path        = os.path.join(LOG_DIR, "trade_journal.jsonl")
        self._entry      = None

    # ── Entry ────────────────────────────────────────────────

    def log_entry(self, episode, step, price, features, confidence):
        self._entry = {
            "episode":          episode,
            "entry_step":       step,
            "entry_price":      float(price),
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "features":         self._feat_dict(features),
            "entry_confidence": confidence,
        }

    # ── Exit ─────────────────────────────────────────────────

    def log_exit(self, exit_price, pnl, candles_held, reason="sell"):
        if self._entry is None:
            return

        label    = self._label(pnl)
        priority = self._priority(pnl, label)

        record = {
            **self._entry,
            "exit_price":   float(exit_price),
            "pnl":          round(float(pnl), 2),
            "candles_held": int(candles_held),
            "label":        label,
            "priority":     priority,
            "exit_reason":  reason,
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._entry = None
        return record

    # ── Read ─────────────────────────────────────────────────

    def load_all(self):
        if not os.path.exists(self.path):
            return []
        trades = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
        return trades

    def count(self):
        return len(self.load_all())

    def big_wins(self):
        return [t for t in self.load_all() if t.get("pnl", 0) > 20]

    def high_priority(self):
        return [t for t in self.load_all() if t.get("priority", 0) >= 7]

    # ── Helpers ──────────────────────────────────────────────

    def _label(self, pnl):
        if   pnl >  20: return "BIG_WIN"
        elif pnl >   0: return "SMALL_WIN"
        elif pnl > -15: return "SMALL_LOSS"
        else:           return "BIG_LOSS"

    def _priority(self, pnl, label):
        if label == "BIG_WIN":
            return min(10, max(7, int(abs(pnl) / 5)))
        elif label == "BIG_LOSS":
            return min(10, max(6, int(abs(pnl) / 5)))
        elif label == "SMALL_WIN":
            return 3
        else:
            return 2

    def _feat_dict(self, features):
        names = [
            "rsi","macd","macd_sig","macd_hist",
            "boll_pos","boll_width","boll_pct","atr","vwap_dev","taker_ratio",
            "price_velocity","price_acceleration","price_jerk",
            "price_vel_avg","price_acc_avg","price_jerk_vol",
            "vol_velocity","vol_acceleration","vol_jerk",
            "vol_vel_avg","vol_acc_avg","vol_jerk_vol",
            "hurst","entropy","zscore","price_vol_corr",
            "ob_spread","ob_imbalance","ob_bid_wall","ob_ask_wall",
            "ob_vol_ratio","ob_spread2","trade_pressure",
            "time_sin_hour","time_cos_hour","time_sin_dow","time_cos_dow",
            "position_held","balance_ratio","steps_held",
            "unrealized_pct","consec_loss",
        ]
        if hasattr(features, "tolist"):
            features = features.tolist()
        return {names[i]: round(float(features[i]), 4)
                for i in range(min(len(names), len(features)))}