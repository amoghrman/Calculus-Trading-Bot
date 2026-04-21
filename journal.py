# ============================================================
#  journal.py — Trading Journal (Fixed Version)
#  Logs every trade with correct P&L, price, features
#  Builds database for pattern extraction + Phase 3 replay
# ============================================================

import os
import json
import numpy as np
from datetime import datetime
from config import LOG_DIR

JOURNAL_FILE = os.path.join(LOG_DIR, "trade_journal.jsonl")
SUMMARY_FILE = os.path.join(LOG_DIR, "pattern_summary.json")

# Trade classification thresholds
BIG_WIN    = "BIG_WIN"      # pnl > $20
SMALL_WIN  = "SMALL_WIN"    # pnl $0 to $20
SMALL_LOSS = "SMALL_LOSS"   # pnl $0 to -$15
BIG_LOSS   = "BIG_LOSS"     # pnl < -$15


class TradeJournal:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        # Entry state — set on BUY, cleared on SELL
        self.entry_episode    = None
        self.entry_step       = None
        self.entry_price      = None
        self.entry_features   = None
        self.entry_confidence = None
        self.entry_time       = None
        self.candles_held     = 0
        self.is_in_trade      = False

    # ----------------------------------------------------------
    # Called AFTER env.step() confirms BUY executed
    # ----------------------------------------------------------
    def log_entry(
        self,
        episode:    int,
        step:       int,
        price:      float,
        features:   np.ndarray,
        confidence: dict,
    ):
        self.entry_episode    = episode
        self.entry_step       = step
        self.entry_price      = round(price, 2)
        self.entry_features   = features.tolist() if features is not None else []
        self.entry_confidence = confidence
        self.entry_time       = datetime.utcnow().isoformat()
        self.candles_held     = 0
        self.is_in_trade      = True
        print(f"[Journal] Entry logged: BUY @ ${price:.2f} | Ep {episode} Step {step}")

    # ----------------------------------------------------------
    # Called every candle while holding position
    # ----------------------------------------------------------
    def tick(self):
        if self.is_in_trade:
            self.candles_held += 1

    # ----------------------------------------------------------
    # Called AFTER env.step() confirms SELL executed
    # Receives actual exit price and actual P&L from env
    # ----------------------------------------------------------
    def log_exit(
        self,
        exit_price:      float,
        pnl:             float,   # actual P&L from env (delta)
        episode:         int,
        step:            int,
        exit_confidence: dict,
    ):
        if not self.is_in_trade or self.entry_price is None:
            return None

        pnl = round(pnl, 2)

        # Classify trade
        if pnl >= 20:
            label = BIG_WIN
        elif pnl >= 0:
            label = SMALL_WIN
        elif pnl >= -15:
            label = SMALL_LOSS
        else:
            label = BIG_LOSS

        # Priority for Phase 3 replay
        priority = self._compute_priority(label, pnl)

        # Named feature dict
        feat = self._name_features(self.entry_features)

        entry = {
            "timestamp":        self.entry_time,
            "exit_timestamp":   datetime.utcnow().isoformat(),
            "episode":          self.entry_episode,
            "entry_step":       self.entry_step,
            "exit_step":        step,
            "entry_price":      self.entry_price,
            "exit_price":       round(exit_price, 2),
            "pnl":              pnl,
            "candles_held":     self.candles_held,
            "label":            label,
            "priority":         priority,
            "entry_confidence": self.entry_confidence,
            "exit_confidence":  exit_confidence,
            "features":         feat,
        }

        # Append to journal
        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"[Journal] Exit logged: SELL @ ${exit_price:.2f} | "
            f"PnL: {'+' if pnl >= 0 else ''}${pnl:.2f} | "
            f"Held: {self.candles_held} candles | "
            f"Label: {label} | Priority: {priority}x"
        )

        # Reset state
        self.is_in_trade      = False
        self.entry_price      = None
        self.entry_features   = None
        self.candles_held     = 0

        return entry

    # ----------------------------------------------------------
    # Priority scoring for Phase 3 weighted replay
    # Big wins AND big losses get high priority
    # Small noise trades get low priority
    # ----------------------------------------------------------
    def _compute_priority(self, label: str, pnl: float) -> int:
        if label == BIG_WIN:
            # Scale priority with win size
            return min(10, max(7, int(abs(pnl) / 5)))
        elif label == SMALL_WIN:
            return min(6, max(3, int(abs(pnl) / 3)))
        elif label == SMALL_LOSS:
            return 2
        else:   # BIG_LOSS — study mistakes hard
            return min(10, max(6, int(abs(pnl) / 5)))

    # ----------------------------------------------------------
    # Name all 42 market features + 5 agent state features
    # ----------------------------------------------------------
    def _name_features(self, feat_list):
        names = [
            # Classical indicators
            "rsi", "macd", "macd_signal", "macd_hist",
            "boll_position", "boll_bandwidth", "boll_pct",
            "atr", "vwap_dev", "taker_ratio",
            # Calculus — price
            "price_velocity", "price_acceleration", "price_jerk",
            "price_vel_avg", "price_acc_avg", "price_jerk_vol",
            # Calculus — volume
            "vol_velocity", "vol_acceleration", "vol_jerk",
            "vol_vel_avg", "vol_acc_avg", "vol_jerk_vol",
            # Statistical
            "hurst", "entropy", "zscore", "price_vol_corr",
            # Order book
            "ob_spread", "ob_imbalance", "ob_bid_wall",
            "ob_ask_wall", "ob_vol_ratio", "ob_spread2",
            # Trade pressure
            "trade_pressure",
            # Time
            "time_sin_hour", "time_cos_hour",
            "time_sin_dow",  "time_cos_dow",
            # Agent state
            "position_held", "balance_ratio",
            "steps_held",    "unrealized_pct", "consec_loss",
        ]
        result = {}
        for i, val in enumerate(feat_list):
            name = names[i] if i < len(names) else f"feat_{i}"
            result[name] = round(float(val), 4)
        return result

    # ----------------------------------------------------------
    # Load today's trades
    # ----------------------------------------------------------
    def load_today(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        trades = []
        if not os.path.exists(JOURNAL_FILE):
            return trades
        with open(JOURNAL_FILE, "r") as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if e.get("timestamp", "").startswith(today):
                        trades.append(e)
                except Exception:
                    continue
        return trades

    # ----------------------------------------------------------
    # Load all trades (last 2000 for pattern analysis)
    # ----------------------------------------------------------
    def load_all(self):
        trades = []
        if not os.path.exists(JOURNAL_FILE):
            return trades
        with open(JOURNAL_FILE, "r") as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except Exception:
                    continue
        return trades[-2000:]

    # ----------------------------------------------------------
    # Load high priority trades for Phase 3 replay
    # Returns list of (features, label, priority) tuples
    # ----------------------------------------------------------
    def load_priority_trades(self, min_priority=5):
        all_trades = self.load_all()
        priority_trades = [
            t for t in all_trades
            if t.get("priority", 0) >= min_priority
        ]
        print(
            f"[Journal] Loaded {len(priority_trades)} priority trades "
            f"(priority >= {min_priority}) from {len(all_trades)} total"
        )
        return priority_trades