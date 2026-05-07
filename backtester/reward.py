#!/usr/bin/env python3
# ============================================================
#  reward.py — Clean Reward Function
#  Stop loss handled in env. No filters here.
# ============================================================

import numpy as np
from collections import deque


class RewardFunction:

    def __init__(self):
        self.pnl_history      = deque(maxlen=100)
        self.peak_value       = None
        self.prev_value       = None
        self.consecutive_loss = 0
        self.hold_steps       = 0

    def reset(self, initial_balance: float):
        self.peak_value       = initial_balance
        self.prev_value       = initial_balance
        self.consecutive_loss = 0
        self.hold_steps       = 0

    def compute(
        self,
        total_value:  float,
        action:       int,
        position:     float,
        trade_closed: bool,
        trade_pnl:    float,
    ) -> float:

        if self.prev_value is None:
            self.prev_value = total_value

        reward = 0.0

        # 1. Portfolio delta — primary signal
        delta   = (total_value - self.prev_value) / max(self.prev_value, 1.0)
        reward += delta * 1000

        # 2. Trade result
        if trade_closed:
            if trade_pnl > 0:
                reward += 5.0 + min(10.0, trade_pnl / 10.0)
                self.consecutive_loss = 0
            else:
                reward -= 2.0 + min(4.0, abs(trade_pnl) / 20.0)
                self.consecutive_loss += 1
            self.pnl_history.append(trade_pnl)
            self.hold_steps = 0

        # 3. Holding rewards
        if position > 0:
            self.hold_steps += 1
            if delta > 0:
                reward += 0.3   # reward holding a winner per candle
            elif self.hold_steps > 20:
                reward -= 0.1 * (self.hold_steps - 20)  # penalise losers

        else:
            self.hold_steps = 0

        # 4. Consecutive loss streak penalty
        if self.consecutive_loss >= 3:
            reward -= 1.0 * (self.consecutive_loss - 2)

        # 5. Drawdown protection
        if self.peak_value and total_value < self.peak_value * 0.90:
            reward -= 5.0

        # Update tracking
        self.prev_value = total_value
        if total_value > (self.peak_value or 0):
            self.peak_value = total_value

        return float(np.clip(reward, -15.0, 15.0))

    def get_stats(self) -> dict:
        return {
            "consecutive_loss": self.consecutive_loss,
            "peak_value":       self.peak_value or 0.0,
            "avg_pnl": float(np.mean(self.pnl_history)) if self.pnl_history else 0.0,
        }