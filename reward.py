# ============================================================
#  reward.py — Reward function for the RL agent
#
#  This is the most critical file. It defines what "good trading"
#  means. Bad reward = agent learns to gamble or do nothing.
#
#  Components:
#    + Realized P&L         (profit from closed trades)
#    + Unrealized P&L       (open position value, smaller weight)
#    + Sharpe bonus         (consistency rewarded)
#    - Drawdown penalty     (punish big losses hard)
#    - Overtrading penalty  (punish too many trades)
#    - Hold cost            (slight cost to holding too long)
# ============================================================

import numpy as np
from collections import deque


class RewardFunction:

    def __init__(self):
        self.pnl_history      = deque(maxlen=100)   # Last 100 trade P&Ls
        self.step_returns     = deque(maxlen=200)   # Step-level returns
        self.peak_balance     = None
        self.consecutive_loss = 0
        self.last_action      = 0
        self.hold_steps       = 0

    def reset(self, initial_balance: float):
        self.peak_balance     = initial_balance
        self.consecutive_loss = 0
        self.hold_steps       = 0

    def compute(
        self,
        prev_balance:    float,
        curr_balance:    float,
        unrealized_pnl:  float,
        action:          int,         # 0=Hold, 1=Buy, 2=Sell
        position:        float,       # Current position size (0 or >0)
        trade_closed:    bool,
        trade_pnl:       float,       # P&L of just-closed trade
        step:            int,
    ) -> float:

        reward = 0.0

        # ---- Update peak balance ----------------------------
        self.peak_balance = max(self.peak_balance, curr_balance + unrealized_pnl)

        # ---- 1. Step Return (primary signal) ----------------
        # Normalized by previous balance so scale is consistent
        step_return = (curr_balance - prev_balance) / prev_balance
        self.step_returns.append(step_return)
        reward += step_return * 100   # Scale up for meaningful gradients

        # ---- 2. Unrealized P&L (smaller weight) -------------
        # Encourage agent to think about open positions
        unrealized_return = unrealized_pnl / (prev_balance + 1e-8)
        reward += unrealized_return * 20   # 20% weight vs realized

        # ---- 3. Trade completion bonus/penalty --------------
        if trade_closed:
            trade_return = trade_pnl / prev_balance
            self.pnl_history.append(trade_return)

            if trade_pnl > 0:
                self.consecutive_loss = 0
                # Bonus for winning trade
                reward += trade_return * 50
            else:
                self.consecutive_loss += 1
                # Extra penalty for repeated losses
                loss_multiplier = 1 + 0.2 * min(self.consecutive_loss, 5)
                reward += trade_return * 50 * loss_multiplier

        # ---- 4. Sharpe-like Consistency Bonus ---------------
        # Reward steady positive returns over volatile ones
        if len(self.step_returns) >= 20:
            returns = list(self.step_returns)[-20:]
            mean_r  = np.mean(returns)
            std_r   = np.std(returns)
            if std_r > 0:
                sharpe = mean_r / std_r
                reward += np.clip(sharpe, -1, 1) * 0.5

        # ---- 5. Drawdown Penalty ----------------------------
        current_total = curr_balance + unrealized_pnl
        drawdown = (self.peak_balance - current_total) / self.peak_balance
        if drawdown > 0.05:    # Start penalizing after 5% drawdown
            reward -= drawdown * 200   # Heavy punishment

        # ---- 6. Overtrading Penalty -------------------------
        if action != 0 and action == self.last_action:
            # Penalize same action twice in a row (flip-flopping)
            reward -= 0.2

        # ---- 7. Hold Cost -----------------------------------
        # Slight penalty for holding position too long without P&L
        if position > 0:
            self.hold_steps += 1
            if self.hold_steps > 20:   # 20 x 5min = 1h40m
                reward -= 0.05 * (self.hold_steps - 20)
        else:
            self.hold_steps = 0

        # ---- 8. Idle Penalty --------------------------------
        # Punish agent for doing nothing in trending markets
        # (only mild — we don't want it trading for the sake of it)
        if action == 0 and position == 0:
            reward -= 0.01   # Very small cost to being flat

        self.last_action = action
        return float(np.clip(reward, -10, 10))

    # ----------------------------------------------------------
    # Stats for monitoring
    # ----------------------------------------------------------
    def get_stats(self):
        if not self.pnl_history:
            return {}
        pnls = list(self.pnl_history)
        wins = [p for p in pnls if p > 0]
        return {
            "win_rate":    len(wins) / len(pnls),
            "avg_win":     np.mean(wins) if wins else 0,
            "avg_loss":    np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0,
            "profit_factor": abs(sum(wins) / (sum(p for p in pnls if p < 0) + 1e-8)),
            "consecutive_losses": self.consecutive_loss,
        }