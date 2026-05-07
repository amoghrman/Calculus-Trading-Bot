#!/usr/bin/env python3
# ============================================================
#  trading_env.py — Backtest Environment
#  Gymnasium env fed from historical parquet data
#  ONLY rule: stop loss at -1.5%
#  Everything else: let the agent learn
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime, timezone

from config import (
    PAPER_INITIAL_BALANCE, MAX_POSITION_SIZE, TRADE_FEE,
    STOP_LOSS_PCT, MAX_EPISODE_STEPS, LOOKBACK,
)
from feature_engine import FeatureEngine, compute_with_time
from reward import RewardFunction


OBS_DIM = 47


class BacktestEnv(gym.Env):
    """
    Backtest trading environment.
    hist_data: list of dicts with OHLCV keys + open_time
    train: if True slice from train set, else test set
    """

    metadata = {"render_modes": []}

    def __init__(self, hist_data: list, train: bool = True):
        super().__init__()

        self.hist_data   = hist_data
        self.train       = train
        self.feat_engine = FeatureEngine()
        self.reward_fn   = RewardFunction()

        self.action_space      = spaces.Discrete(3)   # 0=Hold 1=Buy 2=Sell
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(OBS_DIM,), dtype=np.float32
        )

        # Internal state
        self.balance      = PAPER_INITIAL_BALANCE
        self.position     = 0.0
        self.entry_price  = 0.0
        self.steps_held   = 0
        self.peak_balance = PAPER_INITIAL_BALANCE
        self.step_count   = 0
        self.cursor       = 0
        self.episode_pnl  = 0.0
        self.trade_count  = 0
        self.total_pnl    = 0.0

    # ── Gym Interface ────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random start within the dataset
        max_start = len(self.hist_data) - MAX_EPISODE_STEPS - LOOKBACK
        if max_start <= LOOKBACK:
            self.cursor = LOOKBACK
        else:
            self.cursor = int(np.random.randint(LOOKBACK, max_start))

        self.balance      = PAPER_INITIAL_BALANCE
        self.position     = 0.0
        self.entry_price  = 0.0
        self.steps_held   = 0
        self.peak_balance = PAPER_INITIAL_BALANCE
        self.step_count   = 0
        self.episode_pnl  = 0.0
        self.trade_count  = 0
        self.reward_fn.reset(PAPER_INITIAL_BALANCE)

        return self._get_obs(), {}

    def step(self, action: int):
        current = self.hist_data[self.cursor]
        price   = float(current["close"])

        trade_closed = False
        trade_pnl    = 0.0

        # ── ONLY HARD RULE: Stop Loss at -1.5% ───────────────
        if self.position > 0 and self.entry_price > 0:
            loss_pct = (self.entry_price - price) / self.entry_price
            if loss_pct >= STOP_LOSS_PCT:
                action = 2   # force sell

        # ── State Machine ─────────────────────────────────────
        if action == 1 and self.position == 0:
            self._open_long(price)

        elif action == 2 and self.position > 0:
            trade_pnl, trade_closed = self._close_long(price)

        elif self.position > 0:
            self.steps_held += 1

        # ── Portfolio Value ───────────────────────────────────
        if self.position > 0:
            pos_value      = self.position * price
            unrealized_pnl = (price - self.entry_price) * self.position
            total_value    = self.balance + pos_value
        else:
            unrealized_pnl = 0.0
            total_value    = self.balance

        self.peak_balance = max(self.peak_balance, total_value)

        # ── Reward ───────────────────────────────────────────
        reward = self.reward_fn.compute(
            total_value  = total_value,
            action       = action,
            position     = self.position,
            trade_closed = trade_closed,
            trade_pnl    = trade_pnl,
        )

        # ── Episode Termination ───────────────────────────────
        self.cursor     += 1
        self.step_count += 1
        terminated = False
        truncated  = False

        drawdown = (self.peak_balance - total_value) / (self.peak_balance + 1e-8)
        if drawdown >= 0.20:           # 20% drawdown = episode over
            terminated = True
            reward    -= 10.0

        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        if self.cursor >= len(self.hist_data) - 1:
            truncated = True

        # Update episode tracking
        if trade_closed:
            self.episode_pnl  += trade_pnl
            self.total_pnl    += trade_pnl
            self.trade_count  += 1

        info = {
            "price":         price,
            "balance":       self.balance,
            "total_value":   total_value,
            "trade_closed":  trade_closed,
            "trade_pnl":     trade_pnl,
            "drawdown":      drawdown,
            "unrealized":    unrealized_pnl,
            **self.reward_fn.get_stats(),
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ── Private ───────────────────────────────────────────────

    def _open_long(self, price):
        usdt              = self.balance * MAX_POSITION_SIZE
        fee               = usdt * TRADE_FEE
        self.position     = (usdt - fee) / price
        self.entry_price  = price
        self.balance     -= usdt
        self.steps_held   = 0

    def _close_long(self, price):
        proceeds     = self.position * price
        fee          = proceeds * TRADE_FEE
        net          = proceeds - fee
        pnl          = net - (self.position * self.entry_price)
        self.balance += net
        self.position     = 0.0
        self.entry_price  = 0.0
        self.steps_held   = 0
        return pnl, True

    def _get_obs(self):
        candles = self.hist_data[max(0, self.cursor - LOOKBACK): self.cursor]
        if not candles:
            return np.zeros(OBS_DIM, dtype=np.float32)

        # Get timestamp for time features
        try:
            ts   = candles[-1]["open_time"] / 1000
            dt   = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour = dt.hour
            dow  = dt.weekday()
        except Exception:
            hour, dow = 0, 0

        feats = compute_with_time(candles, hour, dow)

        # Agent state (last 5 features, indices 37-41)
        total_val    = self.balance + (self.position * float(candles[-1]["close"])
                       if self.position > 0 else 0.0)
        bal_ratio    = self.balance / PAPER_INITIAL_BALANCE
        steps_norm   = min(self.steps_held / 60.0, 1.0)
        unreal_pct   = 0.0
        if self.position > 0 and self.entry_price > 0:
            unreal_pct = (float(candles[-1]["close"]) - self.entry_price) / self.entry_price
        consec_norm  = min(self.reward_fn.consecutive_loss / 10.0, 1.0)

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:42] = feats[:42]
        obs[42]  = 1.0 if self.position > 0 else 0.0
        obs[43]  = np.clip(bal_ratio, 0, 2)
        obs[44]  = steps_norm
        obs[45]  = np.clip(unreal_pct * 10, -3, 3)
        obs[46]  = consec_norm

        return np.clip(obs, -10, 10).astype(np.float32)