# ============================================================
#  trading_env.py — Custom Gymnasium paper trading environment
#
#  The agent interacts here:
#    State:   Feature vector (37 values)
#    Action:  0=Hold, 1=Buy, 2=Sell/Close
#    Reward:  From reward.py
#
#  Simulates:
#    - Paper balance tracking
#    - Position management
#    - Fees and slippage
#    - Circuit breaker (stops if drawdown too large)
# ============================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from reward import RewardFunction
from feature_engine import FeatureEngine
from config import (
    PAPER_INITIAL_BALANCE, MAX_POSITION_SIZE,
    TRADE_FEE, MAX_DRAWDOWN_PCT, DAILY_LOSS_LIMIT_PCT,
    MAX_CONSECUTIVE_LOSS
)


class TradingEnv(gym.Env):
    """
    Paper trading environment. Works in two modes:
      - Backtest mode: pass historical candles, steps through them
      - Live mode:     connected to BinanceStream, steps in real-time
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, stream=None, historical_candles=None, mode="live"):
        super().__init__()
        self.stream    = stream
        self.mode      = mode
        self.hist_data = historical_candles   # For backtest

        self.feature_engine = FeatureEngine()
        self.reward_fn      = RewardFunction()

        n_features = self.feature_engine.n_features

        # ---- Action Space: Hold / Buy / Sell ----------------
        self.action_space = spaces.Discrete(3)

        # ---- Observation Space: feature vector --------------
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(n_features,),
            dtype=np.float32
        )

        # ---- Paper trading state ----------------------------
        self.balance     = PAPER_INITIAL_BALANCE
        self.position    = 0.0    # BTC held
        self.entry_price = 0.0
        self.peak_balance= PAPER_INITIAL_BALANCE
        self.day_start_balance = PAPER_INITIAL_BALANCE

        # ---- Episode tracking -------------------------------
        self.step_count   = 0
        self.trade_count  = 0
        self.total_pnl    = 0.0
        self.trade_log    = []

        # ---- Backtest index ---------------------------------
        self._hist_idx = FEATURE_WINDOW + 5 if historical_candles else 0

    # ----------------------------------------------------------
    # reset() — called at start of each episode
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance         = PAPER_INITIAL_BALANCE
        self.position        = 0.0
        self.entry_price     = 0.0
        self.peak_balance    = PAPER_INITIAL_BALANCE
        self.day_start_balance = PAPER_INITIAL_BALANCE
        self.step_count      = 0
        self.trade_count     = 0
        self.total_pnl       = 0.0
        self.trade_log       = []
        self._hist_idx       = FEATURE_WINDOW + 5

        self.reward_fn.reset(PAPER_INITIAL_BALANCE)

        obs = self._get_observation()
        return obs, {}

    # ----------------------------------------------------------
    # step() — agent takes one action
    # ----------------------------------------------------------
    def step(self, action: int):
        prev_balance    = self.balance
        current_price   = self._get_current_price()
        trade_closed    = False
        trade_pnl       = 0.0

        # ---- Execute action ---------------------------------
        if action == 1 and self.position == 0:    # BUY
            self._open_long(current_price)

        elif action == 2 and self.position > 0:   # SELL / CLOSE
            trade_pnl, trade_closed = self._close_long(current_price)

        # ---- Compute unrealized P&L -------------------------
        unrealized_pnl = 0.0
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position

        # ---- Reward -----------------------------------------
        reward = self.reward_fn.compute(
            prev_balance   = prev_balance,
            curr_balance   = self.balance,
            unrealized_pnl = unrealized_pnl,
            action         = action,
            position       = self.position,
            trade_closed   = trade_closed,
            trade_pnl      = trade_pnl,
            step           = self.step_count,
        )

        # ---- Update tracking --------------------------------
        self.step_count  += 1
        total_value       = self.balance + unrealized_pnl
        self.peak_balance = max(self.peak_balance, total_value)

        # ---- Check termination conditions -------------------
        terminated, truncated = False, False
        info = {}

        drawdown = (self.peak_balance - total_value) / self.peak_balance
        daily_loss = (self.day_start_balance - total_value) / self.day_start_balance

        if drawdown >= MAX_DRAWDOWN_PCT:
            terminated = True
            info["reason"] = "circuit_breaker_drawdown"
            reward -= 5.0   # Extra penalty for blowing up

        if daily_loss >= DAILY_LOSS_LIMIT_PCT:
            terminated = True
            info["reason"] = "daily_loss_limit"

        if self.reward_fn.consecutive_loss >= MAX_CONSECUTIVE_LOSS:
            terminated = True
            info["reason"] = "consecutive_loss_limit"

        if self.mode == "backtest" and self._hist_idx >= len(self.hist_data) - 1:
            truncated = True
            info["reason"] = "end_of_data"

        # ---- Advance data pointer ---------------------------
        if self.mode == "backtest":
            self._hist_idx += 1

        # ---- Build info dict --------------------------------
        info.update({
            "balance":       self.balance,
            "position":      self.position,
            "unrealized":    unrealized_pnl,
            "total_value":   total_value,
            "drawdown":      drawdown,
            "trade_count":   self.trade_count,
            "total_pnl":     self.total_pnl,
            "step":          self.step_count,
            "price":         current_price,
        })
        info.update(self.reward_fn.get_stats())

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------
    # Trade execution
    # ----------------------------------------------------------
    def _open_long(self, price: float):
        slippage   = price * 0.0002    # 0.02% slippage on 5-min (realistic)
        exec_price = price + slippage
        fee        = self.balance * MAX_POSITION_SIZE * TRADE_FEE

        usdt_to_use    = self.balance * MAX_POSITION_SIZE - fee
        self.position  = usdt_to_use / exec_price
        self.balance  -= (usdt_to_use + fee)
        self.entry_price = exec_price

        self.trade_log.append({
            "side": "buy", "price": exec_price,
            "size": self.position, "step": self.step_count
        })

    def _close_long(self, price: float):
        slippage   = price * 0.0002
        exec_price = price - slippage
        proceeds   = self.position * exec_price
        fee        = proceeds * TRADE_FEE
        pnl        = proceeds - fee - (self.entry_price * self.position)

        self.balance   += proceeds - fee
        self.total_pnl += pnl
        self.trade_count += 1

        self.trade_log.append({
            "side": "sell", "price": exec_price,
            "size": self.position, "pnl": pnl, "step": self.step_count
        })

        self.position    = 0.0
        self.entry_price = 0.0
        return pnl, True

    # ----------------------------------------------------------
    # Observation builder
    # ----------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        if self.mode == "backtest":
            candles  = self.hist_data[max(0, self._hist_idx-60):self._hist_idx]
            ob       = None   # No live order book in backtest
            pressure = 0.5
        else:
            candles  = self.stream.get_candles()
            ob       = self.stream.get_order_book_features()
            pressure = self.stream.get_trade_pressure()

        obs = self.feature_engine.compute(candles, ob, pressure)
        if obs is None:
            return np.zeros(self.feature_engine.n_features, dtype=np.float32)

        # Append portfolio state to observation
        # (agent needs to know its own position)
        n = len(obs)
        full_obs = np.zeros(self.feature_engine.n_features, dtype=np.float32)
        full_obs[:n] = obs
        return full_obs

    def _get_current_price(self) -> float:
        if self.mode == "backtest":
            return self.hist_data[self._hist_idx]["close"]
        else:
            candles = self.stream.get_candles()
            return candles[-1]["close"] if candles else 0.0

    # ----------------------------------------------------------
    # Render
    # ----------------------------------------------------------
    def render(self):
        price = self._get_current_price()
        unreal = (price - self.entry_price) * self.position if self.position > 0 else 0
        total  = self.balance + unreal
        print(
            f"[Env] Step {self.step_count:5d} | "
            f"Price: {price:10.2f} | "
            f"Balance: {self.balance:10.2f} | "
            f"Position: {self.position:.6f} | "
            f"Total: {total:10.2f} | "
            f"PnL: {self.total_pnl:+.2f}"
        )