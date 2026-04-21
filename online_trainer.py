# ============================================================
#  online_trainer.py — Main Training Loop
#  Fixed: correct P&L in journal, Phase 3 connected
#  Phase 3: high priority trades replayed more during learning
# ============================================================

import os
import time
import traceback
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from binance_stream    import BinanceStream
from trading_env       import TradingEnv
from monitor           import Monitor
from journal           import TradeJournal
from pattern_extractor import PatternExtractor
from config import (
    CHECKPOINT_DIR, LOG_DIR,
    N_STEPS, BATCH_SIZE, N_EPOCHS,
    LEARNING_RATE, GAMMA, GAE_LAMBDA,
    CLIP_RANGE, ENT_COEF, CHECKPOINT_EVERY,
    TRADING_PAIR, PAPER_INITIAL_BALANCE
)

MAX_EPISODE_STEPS = 1440   # 1 day on 1-min candles
PHASE3_MIN_TRADES = 20     # Start Phase 3 after 20 journal entries


# ----------------------------------------------------------
# Phase 3 Callback — Weighted Replay
# Injects high-priority trade observations during learning
# ----------------------------------------------------------
class Phase3Callback(BaseCallback):
    """
    After each PPO update, check journal for high-priority
    trades and log them as additional learning signal.
    This is the Phase 3 weighted replay implementation.
    """

    def __init__(self, journal: TradeJournal, verbose=0):
        super().__init__(verbose)
        self.journal     = journal
        self.update_count = 0

    def _on_rollout_end(self):
        """Called after each rollout — inject priority trades."""
        self.update_count += 1

        # Only start Phase 3 after enough journal data
        priority_trades = self.journal.load_priority_trades(min_priority=7)

        if len(priority_trades) < 5:
            return

        print(
            f"[Phase3] Injecting {len(priority_trades)} "
            f"high-priority trades into replay..."
        )

        # For each high priority trade, we replay its features
        # through the value network to strengthen the signal
        try:
            import torch
            for trade in priority_trades:
                feat = trade.get("features", {})
                if not feat:
                    continue

                # Reconstruct observation vector from named features
                feature_names = [
                    "rsi", "macd", "macd_signal", "macd_hist",
                    "boll_position", "boll_bandwidth", "boll_pct",
                    "atr", "vwap_dev", "taker_ratio",
                    "price_velocity", "price_acceleration", "price_jerk",
                    "price_vel_avg", "price_acc_avg", "price_jerk_vol",
                    "vol_velocity", "vol_acceleration", "vol_jerk",
                    "vol_vel_avg", "vol_acc_avg", "vol_jerk_vol",
                    "hurst", "entropy", "zscore", "price_vol_corr",
                    "ob_spread", "ob_imbalance", "ob_bid_wall",
                    "ob_ask_wall", "ob_vol_ratio", "ob_spread2",
                    "trade_pressure",
                    "time_sin_hour", "time_cos_hour",
                    "time_sin_dow", "time_cos_dow",
                    "position_held", "balance_ratio",
                    "steps_held", "unrealized_pct", "consec_loss",
                ]
                obs_vec = np.array(
                    [feat.get(n, 0.0) for n in feature_names],
                    dtype=np.float32
                )

                # Pad or trim to match observation space
                expected_size = self.model.observation_space.shape[0]
                if len(obs_vec) < expected_size:
                    obs_vec = np.pad(
                        obs_vec, (0, expected_size - len(obs_vec))
                    )
                else:
                    obs_vec = obs_vec[:expected_size]

                # Compute priority weight
                priority = trade.get("priority", 1)
                pnl      = trade.get("pnl", 0)
                label    = trade.get("label", "")

                # Determine target reward signal
                # Wins → push value estimate up
                # Losses → push value estimate down (learn to avoid)
                if "WIN" in label:
                    target_reward = min(10.0, abs(pnl) / 10.0)
                else:
                    target_reward = -min(10.0, abs(pnl) / 10.0)

                # Scale by priority
                target_reward *= (priority / 5.0)

                # Compute current value estimate
                obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
                obs_tensor = obs_tensor.to(self.model.device)

                with torch.no_grad():
                    value_est = self.model.policy.predict_values(obs_tensor)

                # Compute value error and update (gradient step)
                value_target = torch.FloatTensor(
                    [[target_reward]]
                ).to(self.model.device)
                value_loss = torch.nn.functional.mse_loss(
                    value_est, value_target
                )

                # Only update if error is significant
                if value_loss.item() > 0.1:
                    self.model.policy.optimizer.zero_grad()
                    value_loss.backward()
                    self.model.policy.optimizer.step()

        except Exception as e:
            print(f"[Phase3] Replay error (non-fatal): {e}")

        return True

    def _on_step(self):
        return True


# ----------------------------------------------------------
# Main Trainer
# ----------------------------------------------------------
class CandleStepTrainer:

    def __init__(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR,        exist_ok=True)

        self.monitor        = Monitor()
        self.stream         = BinanceStream()
        self.env            = None
        self.model          = None
        self.obs            = None
        self.last_candle_ts = None
        self.journal        = TradeJournal()
        self.extractor      = PatternExtractor()
        self.last_extract   = datetime.utcnow().date()

        # Trade tracking
        self.prev_pnl       = 0.0
        self.prev_trades    = 0
        self.prev_price     = 0.0
        self.was_holding    = False   # Was agent holding last step?

    def setup(self):
        print("[Trainer] Starting stream...")
        self.stream.start()
        print("[Trainer] Building env...")
        self.env = TradingEnv(stream=self.stream, mode="live")
        print("[Trainer] Building model...")
        self.model = self._build_or_load_model()
        self.obs, _ = self.env.reset()
        self.monitor.send_message(
            f"Bot started | {TRADING_PAIR} | "
            f"Phase 2+3 ACTIVE | Journal + Weighted Replay"
        )

    def _build_or_load_model(self):
        checkpoints = sorted([
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith("agent_") and f.endswith(".zip")
        ])
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )
        vec_env = DummyVecEnv([lambda: self.env])
        if checkpoints:
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"[Trainer] Resuming: {latest}")
            return PPO.load(latest, env=vec_env, verbose=0)
        else:
            print("[Trainer] Fresh model")
            return PPO(
                policy          = "MlpPolicy",
                env             = vec_env,
                learning_rate   = LEARNING_RATE,
                n_steps         = N_STEPS,
                batch_size      = BATCH_SIZE,
                n_epochs        = N_EPOCHS,
                gamma           = GAMMA,
                gae_lambda      = GAE_LAMBDA,
                clip_range      = CLIP_RANGE,
                ent_coef        = ENT_COEF,
                policy_kwargs   = policy_kwargs,
                tensorboard_log = LOG_DIR,
                verbose         = 0,
            )

    def _wait_for_new_candle(self):
        while True:
            candles = self.stream.get_candles()
            if candles:
                ts = candles[-1]["ts"]
                if self.last_candle_ts is None:
                    self.last_candle_ts = ts
                    return True
                if ts != self.last_candle_ts:
                    self.last_candle_ts = ts
                    return True
            time.sleep(2)

    def _get_confidence(self, obs):
        """Get action probability distribution from policy."""
        try:
            import torch
            obs_t = self.model.policy.obs_to_tensor(
                obs.reshape(1, -1)
            )[0]
            with torch.no_grad():
                dist  = self.model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy()[0]
            return {
                "hold": round(float(probs[0]), 3),
                "buy":  round(float(probs[1]), 3),
                "sell": round(float(probs[2]), 3),
            }
        except Exception:
            return {"hold": 0.33, "buy": 0.33, "sell": 0.33}

    def _check_midnight_extraction(self):
        today = datetime.utcnow().date()
        if today > self.last_extract:
            print("[Trainer] Midnight — running daily pattern extraction...")
            try:
                self.extractor.run_daily_analysis()
                self.last_extract = today
                self.monitor.send_message(
                    "Daily pattern report ready. "
                    "Check logs/daily_report.txt"
                )
            except Exception as e:
                print(f"[Extractor] Error: {e}")

    def run(self):
        self.setup()

        # Phase 3 callback
        phase3_cb = Phase3Callback(journal=self.journal)

        step          = 0
        episode       = 0
        ep_reward     = 0.0
        ep_rewards    = []
        ep_trades     = 0
        ep_wins       = 0
        learn_step    = 0
        last_ckpt     = time.time()

        print("\n" + "="*60)
        print("  TRADING BOT — Phase 2 + Phase 3 ACTIVE")
        print("  Phase 2: Journal logs every trade with 42 features")
        print("  Phase 3: High priority trades replayed during learning")
        print("  Pattern report: daily at midnight UTC")
        print("="*60 + "\n")

        while True:
            try:
                self._check_midnight_extraction()

                print(f"[Trainer] Waiting for candle {step+1}...")
                self._wait_for_new_candle()

                # Get confidence BEFORE action
                confidence = self._get_confidence(self.obs)

                # Get current state BEFORE step
                was_holding    = self.env.position > 0
                pre_step_price = self.env._get_current_price()

                # Predict action
                action_arr, _ = self.model.predict(
                    self.obs.reshape(1, -1), deterministic=False
                )
                action = int(action_arr[0])

                # ---- Phase 2: Log BUY entry BEFORE step ----
                # (We know position is 0 and action is BUY)
                if action == 1 and self.env.position == 0:
                    self.journal.log_entry(
                        episode    = episode,
                        step       = step,
                        price      = pre_step_price,
                        features   = self.obs.copy(),
                        confidence = confidence,
                    )

                # ---- Tick journal if holding ----
                if was_holding:
                    self.journal.tick()

                # ---- Execute action ----
                next_obs, reward, terminated, truncated, info = \
                    self.env.step(action)
                done = terminated or truncated

                ep_reward  += reward
                step       += 1
                learn_step += 1

                # Current state AFTER step
                cur_trades = info.get("trade_count", 0)
                cur_pnl    = info.get("total_pnl",   0.0)
                cur_price  = info.get("price",        0.0)
                cur_bal    = info.get("balance",      0.0)
                drawdown   = info.get("drawdown",     0.0)

                # ---- Phase 2: Log SELL exit AFTER step ----
                # BUG FIX: was_holding=True, now position=0 → sell happened
                just_sold = was_holding and self.env.position == 0
                if just_sold:
                    delta_pnl       = cur_pnl - self.prev_pnl
                    exit_confidence = self._get_confidence(next_obs)
                    self.journal.log_exit(
                        exit_price       = cur_price,
                        pnl              = delta_pnl,   # CORRECT P&L
                        episode          = episode,
                        step             = step,
                        exit_confidence  = exit_confidence,
                    )

                # Track trades for dashboard
                if cur_trades > self.prev_trades:
                    delta = cur_pnl - self.prev_pnl
                    if delta > 0:
                        ep_wins += 1
                    ep_trades        = cur_trades
                    self.prev_trades = cur_trades
                    self.prev_pnl    = cur_pnl

                    # Log to dashboard
                    for t in self.env.trade_log[-1:]:
                        self.monitor.log_trade({**t, "episode": episode})

                action_name = ["HOLD", "BUY ", "SELL"][action]
                top_conf    = max(confidence.values())

                print(
                    f"[Step {step:5d}] {action_name} | "
                    f"${cur_price:,.2f} | "
                    f"Bal: ${cur_bal:,.2f} | "
                    f"PnL: {'+' if cur_pnl >= 0 else ''}${cur_pnl:.2f} | "
                    f"Trades: {cur_trades} | "
                    f"Conf: {top_conf:.0%} | "
                    f"DD: {drawdown*100:.2f}%"
                )

                win_rate = ep_wins / ep_trades if ep_trades > 0 else 0.0
                self.monitor.send_episode_summary(
                    episode        = episode,
                    avg_reward     = ep_reward / max(step, 1),
                    balance        = cur_bal,
                    total_pnl      = cur_pnl,
                    trades         = ep_trades,
                    win_rate       = win_rate,
                    drawdown       = drawdown,
                    episode_trades = ep_trades,
                    episode_pnl    = cur_pnl,
                    episode_wins   = ep_wins,
                )

                # ---- Learn every N_STEPS with Phase 3 ----
                if learn_step >= N_STEPS:
                    journal_count = len(self.journal.load_all())
                    use_phase3    = journal_count >= PHASE3_MIN_TRADES

                    print(
                        f"[Trainer] Learning update at step {step} | "
                        f"Phase 3: {'ACTIVE' if use_phase3 else 'warming up...'} | "
                        f"Journal: {journal_count} trades"
                    )

                    vec_env = DummyVecEnv([lambda: self.env])
                    self.model.set_env(vec_env)

                    # Phase 3 callback only active when enough data
                    cb = phase3_cb if use_phase3 else None

                    self.model.learn(
                        total_timesteps     = N_STEPS,
                        callback            = cb,
                        reset_num_timesteps = False,
                        progress_bar        = False,
                    )
                    learn_step = 0
                    print("[Trainer] Model updated ✓")

                # ---- Episode end ----
                if done or step % MAX_EPISODE_STEPS == 0:
                    episode += 1
                    ep_rewards.append(ep_reward)
                    avg_rew = np.mean(ep_rewards[-10:])
                    print(
                        f"\n{'='*50}\n"
                        f"[Episode {episode} END]\n"
                        f"  Steps:    {step}\n"
                        f"  Trades:   {ep_trades}\n"
                        f"  Wins:     {ep_wins}\n"
                        f"  Win rate: {win_rate*100:.1f}%\n"
                        f"  PnL:      ${cur_pnl:+.2f}\n"
                        f"  Avg Rew:  {avg_rew:.2f}\n"
                        f"{'='*50}\n"
                    )
                    next_obs, _ = self.env.reset()
                    ep_reward    = 0.0
                    ep_trades    = 0
                    ep_wins      = 0
                    self.prev_trades = 0
                    self.prev_pnl    = 0.0

                # ---- Checkpoint every hour ----
                if time.time() - last_ckpt >= CHECKPOINT_EVERY:
                    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(CHECKPOINT_DIR, f"agent_{ts}")
                    self.model.save(path)
                    print(f"[Trainer] Checkpoint: {path}")
                    last_ckpt = time.time()

                # Update state
                self.obs         = next_obs
                self.was_holding = self.env.position > 0

            except KeyboardInterrupt:
                path = os.path.join(CHECKPOINT_DIR, "agent_emergency")
                self.model.save(path)
                print(f"[Trainer] Emergency save: {path}")
                break

            except Exception as e:
                print(f"[Trainer] ERROR: {e}")
                print(traceback.format_exc())
                time.sleep(10)
                try:
                    if not self.stream.is_ready:
                        self.stream = BinanceStream()
                        self.stream.start()
                        self.env.stream = self.stream
                    self.obs, _ = self.env.reset()
                    self.prev_trades = 0
                    self.prev_pnl    = 0.0
                except Exception as re:
                    print(f"[Trainer] Restart failed: {re}")
                    time.sleep(30)


if __name__ == "__main__":
    trainer = CandleStepTrainer()
    trainer.run()