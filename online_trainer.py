
import os
import time
import traceback
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer

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

MAX_EPISODE_STEPS  = 1440
PHASE3_MIN_TRADES  = 20


class CandleStepTrainer:

    def __init__(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR,        exist_ok=True)
        self.monitor        = Monitor()
        self.stream         = BinanceStream()
        self.env            = None
        self.vec_env        = None   # persistent — never recreated
        self.model          = None
        self.obs            = None
        self.last_candle_ts = None
        self.journal        = TradeJournal()
        self.extractor      = PatternExtractor()
        self.last_extract   = datetime.utcnow().date()
        self.prev_pnl       = 0.0
        self.prev_trades    = 0
        self.was_holding    = False

    def setup(self):
        print("[Trainer] Starting stream...")
        self.stream.start()
        print("[Trainer] Building env...")
        self.env = TradingEnv(stream=self.stream, mode="live")

        # Create vec_env ONCE — never recreated
        self.vec_env = DummyVecEnv([lambda: self.env])

        print("[Trainer] Building model...")
        self.model = self._build_or_load_model()

        # Get initial observation WITHOUT resetting env
        # (env already initialized in __init__)
        self.obs, _ = self.env.reset()

        self.monitor.send_message(
            f"Bot started | {TRADING_PAIR} | Phase 2+3 ACTIVE"
        )

    def _build_or_load_model(self):
        checkpoints = sorted([
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith("agent_") and f.endswith(".zip")
        ])
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )
        if checkpoints:
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"[Trainer] Resuming: {latest}")
            return PPO.load(latest, env=self.vec_env, verbose=0)
        else:
            print("[Trainer] Fresh model")
            return PPO(
                policy          = "MlpPolicy",
                env             = self.vec_env,
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
        try:
            import torch
            # MUST set train mode for gradients
            self.model.policy.set_training_mode(False)
            obs_t = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
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

    def _run_phase3(self):
        """
        Fixed Phase 3 — properly enables gradients
        Updates value head on priority trades
        Does NOT interfere with environment state
        """
        try:
            import torch

            priority_trades = self.journal.load_priority_trades(min_priority=7)
            if len(priority_trades) < 5:
                return

            print(f"[Phase3] Injecting {len(priority_trades)} priority trades...")

            # Switch to TRAINING mode for gradient updates
            self.model.policy.set_training_mode(True)

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

            expected_size = self.model.observation_space.shape[0]
            updates_done  = 0

            for trade in priority_trades:
                feat = trade.get("features", {})
                if not feat:
                    continue

                obs_vec = np.array(
                    [feat.get(n, 0.0) for n in feature_names],
                    dtype=np.float32
                )

                # Pad to match observation space
                if len(obs_vec) < expected_size:
                    obs_vec = np.pad(obs_vec, (0, expected_size - len(obs_vec)))
                else:
                    obs_vec = obs_vec[:expected_size]

                pnl      = trade.get("pnl", 0)
                priority = trade.get("priority", 1)
                label    = trade.get("label", "")

                # Target value scaled by priority and P&L size
                scale = priority / 5.0
                if "WIN" in label:
                    target_val = min(8.0, abs(pnl) / 8.0) * scale
                else:
                    target_val = -min(8.0, abs(pnl) / 8.0) * scale

                obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
                obs_tensor = obs_tensor.to(self.model.device)

                # Compute value with gradients ENABLED
                value_pred = self.model.policy.predict_values(obs_tensor)
                target_tensor = torch.FloatTensor([[target_val]]).to(self.model.device)

                loss = torch.nn.functional.mse_loss(value_pred, target_tensor)

                if loss.item() > 0.05:
                    self.model.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(
                        self.model.policy.parameters(), max_norm=0.5
                    )
                    self.model.policy.optimizer.step()
                    updates_done += 1

            # Switch BACK to eval mode
            self.model.policy.set_training_mode(False)
            print(f"[Phase3] Done — {updates_done} value updates applied")

        except Exception as e:
            print(f"[Phase3] Error: {e}")
            # Always restore eval mode
            try:
                self.model.policy.set_training_mode(False)
            except Exception:
                pass

    def _do_learning_update(self, step):
        """
        Fixed learning update — does NOT reset environment
        Collects rollout from buffer without env.reset()
        """
        journal_count = len(self.journal.load_all())
        use_phase3    = journal_count >= PHASE3_MIN_TRADES

        print(
            f"[Trainer] Learning update at step {step} | "
            f"Phase 3: {'ACTIVE' if use_phase3 else 'warming up'} | "
            f"Journal: {journal_count} trades"
        )

        try:
            # Use existing vec_env — no reset
            self.model.set_env(self.vec_env)

            # Learn without resetting timestep counter
            # reset_num_timesteps=False is critical
            self.model.learn(
                total_timesteps     = N_STEPS,
                reset_num_timesteps = False,
                progress_bar        = False,
            )

            # Run Phase 3 AFTER standard PPO update
            if use_phase3:
                self._run_phase3()

            print("[Trainer] Model updated ✓")

        except Exception as e:
            print(f"[Trainer] Learning error: {e}")

    def _check_midnight_extraction(self):
        today = datetime.utcnow().date()
        if today > self.last_extract:
            print("[Trainer] Midnight — running pattern extraction...")
            try:
                self.extractor.run_daily_analysis()
                self.monitor.send_message("Daily pattern report ready.")
            except Exception as e:
                print(f"[Extractor] Error: {e}")
            finally:
                self.last_extract = datetime.utcnow().date()

    def run(self):
        self.setup()

        step          = 0
        episode       = 0
        ep_reward     = 0.0
        ep_rewards    = []
        ep_trades     = 0
        ep_wins       = 0
        learn_step    = 0
        last_ckpt     = time.time()

        print("\n" + "="*60)
        print("  TRADING BOT — Phase 2+3 | Bug fixes applied")
        print("  - Env NO LONGER resets during learning update")
        print("  - Phase 3 gradients properly enabled")
        print("  - Persistent vec_env throughout session")
        print("="*60 + "\n")

        while True:
            try:
                self._check_midnight_extraction()

                print(f"[Trainer] Waiting for candle {step+1}...")
                self._wait_for_new_candle()

                # Get confidence (eval mode)
                confidence = self._get_confidence(self.obs)

                # State before action
                was_holding    = self.env.position > 0
                pre_step_price = self.env._get_current_price()

                # Predict
                action_arr, _ = self.model.predict(
                    self.obs.reshape(1, -1), deterministic=False
                )
                action = int(action_arr[0])

                # Journal BUY entry
                if action == 1 and self.env.position == 0:
                    self.journal.log_entry(
                        episode    = episode,
                        step       = step,
                        price      = pre_step_price,
                        features   = self.obs.copy(),
                        confidence = confidence,
                    )

                # Journal tick if holding
                if was_holding:
                    self.journal.tick()

                # Execute
                next_obs, reward, terminated, truncated, info = \
                    self.env.step(action)
                done = terminated or truncated

                ep_reward  += reward
                step       += 1
                learn_step += 1

                cur_trades = info.get("trade_count", 0)
                cur_pnl    = info.get("total_pnl",   0.0)
                cur_price  = info.get("price",        0.0)
                cur_bal    = info.get("balance",      0.0)
                drawdown   = info.get("drawdown",     0.0)

                # Journal SELL exit — AFTER step so price is correct
                just_sold = was_holding and self.env.position == 0
                if just_sold:
                    delta_pnl = cur_pnl - self.prev_pnl
                    self.journal.log_exit(
                        exit_price       = cur_price,
                        pnl              = delta_pnl,
                        episode          = episode,
                        step             = step,
                        exit_confidence  = self._get_confidence(next_obs),
                    )

                # Track trades
                if cur_trades > self.prev_trades:
                    if (cur_pnl - self.prev_pnl) > 0:
                        ep_wins += 1
                    ep_trades        = cur_trades
                    self.prev_trades = cur_trades
                    self.prev_pnl    = cur_pnl
                    for t in self.env.trade_log[-1:]:
                        self.monitor.log_trade({**t, "episode": episode})

                action_name = ["HOLD", "BUY ", "SELL"][action]
                top_conf    = max(confidence.values())

                print(
                    f"[Step {step:5d}] {action_name} | "
                    f"${cur_price:,.2f} | "
                    f"Bal: ${cur_bal:,.2f} | "
                    f"PnL: {'+' if cur_pnl>=0 else ''}${cur_pnl:.2f} | "
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

                # Learning update — fixed, no env reset
                if learn_step >= N_STEPS:
                    self._do_learning_update(step)
                    learn_step = 0

                # Episode end
                if done or step % MAX_EPISODE_STEPS == 0:
                    episode += 1
                    ep_rewards.append(ep_reward)
                    avg_rew = np.mean(ep_rewards[-10:])
                    print(
                        f"\n{'='*50}\n"
                        f"[Episode {episode} END]\n"
                        f"  Trades:   {ep_trades}\n"
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

                # Checkpoint
                if time.time() - last_ckpt >= CHECKPOINT_EVERY:
                    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(CHECKPOINT_DIR, f"agent_{ts}")
                    self.model.save(path)
                    print(f"[Trainer] Checkpoint: {path}")
                    last_ckpt = time.time()

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
