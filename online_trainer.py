# ============================================================
#  online_trainer.py — The main always-training loop
#
#  This is the heart of the system.
#  It runs forever on the VPS:
#    1. Agent observes live market
#    2. Decides action
#    3. Paper trade executes
#    4. Reward computed
#    5. Model updates (online learning)
#    6. Checkpoint saved every hour
#    7. Monitor sends you updates
# ============================================================

import os
import time
import traceback
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from binance_stream import BinanceStream
from trading_env    import TradingEnv
from monitor        import Monitor
from config import (
    ALGORITHM, CHECKPOINT_DIR, LOG_DIR,
    TRAIN_EVERY_STEPS, CHECKPOINT_EVERY,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF,
    MAX_DRAWDOWN_PCT, TRADING_PAIR
)


# ----------------------------------------------------------
# Custom callback for live monitoring
# ----------------------------------------------------------
class LiveCallback(BaseCallback):

    def __init__(self, monitor: Monitor, env: TradingEnv, verbose=0):
        super().__init__(verbose)
        self.monitor  = monitor
        self.env      = env
        self.last_ckpt = time.time()
        self.episode_rewards = []
        self._ep_reward = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]

        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        if done:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0

            # Send summary to Telegram every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                self.monitor.send_episode_summary(
                    episode    = len(self.episode_rewards),
                    avg_reward = np.mean(self.episode_rewards[-10:]),
                    balance    = info.get("balance", 0),
                    total_pnl  = info.get("total_pnl", 0),
                    trades     = info.get("trade_count", 0),
                    win_rate   = info.get("win_rate", 0),
                    drawdown   = info.get("drawdown", 0),
                )

        # Checkpoint every hour
        if time.time() - self.last_ckpt >= CHECKPOINT_EVERY:
            self._save_checkpoint()
            self.last_ckpt = time.time()

        return True   # Keep training

    def _save_checkpoint(self):
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CHECKPOINT_DIR, f"agent_{ts}")
        self.model.save(path)
        print(f"[Trainer] Checkpoint saved: {path}")
        self.monitor.send_message(f"💾 Checkpoint saved: {ts}")


# ----------------------------------------------------------
# Main Trainer Class
# ----------------------------------------------------------
class OnlineTrainer:

    def __init__(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR,        exist_ok=True)

        self.monitor = Monitor()
        self.stream  = BinanceStream()
        self.env     = None
        self.model   = None

    def setup(self):
        print("[Trainer] Starting Binance stream...")
        self.stream.start()

        print("[Trainer] Setting up trading environment...")
        self.env = TradingEnv(stream=self.stream, mode="live")

        print("[Trainer] Initializing RL agent...")
        self.model = self._build_or_load_model()

        self.monitor.send_message(
            f"🤖 Trading bot started!\n"
            f"Pair: {TRADING_PAIR}\n"
            f"Algorithm: {ALGORITHM}\n"
            f"Mode: Online learning (paper trading)\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

    def _build_or_load_model(self):
        # Look for latest checkpoint
        checkpoints = sorted([
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith("agent_") and f.endswith(".zip")
        ])

        if checkpoints:
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"[Trainer] Loading checkpoint: {latest}")
            if ALGORITHM == "PPO":
                model = PPO.load(latest, env=self.env, verbose=1)
            else:
                model = SAC.load(latest, env=self.env, verbose=1)
            self.monitor.send_message(f"📂 Resumed from checkpoint: {checkpoints[-1]}")
        else:
            print("[Trainer] No checkpoint found — starting fresh...")
            model = self._create_new_model()
            self.monitor.send_message("🌱 Fresh model initialized — learning from scratch!")

        return model

    def _create_new_model(self):
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
        )

        if ALGORITHM == "PPO":
            return PPO(
                policy        = "MlpPolicy",
                env           = self.env,
                learning_rate = LEARNING_RATE,
                n_steps       = N_STEPS,
                batch_size    = BATCH_SIZE,
                n_epochs      = N_EPOCHS,
                gamma         = GAMMA,
                gae_lambda    = GAE_LAMBDA,
                clip_range    = CLIP_RANGE,
                ent_coef      = ENT_COEF,
                policy_kwargs = policy_kwargs,
                tensorboard_log = LOG_DIR,
                verbose       = 1,
            )
        else:   # SAC
            return SAC(
                policy          = "MlpPolicy",
                env             = self.env,
                learning_rate   = LEARNING_RATE,
                batch_size      = BATCH_SIZE,
                gamma           = GAMMA,
                policy_kwargs   = policy_kwargs,
                tensorboard_log = LOG_DIR,
                verbose         = 1,
            )

    # ----------------------------------------------------------
    # Main forever loop
    # ----------------------------------------------------------
    def run(self):
        self.setup()

        callback = LiveCallback(monitor=self.monitor, env=self.env)

        print("\n" + "="*60)
        print("  ONLINE TRAINING STARTED — Running forever on VPS")
        print("  The agent will trade, learn, and improve over time.")
        print("="*60 + "\n")

        while True:
            try:
                # Train for TRAIN_EVERY_STEPS, then loop back
                # This gives us continuous online learning
                self.model.learn(
                    total_timesteps  = TRAIN_EVERY_STEPS,
                    callback         = callback,
                    reset_num_timesteps = False,   # Don't reset step counter
                    progress_bar     = False,
                )

            except KeyboardInterrupt:
                print("\n[Trainer] Interrupted by user. Saving final model...")
                self._emergency_save()
                break

            except Exception as e:
                error_msg = traceback.format_exc()
                print(f"[Trainer] ERROR: {e}")
                print(error_msg)
                self.monitor.send_message(
                    f"⚠️ Error in training loop:\n{str(e)[:200]}\nRestarting in 30s..."
                )
                time.sleep(30)
                # Auto-recover: reinitialize stream if needed
                try:
                    if not self.stream.is_ready:
                        print("[Trainer] Restarting stream...")
                        self.stream = BinanceStream()
                        self.stream.start()
                        self.env.stream = self.stream
                except Exception as restart_err:
                    print(f"[Trainer] Restart failed: {restart_err}")
                    time.sleep(60)

    def _emergency_save(self):
        path = os.path.join(CHECKPOINT_DIR, "agent_emergency")
        self.model.save(path)
        print(f"[Trainer] Emergency save: {path}")
        self.monitor.send_message("🔴 Bot stopped. Emergency checkpoint saved.")


# ----------------------------------------------------------
# Entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    trainer = OnlineTrainer()
    trainer.run()