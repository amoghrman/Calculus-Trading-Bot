#!/usr/bin/env python3
# ============================================================
#  backtest_runner.py — Main Backtest Training Loop
#  Usage: python3 backtest_runner.py
#
#  Trains PPO on full BTCUSDT history at maximum speed
#  Phase 2 journal + Phase 3 replay active
#  Saves checkpoint every episode
#  Stats written to logs/ for UI to read
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    LOG_DIR, CHECKPOINT_DIR, DATA_DIR,
    PAPER_INITIAL_BALANCE, LEARNING_RATE, N_STEPS, BATCH_SIZE,
    N_EPOCHS, GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF,
    TRAIN_END_DATE, TEST_START_DATE, PHASE3_MIN_BIG_WINS,
)
from trading_env import BacktestEnv
from journal import TradeJournal
from phase3 import Phase3Trainer

PARQUET   = os.path.join(DATA_DIR, "btcusdt_1m.parquet")
STATS_FILE = os.path.join(LOG_DIR, "backtest_stats.json")
BEST_CKPT  = os.path.join(CHECKPOINT_DIR, "best_model")
LIVE_CKPT  = os.path.join(CHECKPOINT_DIR, "transfer_to_live")


def load_data():
    print("Loading historical data...")
    df = pd.read_parquet(PARQUET)
    rows = df.to_dict("records")
    print(f"  Loaded {len(rows):,} candles")

    # Split train / test
    split_ts = int(pd.Timestamp(TRAIN_END_DATE).timestamp() * 1000)
    train = [r for r in rows if r["open_time"] <= split_ts]
    test  = [r for r in rows if r["open_time"] >  split_ts]
    print(f"  Train: {len(train):,} candles | Test: {len(test):,} candles")
    return train, test


def make_model(env):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate = LEARNING_RATE,
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF,
        policy_kwargs = dict(net_arch=[256, 256, 128]),
        verbose       = 0,
    )


def run_episode(env_raw, model, journal, episode):
    """Run one episode, log trades, return stats."""
    obs, _  = env_raw.reset()
    done    = False
    wins    = 0
    losses  = 0
    pnl     = 0.0
    trades  = 0
    in_trade = False
    entry_price = 0.0
    entry_step  = 0
    entry_conf  = {}
    entry_obs   = None

    while not done:
        action_arr, _ = model.predict(obs.reshape(1, -1), deterministic=False)
        action = int(action_arr[0])

        # Confidence
        with torch.no_grad():
            obs_t    = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            dist     = model.policy.get_distribution(obs_t)
            probs    = dist.distribution.probs.squeeze().numpy()
        conf = {"hold": float(probs[0]),
                "buy":  float(probs[1]),
                "sell": float(probs[2])}

        # Journal entry
        if action == 1 and not in_trade:
            entry_price = env_raw.hist_data[env_raw.cursor - 1]["close"]
            entry_step  = env_raw.step_count
            entry_obs   = obs.copy()
            entry_conf  = conf
            in_trade    = True
            journal.log_entry(
                episode    = episode,
                step       = entry_step,
                price      = entry_price,
                features   = entry_obs,
                confidence = entry_conf,
            )

        prev_position = env_raw.position
        obs, reward, terminated, truncated, info = env_raw.step(action)
        done = terminated or truncated

        # Journal exit
        if in_trade and info.get("trade_closed"):
            t = journal.log_exit(
                exit_price   = env_raw.hist_data[env_raw.cursor - 1]["close"],
                pnl          = info["trade_pnl"],
                candles_held = env_raw.steps_held,
                reason       = "sell",
            )
            if t:
                if t["pnl"] > 0:
                    wins += 1
                else:
                    losses += 1
                pnl    += t["pnl"]
                trades += 1
            in_trade = False

    return {
        "episode": episode,
        "trades":  trades,
        "wins":    wins,
        "losses":  losses,
        "pnl":     round(pnl, 2),
        "win_rate": round(wins / trades * 100, 1) if trades else 0.0,
        "balance": round(env_raw.balance, 2),
    }


def evaluate_on_test(test_data, model, journal, n_episodes=5):
    """Run model on unseen test data."""
    results = []
    for i in range(n_episodes):
        env = BacktestEnv(test_data, train=False)
        r   = run_episode(env, model, journal, episode=-1)
        results.append(r)
    return {
        "test_win_rate": np.mean([r["win_rate"] for r in results]),
        "test_avg_pnl":  np.mean([r["pnl"]      for r in results]),
        "test_episodes": n_episodes,
    }


def save_stats(episode, ep_result, history, test_result=None, speed=0):
    trades = journal_global.load_all()
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    stats = {
        "last_updated":    datetime.now(timezone.utc).isoformat(),
        "episode":         episode,
        "speed_cps":       round(speed, 0),
        "total_candles":   episode * 1440,

        # Episode result
        "ep_trades":       ep_result["trades"],
        "ep_wins":         ep_result["wins"],
        "ep_pnl":          ep_result["pnl"],
        "ep_win_rate":     ep_result["win_rate"],

        # All-time journal
        "all_trades":      len(trades),
        "all_wins":        len(wins),
        "all_win_rate":    round(len(wins)/len(trades)*100, 1) if trades else 0,
        "all_pnl":         round(sum(t["pnl"] for t in trades), 2),
        "avg_win":         round(sum(t["pnl"] for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss":        round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0,
        "best_trade":      round(max((t["pnl"] for t in trades), default=0), 2),
        "worst_trade":     round(min((t["pnl"] for t in trades), default=0), 2),
        "big_wins":        sum(1 for t in trades if t["pnl"] > 20),
        "high_priority":   sum(1 for t in trades if t.get("priority", 0) >= 7),
        "avg_hold_wins":   round(np.mean([t["candles_held"] for t in wins]), 1) if wins else 0,
        "avg_hold_losses": round(np.mean([t["candles_held"] for t in losses]), 1) if losses else 0,

        # History for chart
        "history":         history[-200:],

        # Test
        "test":            test_result or {},
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


# Global journal for save_stats access
journal_global = TradeJournal()


def main():
    global journal_global
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_data, test_data = load_data()

    journal    = journal_global
    phase3     = Phase3Trainer()
    history    = []
    best_wr    = 0.0
    episode    = 0
    start_time = time.time()

    print("\n=== Backtest Training Started ===")
    print(f"Train: {len(train_data):,} candles | Test: {len(test_data):,} candles")
    print("Ctrl+C to stop. Checkpoints saved every episode.\n")

    # Build initial env + model
    env_fn  = lambda: BacktestEnv(train_data)
    vec_env = DummyVecEnv([env_fn])
    model   = make_model(vec_env)

    # Resume if checkpoint exists
    latest = _latest_checkpoint()
    if latest:
        print(f"Resuming from {latest}")
        model = PPO.load(latest, env=vec_env)

    try:
        while True:
            ep_start = time.time()

            # ── Train one episode via PPO ─────────────────────
            model.set_env(vec_env)
            model.learn(total_timesteps=1440, reset_num_timesteps=False)

            # ── Run one episode for stats ─────────────────────
            env_raw = BacktestEnv(train_data)
            result  = run_episode(env_raw, model, journal, episode)

            # ── Phase 3 ───────────────────────────────────────
            big_wins = journal.big_wins()
            if len(big_wins) >= PHASE3_MIN_BIG_WINS:
                phase3.train(model, big_wins)
                print(f"  [Phase3] Updated with {len(big_wins)} big wins")

            # ── Save checkpoint ───────────────────────────────
            ckpt_path = os.path.join(
                CHECKPOINT_DIR,
                f"ep_{episode:05d}_wr{result['win_rate']:.0f}"
            )
            model.save(ckpt_path)

            if result["win_rate"] > best_wr:
                best_wr = result["win_rate"]
                model.save(BEST_CKPT)
                model.save(LIVE_CKPT)
                print(f"  ★ New best! WR={best_wr:.1f}% — saved to transfer_to_live")

            # ── Speed ─────────────────────────────────────────
            elapsed = time.time() - ep_start
            speed   = 1440 / elapsed if elapsed > 0 else 0

            # ── Stats file ────────────────────────────────────
            history.append({
                "ep":       episode,
                "wr":       result["win_rate"],
                "pnl":      result["pnl"],
                "trades":   result["trades"],
            })

            test_result = None
            if episode % 50 == 0 and episode > 0:
                print("  [Test] Evaluating on unseen data...")
                test_result = evaluate_on_test(test_data, model, journal)
                print(f"  [Test] WR={test_result['test_win_rate']:.1f}%  "
                      f"AvgPnL=${test_result['test_avg_pnl']:.2f}")

            save_stats(episode, result, history, test_result, speed)

            # ── Console output ────────────────────────────────
            total_trades = journal.count()
            print(
                f"Ep {episode:4d} | "
                f"WR:{result['win_rate']:5.1f}% | "
                f"PnL:${result['pnl']:+8.2f} | "
                f"T:{result['trades']:3d} | "
                f"AllT:{total_trades:5d} | "
                f"BigW:{len(big_wins):3d} | "
                f"{speed:5.0f} c/s"
            )

            episode += 1

    except KeyboardInterrupt:
        print("\nStopped. Saving final checkpoint...")
        model.save(os.path.join(CHECKPOINT_DIR, "final"))
        print("Done.")


def _latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    files.sort()
    return os.path.join(CHECKPOINT_DIR, files[-1][:-4])


if __name__ == "__main__":
    main()