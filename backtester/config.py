# ============================================================
#  config.py — Backtest Configuration
# ============================================================

# Paths
DATA_DIR       = "data"
LOG_DIR        = "logs"
CHECKPOINT_DIR = "checkpoints"

# Trading
TRADING_PAIR          = "BTCUSDT"
PAPER_INITIAL_BALANCE = 10_000.0
MAX_POSITION_SIZE     = 0.95
TRADE_FEE             = 0.001
STOP_LOSS_PCT         = 0.015   # 1.5% hard stop loss

# Features
FEATURE_WINDOW = 14
LOOKBACK       = 60

# PPO Hyperparameters
ALGORITHM     = "PPO"
LEARNING_RATE = 3e-4
N_STEPS       = 512
BATCH_SIZE    = 64
N_EPOCHS      = 10
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.01

# Episode
MAX_EPISODE_STEPS = 1440   # 1 day of 1-min candles

# Phase 3
PHASE3_MIN_BIG_WINS = 10   # need 10 big wins before Phase 3

# Data download
START_DATE = "2019-09-01"   # Binance BTCUSDT start
INTERVAL   = "1m"

# Train/Test split
TRAIN_END_DATE = "2024-12-31"   # train on this
TEST_START_DATE = "2025-01-01"  # validate on this