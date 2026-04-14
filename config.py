# ============================================================
#  config.py — Central configuration for the trading bot
# ============================================================

import os


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


# --- Binance ---
BINANCE_API_KEY    = _get_env("BINANCE_API_KEY")
BINANCE_API_SECRET = _get_env("BINANCE_API_SECRET")
TRADING_PAIR       = _get_env("TRADING_PAIR", "BTCUSDT")   # Change to any pair
TIMEFRAME          = _get_env("TIMEFRAME", "5m")           # 5-minute candles
ORDER_BOOK_DEPTH   = int(_get_env("ORDER_BOOK_DEPTH", "20"))

# --- Paper Trading ---
PAPER_INITIAL_BALANCE = float(_get_env("PAPER_INITIAL_BALANCE", "10000"))
MAX_POSITION_SIZE     = float(_get_env("MAX_POSITION_SIZE", "0.99"))
TRADE_FEE             = float(_get_env("TRADE_FEE", "0.001"))

# --- Features ---
LOOKBACK_CANDLES    = int(_get_env("LOOKBACK_CANDLES", "60"))
FEATURE_WINDOW      = int(_get_env("FEATURE_WINDOW", "14"))

# --- RL Agent ---
ALGORITHM           = _get_env("ALGORITHM", "PPO")         # "PPO" or "SAC"
LEARNING_RATE       = float(_get_env("LEARNING_RATE", "3e-4"))
N_STEPS             = int(_get_env("N_STEPS", "2048"))
BATCH_SIZE          = int(_get_env("BATCH_SIZE", "64"))
N_EPOCHS            = int(_get_env("N_EPOCHS", "10"))
GAMMA               = float(_get_env("GAMMA", "0.99"))
GAE_LAMBDA          = float(_get_env("GAE_LAMBDA", "0.95"))
CLIP_RANGE          = float(_get_env("CLIP_RANGE", "0.2"))
ENT_COEF            = float(_get_env("ENT_COEF", "0.01"))

# --- Online Training ---
TRAIN_EVERY_STEPS   = int(_get_env("TRAIN_EVERY_STEPS", "512"))
CHECKPOINT_EVERY    = int(_get_env("CHECKPOINT_EVERY", "3600"))
REPLAY_BUFFER_SIZE  = int(_get_env("REPLAY_BUFFER_SIZE", "50000"))

# --- Risk / Circuit Breaker ---
MAX_DRAWDOWN_PCT     = float(_get_env("MAX_DRAWDOWN_PCT", "0.20"))
DAILY_LOSS_LIMIT_PCT = float(_get_env("DAILY_LOSS_LIMIT_PCT", "0.05"))
MAX_CONSECUTIVE_LOSS = int(_get_env("MAX_CONSECUTIVE_LOSS", "10"))

# --- Monitoring ---
DASHBOARD_PORT      = int(_get_env("DASHBOARD_PORT", "8080"))

# --- Paths ---
LOG_DIR             = _get_env("LOG_DIR", "logs/")
CHECKPOINT_DIR      = _get_env("CHECKPOINT_DIR", "checkpoints/")
DATA_DIR            = _get_env("DATA_DIR", "data/")
