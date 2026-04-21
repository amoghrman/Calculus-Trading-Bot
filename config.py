
# ============================================================
#  config.py — Central configuration for the trading bot
# ============================================================

# --- Binance ---
BINANCE_API_KEY    = ""
BINANCE_API_SECRET = ""
TRADING_PAIR       = "BTCUSDT"
TIMEFRAME          = "5m"
ORDER_BOOK_DEPTH   = 20

# --- Paper Trading ---
PAPER_INITIAL_BALANCE = 10_000
MAX_POSITION_SIZE     = 0.95
TRADE_FEE             = 0.001

# --- Features ---
LOOKBACK_CANDLES    = 60
FEATURE_WINDOW      = 14

# --- RL Agent ---
ALGORITHM           = "PPO"
LEARNING_RATE       = 3e-4
N_STEPS             = 2048
BATCH_SIZE          = 64
N_EPOCHS            = 10
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
CLIP_RANGE          = 0.2
ENT_COEF            = 0.1

# --- Online Training ---
TRAIN_EVERY_STEPS   = 512
CHECKPOINT_EVERY    = 3600
REPLAY_BUFFER_SIZE  = 50_000

# --- Risk / Circuit Breaker ---
MAX_DRAWDOWN_PCT     = 0.80
DAILY_LOSS_LIMIT_PCT = 0.50
MAX_CONSECUTIVE_LOSS = 20

# --- Monitoring ---
DASHBOARD_PORT      = 8080

# --- Paths ---
LOG_DIR             = "logs/"
CHECKPOINT_DIR      = "checkpoints/"
DATA_DIR            = "data/"
EOF