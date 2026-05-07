#!/bin/bash
# ============================================================
#  copy_to_live.sh
#  Copies best backtest model to live bot and restarts it
#  Usage: bash copy_to_live.sh
# ============================================================

BACKTEST_CKPT="/root/btc-backtest/checkpoints/transfer_to_live.zip"
LIVE_DIR="/root/trading-bot/checkpoints"
LIVE_COMPOSE="/root/trading-bot"

echo "=== Transfer Backtest Model to Live Bot ==="

if [ ! -f "$BACKTEST_CKPT" ]; then
    echo "ERROR: No transfer checkpoint found at $BACKTEST_CKPT"
    echo "       Run backtest_runner.py first and wait for a big win."
    exit 1
fi

echo "Source: $BACKTEST_CKPT"
echo "Dest:   $LIVE_DIR/"

# Stop live bot
echo "Stopping live bot..."
cd "$LIVE_COMPOSE" && docker-compose down

# Copy checkpoint
mkdir -p "$LIVE_DIR"
cp "$BACKTEST_CKPT" "$LIVE_DIR/backtest_transfer.zip"
echo "Copied checkpoint."

# Rename to what the live bot expects
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp "$BACKTEST_CKPT" "$LIVE_DIR/agent_${TIMESTAMP}.zip"
echo "Also saved as agent_${TIMESTAMP}.zip"

# Restart live bot
echo "Starting live bot..."
cd "$LIVE_COMPOSE" && docker-compose up -d
echo ""
echo "Done! Live bot restarted with backtest model."
echo "Check logs: docker-compose -f $LIVE_COMPOSE/docker-compose.yml logs -f"