# 🤖 ML Trading Bot — Self-Learning Paper Trader

A Reinforcement Learning agent that trades BTC/USDT on 5-minute candles,
learns continuously, and runs 24/7 on your VPS.

---

## Project Structure

```
trading-bot/
├── config.py            ← All settings (API keys, parameters)
├── binance_stream.py    ← Live data from Binance WebSocket
├── feature_engine.py    ← 37-feature signal pipeline
├── trading_env.py       ← Paper trading Gymnasium environment
├── reward.py            ← Reward shaping (what "good trading" means)
├── online_trainer.py    ← Main forever-training loop (entry point)
├── monitor.py           ← Telegram alerts + web dashboard
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Step 1 — Configure

Edit `config.py`:

```python
BINANCE_API_KEY    = "your_key"        # Read-only key is fine (paper trading)
BINANCE_API_SECRET = "your_secret"
TRADING_PAIR       = "BTCUSDT"
```

---

## Step 2 — Deploy on Hostinger VPS

SSH into your VPS:
```bash
ssh root@your_vps_ip
```

Install Docker:
```bash
curl -fsSL https://get.docker.com | sh
apt install docker-compose -y
```

Upload project (from your local machine):
```bash
scp -r trading-bot/ root@your_vps_ip:/root/
```

Or clone from GitHub if you push it there:
```bash
git clone https://github.com/you/trading-bot.git
cd trading-bot
```

Build and start:
```bash
cd trading-bot
docker-compose up -d --build
```

---

## Step 3 — Monitor

View live logs:
```bash
docker-compose logs -f
```

Open dashboard in browser:
```
http://YOUR_VPS_IP:8080
```

The dashboard auto-refreshes every 10 seconds and shows:
- Live balance and total P&L
- Win rate, drawdown, trade count
- Equity curve chart
- Recent trades log
- Agent status and circuit breaker state

---

## Step 4 — Managing the Bot

Stop:
```bash
docker-compose down
```

Restart (after config changes):
```bash
docker-compose restart
```

View resource usage:
```bash
docker stats ml-trading-bot
```

The model auto-saves to `checkpoints/` every hour.
If the bot restarts, it picks up from the latest checkpoint.

---

## How the Learning Works

```
Day 1-3:   Random actions. Loses paper money. Learning cost structure.
Week 1-2:  Starts avoiding obvious mistakes. Losses shrink.
Week 2-4:  Finds repeating patterns. Win rate climbs.
Month 2+:  Adapts to market regimes. Becoming consistent.
```

The agent is never "done training" — it always improves.

---

## Binance API Key Setup

For paper trading, create a **read-only** API key on Binance:
1. Binance → Profile → API Management
2. Create API → Enable "Read Info" only
3. DISABLE "Enable Trading" and "Enable Withdrawals"
4. Restrict to your VPS IP for security

This is safe — even if the key leaks, no trades can be made.

---

## Circuit Breakers

The bot automatically pauses trading (but keeps learning) if:
- Paper balance drops 20% from peak
- Daily loss exceeds 5%
- 10 consecutive losing trades

You'll get a Telegram alert. The model keeps training on data.
Resume by restarting: `docker-compose restart`

---

## Going Live (Later)

When the paper results look consistently profitable over 2-3 months:
1. Create a trading-enabled Binance API key
2. Set `LIVE_TRADING = True` in config.py (not yet implemented — add when ready)
3. Start with very small position sizes
4. Keep monitoring closely

**Do not go live early. Paper trade for months.**