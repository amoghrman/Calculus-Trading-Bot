# ============================================================
#  monitor.py — Web dashboard only
#
#  Live dashboard at http://YOUR_VPS_IP:8080
#    - Real-time balance & P&L
#    - Equity curve
#    - Win rate, drawdown, trade count
#    - Recent trades log
#    - Agent status & circuit breaker state
# ============================================================

import json
import threading
from datetime import datetime
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import DASHBOARD_PORT


# ----------------------------------------------------------
# Dashboard HTTP Handler
# ----------------------------------------------------------
class DashboardState:
    """Shared state for the dashboard."""
    def __init__(self):
        self.balance      = 10000.0
        self.total_pnl    = 0.0
        self.trade_count  = 0
        self.win_rate     = 0.0
        self.drawdown     = 0.0
        self.episode      = 0
        self.avg_reward   = 0.0
        self.equity_curve = deque(maxlen=500)
        self.recent_trades= deque(maxlen=50)
        self.last_update  = datetime.utcnow().isoformat()
        self.status       = "Starting..."

_state = DashboardState()


class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass   # Suppress request logs

    def do_GET(self):
        if self.path == "/api/state":
            self._json_response({
                "balance":      _state.balance,
                "total_pnl":    _state.total_pnl,
                "trade_count":  _state.trade_count,
                "win_rate":     _state.win_rate,
                "drawdown":     _state.drawdown,
                "episode":      _state.episode,
                "avg_reward":   _state.avg_reward,
                "equity_curve": list(_state.equity_curve),
                "recent_trades":list(_state.recent_trades),
                "last_update":  _state.last_update,
                "status":       _state.status,
            })
        else:
            self._html_response(self._build_html())

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _build_html(self):
        return """<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1   { color: #58a6ff; }
        .card { background: #161b22; border: 1px solid #30363d;
                border-radius: 8px; padding: 16px; margin: 10px 0; }
        .green { color: #3fb950; }
        .red   { color: #f85149; }
        .grid  { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .metric { text-align: center; }
        .label  { color: #8b949e; font-size: 12px; }
        .value  { font-size: 24px; font-weight: bold; }
        canvas  { width: 100%; height: 200px; }
    </style>
</head>
<body>
    <h1>🤖 ML Trading Bot — Live Dashboard</h1>
    <div id="status" class="card"></div>
    <div class="grid" id="metrics"></div>
    <div class="card">
        <h3>Equity Curve</h3>
        <canvas id="chart"></canvas>
    </div>
    <div class="card">
        <h3>Recent Trades</h3>
        <div id="trades"></div>
    </div>

<script>
async function update() {
    const r = await fetch('/api/state');
    const d = await r.json();

    document.getElementById('status').innerHTML =
        `<b>Status:</b> ${d.status} &nbsp;|&nbsp; <b>Last update:</b> ${d.last_update} UTC`;

    const pnlColor = d.total_pnl >= 0 ? 'green' : 'red';
    document.getElementById('metrics').innerHTML = `
        <div class="card metric">
            <div class="label">Balance (Paper USDT)</div>
            <div class="value">$${d.balance.toFixed(2)}</div>
        </div>
        <div class="card metric">
            <div class="label">Total P&L</div>
            <div class="value ${pnlColor}">${d.total_pnl >= 0 ? '+' : ''}$${d.total_pnl.toFixed(2)}</div>
        </div>
        <div class="card metric">
            <div class="label">Win Rate</div>
            <div class="value">${(d.win_rate * 100).toFixed(1)}%</div>
        </div>
        <div class="card metric">
            <div class="label">Total Trades</div>
            <div class="value">${d.trade_count}</div>
        </div>
        <div class="card metric">
            <div class="label">Current Drawdown</div>
            <div class="value red">${(d.drawdown * 100).toFixed(2)}%</div>
        </div>
        <div class="card metric">
            <div class="label">Episodes</div>
            <div class="value">${d.episode}</div>
        </div>
    `;

    // Draw equity curve
    const curve = d.equity_curve;
    if (curve.length > 1) {
        const canvas = document.getElementById('chart');
        const ctx    = canvas.getContext('2d');
        canvas.width  = canvas.offsetWidth;
        canvas.height = 200;
        const min = Math.min(...curve);
        const max = Math.max(...curve);
        const range = max - min || 1;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        curve.forEach((v, i) => {
            const x = (i / (curve.length - 1)) * canvas.width;
            const y = canvas.height - ((v - min) / range) * canvas.height;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // Trades
    const trades = d.recent_trades.slice().reverse();
    document.getElementById('trades').innerHTML = trades.map(t =>
        `<div style="border-bottom:1px solid #30363d;padding:4px 0">
            <span class="${t.pnl >= 0 ? 'green' : 'red'}">${t.side.toUpperCase()}</span>
            &nbsp; Price: ${t.price?.toFixed(2)} &nbsp;|&nbsp;
            PnL: ${t.pnl >= 0 ? '+' : ''}${t.pnl?.toFixed(2)}
            &nbsp; @ Step ${t.step}
        </div>`
    ).join('');
}
update();
setInterval(update, 10000);
</script>
</body>
</html>"""


# ----------------------------------------------------------
# Main Monitor Class
# ----------------------------------------------------------
class Monitor:

    def __init__(self):
        self.state = _state
        self._start_dashboard()

    def _start_dashboard(self):
        def _run():
            server = HTTPServer(("0.0.0.0", DASHBOARD_PORT), DashboardHandler)
            print(f"[Monitor] Dashboard running at http://0.0.0.0:{DASHBOARD_PORT}")
            server.serve_forever()
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def send_message(self, text: str):
        """Logs to console only."""
        print(f"[Monitor] {text}")

    def send_episode_summary(
        self, episode, avg_reward, balance,
        total_pnl, trades, win_rate, drawdown
    ):
        self.state.episode     = episode
        self.state.avg_reward  = avg_reward
        self.state.balance     = balance
        self.state.total_pnl   = total_pnl
        self.state.trade_count = trades
        self.state.win_rate    = win_rate
        self.state.drawdown    = drawdown
        self.state.last_update = datetime.utcnow().isoformat()
        self.state.equity_curve.append(balance)
        self.state.status      = "Trading & Learning"
        print(
            f"[Monitor] Ep {episode} | Balance: ${balance:.2f} | "
            f"PnL: {'+' if total_pnl >= 0 else ''}${total_pnl:.2f} | "
            f"Win: {win_rate*100:.1f}% | DD: {drawdown*100:.2f}%"
        )

    def log_trade(self, trade: dict):
        self.state.recent_trades.append(trade)
        side  = trade.get("side", "?").upper()
        price = trade.get("price", 0)
        pnl   = trade.get("pnl", 0)
        if pnl is not None:
            print(f"[Trade] {side} @ ${price:.2f} | PnL: {'+' if pnl >= 0 else ''}${pnl:.2f}")

    def circuit_breaker_alert(self, reason: str, balance: float, drawdown: float):
        self.state.status = f"⛔ PAUSED — {reason}"
        print(
            f"[CIRCUIT BREAKER] {reason} | "
            f"Balance: ${balance:.2f} | Drawdown: {drawdown*100:.2f}%"
        )