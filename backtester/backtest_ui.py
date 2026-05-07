#!/usr/bin/env python3
# ============================================================
#  backtest_ui.py — Backtest Dashboard (port 8081)
#  Usage: python3 backtest_ui.py
#  Shows live training stats from logs/backtest_stats.json
# ============================================================

import os
import json
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import LOG_DIR, CHECKPOINT_DIR

STATS_FILE = os.path.join(LOG_DIR, "backtest_stats.json")
LIVE_CKPT  = os.path.join(CHECKPOINT_DIR, "transfer_to_live.zip")
LIVE_BOT   = "/root/trading-bot/checkpoints"
PORT       = 8081


def load_stats():
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Backtest Dashboard</title>
<meta http-equiv="refresh" content="10">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; padding: 16px; }
  h1 { color: #58a6ff; font-size: 1.4em; margin-bottom: 4px; }
  .sub { color: #8b949e; font-size: 0.8em; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .label { color: #8b949e; font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px; }
  .value { font-size: 1.6em; font-weight: bold; margin-top: 4px; }
  .green { color: #3fb950; }
  .red   { color: #f85149; }
  .blue  { color: #58a6ff; }
  .gold  { color: #d29922; }
  .section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .section h2 { color: #58a6ff; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  .row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 0.85em; }
  .row:last-child { border-bottom: none; }
  canvas { width: 100%; height: 200px; }
  .btn { background: #238636; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9em; width: 100%; margin-top: 8px; }
  .btn:hover { background: #2ea043; }
  .btn-red { background: #da3633; }
  .progress { background: #21262d; border-radius: 4px; height: 8px; margin-top: 6px; }
  .progress-bar { background: #58a6ff; height: 8px; border-radius: 4px; transition: width 0.3s; }
</style>
</head>
<body>

<h1>BTC Backtest — Training Dashboard</h1>
<div class="sub">Port 8081 | Auto-refreshes every 10s | {updated}</div>

<div class="grid">
  <div class="card">
    <div class="label">Episode</div>
    <div class="value blue">{episode}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate (Episode)</div>
    <div class="value {wr_color}">{ep_win_rate}%</div>
  </div>
  <div class="card">
    <div class="label">Episode PnL</div>
    <div class="value {pnl_color}">${ep_pnl}</div>
  </div>
  <div class="card">
    <div class="label">Speed</div>
    <div class="value blue">{speed} c/s</div>
  </div>
  <div class="card">
    <div class="label">All-Time Win Rate</div>
    <div class="value {all_wr_color}">{all_win_rate}%</div>
  </div>
  <div class="card">
    <div class="label">Big Wins (&gt;$20)</div>
    <div class="value gold">{big_wins}</div>
  </div>
</div>

<div class="section">
  <h2>Win Rate Over Episodes</h2>
  <canvas id="chart"></canvas>
</div>

<div class="section">
  <h2>Journal Stats (All Time)</h2>
  <div class="row"><span>Total Trades</span><span class="blue">{all_trades}</span></div>
  <div class="row"><span>Wins / Losses</span><span>{all_wins} / {all_losses}</span></div>
  <div class="row"><span>All-Time PnL</span><span class="{total_pnl_color}">${all_pnl}</span></div>
  <div class="row"><span>Avg Win</span><span class="green">+${avg_win}</span></div>
  <div class="row"><span>Avg Loss</span><span class="red">${avg_loss}</span></div>
  <div class="row"><span>Best Trade</span><span class="green">+${best_trade}</span></div>
  <div class="row"><span>Worst Trade</span><span class="red">${worst_trade}</span></div>
  <div class="row"><span>Avg Hold (Wins)</span><span class="green">{avg_hold_wins} candles</span></div>
  <div class="row"><span>Avg Hold (Losses)</span><span class="red">{avg_hold_losses} candles</span></div>
  <div class="row"><span>High Priority (&gt;=7)</span><span class="gold">{high_priority}</span></div>
  <div class="row"><span>Phase 3 Status</span><span class="{p3_color}">{p3_status}</span></div>
</div>

<div class="section">
  <h2>Test Set Performance (Unseen Data)</h2>
  {test_html}
</div>

<div class="section">
  <h2>Transfer to Live Bot</h2>
  <div class="row"><span>Best Win Rate</span><span class="gold">{best_info}</span></div>
  <form method="POST" action="/transfer">
    <button class="btn" type="submit">Transfer Best Model to Live Bot</button>
  </form>
  <div style="color:#8b949e;font-size:0.75em;margin-top:6px;">
    Copies best checkpoint to /root/trading-bot/checkpoints/
  </div>
</div>

<script>
const history = {history_json};
if (history.length > 1) {{
  const canvas = document.getElementById('chart');
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = 200;
  const w = canvas.width;
  const h = canvas.height;
  const pad = 30;
  const wrs = history.map(d => d.wr);
  const minWR = Math.min(...wrs, 0);
  const maxWR = Math.max(...wrs, 50);
  const xScale = ep => pad + (ep / (history.length - 1)) * (w - 2*pad);
  const yScale = wr => h - pad - ((wr - minWR) / (maxWR - minWR + 1)) * (h - 2*pad);

  // Grid
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  [0, 25, 50].forEach(wr => {{
    const y = yScale(wr);
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w-pad, y); ctx.stroke();
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.fillText(wr + '%', 2, y + 4);
  }});

  // Win rate line
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((d, i) => {{
    const x = xScale(i);
    const y = yScale(d.wr);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // 40% target line
  ctx.strokeStyle = '#3fb950';
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 1;
  const y40 = yScale(40);
  ctx.beginPath(); ctx.moveTo(pad, y40); ctx.lineTo(w-pad, y40); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#3fb950'; ctx.font = '10px monospace';
  ctx.fillText('Target 40%', w-80, y40-4);
}}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        s    = load_stats()
        hist = s.get("history", [])
        ep   = s.get("episode", 0)
        wr   = s.get("ep_win_rate", 0)
        pnl  = s.get("ep_pnl", 0)
        awr  = s.get("all_win_rate", 0)
        bw   = s.get("big_wins", 0)

        p3_status = f"ACTIVE ({bw} big wins)" if bw >= 10 else f"Warming ({bw}/10 big wins)"
        p3_color  = "green" if bw >= 10 else "gold"

        test = s.get("test", {})
        if test:
            test_html = f"""
            <div class="row"><span>Test Win Rate</span>
              <span class="{'green' if test.get('test_win_rate',0)>=30 else 'red'}">
              {test.get('test_win_rate',0):.1f}%</span></div>
            <div class="row"><span>Test Avg PnL</span>
              <span class="{'green' if test.get('test_avg_pnl',0)>0 else 'red'}">
              ${test.get('test_avg_pnl',0):.2f}</span></div>
            <div class="row"><span>Episodes Tested</span>
              <span>{test.get('test_episodes',0)}</span></div>"""
        else:
            test_html = '<div class="row"><span>No test run yet (runs every 50 episodes)</span></div>'

        html = HTML.format(
            updated        = s.get("last_updated", "—")[:19].replace("T", " "),
            episode        = ep,
            ep_win_rate    = wr,
            ep_pnl         = f"{pnl:+.2f}",
            speed          = int(s.get("speed_cps", 0)),
            all_win_rate   = awr,
            big_wins       = bw,
            all_trades     = s.get("all_trades", 0),
            all_wins       = s.get("all_wins", 0),
            all_losses     = s.get("all_trades", 0) - s.get("all_wins", 0),
            all_pnl        = f"{s.get('all_pnl', 0):+.2f}",
            avg_win        = f"{s.get('avg_win', 0):.2f}",
            avg_loss       = f"{s.get('avg_loss', 0):.2f}",
            best_trade     = f"{s.get('best_trade', 0):.2f}",
            worst_trade    = f"{s.get('worst_trade', 0):.2f}",
            avg_hold_wins  = s.get("avg_hold_wins", 0),
            avg_hold_losses= s.get("avg_hold_losses", 0),
            high_priority  = s.get("high_priority", 0),
            p3_status      = p3_status,
            p3_color       = p3_color,
            best_info      = f"{s.get('ep_win_rate', 0):.1f}% (ep {ep})",
            test_html      = test_html,
            history_json   = json.dumps(hist),
            wr_color       = "green" if wr >= 30 else ("gold" if wr >= 15 else "red"),
            pnl_color      = "green" if pnl > 0 else "red",
            all_wr_color   = "green" if awr >= 30 else ("gold" if awr >= 15 else "red"),
            total_pnl_color= "green" if s.get("all_pnl", 0) > 0 else "red",
        )

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        if self.path == "/transfer":
            _transfer_to_live()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()

    def log_message(self, *args):
        pass   # suppress access logs


def _transfer_to_live():
    if not os.path.exists(LIVE_CKPT):
        return
    os.makedirs(LIVE_BOT, exist_ok=True)
    dest = os.path.join(LIVE_BOT, "backtest_transfer.zip")
    shutil.copy2(LIVE_CKPT, dest)
    print(f"[Transfer] Copied to {dest}")


if __name__ == "__main__":
    print(f"Backtest Dashboard at http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()