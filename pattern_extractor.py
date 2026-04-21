# ============================================================
#  pattern_extractor.py — Daily Pattern Analysis
#  Runs at midnight UTC via cron
#  Reads journal → finds patterns → generates report
#  Updates Phase 3 replay weights
# ============================================================

import os
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from journal import TradeJournal, BIG_WIN, SMALL_WIN, SMALL_LOSS, BIG_LOSS
from config import LOG_DIR

REPORT_FILE  = os.path.join(LOG_DIR, "daily_report.txt")
WEIGHTS_FILE = os.path.join(LOG_DIR, "replay_weights.json")
MIN_SAMPLES  = 5   # Minimum trades before calling a pattern


class PatternExtractor:

    def __init__(self):
        self.journal = TradeJournal()

    # ----------------------------------------------------------
    # Main entry — called at midnight by cron + by trainer
    # ----------------------------------------------------------
    def run_daily_analysis(self):
        print("[Extractor] Starting daily pattern analysis...")
        all_trades   = self.journal.load_all()
        today_trades = self.journal.load_today()

        if len(all_trades) < MIN_SAMPLES:
            print(f"[Extractor] Only {len(all_trades)} trades. Need {MIN_SAMPLES}+.")
            return None, {}

        report  = self._generate_report(all_trades, today_trades)
        weights = self._compute_replay_weights(all_trades)

        # Save report
        with open(REPORT_FILE, "w") as f:
            f.write(report)

        # Save weights for Phase 3 trainer
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)

        print(report)
        print(f"[Extractor] Report → {REPORT_FILE}")
        print(f"[Extractor] Weights → {WEIGHTS_FILE}")
        return report, weights

    # ----------------------------------------------------------
    # Human-readable report
    # ----------------------------------------------------------
    def _generate_report(self, all_trades, today_trades):
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Today stats
        wins   = [t for t in today_trades if t["pnl"] > 0]
        losses = [t for t in today_trades if t["pnl"] <= 0]
        total  = len(today_trades)
        wr     = len(wins) / total * 100 if total > 0 else 0
        net    = sum(t["pnl"] for t in today_trades)
        avg_w  = np.mean([t["pnl"] for t in wins])   if wins   else 0
        avg_l  = np.mean([t["pnl"] for t in losses]) if losses else 0

        # Win patterns from all history
        win_patterns  = self._find_patterns(all_trades, target="win")
        lose_patterns = self._find_patterns(all_trades, target="loss")

        # Holding time analysis
        w_holds = [t["candles_held"] for t in all_trades if t["pnl"] > 0]
        l_holds = [t["candles_held"] for t in all_trades if t["pnl"] <= 0]

        # Time of day analysis
        hour_stats = self._hour_analysis(all_trades)

        # All time stats
        all_wins = [t for t in all_trades if t["pnl"] > 0]
        all_wr   = len(all_wins) / len(all_trades) * 100 if all_trades else 0

        sep = "=" * 55

        r = f"""
{sep}
  ML TRADING BOT — DAILY PATTERN REPORT
  {now}
{sep}

TODAY'S PERFORMANCE:
  Trades:     {total}
  Wins:       {len(wins)} ({wr:.1f}%)
  Losses:     {len(losses)} ({100-wr:.1f}%)
  Net PnL:    ${net:+.2f}
  Avg win:    ${avg_w:+.2f}
  Avg loss:   ${avg_l:+.2f}
"""
        if today_trades:
            best  = max(today_trades, key=lambda t: t["pnl"])
            worst = min(today_trades, key=lambda t: t["pnl"])
            r += f"  Best:       ${best['pnl']:+.2f} (held {best['candles_held']} candles)\n"
            r += f"  Worst:      ${worst['pnl']:+.2f} (held {worst['candles_held']} candles)\n"

        r += f"""
{sep}
  WINNING PATTERNS (from all {len(all_trades)} historical trades):
{sep}
"""
        if win_patterns:
            for i, p in enumerate(win_patterns[:3], 1):
                conf = "HIGH ✅" if p["samples"] >= 15 else \
                       "MODERATE ⚠️" if p["samples"] >= 5 else "LOW ❌"
                r += f"""
Pattern W{i} — "{p['name']}"
  Samples:    {p['samples']} trades
  Win rate:   {p['win_rate']:.1f}%
  Avg PnL:    ${p['avg_pnl']:+.2f}
  Confidence: {conf}
  Signal:     {p['signal']}
  Replay:     {p['weight']}x priority in Phase 3
"""
        else:
            r += "  Not enough data yet.\n"

        r += f"""
{sep}
  LOSING PATTERNS (AVOID THESE):
{sep}
"""
        if lose_patterns:
            for i, p in enumerate(lose_patterns[:3], 1):
                conf = "HIGH ✅" if p["samples"] >= 15 else \
                       "MODERATE ⚠️" if p["samples"] >= 5 else "LOW ❌"
                r += f"""
Pattern L{i} — "{p['name']}"
  Samples:    {p['samples']} trades
  Loss rate:  {p['loss_rate']:.1f}%
  Avg PnL:    ${p['avg_pnl']:+.2f}
  Confidence: {conf}
  Signal:     {p['signal']}
  Avoid:      {p['weight']}x negative replay in Phase 3
"""
        else:
            r += "  Not enough data yet.\n"

        r += f"""
{sep}
  HOLDING TIME ANALYSIS:
{sep}
  Avg candles held — WINS:   {np.mean(w_holds):.1f if w_holds else 'N/A'}
  Avg candles held — LOSSES: {np.mean(l_holds):.1f if l_holds else 'N/A'}
"""
        if w_holds and l_holds:
            if np.mean(w_holds) > np.mean(l_holds):
                r += "  Agent holds winners longer than losers ✅\n"
            else:
                r += "  Agent cutting winners too early ❌\n"

        r += f"""
{sep}
  TIME OF DAY ANALYSIS (UTC):
{sep}
"""
        best_h, worst_h = hour_stats
        if best_h:
            r += "  Best hours:\n"
            for h, wr_h, pnl_h, n in best_h[:3]:
                r += f"    {h:02d}:00 — {wr_h:.0f}% wins | ${pnl_h:+.2f} net | {n} trades\n"
        if worst_h:
            r += "  Worst hours:\n"
            for h, wr_h, pnl_h, n in worst_h[:3]:
                r += f"    {h:02d}:00 — {wr_h:.0f}% wins | ${pnl_h:+.2f} net | {n} trades\n"

        r += f"""
{sep}
  OVERALL PROGRESS:
{sep}
  All-time win rate:  {all_wr:.1f}%
  Total trades:       {len(all_trades)}
  Target win rate:    40.0%
  Gap remaining:      {max(0, 40 - all_wr):.1f}%

  Phase 3 Status:
  High priority trades (>=7): {len([t for t in all_trades if t.get('priority',0) >= 7])}
  Medium priority (4-6):      {len([t for t in all_trades if 4 <= t.get('priority',0) < 7])}
  Low priority (<4):          {len([t for t in all_trades if t.get('priority',0) < 4])}
{sep}
"""
        return r

    # ----------------------------------------------------------
    # Find statistically significant patterns
    # ----------------------------------------------------------
    def _find_patterns(self, trades, target="win"):
        candidates = [
            ("rsi", -0.3, "below", "RSI Oversold Entry"),
            ("rsi",  0.4, "above", "RSI Overbought (Avoid Long)"),
            ("price_velocity",  0.3, "above", "Positive Momentum Entry"),
            ("price_velocity", -0.3, "below", "Negative Momentum (Falling)"),
            ("price_acceleration", 0.2, "above", "Accelerating Upward"),
            ("ob_imbalance",  0.3, "above", "Buyer Dominated Order Book"),
            ("ob_imbalance", -0.3, "below", "Seller Dominated Order Book"),
            ("entropy",  0.3, "above", "High Chaos (Avoid)"),
            ("entropy", -0.3, "below", "Low Entropy (Predictable Market)"),
            ("hurst",  0.1, "above", "Trending Market"),
            ("zscore", -1.5, "below", "Price Below Mean (Oversold)"),
            ("zscore",  1.5, "above", "Price Above Mean (Overbought)"),
            ("trade_pressure", 0.2, "above", "Buy Pressure Dominant"),
        ]

        patterns = []
        for feat, thresh, direction, name in candidates:
            p = self._analyze_pattern(
                trades, feat, thresh, direction, name, target
            )
            if p:
                patterns.append(p)

        # Sort
        if target == "win":
            patterns.sort(key=lambda x: x["win_rate"], reverse=True)
        else:
            patterns.sort(key=lambda x: x["loss_rate"], reverse=True)

        return patterns

    def _analyze_pattern(self, trades, feature, threshold, direction, name, target):
        matching = []
        for t in trades:
            val = t.get("features", {}).get(feature)
            if val is None:
                continue
            if direction == "above" and val > threshold:
                matching.append(t)
            elif direction == "below" and val < threshold:
                matching.append(t)

        if len(matching) < MIN_SAMPLES:
            return None

        wins      = [t for t in matching if t["pnl"] > 0]
        losses    = [t for t in matching if t["pnl"] <= 0]
        win_rate  = len(wins) / len(matching) * 100
        loss_rate = len(losses) / len(matching) * 100
        avg_pnl   = np.mean([t["pnl"] for t in matching])

        if target == "win"  and win_rate  < 55:
            return None
        if target == "loss" and loss_rate < 60:
            return None

        weight = min(10, max(2, int(
            win_rate / 10 if target == "win" else loss_rate / 10
        )))

        dir_word = ">" if direction == "above" else "<"
        signal = f"{feature} {dir_word} {threshold}"

        return {
            "name":      name,
            "feature":   feature,
            "threshold": threshold,
            "direction": direction,
            "samples":   len(matching),
            "win_rate":  win_rate,
            "loss_rate": loss_rate,
            "avg_pnl":   avg_pnl,
            "weight":    weight,
            "signal":    signal,
        }

    # ----------------------------------------------------------
    # Time of day analysis
    # ----------------------------------------------------------
    def _hour_analysis(self, trades):
        hour_data = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for t in trades:
            try:
                h = int(t["timestamp"][11:13])
                if t["pnl"] > 0:
                    hour_data[h]["wins"] += 1
                else:
                    hour_data[h]["losses"] += 1
                hour_data[h]["pnl"] += t["pnl"]
            except Exception:
                continue

        best, worst = [], []
        for h, d in hour_data.items():
            n = d["wins"] + d["losses"]
            if n < 3:
                continue
            wr = d["wins"] / n * 100
            if wr >= 50:
                best.append((h, wr, d["pnl"], n))
            elif wr <= 25:
                worst.append((h, wr, d["pnl"], n))

        best.sort(key=lambda x: x[1], reverse=True)
        worst.sort(key=lambda x: x[1])
        return best, worst

    # ----------------------------------------------------------
    # Compute Phase 3 replay weights for each trade
    # Key: "episode_step" → Value: priority score
    # ----------------------------------------------------------
    def _compute_replay_weights(self, trades):
        weights = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_trades": len(trades),
            "trades": {}
        }
        for t in trades:
            key = f"{t['episode']}_{t['entry_step']}"
            weights["trades"][key] = {
                "priority": t.get("priority", 1),
                "label":    t.get("label", "UNKNOWN"),
                "pnl":      t.get("pnl", 0),
            }

        # Summary stats
        priorities = [t.get("priority", 1) for t in trades]
        weights["avg_priority"]  = round(np.mean(priorities), 2)
        weights["high_priority_count"] = sum(1 for p in priorities if p >= 7)

        return weights


# ----------------------------------------------------------
# Direct execution for testing / cron
# ----------------------------------------------------------
if __name__ == "__main__":
    extractor = PatternExtractor()
    report, weights = extractor.run_daily_analysis()