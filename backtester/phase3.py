#!/usr/bin/env python3
# ============================================================
#  phase3.py — Weighted Experience Replay
#  Injects high-priority winning trades into learning
# ============================================================

import numpy as np
import torch


FEATURE_NAMES = [
    "rsi","macd","macd_sig","macd_hist",
    "boll_pos","boll_width","boll_pct","atr","vwap_dev","taker_ratio",
    "price_velocity","price_acceleration","price_jerk",
    "price_vel_avg","price_acc_avg","price_jerk_vol",
    "vol_velocity","vol_acceleration","vol_jerk",
    "vol_vel_avg","vol_acc_avg","vol_jerk_vol",
    "hurst","entropy","zscore","price_vol_corr",
    "ob_spread","ob_imbalance","ob_bid_wall","ob_ask_wall",
    "ob_vol_ratio","ob_spread2","trade_pressure",
    "time_sin_hour","time_cos_hour","time_sin_dow","time_cos_dow",
    "position_held","balance_ratio","steps_held",
    "unrealized_pct","consec_loss",
]


class Phase3Trainer:

    def train(self, model, big_wins: list, lr: float = 1e-4):
        if not big_wins:
            return

        model.policy.set_training_mode(True)

        opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
        losses_done = 0

        for trade in big_wins:
            feats = trade.get("features", {})
            if not feats:
                continue

            obs_list = [feats.get(n, 0.0) for n in FEATURE_NAMES]
            # Pad to 47
            while len(obs_list) < 47:
                obs_list.append(0.0)
            obs_list = obs_list[:47]

            obs_t = torch.tensor([obs_list], dtype=torch.float32)

            # Expected value: normalize pnl to reasonable target
            pnl      = trade.get("pnl", 0.0)
            target_v = torch.tensor([[min(pnl / 100.0, 1.0)]], dtype=torch.float32)

            try:
                _, value, _ = model.policy.evaluate_actions(
                    obs_t, torch.tensor([1])  # action=buy
                )
                loss = torch.nn.functional.mse_loss(value, target_v)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                opt.step()
                losses_done += 1
            except Exception:
                pass

        model.policy.set_training_mode(False)