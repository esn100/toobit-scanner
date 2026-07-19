"""
Post-signal outcome evaluation (Layer 10 of the PumpHunter pipeline).

Given a past signal and the price action since, compute:
  - peak profit % (max high over the look-ahead window)
  - max drawdown % (min low vs entry)
  - holding time at peak
  - final return %
  - outcome_score (0..100)
  - label (1 = success if outcome_score >= 60, else 0)
"""
from __future__ import annotations
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _entry_price_from_context(row: Dict) -> Optional[float]:
    """We didn't always store entry_price, so fall back to context."""
    return row.get("entry_price")


def compute_outcome_metrics(
    df_after: pd.DataFrame,
    entry_price: float,
    forward_bars: int = 3,  # 3 * 4h = 12 hours
) -> Dict:
    """
    Given a dataframe containing candles from signal-time forward,
    compute peak/drawdown/return metrics.
    """
    if df_after.empty or entry_price <= 0:
        return {
            "peak_profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_return_pct": 0.0,
            "holding_bars": 0,
            "outcome_score": 0.0,
            "label": 0,
        }
    sub = df_after.head(forward_bars)
    if sub.empty:
        return {
            "peak_profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_return_pct": 0.0,
            "holding_bars": 0,
            "outcome_score": 0.0,
            "label": 0,
        }
    highs = sub["high"].astype(float)
    lows = sub["low"].astype(float)
    closes = sub["close"].astype(float)
    peak = float(((highs - entry_price) / entry_price * 100.0).max())
    dd = float(((lows - entry_price) / entry_price * 100.0).min())
    final = float(((closes.iloc[-1] - entry_price) / entry_price * 100.0))
    # Outcome score: blend of peak, final, and drawdown protection
    # - Heavy reward for high peak profit
    # - Penalty for large drawdown
    # - Mild reward for sustained hold
    score = 0.0
    score += min(60.0, max(0.0, peak * 8.0))  # up to 60 pts for 7.5%+ peak
    score += min(25.0, max(0.0, final * 4.0))  # up to 25 pts for 6%+ final
    if dd < -3.0:
        score -= min(20.0, abs(dd) * 3.0)      # up to -20 pts for drawdown
    if peak >= 2.0 and final <= -1.0:
        # Reverted after pump - classic fakeout
        score -= 5.0
    score = max(0.0, min(100.0, score))
    label = 1 if score >= 60 else 0
    return {
        "peak_profit_pct": round(peak, 4),
        "max_drawdown_pct": round(dd, 4),
        "final_return_pct": round(final, 4),
        "holding_bars": int(len(sub)),
        "outcome_score": round(score, 2),
        "label": int(label),
    }


def evaluate_pending_signals(
    history_path: str,
    toobit_client,
    forward_bars: int = 3,
    max_eval: int = 30,
) -> int:
    """
    For every (symbol, timestamp) in the history with no label, fetch
    recent candles, compute outcome, and update the history.
    Returns the count of signals resolved.
    """
    from ml_engine import _load_history, update_signal_outcome
    df = _load_history(history_path)
    if df.empty:
        return 0
    pending = df[df["label"].isna()]
    if pending.empty:
        return 0
    resolved = 0
    for _, row in pending.head(max_eval).iterrows():
        sym = row["symbol"]
        try:
            df_k = toobit_client.get_klines(sym, "4h", forward_bars + 5)
        except Exception:
            continue
        if df_k.empty:
            continue
        entry = float(row.get("entry_price") or 0.0)
        if entry <= 0.0:
            # Without an entry price we cannot evaluate, skip
            continue
        m = compute_outcome_metrics(df_k, entry, forward_bars)
        if update_signal_outcome(
            history_path, sym, row["timestamp"],
            m["outcome_score"], m["label"],
        ):
            resolved += 1
        time.sleep(0.3)
    return resolved
