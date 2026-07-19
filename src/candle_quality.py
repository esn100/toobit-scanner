"""
Candle quality analysis (Layer 3 of the PumpHunter pipeline).

For each bar we look at:
  - body ratio (|close-open| / range)
  - wick lengths (upper and lower)
  - close position within the bar (1 = top, 0 = bottom)
  - consecutive power candles (large body + close near high)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _bar_metrics(row) -> dict:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    body_ratio = body / rng
    upper_wick = (h - max(c, o)) / rng
    lower_wick = (min(c, o) - l) / rng
    close_position = ((c - l) / rng) if rng > 0 else 0.5
    return {
        "body_ratio": float(body_ratio),
        "upper_wick": float(upper_wick),
        "lower_wick": float(lower_wick),
        "close_position": float(close_position),
    }


def candle_quality_features(df: pd.DataFrame) -> dict:
    """
    Aggregate candle-quality features for the last few bars.
    """
    if df.empty or len(df) < 3:
        return {"candle_strength": 0.5, "candle_score": 50.0,
                "big_wick_top": False, "bullish_close": False,
                "power_streak": 0}
    last = _bar_metrics(df.iloc[-1])
    # Power candle: body_ratio > 0.65, close_position > 0.7, upper_wick small
    bullish_close = last["close_position"] > 0.6
    big_wick_top = last["upper_wick"] > 0.45
    # Power streak: count of recent bars that qualify as power candles
    streak = 0
    for i in range(-1, -min(6, len(df)) - 1, -1):
        m = _bar_metrics(df.iloc[i])
        if m["body_ratio"] > 0.6 and m["close_position"] > 0.6 and m["upper_wick"] < 0.3:
            streak += 1
        else:
            break
    # Aggregate strength = mean body_ratio of last 3 bars
    recent = [_bar_metrics(df.iloc[i]) for i in range(-3, 0)]
    avg_body = float(np.mean([m["body_ratio"] for m in recent]))
    avg_close_pos = float(np.mean([m["close_position"] for m in recent]))
    strength = 0.5 * avg_body + 0.5 * avg_close_pos
    score = 50.0 + 50.0 * (strength - 0.5)
    if big_wick_top:
        score -= 15.0
    if streak >= 2:
        score += 8.0
    return {
        "candle_strength": float(strength),
        "candle_score": float(max(0.0, min(100.0, score))),
        "big_wick_top": bool(big_wick_top),
        "bullish_close": bool(bullish_close),
        "power_streak": int(streak),
        "last_body_ratio": last["body_ratio"],
        "last_upper_wick": last["upper_wick"],
    }
