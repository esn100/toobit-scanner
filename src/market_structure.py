"""
Market Structure analysis (Layer 3 of the PumpHunter pipeline).

Detects:
  - Higher Highs / Higher Lows (uptrend structure)
  - Lower Highs / Lower Lows (downtrend structure)
  - Break of Structure (BOS) - close beyond a recent swing
  - Range / consolidation state

The output is a dict of booleans + scores that feeds into both the
rule-based scorer and the ML model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple


def find_swing_points(
    series: pd.Series, order: int = 5
) -> Tuple[List[int], List[int]]:
    """
    Find local maxima and minima indices in a series.
    A point at i is a swing high if it's the max in [i-order..i+order].
    """
    vals = series.values
    n = len(vals)
    highs: List[int] = []
    lows: List[int] = []
    for i in range(order, n - order):
        window = vals[i - order:i + order + 1]
        if vals[i] == window.max():
            highs.append(i)
        if vals[i] == window.min():
            lows.append(i)
    return highs, lows


def structure_features(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Compute market structure features for the last `lookback` bars.
    """
    out = {
        "higher_highs": False,
        "higher_lows": False,
        "lower_highs": False,
        "lower_lows": False,
        "bos_up": False,
        "bos_down": False,
        "in_range": False,
        "structure_score": 50.0,
    }
    if len(df) < 20:
        return out
    sub = df.tail(lookback).reset_index(drop=True)
    high_idx, low_idx = find_swing_points(sub["high"], order=3)
    _, low_idx2 = find_swing_points(sub["low"], order=3)

    # Use closes to determine BOS (Break of Structure)
    if len(sub) >= 20:
        recent_high = float(sub["high"].tail(20).max())
        prior_high = float(sub["high"].iloc[:-20].max()) if len(sub) > 20 else recent_high
        recent_low = float(sub["low"].tail(20).min())
        prior_low = float(sub["low"].iloc[:-20].min()) if len(sub) > 20 else recent_low
        last_close = float(sub["close"].iloc[-1])
        out["bos_up"] = bool(last_close > prior_high and recent_high > prior_high)
        out["bos_down"] = bool(last_close < prior_low and recent_low < prior_low)

    # Higher highs / higher lows over the last few swing points
    if len(high_idx) >= 3:
        h_vals = [float(sub["high"].iloc[i]) for i in high_idx[-3:]]
        if h_vals[0] < h_vals[1] < h_vals[2]:
            out["higher_highs"] = True
        elif h_vals[0] > h_vals[1] > h_vals[2]:
            out["lower_highs"] = True
    if len(low_idx2) >= 3:
        l_vals = [float(sub["low"].iloc[i]) for i in low_idx2[-3:]]
        if l_vals[0] < l_vals[1] < l_vals[2]:
            out["higher_lows"] = True
        elif l_vals[0] > l_vals[1] > l_vals[2]:
            out["lower_lows"] = True

    # Range detection: small range over last 20 bars
    if len(sub) >= 20:
        rng = float(sub["high"].tail(20).max() - sub["low"].tail(20).min())
        avg_price = float(sub["close"].tail(20).mean())
        rng_pct = (rng / max(avg_price, 1e-12)) * 100.0
        out["range_pct"] = rng_pct
        out["in_range"] = bool(rng_pct < 6.0)

    # Score
    score = 50.0
    if out["higher_highs"] and out["higher_lows"]:
        score = 90.0  # perfect uptrend structure
    elif out["lower_lows"]:
        score = 15.0
    if out["bos_up"]:
        score = max(score, 85.0)
    elif out["bos_down"]:
        score = min(score, 15.0)
    if out["in_range"]:
        score = 50.0  # neutral
    out["structure_score"] = float(max(0.0, min(100.0, score)))
    return out
