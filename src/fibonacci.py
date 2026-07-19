"""
Fibonacci retracement and extension levels.

For each significant swing, compute fib levels. When price is near
a key fib level (0.382, 0.5, 0.618, 0.786), it can act as support
or resistance. We also compute fib extensions (1.272, 1.618, 2.618)
for target projections.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# Key Fibonacci ratios
RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
EXTENSION_LEVELS = [1.0, 1.272, 1.618, 2.0, 2.618]


def find_swings(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float, str]:
    """
    Find the most recent significant swing high and low.
    Returns (high, low, direction).
    """
    if df.empty or len(df) < 10:
        return 0.0, 0.0, "none"
    sub = df.tail(lookback)
    high = float(sub["high"].max())
    low = float(sub["low"].min())
    # Direction: where is the latest close?
    last_close = float(sub["close"].iloc[-1])
    midpoint = (high + low) / 2
    direction = "up" if last_close > midpoint else "down"
    return high, low, direction


def compute_fib_levels(
    df: pd.DataFrame, lookback: int = 50
) -> Dict:
    """
    Compute Fibonacci retracement levels from the most recent swing.
    """
    high, low, direction = find_swings(df, lookback)
    if high <= 0 or low <= 0 or high == low:
        return {"levels": {}, "direction": direction,
                "high": high, "low": low, "closest_level": None}
    diff = high - low
    # If uptrend (price moved up), retracement goes down from high
    # If downtrend, retracement goes up from low
    levels = {}
    if direction == "up":
        for r in RETRACEMENT_LEVELS:
            levels[f"fib_{r:.3f}"] = high - diff * r
    else:
        for r in RETRACEMENT_LEVELS:
            levels[f"fib_{r:.3f}"] = low + diff * r
    # Current price vs levels
    current_price = float(df["close"].iloc[-1])
    closest = None
    closest_dist = float("inf")
    for name, price in levels.items():
        d = abs(price - current_price) / current_price
        if d < closest_dist:
            closest_dist = d
            closest = name
    # Compute extensions
    extensions = {}
    if direction == "up":
        # Extension above the high
        for r in EXTENSION_LEVELS:
            extensions[f"ext_{r:.3f}"] = high + diff * r
    else:
        # Extension below the low
        for r in EXTENSION_LEVELS:
            extensions[f"ext_{r:.3f}"] = low - diff * r
    return {
        "levels": levels,
        "extensions": extensions,
        "direction": direction,
        "high": high,
        "low": low,
        "current_price": current_price,
        "closest_level": closest,
        "distance_to_closest": round(closest_dist * 100, 3),
    }


def fib_score(fib_data: Dict, tolerance_pct: float = 1.5) -> float:
    """
    Score 0-100 based on proximity to key fib levels.
    Price near 0.382/0.5/0.618 retracement is a strong signal.
    """
    if not fib_data or not fib_data.get("levels"):
        return 50.0
    current = fib_data.get("current_price", 0)
    levels = fib_data["levels"]
    if current <= 0:
        return 50.0
    # Key levels: 0.382, 0.5, 0.618 (golden pocket)
    key_levels = ["fib_0.382", "fib_0.500", "fib_0.618"]
    score = 0.0
    for name in key_levels:
        if name not in levels:
            continue
        level_price = levels[name]
        dist_pct = abs(level_price - current) / current * 100
        if dist_pct <= tolerance_pct:
            # Strong: very close to key fib
            score = max(score, 100.0 - (dist_pct / tolerance_pct) * 30)
        elif dist_pct <= tolerance_pct * 2:
            score = max(score, 70.0 - (dist_pct - tolerance_pct) * 5)
    if score == 0:
        # Check if price is between two key levels (in the "golden pocket")
        if "fib_0.500" in levels and "fib_0.618" in levels:
            mid = (levels["fib_0.500"] + levels["fib_0.618"]) / 2
            dist_pct = abs(mid - current) / current * 100
            if dist_pct <= tolerance_pct:
                score = 80.0
    return min(100.0, max(0.0, score))


def fib_extension_target(fib_data: Dict, prefer: str = "1.618") -> float:
    """
    Return a price target from Fibonacci extensions.
    prefer can be '1.0', '1.272', '1.618', '2.0', '2.618'.
    """
    if not fib_data or not fib_data.get("extensions"):
        return 0.0
    return float(fib_data["extensions"].get(f"ext_{prefer}", 0.0))
