"""
Helper functions to integrate Elliott/Fib/Ichimoku into the score.
"""
from __future__ import annotations
from typing import Dict


def advanced_score_boost(pack: Dict, direction: str) -> float:
    """
    Boost/penalty based on Elliott Wave, Fibonacci, and Ichimoku.
    Returns a value between -30 and +30.
    """
    boost = 0.0
    # Elliott Wave
    elliott = pack.get("elliott", {})
    if elliott and elliott.get("wave") != "none":
        ed = elliott.get("details", {})
        is_up = ed.get("is_uptrend", True)
        elliott_base = elliott.get("score", 50) - 50
        if direction == "LONG" and is_up:
            boost += elliott_base * 0.3
        elif direction == "SHORT" and not is_up:
            boost += elliott_base * 0.3
        else:
            boost -= 5
    # Fibonacci
    fib = pack.get("fib", {})
    if fib and fib.get("current_price", 0) > 0:
        from fibonacci import fib_score
        f_score = fib_score(fib) - 50
        if (direction == "LONG" and fib.get("direction") == "up") or \
           (direction == "SHORT" and fib.get("direction") == "down"):
            boost += f_score * 0.2
        if fib.get("distance_to_closest", 100) < 1.5:
            boost += 8
    # Ichimoku
    ichi = pack.get("ichimoku", {})
    if ichi and ichi.get("current_price", 0) > 0:
        from ichimoku import ichimoku_score
        i_score = ichimoku_score(ichi) - 50
        if direction == "LONG" and ichi.get("price_vs_cloud") == "above":
            boost += i_score * 0.25
        elif direction == "SHORT" and ichi.get("price_vs_cloud") == "below":
            boost += i_score * 0.25
        elif direction == "LONG" and ichi.get("price_vs_cloud") == "below":
            boost -= 10
        elif direction == "SHORT" and ichi.get("price_vs_cloud") == "above":
            boost -= 10
        if direction == "LONG" and ichi.get("tk_cross") == "bullish":
            boost += 5
        elif direction == "SHORT" and ichi.get("tk_cross") == "bearish":
            boost += 5
    return max(-30, min(30, boost))
