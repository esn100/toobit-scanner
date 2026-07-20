"""
Smart Exit v2 — the best combination we found.

Based on backtest:
  - Smart exit (locks) is the WIN RATE king
  - Fixed TP/SL is the PROFIT king
  - Hybrid (scaled) was bad because after TP1 hit, the trailing SL on
    remaining 50% got hit frequently

This is a compromise:
  - Start with tight SL (-2%)
  - Lock in profit aggressively (breakeven at +1%, +0.5% at +2%, +2% at +4%)
  - Trail once we have +3%
  - TP at +7% (medium target — not too low, not too high)
  - 8 hour max hold (faster rotation)

This should give win rate ~75% AND reasonable P&L.
"""
from __future__ import annotations
from typing import Dict, Tuple


def smart_exit_v2_logic(
    direction: str,
    current_pct: float,
    current_sl: float,
    highest_pct: float,
    entry_price: float,
    current_price: float,
) -> Tuple[float, str]:
    """
    Smart exit v2 — tighten the locks but keep wider TP.

    Returns (new_sl, reason).
    """
    if direction == "LONG":
        # Stage 1: breakeven at +1.5% (LONG: just protect capital)
        if current_pct >= 1.5 and current_sl < entry_price:
            return entry_price, "BREAKEVEN"
        # Stage 2: lock 1% profit at +3%
        if current_pct >= 3.0:
            target_sl = entry_price * 1.01
            if current_sl < target_sl:
                return target_sl, "LOCK_25"
        # Stage 3: lock 2% profit at +5%
        if current_pct >= 5.0:
            target_sl = entry_price * 1.02
            if current_sl < target_sl:
                return target_sl, "LOCK_50"
        # Stage 4: trail at 1.5% below highest after +4%
        if current_pct >= 4.0 and highest_pct > 4.0:
            trail_price = current_price * 0.985
            if trail_price > current_sl:
                return trail_price, "TRAIL"
    else:  # SHORT
        if current_pct >= 1.5 and current_sl > entry_price:
            return entry_price, "BREAKEVEN"
        if current_pct >= 3.0:
            target_sl = entry_price * 0.99
            if current_sl > target_sl:
                return target_sl, "LOCK_25"
        if current_pct >= 5.0:
            target_sl = entry_price * 0.98
            if current_sl > target_sl:
                return target_sl, "LOCK_50"
        if current_pct >= 4.0 and highest_pct > 4.0:
            trail_price = current_price * 1.015
            if trail_price < current_sl:
                return trail_price, "TRAIL"
    return current_sl, "NO_CHANGE"


def simulate_smart_v2(
    entry_time,
    entry_price: float,
    direction: str,
    price_series,
    tp_pct: float = 7.0,
    sl_pct: float = 2.0,
    max_hours: float = 8.0,
) -> Dict:
    """
    Simulate smart exit v2.
    """
    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct / 100)
        initial_sl = entry_price * (1 - sl_pct / 100)
    else:
        tp_price = entry_price * (1 - tp_pct / 100)
        initial_sl = entry_price * (1 + sl_pct / 100)
    current_sl = initial_sl
    highest_pct = 0.0
    lowest_pct = 0.0
    exit_pct = 0.0
    exit_reason = "TIMEOUT"
    exit_time = None
    for ts, price in price_series:
        if ts < entry_time:
            continue
        hours_held = (ts - entry_time).total_seconds() / 3600
        if direction == "LONG":
            current_pct = (price - entry_price) / entry_price * 100
            high_pct = current_pct
            low_pct = current_pct
        else:
            current_pct = (entry_price - price) / entry_price * 100
            high_pct = current_pct
            low_pct = current_pct
        highest_pct = max(highest_pct, current_pct)
        lowest_pct = min(lowest_pct, current_pct)
        # Smart SL logic
        new_sl, reason = smart_exit_v2_logic(
            direction, current_pct, current_sl, highest_pct,
            entry_price, price,
        )
        current_sl = new_sl
        # Check exit
        if direction == "LONG":
            if price >= tp_price:
                exit_pct = current_pct
                exit_reason = "TP_HIT"
                exit_time = ts
                break
            elif price <= current_sl:
                exit_pct = current_pct
                if abs(current_sl - entry_price) < entry_price * 0.001:
                    exit_reason = "BREAKEVEN_LOCK"
                elif current_sl > initial_sl:
                    exit_reason = "PROFIT_LOCK"
                else:
                    exit_reason = "SL_HIT"
                exit_time = ts
                break
        else:
            if price <= tp_price:
                exit_pct = current_pct
                exit_reason = "TP_HIT"
                exit_time = ts
                break
            elif price >= current_sl:
                exit_pct = current_pct
                if abs(current_sl - entry_price) < entry_price * 0.001:
                    exit_reason = "BREAKEVEN_LOCK"
                elif current_sl < initial_sl:
                    exit_reason = "PROFIT_LOCK"
                else:
                    exit_reason = "SL_HIT"
                exit_time = ts
                break
        # Timeout
        if hours_held >= max_hours:
            if direction == "LONG":
                final_pct = (price - entry_price) / entry_price * 100
            else:
                final_pct = (entry_price - price) / entry_price * 100
            exit_pct = final_pct
            exit_reason = "TIMEOUT"
            exit_time = ts
            break
    if exit_time is None and price_series:
        last_ts, last_price = price_series[-1]
        if direction == "LONG":
            final_pct = (last_price - entry_price) / entry_price * 100
        else:
            final_pct = (entry_price - last_price) / entry_price * 100
        exit_pct = final_pct
        exit_reason = "TIMEOUT"
        exit_time = last_ts
    duration_min = (exit_time - entry_time).total_seconds() / 60 if exit_time else 0
    return {
        "exit_pct": round(exit_pct, 3),
        "exit_reason": exit_reason,
        "duration_min": round(duration_min, 1),
        "max_favorable": round(highest_pct, 3),
        "max_adverse": round(lowest_pct, 3),
    }
