"""
Hybrid exit strategy: best of both worlds.

Combines:
  1. Smart exit (breakeven, locks, trail) for safety
  2. Larger TP target (8-10% instead of 5%) for more profit
  3. Tighter SL initially (2% instead of 3%) for protection
  4. Time-based exit at 8h (instead of 12h) to avoid prolonged drawdown
  5. Optional: scale out 50% at TP1, hold 50% for TP2

Rules:
  - Entry: open position
  - +1.5%: SL → entry (breakeven)
  - +3%:   SL → entry+1% (lock small profit)
  - +5%:   SL → entry+2.5% (lock bigger)
  - +8%:   SL → entry+4% (lock most)
  - TP1:   +6% → exit 50%, SL moves to entry+3% on remaining 50%
  - TP2:   +10% → exit remaining 50%
  - SL:    -2% (initial, tight)
  - Time:  8 hours max hold
"""
from __future__ import annotations
from typing import Dict, Tuple


def hybrid_exit_logic(
    direction: str,
    current_pct: float,
    current_sl: float,
    highest_pct: float,
    entry_price: float,
    current_price: float,
    tp1_pct: float = 6.0,
    tp2_pct: float = 10.0,
    sl_pct: float = 2.0,
    max_hours: float = 8.0,
    hours_held: float = 0.0,
) -> Tuple[float, str, float]:
    """
    Compute the new SL and exit signal based on hybrid strategy.
    Returns (new_sl, exit_reason, new_tp2_target).
    """
    if direction == "LONG":
        tp1_price = entry_price * (1 + tp1_pct / 100)
        tp2_price = entry_price * (1 + tp2_pct / 100)
        initial_sl = entry_price * (1 - sl_pct / 100)
        # Stage 1: breakeven at +1.5%
        if current_pct >= 1.5 and current_sl < entry_price:
            return entry_price, "PROGRESS", tp2_price
        # Stage 2: lock 1% profit at +3%
        if current_pct >= 3.0:
            target_sl = entry_price * 1.01
            if current_sl < target_sl:
                return target_sl, "PROGRESS", tp2_price
        # Stage 3: TP1 hit at +6% — exit 50% (we'll handle this in tracker)
        if current_pct >= 6.0:
            target_sl = entry_price * 1.025
            if current_sl < target_sl:
                return target_sl, "TP1_HIT", tp2_price
        # Stage 4: lock 4% profit at +8%
        if current_pct >= 8.0:
            target_sl = entry_price * 1.04
            if current_sl < target_sl:
                return target_sl, "PROGRESS", tp2_price
        # Stage 5: trail at 2% below highest after +5%
        if current_pct >= 5.0 and highest_pct > 5.0:
            trail_price = current_price * 0.98
            if trail_price > current_sl:
                return trail_price, "TRAILING", tp2_price
    else:  # SHORT
        tp1_price = entry_price * (1 - tp1_pct / 100)
        tp2_price = entry_price * (1 - tp2_pct / 100)
        initial_sl = entry_price * (1 + sl_pct / 100)
        if current_pct >= 1.5 and current_sl > entry_price:
            return entry_price, "PROGRESS", tp2_price
        if current_pct >= 3.0:
            target_sl = entry_price * 0.99
            if current_sl > target_sl:
                return target_sl, "PROGRESS", tp2_price
        if current_pct >= 6.0:
            target_sl = entry_price * 0.975
            if current_sl > target_sl:
                return target_sl, "TP1_HIT", tp2_price
        if current_pct >= 8.0:
            target_sl = entry_price * 0.96
            if current_sl > target_sl:
                return target_sl, "PROGRESS", tp2_price
        if current_pct >= 5.0 and highest_pct > 5.0:
            trail_price = current_price * 1.02
            if trail_price < current_sl:
                return trail_price, "TRAILING", tp2_price
    return current_sl, "NO_CHANGE", tp2_price


def simulate_hybrid_exit(
    entry_time,
    entry_price: float,
    direction: str,
    price_series,
    tp1_pct: float = 6.0,
    tp2_pct: float = 10.0,
    sl_pct: float = 2.0,
    max_hours: float = 8.0,
) -> Dict:
    """
    Simulate a trade with hybrid exit strategy.

    Returns dict with exit details.
    """
    if direction == "LONG":
        tp1_price = entry_price * (1 + tp1_pct / 100)
        tp2_price = entry_price * (1 + tp2_pct / 100)
        initial_sl = entry_price * (1 - sl_pct / 100)
    else:
        tp1_price = entry_price * (1 - tp1_pct / 100)
        tp2_price = entry_price * (1 - tp2_pct / 100)
        initial_sl = entry_price * (1 + sl_pct / 100)
    current_sl = initial_sl
    highest_pct = 0.0
    lowest_pct = 0.0
    size_remaining = 1.0
    partial_pnl = 0.0
    exit_pct = 0.0
    exit_reason = "TIMEOUT"
    exit_time = None
    tp1_hit = False
    for ts, price in price_series:
        if ts < entry_time:
            continue
        # Calculate hours held
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
        # Check TP2 first (full exit)
        if direction == "LONG" and price >= tp2_price and tp1_hit:
            # Exit remaining 50%
            final_pnl = partial_pnl + 0.5 * current_pct
            exit_pct = final_pnl
            exit_reason = "TP2_HIT"
            exit_time = ts
            break
        elif direction == "SHORT" and price <= tp2_price and tp1_hit:
            final_pnl = partial_pnl + 0.5 * current_pct
            exit_pct = final_pnl
            exit_reason = "TP2_HIT"
            exit_time = ts
            break
        # Check TP1 (50% exit)
        if not tp1_hit:
            if direction == "LONG" and price >= tp1_price:
                # Exit 50%
                partial_pnl += 0.5 * current_pct
                size_remaining = 0.5
                tp1_hit = True
                # Move SL to entry + 3% on remaining
                current_sl = entry_price * 1.03
            elif direction == "SHORT" and price <= tp1_price:
                partial_pnl += 0.5 * current_pct
                size_remaining = 0.5
                tp1_hit = True
                current_sl = entry_price * 0.97
        # Smart SL logic
        if size_remaining > 0:
            new_sl, reason, _ = hybrid_exit_logic(
                direction, current_pct, current_sl, highest_pct,
                entry_price, price,
                tp1_pct=tp1_pct, tp2_pct=tp2_pct, sl_pct=sl_pct,
                max_hours=max_hours, hours_held=hours_held,
            )
            current_sl = new_sl
        # Check SL hit
        if direction == "LONG" and price <= current_sl:
            if tp1_hit:
                final_pnl = partial_pnl + 0.5 * current_pct
            else:
                final_pnl = current_pct
            exit_pct = final_pnl
            if abs(current_sl - entry_price) < entry_price * 0.002:
                exit_reason = "BREAKEVEN_LOCK"
            elif current_sl > initial_sl:
                exit_reason = "PROFIT_LOCK"
            else:
                exit_reason = "SL_HIT"
            exit_time = ts
            break
        elif direction == "SHORT" and price >= current_sl:
            if tp1_hit:
                final_pnl = partial_pnl + 0.5 * current_pct
            else:
                final_pnl = current_pct
            exit_pct = final_pnl
            if abs(current_sl - entry_price) < entry_price * 0.002:
                exit_reason = "BREAKEVEN_LOCK"
            elif current_sl < initial_sl:
                exit_reason = "PROFIT_LOCK"
            else:
                exit_reason = "SL_HIT"
            exit_time = ts
            break
        # Check timeout
        if hours_held >= max_hours:
            if tp1_hit:
                final_pnl = partial_pnl + 0.5 * current_pct
            else:
                final_pnl = current_pct
            exit_pct = final_pnl
            exit_reason = "TIMEOUT"
            exit_time = ts
            break
    if exit_time is None:
        if price_series:
            last_ts, last_price = price_series[-1]
            if direction == "LONG":
                final_pct = (last_price - entry_price) / entry_price * 100
            else:
                final_pct = (entry_price - last_price) / entry_price * 100
            if tp1_hit:
                exit_pct = partial_pnl + 0.5 * final_pct
            else:
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
        "tp1_hit": tp1_hit,
        "size_remaining": size_remaining,
    }
