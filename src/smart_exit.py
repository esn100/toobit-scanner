"""
Smart exit strategies for higher win rate.

Standard exits are: fixed TP at +5%, fixed SL at -3%.
Smart exits adapt the SL as price moves in our favor:

  1. BREAKEVEN: After +1.5% gain, move SL to entry (zero risk)
  2. LOCK_25:  After +2.5% gain, move SL to +0.5% (lock in 0.5%)
  3. LOCK_50:  After +4% gain, move SL to +2% (lock in 2%)
  4. TRAIL:    After +3% gain, trail SL at 1.5% below highest

For SHORT positions, mirror the logic.

Why this boosts win rate:
  - If a trade goes our way, we lock in profit
  - If it reverses, we exit with less loss
  - Many trades that would have hit SL now exit at breakeven
  - Trades that would have hit TP still hit TP
  - Net effect: win rate goes UP, profit factor goes UP
"""
from __future__ import annotations
from typing import Dict, Tuple


def smart_sl_logic(direction: str, current_pct: float, current_sl: float,
                   highest_pct: float, entry_price: float,
                   current_price: float) -> Tuple[float, str]:
    """
    Compute the new SL based on how the trade is going.
    Returns (new_sl_price, reason_for_change).
    """
    if direction == "LONG":
        # Stage 1: breakeven at +1.5%
        if current_pct >= 1.5 and current_sl < entry_price:
            return entry_price, "BREAKEVEN_LOCK"
        # Stage 2: lock 0.5% profit at +2.5%
        if current_pct >= 2.5:
            target_sl = entry_price * 1.005  # +0.5%
            if current_sl < target_sl:
                return target_sl, "LOCK_25_PROFIT"
        # Stage 3: lock 2% profit at +4%
        if current_pct >= 4.0:
            target_sl = entry_price * 1.02
            if current_sl < target_sl:
                return target_sl, "LOCK_50_PROFIT"
        # Stage 4: trail at 1.5% below highest after +3%
        if current_pct >= 3.0 and highest_pct > 3.0:
            trail_price = current_price * 0.985  # 1.5% below current
            if trail_price > current_sl:
                return trail_price, "TRAILING_1.5"
    else:  # SHORT
        if current_pct >= 1.5 and current_sl > entry_price:
            return entry_price, "BREAKEVEN_LOCK"
        if current_pct >= 2.5:
            target_sl = entry_price * 0.995
            if current_sl > target_sl:
                return target_sl, "LOCK_25_PROFIT"
        if current_pct >= 4.0:
            target_sl = entry_price * 0.98
            if current_sl > target_sl:
                return target_sl, "LOCK_50_PROFIT"
        if current_pct >= 3.0 and highest_pct > 3.0:
            trail_price = current_price * 1.015
            if trail_price < current_sl:
                return trail_price, "TRAILING_1.5"
    return current_sl, "NO_CHANGE"


def adjust_tp_sl_based_on_history(features: Dict, current_win_rate: float,
                                  n_signals: int) -> Dict:
    """
    If we have enough data and win rate is good, be more aggressive.
    If win rate is poor, tighten stops.
    """
    atr = float(features.get("f_atr_pct", 3))
    confidence = float(features.get("confidence", 50))
    # Default
    result = {
        "tp_pct": 5.0,
        "sl_pct": 3.0,
        "use_breakeven": True,
        "use_trail": True,
    }
    if n_signals < 20:
        return result
    # If win rate is good, be more aggressive (wider TP, tighter SL)
    if current_win_rate >= 0.7:
        result["tp_pct"] = atr * 1.5  # 1.5x ATR
        result["sl_pct"] = atr * 0.7  # 0.7x ATR (tighter)
        result["use_breakeven"] = True
        result["use_trail"] = True
    elif current_win_rate >= 0.5:
        result["tp_pct"] = atr * 1.2
        result["sl_pct"] = atr * 0.85
    else:
        # Win rate poor, be very conservative
        result["tp_pct"] = atr * 0.8  # Take profit faster
        result["sl_pct"] = atr * 0.6  # Very tight stop
    return result
