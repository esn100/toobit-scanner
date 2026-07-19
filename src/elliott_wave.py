"""
Elliott Wave detection using ZigZag indicator + Hurst exponent.

Elliott Wave theory: markets move in 5-3 patterns (5 impulse + 3 corrective).
We use a fractal ZigZag to identify swing points, then check:
  - 5-wave impulse (3 of which are trending, 2 corrective)
  - 3-wave ABC correction
  - Wave 3 is typically the longest (1.618x of wave 1)
  - Wave 5 often shows divergence with RSI

This is an "auto-detect" approach using:
  1. ZigZag to find swing points
  2. Hurst exponent to identify trending vs ranging
  3. Fibonacci ratio analysis between waves
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def zigzag(df: pd.DataFrame, threshold: float = 0.05) -> List[Dict]:
    """
    ZigZag indicator: identifies significant swing points.
    A new swing is confirmed when price reverses by `threshold` * last_swing.
    """
    if df.empty or len(df) < 5:
        return []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["open_time"].values if "open_time" in df.columns else range(len(df))
    swings = []
    last_swing_idx = 0
    last_swing_price = closes[0]
    direction = 0  # 1 = up, -1 = down
    for i in range(1, len(closes)):
        if direction >= 0:
            # Looking for swing high
            if highs[i] > highs[last_swing_idx]:
                # Update the high
                last_swing_idx = i
                last_swing_price = highs[i]
            elif (last_swing_price - lows[i]) / last_swing_price >= threshold:
                # Confirmed swing high
                swings.append({
                    "idx": last_swing_idx,
                    "price": float(last_swing_price),
                    "type": "high",
                    "time": times[last_swing_idx],
                })
                # Now look for swing low
                direction = -1
                last_swing_idx = i
                last_swing_price = lows[i]
        if direction <= 0:
            # Looking for swing low
            if lows[i] < lows[last_swing_idx]:
                last_swing_idx = i
                last_swing_price = lows[i]
            elif (highs[i] - last_swing_price) / last_swing_price >= threshold:
                swings.append({
                    "idx": last_swing_idx,
                    "price": float(last_swing_price),
                    "type": "low",
                    "time": times[last_swing_idx],
                })
                direction = 1
                last_swing_idx = i
                last_swing_price = highs[i]
    return swings


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute Hurst exponent using rescaled range (R/S) method.
    H > 0.5: trending (persistent)
    H < 0.5: mean-reverting
    H = 0.5: random walk
    """
    if len(series) < 20:
        return 0.5
    series = series[~np.isnan(series)]
    if len(series) < 20:
        return 0.5
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    for lag in lags:
        # Compute standard deviation of differences
        diffs = series[lag:] - series[:-lag]
        if len(diffs) < 2:
            continue
        tau.append(np.std(diffs))
    if not tau or len(tau) < 3:
        return 0.5
    lags_arr = np.array(list(lags)[:len(tau)])
    tau = np.array(tau)
    # Avoid log(0)
    valid = (tau > 0) & (lags_arr > 0)
    if valid.sum() < 3:
        return 0.5
    try:
        log_lags = np.log(lags_arr[valid])
        log_tau = np.log(tau[valid])
        hurst = float(np.polyfit(log_lags, log_tau, 1)[0])
        return max(0.0, min(1.0, hurst))
    except Exception:
        return 0.5


def detect_elliott_waves(df: pd.DataFrame, threshold: float = 0.05) -> Dict:
    """
    Detect Elliott Wave patterns in the dataframe.
    Returns: wave structure, current position, target, signal.
    """
    if df.empty or len(df) < 30:
        return {"wave": "none", "score": 50.0, "details": {}}
    swings = zigzag(df, threshold)
    if len(swings) < 4:
        return {"wave": "none", "score": 50.0, "details": {}}
    closes = df["close"].values
    # Hurst exponent
    hurst = hurst_exponent(closes)
    # Try to identify 5-wave impulse
    # Take the last 5-7 swings
    recent = swings[-7:]
    if len(recent) < 5:
        return {"wave": "none", "score": 50.0,
                "details": {"hurst": hurst, "swings": len(swings)}}
    # Extract alternating high/low sequence
    high_low = [(s["price"], s["type"]) for s in recent]
    # Check if pattern is alternating
    types = [s[1] for s in high_low]
    is_alternating = all(types[i] != types[i + 1] for i in range(len(types) - 1))
    if not is_alternating:
        # Try to filter to alternating only
        filtered = [high_low[0]]
        for s in high_low[1:]:
            if s[1] != filtered[-1][1]:
                filtered.append(s)
        high_low = filtered[-7:]
    if len(high_low) < 5:
        return {"wave": "none", "score": 50.0,
                "details": {"hurst": hurst}}
    # Identify impulse waves (5 waves: low-high-low-high-low or high-low-high-low-high)
    # Pattern 1: starting low -> 5 waves
    # Wave 1: low to first high
    # Wave 2: first high to second low
    # Wave 3: second low to second high (longest)
    # Wave 4: second high to third low
    # Wave 5: third low to third high
    if high_low[0][1] == "low":
        # 5-wave: low, high, low, high, low, high, low
        # For uptrend: wave 1 = low[0] to high[1], wave 2 = high[1] to low[2]...
        try:
            w1 = high_low[1][0] - high_low[0][0]
            w2 = high_low[1][0] - high_low[2][0]
            w3 = high_low[3][0] - high_low[2][0]
            w4 = high_low[3][0] - high_low[4][0]
            w5 = high_low[5][0] - high_low[4][0]
            waves = [w1, w2, w3, w4, w5]
            wave_prices = [s[0] for s in high_low[:6]]
            wave_types = [s[1] for s in high_low[:6]]
        except (IndexError, ValueError):
            return {"wave": "none", "score": 50.0,
                    "details": {"hurst": hurst}}
    else:
        # Mirror for downtrend
        try:
            w1 = high_low[0][0] - high_low[1][0]
            w2 = high_low[2][0] - high_low[1][0]
            w3 = high_low[2][0] - high_low[3][0]
            w4 = high_low[4][0] - high_low[3][0]
            w5 = high_low[4][0] - high_low[5][0]
            waves = [w1, w2, w3, w4, w5]
            wave_prices = [s[0] for s in high_low[:6]]
            wave_types = [s[1] for s in high_low[:6]]
        except (IndexError, ValueError):
            return {"wave": "none", "score": 50.0,
                    "details": {"hurst": hurst}}
    # Validate Elliott rules:
    # Wave 2 should not exceed wave 1 start
    # Wave 3 should not be the shortest
    # Wave 4 should not enter wave 1 territory
    if w2 >= w1:
        return {"wave": "correction", "score": 50.0,
                "details": {"hurst": hurst, "reason": "wave2_too_deep"}}
    if w3 <= 0 or w3 < w1 * 0.8:
        return {"wave": "correction", "score": 50.0,
                "details": {"hurst": hurst, "reason": "wave3_too_short"}}
    # Fibonacci ratios
    wave2_retrace = w2 / w1 if w1 > 0 else 0
    wave3_ext = w3 / w1 if w1 > 0 else 0
    wave5_ext = w5 / w1 if w1 > 0 else 0
    # Quality score
    score = 50.0
    # Wave 2 retrace in 0.382-0.786 fib range
    if 0.382 <= wave2_retrace <= 0.786:
        score += 15
    # Wave 3 extension > 1.0 (and often ~1.618)
    if wave3_ext >= 1.0:
        score += 15
    if 1.5 <= wave3_ext <= 2.0:
        score += 5
    # Wave 5 extension reasonable
    if 0.8 <= wave5_ext <= 1.8:
        score += 10
    # Hurst confirms trend
    if hurst > 0.55:
        score += 5
    # Determine current position
    current_price = float(closes[-1])
    is_uptrend = high_low[0][1] == "low"
    last_swing = high_low[-1]
    if is_uptrend:
        # Check if we're in wave 5 (price between last low and high)
        if last_swing[1] == "low":
            position = "wave_5_in_progress"
            target = wave_prices[5]  # projected high
        else:
            position = "wave_3_in_progress"
            target = wave_prices[3]
    else:
        if last_swing[1] == "high":
            position = "wave_5_in_progress_down"
            target = wave_prices[5]
        else:
            position = "wave_3_in_progress_down"
            target = wave_prices[3]
    return {
        "wave": "impulse_5" if is_uptrend else "impulse_5_down",
        "score": min(100.0, max(0.0, score)),
        "details": {
            "hurst": round(hurst, 3),
            "waves": [round(w, 4) for w in waves],
            "wave2_retrace": round(wave2_retrace, 3),
            "wave3_extension": round(wave3_ext, 3),
            "wave5_extension": round(wave5_ext, 3),
            "position": position,
            "target": round(target, 6) if target else None,
            "is_uptrend": is_uptrend,
        }
    }


def elliott_score(pack_result: Dict) -> float:
    """
    Convert Elliott Wave result to a 0-100 score contribution.
    Higher if we're in a clear wave 3 or 5 with strong trend.
    """
    if not pack_result or pack_result.get("wave") == "none":
        return 0.0
    base = pack_result.get("score", 50.0)
    position = pack_result.get("details", {}).get("position", "")
    # Bonus for being in a tradable position
    if "wave_3" in position:
        return min(100.0, base * 1.0)
    if "wave_5" in position:
        return min(100.0, base * 0.8)  # wave 5 is later, more risky
    return base * 0.5
