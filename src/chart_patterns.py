"""
Advanced chart pattern recognition (Layer 13).

Detects:
  - Bull Flag (consolidation after a strong upward move)
  - Bear Flag (mirror)
  - Ascending / Descending / Symmetric Triangle
  - Cup and Handle
  - Falling / Rising Wedge
  - Wyckoff Accumulation (simplified)

Each pattern returns:
  - detected (bool)
  - confidence (0..1)
  - target (estimated price target)
  - stop_loss
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


def _find_swing_lows(lows: np.ndarray, order: int = 3) -> List[int]:
    out = []
    for i in range(order, len(lows) - order):
        if lows[i] == lows[i - order:i + order + 1].min():
            out.append(i)
    return out


def _find_swing_highs(highs: np.ndarray, order: int = 3) -> List[int]:
    out = []
    for i in range(order, len(highs) - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            out.append(i)
    return out


# ----------------------------------------------------------------------------
# Bull Flag / Bear Flag
# ----------------------------------------------------------------------------
def detect_flag(df: pd.DataFrame) -> dict:
    """
    Bull flag: a strong pole (price run-up > 8% in < 10 bars) followed
    by a 5-15 bar consolidation against the prevailing trend.
    """
    out = {"bull_flag": False, "bear_flag": False,
           "confidence": 0.0, "target": 0.0, "stop": 0.0}
    if len(df) < 30:
        return out
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    # Look at the last 25 bars
    window = 25
    for start in range(len(df) - window, max(0, len(df) - window - 10), -1):
        pole = close[start + 8] / close[start] - 1
        if pole >= 0.08:  # 8%+ pole
            # Consolidation = 5 bars after pole with low volatility
            consol = close[start + 8: start + 18]
            if len(consol) < 5:
                continue
            consol_range = (consol.max() - consol.min()) / consol.mean()
            if consol_range < 0.04:  # <4% range = tight consolidation
                # Pole target = continuation of the pole move
                target = float(close[-1]) * (1 + pole)
                out["bull_flag"] = True
                out["confidence"] = min(1.0, pole * 5)
                out["target"] = target
                out["stop"] = float(consol.min()) * 0.99
                return out
    # Bear flag (mirror)
    for start in range(len(df) - window, max(0, len(df) - window - 10), -1):
        pole = close[start] / close[start + 8] - 1
        if pole >= 0.08:
            consol = close[start + 8: start + 18]
            if len(consol) < 5:
                continue
            consol_range = (consol.max() - consol.min()) / consol.mean()
            if consol_range < 0.04:
                target = float(close[-1]) * (1 - pole)
                out["bear_flag"] = True
                out["confidence"] = min(1.0, pole * 5)
                out["target"] = target
                out["stop"] = float(consol.max()) * 1.01
                return out
    return out


# ----------------------------------------------------------------------------
# Triangles (Ascending, Descending, Symmetric)
# ----------------------------------------------------------------------------
def detect_triangle(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Detect triangle patterns using linear regression on highs and lows.
    Ascending: flat top, rising bottom  -> bullish breakout
    Descending: falling top, flat bottom -> bearish breakdown
    Symmetric: converging trendlines -> breakout in either direction
    """
    out = {"triangle": "none", "confidence": 0.0, "target": 0.0}
    if len(df) < lookback + 5:
        return out
    sub = df.tail(lookback).reset_index(drop=True)
    highs = sub["high"].values
    lows = sub["low"].values
    x = np.arange(len(highs))
    # Fit lines
    h_slope, h_intercept = np.polyfit(x, highs, 1)
    l_slope, l_intercept = np.polyfit(x, lows, 1)
    h_resid = np.std(highs - (h_slope * x + h_intercept))
    l_resid = np.std(lows - (l_slope * x + l_intercept))
    h_r2 = 1 - h_resid / (np.std(highs) + 1e-12)
    l_r2 = 1 - l_resid / (np.std(lows) + 1e-12)
    if h_r2 < 0.4 or l_r2 < 0.4:
        return out
    close = float(sub["close"].iloc[-1])
    last = len(highs) - 1
    h_now = h_slope * last + h_intercept
    l_now = l_slope * last + l_intercept
    conf = (h_r2 + l_r2) / 2
    # Ascending: top flat, bottom rising
    if abs(h_slope) < 0.0001 * close and l_slope > 0.0001 * close:
        out["triangle"] = "ascending"
        out["confidence"] = float(min(1.0, conf))
        out["target"] = h_now  # measured move = height of the triangle
    # Descending: top falling, bottom flat
    elif h_slope < -0.0001 * close and abs(l_slope) < 0.0001 * close:
        out["triangle"] = "descending"
        out["confidence"] = float(min(1.0, conf))
        out["target"] = l_now
    # Symmetric: both converging
    elif h_slope < -0.0001 * close and l_slope > 0.0001 * close:
        out["triangle"] = "symmetric"
        out["confidence"] = float(min(1.0, conf * 0.7))  # less reliable
        # Direction based on which side is closer
        dist_to_top = abs(close - h_now)
        dist_to_bottom = abs(close - l_now)
        out["target"] = h_now if dist_to_top > dist_to_bottom else l_now
    return out


# ----------------------------------------------------------------------------
# Cup and Handle
# ----------------------------------------------------------------------------
def detect_cup_and_handle(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Cup and handle: U-shaped recovery followed by a small pullback.
    """
    out = {"cup_handle": False, "confidence": 0.0, "target": 0.0}
    if len(df) < lookback:
        return out
    sub = df.tail(lookback).reset_index(drop=True)
    close = sub["close"].values
    # Find the cup: a down then up sequence
    n = len(close)
    third = n // 3
    if third < 5:
        return out
    left_high = float(close[:third].max())
    cup_low = float(close[third:2 * third].min())
    right_high = float(close[2 * third:].max())
    if left_high <= 0 or right_high <= 0:
        return out
    # Cup depth should be 12-35% of left high
    depth_pct = (left_high - cup_low) / left_high
    if not (0.12 <= depth_pct <= 0.35):
        return out
    # Left and right highs should be roughly equal
    symmetry = abs(left_high - right_high) / max(left_high, right_high)
    if symmetry > 0.05:
        return out
    # Handle: small pullback in last 10 bars
    handle = close[-10:]
    if len(handle) < 5:
        return out
    handle_dd = (handle.max() - handle.min()) / handle.max()
    if handle_dd > 0.10 or handle_dd < 0.01:
        return out
    # Target = cup depth projected from breakout
    target = right_high + (left_high - cup_low)
    out["cup_handle"] = True
    out["confidence"] = float(min(1.0, (1 - symmetry) * depth_pct * 3))
    out["target"] = float(target)
    return out


# ----------------------------------------------------------------------------
# Wedge (Rising / Falling)
# ----------------------------------------------------------------------------
def detect_wedge(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Rising wedge: rising highs and rising lows, but lows rising faster
    -> typically bearish reversal.
    Falling wedge: falling highs and falling lows, but highs falling
    faster -> typically bullish reversal.
    """
    out = {"wedge": "none", "confidence": 0.0, "target": 0.0}
    if len(df) < lookback:
        return out
    sub = df.tail(lookback).reset_index(drop=True)
    highs = sub["high"].values
    lows = sub["low"].values
    x = np.arange(len(highs))
    h_slope, _ = np.polyfit(x, highs, 1)
    l_slope, _ = np.polyfit(x, lows, 1)
    h_r2 = 1 - np.std(highs - np.polyval(np.polyfit(x, highs, 1), x)) / (np.std(highs) + 1e-12)
    l_r2 = 1 - np.std(lows - np.polyval(np.polyfit(x, lows, 1), x)) / (np.std(lows) + 1e-12)
    if h_r2 < 0.5 or l_r2 < 0.5:
        return out
    close = float(sub["close"].iloc[-1])
    if h_slope > 0 and l_slope > 0 and l_slope > h_slope * 1.2:
        out["wedge"] = "rising"  # bearish
        out["confidence"] = float(min(1.0, (h_r2 + l_r2) / 2))
        out["target"] = float(lows.min())
    elif h_slope < 0 and l_slope < 0 and h_slope < l_slope * 1.2:
        out["wedge"] = "falling"  # bullish
        out["confidence"] = float(min(1.0, (h_r2 + l_r2) / 2))
        out["target"] = float(highs.max() + (highs.max() - lows.min()))
    return out


# ----------------------------------------------------------------------------
# Wyckoff Accumulation (simplified)
# ----------------------------------------------------------------------------
def detect_wyckoff(df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Simplified Wyckoff: detect a trading range with a Spring (false
    breakdown below support that quickly recovers) followed by a
    Sign of Strength (SOS) breakout above resistance.
    """
    out = {"wyckoff": "none", "spring": False, "sos": False,
           "confidence": 0.0, "target": 0.0}
    if len(df) < lookback:
        return out
    sub = df.tail(lookback).reset_index(drop=True)
    close = sub["close"].values
    low = sub["low"].values
    high = sub["high"].values
    # Range detection
    range_high = float(high.max())
    range_low = float(low.min())
    range_size = range_high - range_low
    if range_size <= 0:
        return out
    # Spring: a bar that wicked below range_low but closed back inside
    last_20 = sub.tail(20)
    for i, row in last_20.iterrows():
        if row["low"] < range_low * 1.01 and row["close"] > range_low:
            out["spring"] = True
            break
    # SOS: a strong bar (body > 2x ATR) closing above range_high
    if len(sub) >= 14:
        atr = float((sub["high"] - sub["low"]).tail(14).mean())
        last = sub.iloc[-1]
        body = abs(last["close"] - last["open"])
        if (last["close"] > range_high * 0.99
                and body > 2 * atr
                and last["close"] > last["open"]):
            out["sos"] = True
    if out["spring"] and out["sos"]:
        out["wyckoff"] = "accumulation"
        out["confidence"] = 0.8
        out["target"] = range_high + range_size  # measured move
    elif out["spring"]:
        out["wyckoff"] = "spring_only"
        out["confidence"] = 0.4
    return out


# ----------------------------------------------------------------------------
# Combined pattern detector
# ----------------------------------------------------------------------------
def detect_all_patterns(df: pd.DataFrame) -> dict:
    """
    Run all pattern detectors and return a consolidated view.
    """
    out = {
        "patterns": [],
        "bullish_score": 0.0,
        "bearish_score": 0.0,
        "primary_pattern": None,
        "target": 0.0,
        "stop": 0.0,
    }
    flag = detect_flag(df)
    if flag["bull_flag"]:
        out["patterns"].append("bull_flag")
        out["bullish_score"] += flag["confidence"] * 30
        out["target"] = max(out["target"], flag["target"])
    if flag["bear_flag"]:
        out["patterns"].append("bear_flag")
        out["bearish_score"] += flag["confidence"] * 30
    triangle = detect_triangle(df)
    if triangle["triangle"] == "ascending":
        out["patterns"].append("ascending_triangle")
        out["bullish_score"] += triangle["confidence"] * 25
    elif triangle["triangle"] == "descending":
        out["patterns"].append("descending_triangle")
        out["bearish_score"] += triangle["confidence"] * 25
    elif triangle["triangle"] == "symmetric":
        out["patterns"].append("symmetric_triangle")
        out["bullish_score"] += triangle["confidence"] * 12
        out["bearish_score"] += triangle["confidence"] * 12
    cup = detect_cup_and_handle(df)
    if cup["cup_handle"]:
        out["patterns"].append("cup_and_handle")
        out["bullish_score"] += cup["confidence"] * 30
        out["target"] = max(out["target"], cup["target"])
    wedge = detect_wedge(df)
    if wedge["wedge"] == "falling":
        out["patterns"].append("falling_wedge")
        out["bullish_score"] += wedge["confidence"] * 25
        out["target"] = max(out["target"], wedge["target"])
    elif wedge["wedge"] == "rising":
        out["patterns"].append("rising_wedge")
        out["bearish_score"] += wedge["confidence"] * 25
    wyckoff = detect_wyckoff(df)
    if wyckoff["wyckoff"] == "accumulation":
        out["patterns"].append("wyckoff_accumulation")
        out["bullish_score"] += wyckoff["confidence"] * 35
        out["target"] = max(out["target"], wyckoff["target"])
    # Pick primary pattern
    if out["bullish_score"] >= out["bearish_score"]:
        out["primary_pattern"] = "bullish" if out["bullish_score"] > 5 else None
    else:
        out["primary_pattern"] = "bearish" if out["bearish_score"] > 5 else None
    return out
