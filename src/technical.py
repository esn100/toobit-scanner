"""
Technical analysis module.
- RSI + divergences
- MACD + divergences
- EMA stack (20/50/100/200)
- Candlestick & chart pattern recognition
- Returns: 0..100 technical score + a per-indicator breakdown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import ta

# ----------------------------------------------------------------------------
# 1. RSI
# ----------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return ta.momentum.RSIIndicator(close, window=period).rsi()


def _detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 30) -> str:
    """
    Returns: 'bullish_div', 'bearish_div', or 'none'.
    Bullish: price makes lower low while indicator makes higher low.
    Bearish: price makes higher high while indicator makes lower high.
    """
    if len(price) < lookback or len(indicator) < lookback:
        return "none"
    p = price.tail(lookback).values
    ind = indicator.tail(lookback).values
    if len(p) < 5 or len(ind) < 5:
        return "none"
    # Find two most recent local extrema
    try:
        p_min_idx = np.argmin(p)
        p_max_idx = np.argmax(p)
        # Use the most recent swing low/high
        recent_low = np.min(p[-10:])
        recent_high = np.max(p[-10:])
        # Compare with prior swing
        prior_low = np.min(p[:-10]) if len(p) > 10 else recent_low
        prior_high = np.max(p[:-10]) if len(p) > 10 else recent_high
        ind_recent_low = np.min(ind[-10:])
        ind_recent_high = np.max(ind[-10:])
        ind_prior_low = np.min(ind[:-10]) if len(ind) > 10 else ind_recent_low
        ind_prior_high = np.max(ind[:-10]) if len(ind) > 10 else ind_recent_high
        # Bullish divergence
        if recent_low < prior_low and ind_recent_low > ind_prior_low:
            return "bullish_div"
        # Bearish divergence
        if recent_high > prior_high and ind_recent_high < ind_prior_high:
            return "bearish_div"
    except (ValueError, IndexError):
        return "none"
    return "none"


def rsi_score(close: pd.Series) -> dict:
    rsi = _rsi(close, 14)
    last = float(rsi.iloc[-1]) if not rsi.empty else 50.0
    div = _detect_divergence(close, rsi)
    # Score 0-100
    score = 50.0
    if last < 30:
        score = 80 + (30 - last)       # very oversold => bullish
    elif last < 40:
        score = 65 + (40 - last)
    elif last > 70:
        score = 30 - (last - 70)       # overbought => bearish
    elif last > 60:
        score = 45 - (last - 60)
    if div == "bullish_div":
        score = min(100.0, score + 15)
    elif div == "bearish_div":
        score = max(0.0, score - 15)
    return {"rsi_value": last, "rsi_divergence": div, "rsi_score": max(0.0, min(100.0, score))}


# ----------------------------------------------------------------------------
# 2. MACD
# ----------------------------------------------------------------------------
def macd_score(close: pd.Series) -> dict:
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_ind.macd()
    signal = macd_ind.macd_signal()
    hist = macd_ind.macd_diff()
    last_macd = float(macd.iloc[-1]) if not macd.empty else 0.0
    last_signal = float(signal.iloc[-1]) if not signal.empty else 0.0
    last_hist = float(hist.iloc[-1]) if not hist.empty else 0.0
    prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
    div = _detect_divergence(close, macd)
    score = 50.0
    # Crossover
    if last_macd > last_signal and prev_hist < 0 <= last_hist:
        score = 80  # bullish crossover above zero
    elif last_macd > last_signal and last_hist > 0 and prev_hist < 0:
        score = 90  # bullish crossover with momentum
    elif last_macd < last_signal and prev_hist > 0 >= last_hist:
        score = 20  # bearish crossover below zero
    elif last_macd < last_signal and last_hist < 0 and prev_hist > 0:
        score = 10
    else:
        # Histogram trend
        if last_hist > 0:
            score = 60
        else:
            score = 40
    if div == "bullish_div":
        score = min(100.0, score + 10)
    elif div == "bearish_div":
        score = max(0.0, score - 10)
    return {
        "macd_value": last_macd,
        "macd_signal": last_signal,
        "macd_hist": last_hist,
        "macd_divergence": div,
        "macd_score": max(0.0, min(100.0, score)),
    }


# ----------------------------------------------------------------------------
# 3. EMA stack (20/50/100/200)
# ----------------------------------------------------------------------------
def ema_score(close: pd.Series) -> dict:
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema100 = ta.trend.EMAIndicator(close, window=100).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()
    last = float(close.iloc[-1])
    vals = {
        "ema20": float(ema20.iloc[-1]) if not ema20.empty else last,
        "ema50": float(ema50.iloc[-1]) if not ema50.empty else last,
        "ema100": float(ema100.iloc[-1]) if not ema100.empty else last,
        "ema200": float(ema200.iloc[-1]) if not ema200.empty else last,
    }
    # Bullish: price > ema20 > ema50 > ema100 > ema200
    perfect_bull = vals["ema20"] > vals["ema50"] > vals["ema100"] > vals["ema200"] and last > vals["ema20"]
    perfect_bear = vals["ema20"] < vals["ema50"] < vals["ema100"] < vals["ema200"] and last < vals["ema20"]
    score = 50.0
    if perfect_bull:
        score = 95.0
    elif perfect_bear:
        score = 10.0
    else:
        # partial scoring: count of "above" relations
        above = sum([
            last > vals["ema20"],
            vals["ema20"] > vals["ema50"],
            vals["ema50"] > vals["ema100"],
            vals["ema100"] > vals["ema200"],
        ])
        score = 50 + (above - 2) * 12  # each step shifts by 12 points
    return {
        **vals,
        "ema_alignment": (
            "bullish" if perfect_bull
            else "bearish" if perfect_bear
            else "mixed"
        ),
        "ema_score": max(0.0, min(100.0, score)),
    }


# ----------------------------------------------------------------------------
# 4. Pattern recognition
# ----------------------------------------------------------------------------
def _find_swing_points(series: pd.Series, order: int = 5) -> tuple:
    """Find local maxima and minima indices."""
    highs, lows = [], []
    vals = series.values
    for i in range(order, len(vals) - order):
        if vals[i] == max(vals[i - order:i + order + 1]):
            highs.append(i)
        if vals[i] == min(vals[i - order:i + order + 1]):
            lows.append(i)
    return highs, lows


def detect_patterns(df: pd.DataFrame) -> dict:
    """
    Detect a handful of common patterns on the 4h timeframe.
    Returns a 0..100 pattern_score and a list of detected patterns.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    detected: list = []
    score = 50.0

    # --- Bullish engulfing (last 1-2 candles) ---
    if len(close) >= 2:
        prev, cur = close.iloc[-2], close.iloc[-1]
        prev_o, cur_o = open_.iloc[-2], open_.iloc[-1]
        if (cur > cur_o) and (prev < prev_o) and (cur_o < prev_o) and (cur > prev):
            detected.append("bullish_engulfing")
            score += 12

    # --- Hammer (last candle) ---
    if len(close) >= 1:
        body = abs(close.iloc[-1] - open_.iloc[-1])
        rng = high.iloc[-1] - low.iloc[-1]
        lower_wick = min(close.iloc[-1], open_.iloc[-1]) - low.iloc[-1]
        if rng > 0 and lower_wick > 2 * body and lower_wick / rng > 0.55:
            detected.append("hammer")
            score += 8

    # --- Double bottom (last 60 bars) ---
    highs, lows = _find_swing_points(low.tail(60), order=4)
    if len(lows) >= 2:
        last_lows = sorted(lows)[-2:]
        v1, v2 = low.iloc[last_lows[0]], low.iloc[last_lows[1]]
        if abs(v1 - v2) / max(v1, v2) < 0.03 and close.iloc[-1] > v1:
            detected.append("double_bottom")
            score += 12

    # --- Higher lows (last 30 bars) ---
    _, lows_recent = _find_swing_points(low.tail(30), order=3)
    if len(lows_recent) >= 3:
        ll_vals = [low.iloc[i] for i in lows_recent[-3:]]
        if ll_vals[0] < ll_vals[1] < ll_vals[2]:
            detected.append("higher_lows")
            score += 6

    # --- Bearish engulfing ---
    if len(close) >= 2:
        prev, cur = close.iloc[-2], close.iloc[-1]
        prev_o, cur_o = open_.iloc[-2], open_.iloc[-1]
        if (cur < cur_o) and (prev > prev_o) and (cur_o > prev_o) and (cur < prev):
            detected.append("bearish_engulfing")
            score -= 12

    # --- Double top ---
    highs_idx, _ = _find_swing_points(high.tail(60), order=4)
    if len(highs_idx) >= 2:
        last_highs = sorted(highs_idx)[-2:]
        v1, v2 = high.iloc[last_highs[0]], high.iloc[last_highs[1]]
        if abs(v1 - v2) / max(v1, v2) < 0.03 and close.iloc[-1] < v1:
            detected.append("double_top")
            score -= 12

    return {
        "patterns": detected,
        "pattern_score": max(0.0, min(100.0, score)),
    }


# ----------------------------------------------------------------------------
# 5. Combined technical score
# ----------------------------------------------------------------------------
def technical_analysis(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {
            "rsi_value": 50.0, "rsi_divergence": "none", "rsi_score": 50.0,
            "macd_value": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "macd_divergence": "none", "macd_score": 50.0,
            "ema20": 0.0, "ema50": 0.0, "ema100": 0.0, "ema200": 0.0,
            "ema_alignment": "mixed", "ema_score": 50.0,
            "patterns": [], "pattern_score": 50.0,
            "technical_score": 50.0,
        }
    rsi = rsi_score(df["close"])
    macd = macd_score(df["close"])
    ema = ema_score(df["close"])
    pat = detect_patterns(df)
    # Aggregate: equal weight for rsi/macd/ema
    tech_score = (rsi["rsi_score"] + macd["macd_score"] + ema["ema_score"]) / 3.0
    return {
        **rsi,
        **macd,
        **ema,
        **pat,
        "technical_score": tech_score,
    }
