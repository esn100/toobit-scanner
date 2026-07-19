"""
Advanced technical indicators (Layer 3 of the PumpHunter pipeline).

Beyond the basic RSI/MACD/EMA in `technical.py`, this module computes:
  - VWAP (session and rolling)
  - ATR (Average True Range) and ATR expansion state
  - Bollinger Bands with squeeze detection
  - Relative Volume (current vs N-bar average)
  - Price momentum (1m/5m/15m-equivalent on 4h, plus multi-bar acceleration)
  - Multi-timeframe alignment helpers

All functions return primitives (float, str, dict) so they can be
consumed directly by the rule-based scorer and the feature extractor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import ta


# ----------------------------------------------------------------------------
# VWAP
# ----------------------------------------------------------------------------
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-style VWAP. We don't have intraday sessions for crypto so
    we use a rolling 20-bar VWAP which approximates what most chart
    platforms call "VWAP" on a perpetual contract.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].rolling(20, min_periods=1).sum()
    cum_vp = (typical * df["volume"]).rolling(20, min_periods=1).sum()
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return vwap.bfill().fillna(df["close"])


def vwap_features(df: pd.DataFrame) -> dict:
    """Return 0..100 vwap_score and helpers."""
    vwap = compute_vwap(df)
    close = float(df["close"].iloc[-1])
    v = float(vwap.iloc[-1]) if not vwap.empty else close
    dist = (close - v) / max(v, 1e-12)  # signed fractional distance
    above = close > v
    score = 50.0
    if above:
        # distance in %; gentle score
        score = 50.0 + min(40.0, dist * 1000.0)
    else:
        score = 50.0 + max(-40.0, dist * 1000.0)
    return {
        "vwap": v,
        "vwap_distance_pct": float(dist * 100.0),
        "price_above_vwap": bool(above),
        "vwap_score": float(max(0.0, min(100.0, score))),
    }


# ----------------------------------------------------------------------------
# ATR + expansion
# ----------------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=period
    ).average_true_range()


def atr_features(df: pd.DataFrame) -> dict:
    a = atr(df, 14)
    if a.empty or a.isna().all():
        return {"atr": 0.0, "atr_pct": 0.0, "atr_expanding": False,
                "atr_score": 50.0}
    last = float(a.iloc[-1])
    avg20 = float(a.tail(20).mean()) if len(a) >= 20 else float(a.mean())
    close = float(df["close"].iloc[-1])
    atr_pct = (last / max(close, 1e-12)) * 100.0
    # Expansion = current ATR > 1.4x of recent baseline
    expanding = bool(last > 1.4 * avg20) if avg20 > 0 else False
    score = 50.0
    if expanding:
        score = 70.0
    elif last < 0.6 * avg20:
        # Squeezed
        score = 35.0
    return {
        "atr": last,
        "atr_pct": float(atr_pct),
        "atr_expanding": expanding,
        "atr_score": score,
    }


# ----------------------------------------------------------------------------
# Bollinger Bands + squeeze
# ----------------------------------------------------------------------------
def bollinger_features(df: pd.DataFrame, period: int = 20, dev: float = 2.0) -> dict:
    bb = ta.volatility.BollingerBands(
        df["close"], window=period, window_dev=dev
    )
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    mid = bb.bollinger_mavg()
    width = (upper - lower) / mid.replace(0, np.nan)
    if upper.empty or df.empty:
        return {"bb_score": 50.0, "bb_squeeze": False,
                "bb_breakout_above": False, "bb_breakout_below": False}
    last_close = float(df["close"].iloc[-1])
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    last_mid = float(mid.iloc[-1])
    last_width = float(width.iloc[-1]) if not width.empty else 0.0
    # Squeeze = current width is at the low end of its recent range
    width_pct = 0.0
    if len(width) >= 60:
        width_pct = float(
            (last_width - width.tail(60).min())
            / max(width.tail(60).max() - width.tail(60).min(), 1e-12)
        )
    squeeze = bool(0.0 <= width_pct <= 0.25)
    breakout_above = bool(last_close > last_upper)
    breakout_below = bool(last_close < last_lower)
    score = 50.0
    if squeeze:
        score = 65.0  # squeeze → expect expansion
    if breakout_above:
        score = 90.0
    elif breakout_below:
        score = 15.0
    return {
        "bb_upper": last_upper,
        "bb_lower": last_lower,
        "bb_mid": last_mid,
        "bb_width_pct": float(last_width * 100.0),
        "bb_squeeze": squeeze,
        "bb_breakout_above": breakout_above,
        "bb_breakout_below": breakout_below,
        "bb_score": score,
    }


# ----------------------------------------------------------------------------
# Relative Volume
# ----------------------------------------------------------------------------
def relative_volume(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Compare current volume to N-bar baseline.
    Returns rvol (ratio) and a score 0..100.
    """
    if len(df) < lookback + 1:
        return {"rvol": 1.0, "rvol_score": 50.0, "volume_spike": False}
    vol = df["volume"]
    baseline = vol.iloc[-(lookback + 1):-1].mean()
    cur = float(vol.iloc[-1])
    rvol = float(cur / baseline) if baseline > 0 else 1.0
    spike = bool(rvol >= 2.0)
    # Map 0.5x..3x linearly to 20..95
    score = float(max(0.0, min(100.0, 20 + (rvol - 0.5) * 30)))
    if spike:
        score = max(score, 80.0)
    if rvol < 0.5:
        score = min(score, 25.0)
    return {
        "rvol": round(rvol, 3),
        "rvol_score": score,
        "volume_spike": spike,
    }


def volume_continuity(df: pd.DataFrame, n: int = 3) -> dict:
    """
    Is volume sustained over the last N bars? True if all N bars are
    above their 20-bar baseline.
    """
    if len(df) < 25:
        return {"volume_continuity": 0, "sustained": False}
    vol = df["volume"]
    base = vol.rolling(20).mean()
    last_n = vol.tail(n)
    last_base = base.tail(n)
    cnt = int((last_n > last_base).sum())
    return {
        "volume_continuity": cnt,
        "sustained": bool(cnt >= max(1, n - 1)),
    }


# ----------------------------------------------------------------------------
# Price momentum
# ----------------------------------------------------------------------------
def momentum_features(df: pd.DataFrame) -> dict:
    """
    Compute short/medium momentum on a 4h dataframe, expressed in %.
    """
    if len(df) < 16:
        return {f"momentum_{k}_pct": 0.0 for k in (1, 3, 6, 12)} | {
            "momentum_acceleration": 0.0,
            "momentum_score": 50.0,
        }
    close = df["close"]
    last = float(close.iloc[-1])
    out = {}
    for n in (1, 3, 6, 12):
        past = float(close.iloc[-(n + 1)])
        out[f"momentum_{n}_pct"] = float((last - past) / max(past, 1e-12) * 100.0)
    # Acceleration: change of change. momentum_3 - average(prev momentum_3)
    if len(close) >= 9:
        cur_3 = out["momentum_3_pct"]
        prev_3 = float(
            (close.iloc[-4] - close.iloc[-7]) / max(close.iloc[-7], 1e-12) * 100.0
        )
        out["momentum_acceleration"] = cur_3 - prev_3
    else:
        out["momentum_acceleration"] = 0.0
    # Score: weighted blend emphasising the medium-term and acceleration
    blended = (
        0.15 * out["momentum_1_pct"]
        + 0.30 * out["momentum_3_pct"]
        + 0.35 * out["momentum_6_pct"]
        + 0.20 * out["momentum_12_pct"]
    )
    score = 50.0 + max(-40.0, min(40.0, blended * 4.0))
    if out["momentum_acceleration"] > 0:
        score += 5
    out["momentum_score"] = float(max(0.0, min(100.0, score)))
    return out


# ----------------------------------------------------------------------------
# Multi-timeframe alignment
# ----------------------------------------------------------------------------
def mtf_alignment(
    df_fast: pd.DataFrame, df_slow: pd.DataFrame
) -> dict:
    """
    Compare the trend bias of two dataframes (typically 1h and 4h).
    Both should have the same columns. We compute a simple trend score
    from EMA stack + momentum.
    """
    def trend_bias(d: pd.DataFrame) -> float:
        if d.empty or len(d) < 30:
            return 0.0
        e20 = float(d["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        e50 = float(d["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        c = float(d["close"].iloc[-1])
        # -1..+1
        bias = 0.0
        if c > e20 > e50:
            bias = 0.6
        elif c > e20:
            bias = 0.3
        elif c < e20 < e50:
            bias = -0.6
        elif c < e20:
            bias = -0.3
        # Add momentum influence
        if len(d) >= 6:
            m = (c - float(d["close"].iloc[-6])) / max(float(d["close"].iloc[-6]), 1e-12)
            bias += max(-0.4, min(0.4, m * 5.0))
        return max(-1.0, min(1.0, bias))

    fast = trend_bias(df_fast)
    slow = trend_bias(df_slow)
    aligned = (fast > 0 and slow > 0) or (fast < 0 and slow < 0)
    same_sign = bool((fast * slow) > 0)
    return {
        "fast_bias": float(fast),
        "slow_bias": float(slow),
        "aligned": aligned,
        "same_sign": same_sign,
        "alignment_score": float(50 + 50 * (fast + slow) / 2.0),
    }
