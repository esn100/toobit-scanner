"""
Pre-filter module for small-cap pump hunting.

Pass 1: Aggressive filtering to narrow the universe down to the top 5%
of candidates. Each filter is fast and uses short lookback windows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def prefilter_score(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                    df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Dict:
    """
    Compute a fast pre-filter score (0..100).
    Returns the score and a dict of component flags.
    """
    out = {"prefilter": 0.0, "flags": {}}
    if df_15m.empty or len(df_15m) < 30:
        return out
    # ---- 1) Volume spike in last 3 candles (15m) ----
    vol_15m = df_15m["volume"].astype(float)
    baseline_15m = vol_15m.iloc[-16:-3].mean()  # last 4 hours excluding spike
    recent_3 = vol_15m.iloc[-3:]
    avg_recent = recent_3.mean()
    rvol_3bar = float(avg_recent / max(baseline_15m, 1e-9))
    out["flags"]["vol_spike_3bar"] = rvol_3bar
    # ---- 2) Price momentum on 1h (last 4 bars = 4h) ----
    close_1h = df_1h["close"].astype(float) if not df_1h.empty else df_15m["close"].astype(float)
    if len(close_1h) >= 5:
        m4h = float((close_1h.iloc[-1] - close_1h.iloc[-5]) / close_1h.iloc[-5] * 100.0)
    else:
        m4h = 0.0
    out["flags"]["momentum_4h_pct"] = m4h
    # ---- 3) Range contraction (4h) - pre-breakout squeeze ----
    if not df_4h.empty and len(df_4h) >= 12:
        rng_recent = float(df_4h["high"].tail(6).max() - df_4h["low"].tail(6).min())
        rng_prior = float(df_4h["high"].iloc[-12:-6].max() - df_4h["low"].iloc[-12:-6].min())
        rng_ratio = rng_recent / max(rng_prior, 1e-9)
    else:
        rng_ratio = 1.0
    out["flags"]["range_ratio"] = rng_ratio
    # ---- 4) BTC correlation (4h, last 24 bars) ----
    if not btc_df.empty and len(btc_df) >= 24 and not df_4h.empty and len(df_4h) >= 24:
        sym_ret = df_4h["close"].pct_change().tail(24).fillna(0).values
        btc_ret = btc_df["close"].pct_change().tail(24).fillna(0).values
        n = min(len(sym_ret), len(btc_ret))
        if n >= 10:
            try:
                corr = float(np.corrcoef(sym_ret[-n:], btc_ret[-n:])[0, 1])
            except Exception:
                corr = 1.0
        else:
            corr = 1.0
    else:
        corr = 1.0
    out["flags"]["btc_correlation"] = corr
    # ---- 5) Trend alignment (15m short-term) ----
    close_15m = df_15m["close"].astype(float)
    if len(close_15m) >= 20:
        ema9 = close_15m.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close_15m.ewm(span=21, adjust=False).mean().iloc[-1]
        trend_aligned = bool(ema9 > ema21)
    else:
        ema9 = ema21 = close_15m.iloc[-1]
        trend_aligned = False
    out["flags"]["trend_aligned_15m"] = trend_aligned
    # ---- 6) Wick rejection in last 15m candle ----
    last = df_15m.iloc[-1]
    rng = float(last["high"] - last["low"])
    upper_wick = float((last["high"] - max(last["close"], last["open"])) / max(rng, 1e-9))
    out["flags"]["upper_wick_pct"] = upper_wick

    # ---- Composite ----
    score = 0.0
    # Volume spike is critical: top 5% have rvol > 2.5
    if rvol_3bar >= 3.0:
        score += 30
    elif rvol_3bar >= 2.0:
        score += 20
    elif rvol_3bar >= 1.5:
        score += 10
    # Positive momentum
    if 1.5 <= m4h <= 15.0:
        score += 20
    elif m4h > 0:
        score += 10
    elif m4h < -8:
        score -= 15  # already dumping
    # Range contraction (squeeze)
    if 0.3 <= rng_ratio <= 0.7:
        score += 15
    # Independent of BTC (low correlation = independent mover)
    if 0 <= corr <= 0.3:
        score += 15
    elif corr > 0.7:
        score -= 5
    # Trend aligned
    if trend_aligned:
        score += 10
    # No upper wick rejection
    if upper_wick < 0.3:
        score += 10
    elif upper_wick > 0.6:
        score -= 10
    out["prefilter"] = float(max(0.0, min(100.0, score)))
    return out


def passes_prefilter(prefilter_result: Dict, min_score: float = 60.0) -> bool:
    """A symbol passes if prefilter_score >= min_score and key flags are set."""
    if prefilter_result["prefilter"] < min_score:
        return False
    f = prefilter_result["flags"]
    # Hard requirements: must have volume spike AND positive momentum
    if f.get("vol_spike_3bar", 0) < 1.5:
        return False
    if f.get("momentum_4h_pct", 0) < 0:
        return False
    return True
