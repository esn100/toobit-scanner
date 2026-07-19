"""
Ichimoku Cloud (Kumo) indicator.

Five lines:
  - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
  - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
  - Senkou Span A: (Tenkan + Kijun) / 2, projected 26 periods ahead
  - Senkou Span B: (52-period high + 52-period low) / 2, projected 26 ahead
  - Chikou Span: Close plotted 26 periods back

Signals:
  - Price above cloud = bullish
  - Price below cloud = bearish
  - Tenkan > Kijun = bullish
  - Tenkan < Kijun = bearish
  - Cloud thickness = trend strength
  - TK cross = signal
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def ichimoku_features(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> Dict:
    """
    Compute Ichimoku components and produce features.
    """
    if df.empty or len(df) < senkou_b_period + displacement:
        return _empty_ichimoku()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    # Tenkan-sen
    tenkan = (high.rolling(tenkan_period).max()
              + low.rolling(tenkan_period).min()) / 2
    # Kijun-sen
    kijun = (high.rolling(kijun_period).max()
             + low.rolling(kijun_period).min()) / 2
    # Senkou Span A: (Tenkan + Kijun) / 2, projected forward
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    # Senkou Span B
    senkou_b = ((high.rolling(senkou_b_period).max()
                 + low.rolling(senkou_b_period).min()) / 2).shift(displacement)
    # Chikou Span: close shifted back
    chikou = close.shift(-displacement)
    last = len(df) - 1
    if last < max(tenkan_period, kijun_period, senkou_b_period):
        return _empty_ichimoku()
    tenkan_v = float(tenkan.iloc[last - displacement])  # current Tenkan (not shifted)
    kijun_v = float(kijun.iloc[last - displacement])
    senkou_a_v = float(senkou_a.iloc[last]) if not pd.isna(senkou_a.iloc[last]) else 0
    senkou_b_v = float(senkou_b.iloc[last]) if not pd.isna(senkou_b.iloc[last]) else 0
    chikou_v = float(close.iloc[last]) if last < len(close) else 0
    current_price = float(close.iloc[last])
    # Determine cloud top/bottom at current time
    if senkou_a_v > senkou_b_v:
        cloud_top = senkou_a_v
        cloud_bottom = senkou_b_v
        cloud_color = "green"  # bullish cloud
    else:
        cloud_top = senkou_b_v
        cloud_bottom = senkou_a_v
        cloud_color = "red"  # bearish cloud
    # Price vs cloud
    if current_price > cloud_top:
        price_vs_cloud = "above"
    elif current_price < cloud_bottom:
        price_vs_cloud = "below"
    else:
        price_vs_cloud = "inside"
    # TK cross
    if tenkan_v > kijun_v:
        tk_cross = "bullish"
    else:
        tk_cross = "bearish"
    # Cloud thickness
    thickness = (cloud_top - cloud_bottom) / current_price * 100 if current_price > 0 else 0
    # Future cloud (Senkou A and B at +26)
    future_a = float(senkou_a.iloc[last - 1]) if last - 1 >= 0 and not pd.isna(senkou_a.iloc[last - 1]) else senkou_a_v
    future_b = float(senkou_b.iloc[last - 1]) if last - 1 >= 0 and not pd.isna(senkou_b.iloc[last - 1]) else senkou_b_v
    future_cloud_color = "green" if future_a > future_b else "red"
    return {
        "tenkan": tenkan_v,
        "kijun": kijun_v,
        "senkou_a": senkou_a_v,
        "senkou_b": senkou_b_v,
        "chikou": chikou_v,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "cloud_thickness_pct": round(thickness, 3),
        "cloud_color": cloud_color,
        "price_vs_cloud": price_vs_cloud,
        "tk_cross": tk_cross,
        "future_cloud_color": future_cloud_color,
        "current_price": current_price,
    }


def _empty_ichimoku() -> Dict:
    return {
        "tenkan": 0.0, "kijun": 0.0, "senkou_a": 0.0, "senkou_b": 0.0,
        "chikou": 0.0, "cloud_top": 0.0, "cloud_bottom": 0.0,
        "cloud_thickness_pct": 0.0, "cloud_color": "neutral",
        "price_vs_cloud": "neutral", "tk_cross": "neutral",
        "future_cloud_color": "neutral", "current_price": 0.0,
    }


def ichimoku_score(ichi: Dict) -> float:
    """
    Convert Ichimoku features to 0-100 score.
    Strongest signal: price above green cloud with bullish TK cross.
    """
    if not ichi or ichi.get("current_price", 0) <= 0:
        return 50.0
    score = 50.0
    # Price vs cloud (most important)
    if ichi["price_vs_cloud"] == "above" and ichi["cloud_color"] == "green":
        score += 25
    elif ichi["price_vs_cloud"] == "above" and ichi["cloud_color"] == "red":
        score += 10  # above red cloud = weak bullish (resistance)
    elif ichi["price_vs_cloud"] == "below" and ichi["cloud_color"] == "red":
        score -= 20
    elif ichi["price_vs_cloud"] == "below" and ichi["cloud_color"] == "green":
        score -= 10
    elif ichi["price_vs_cloud"] == "inside":
        score += 5  # neutral, slight bullish bias if green
    # TK cross
    if ichi["tk_cross"] == "bullish":
        score += 10
    else:
        score -= 5
    # Future cloud (looking ahead 26 bars)
    if ichi["future_cloud_color"] == "green":
        score += 8
    else:
        score -= 3
    # Thick cloud = strong trend
    if ichi["cloud_thickness_pct"] > 3.0:
        if ichi["cloud_color"] == "green":
            score += 8
        else:
            score -= 5
    return min(100.0, max(0.0, score))
