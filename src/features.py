"""
Feature extraction (Layer 4 of the PumpHunter pipeline).

Combines raw indicator values, candle quality, structure, and market
context into a single flat feature dict ready for both rule-based
scoring and ML model ingestion.

The feature names here are the *contract* between this module and
`ml_engine.py`. If you add/remove features, update ML feature list too.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List


FEATURE_NAMES: List[str] = [
    # technical
    "rsi_value",
    "rsi_divergence",          # encoded: 0=none, 1=bull, -1=bear
    "macd_hist",
    "macd_divergence",         # encoded: 0/1/-1
    "ema_alignment",           # encoded: 1=bull, 0=mixed, -1=bear
    # new indicators
    "rvol",
    "volume_spike",            # 0/1
    "vwap_distance_pct",
    "price_above_vwap",        # 0/1
    "atr_pct",
    "atr_expanding",           # 0/1
    "bb_squeeze",              # 0/1
    "bb_breakout_above",       # 0/1
    # momentum
    "momentum_1_pct",
    "momentum_3_pct",
    "momentum_6_pct",
    "momentum_12_pct",
    "momentum_acceleration",
    # structure
    "higher_highs",            # 0/1
    "higher_lows",             # 0/1
    "bos_up",                  # 0/1
    "in_range",                # 0/1
    # candle quality
    "candle_strength",
    "big_wick_top",            # 0/1
    "power_streak",
    # multi-timeframe
    "mtf_alignment",
    # market context
    "btc_state",               # encoded: 1=bull, 0=neutral, -1=bear, -2=risk_off
    "btc_momentum_12_pct",
]


def _enc_divergence(s: str) -> float:
    return 1.0 if s == "bullish_div" else (-1.0 if s == "bearish_div" else 0.0)


def _enc_ema(s: str) -> float:
    return 1.0 if s == "bullish" else (-1.0 if s == "bearish" else 0.0)


def _enc_btc(s: str) -> float:
    return {"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": -1.0}.get(s, -2.0)


def build_features(
    technical: dict,
    indicators: dict,
    structure: dict,
    candle: dict,
    mtf: dict,
    btc: dict,
) -> Dict[str, float]:
    """
    Aggregate all per-symbol analytics into a flat numeric feature dict.
    """
    f: Dict[str, float] = {}
    # technical
    f["rsi_value"] = float(technical.get("rsi_value", 50.0))
    f["rsi_divergence"] = _enc_divergence(technical.get("rsi_divergence", "none"))
    f["macd_hist"] = float(technical.get("macd_hist", 0.0))
    f["macd_divergence"] = _enc_divergence(technical.get("macd_divergence", "none"))
    f["ema_alignment"] = _enc_ema(technical.get("ema_alignment", "mixed"))
    # volume
    f["rvol"] = float(indicators.get("rvol", 1.0))
    f["volume_spike"] = float(bool(indicators.get("volume_spike", False)))
    # vwap
    f["vwap_distance_pct"] = float(indicators.get("vwap_distance_pct", 0.0))
    f["price_above_vwap"] = float(bool(indicators.get("price_above_vwap", False)))
    # atr / bollinger
    f["atr_pct"] = float(indicators.get("atr_pct", 0.0))
    f["atr_expanding"] = float(bool(indicators.get("atr_expanding", False)))
    f["bb_squeeze"] = float(bool(indicators.get("bb_squeeze", False)))
    f["bb_breakout_above"] = float(bool(indicators.get("bb_breakout_above", False)))
    # momentum
    for k in ("momentum_1_pct", "momentum_3_pct", "momentum_6_pct",
              "momentum_12_pct", "momentum_acceleration"):
        f[k] = float(indicators.get(k, 0.0))
    # structure
    f["higher_highs"] = float(bool(structure.get("higher_highs", False)))
    f["higher_lows"] = float(bool(structure.get("higher_lows", False)))
    f["bos_up"] = float(bool(structure.get("bos_up", False)))
    f["in_range"] = float(bool(structure.get("in_range", False)))
    # candle
    f["candle_strength"] = float(candle.get("candle_strength", 0.5))
    f["big_wick_top"] = float(bool(candle.get("big_wick_top", False)))
    f["power_streak"] = float(candle.get("power_streak", 0))
    # mtf
    f["mtf_alignment"] = float(mtf.get("alignment_score", 50.0)) / 100.0
    # market context
    f["btc_state"] = _enc_btc(btc.get("state", "NEUTRAL"))
    f["btc_momentum_12_pct"] = float(btc.get("btc_momentum_12_pct", 0.0))
    return f


def feature_vector(features: Dict[str, float], names: List[str] = None) -> np.ndarray:
    """Return features in a stable order for the ML model."""
    if names is None:
        names = FEATURE_NAMES
    return np.array([float(features.get(n, 0.0)) for n in names], dtype=float)
