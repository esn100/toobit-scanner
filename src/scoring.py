"""
Rule-based scoring (Layer 5 of the PumpHunter pipeline).

Each sub-score is 0..100. Weights come from config (and may be softly
adapted by ml_engine). The composite is also 0..100.

Key features:
  - Bounded sub-scores (no single feature can dominate)
  - Heavy weight on volume (pump hunter)
  - Penalty for BTC bearish state
  - Penalty for overbought RSI
  - Penalty for low-confidence candles (big upper wick)
"""
from __future__ import annotations
from typing import Dict


def rule_based_score(
    technical: dict,
    indicators: dict,
    structure: dict,
    candle: dict,
    mtf: dict,
    btc: dict,
    weights: Dict[str, float],
) -> Dict:
    """
    Combine all sub-scores with the given weights.
    Returns composite_score (0..100) and a breakdown dict.
    """
    # ----- hard rejections (penalties) -----
    rsi_value = float(technical.get("rsi_value", 50.0))
    overbought_penalty = 0.0
    if rsi_value > 80:
        overbought_penalty = 15.0
    elif rsi_value > 70:
        overbought_penalty = 7.0
    elif rsi_value < 20:
        # Deeply oversold is risky in small caps (rug magnets)
        overbought_penalty = 5.0

    # Big upper wick on the last bar = rejection
    wick_penalty = 10.0 if candle.get("big_wick_top") else 0.0

    # In range / choppy
    if structure.get("in_range"):
        range_penalty = 8.0
    else:
        range_penalty = 0.0

    # BTC regime modifier
    btc_state = btc.get("state", "NEUTRAL")
    btc_mod = float(btc.get("score_modifier", 1.0))
    if btc_state == "BEARISH":
        btc_penalty = 12.0
    elif btc_state == "RISK_OFF":
        btc_penalty = 30.0
    else:
        btc_penalty = 0.0
    btc_bonus = 0.0
    if btc_state == "BULLISH":
        btc_bonus = 4.0

    # ----- weighted composite -----
    sub = {
        "technical": float(technical.get("technical_score", 50.0)),
        "momentum": float(indicators.get("momentum_score", 50.0)),
        "volume": float(indicators.get("rvol_score", 50.0)),
        "vwap": float(indicators.get("vwap_score", 50.0)),
        "atr_bb": float(
            0.5 * indicators.get("atr_score", 50.0)
            + 0.5 * indicators.get("bb_score", 50.0)
        ),
        "structure": float(structure.get("structure_score", 50.0)),
        "candle": float(candle.get("candle_score", 50.0)),
        "mtf": float(mtf.get("alignment_score", 50.0)),
        "pattern": float(technical.get("pattern_score", 50.0)),
    }

    w = {
        "technical": weights.get("technical", 12) / 100.0,
        "momentum": weights.get("momentum", 12) / 100.0,
        "volume": weights.get("volume", 18) / 100.0,
        "vwap": weights.get("vwap", 8) / 100.0,
        "atr_bb": weights.get("atr_bb", 6) / 100.0,
        "structure": weights.get("structure", 10) / 100.0,
        "candle": weights.get("candle", 8) / 100.0,
        "mtf": weights.get("mtf", 8) / 100.0,
        "pattern": weights.get("pattern", 8) / 100.0,
    }
    # Normalise weights to 1
    wsum = sum(w.values()) or 1.0
    w = {k: v / wsum for k, v in w.items()}

    base = sum(sub[k] * w[k] for k in w)
    # Apply modifiers
    final = (
        base * btc_mod
        - overbought_penalty
        - wick_penalty
        - range_penalty
        - btc_penalty
        + btc_bonus
    )
    final = max(0.0, min(100.0, final))

    return {
        "composite_score": float(final),
        "sub_scores": sub,
        "penalties": {
            "overbought": overbought_penalty,
            "wick": wick_penalty,
            "range": range_penalty,
            "btc": btc_penalty,
        },
        "btc_bonus": btc_bonus,
    }
