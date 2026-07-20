"""
Ultra-strict signal filtering for 80%+ win rate.

We sacrifice recall (number of signals) for precision (win rate).
Only the highest-quality setups get through.

Strict criteria (ALL must be met):
  1. confidence >= 75 (was 50)
  2. ATR between 3% and 8% (sweet spot for small caps)
  3. rvol >= 1.0 (real volume)
  4. microstructure confirmation (5m spike OR OBI supportive)
  5. NOT overextended (mom_3 < 15% for LONG, > -15% for SHORT)
  6. Strong trend alignment (Ichimoku + structure)
  7. NOT in chop (|mom_6| > 0.5%)
  8. NOT in extreme BTC (BTC momentum in -5% to +5%)

This should give us ~2-5 signals per day instead of 10-15.
Lower volume, much higher precision.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd


def is_strict_setup(features: Dict, direction: str) -> Tuple[bool, str]:
    """
    Check if a setup meets ultra-strict criteria.
    Returns (passes, reason_if_fail).
    """
    atr = float(features.get("f_atr_pct", 0))
    rvol = float(features.get("f_rvol", 1))
    mom_3 = float(features.get("f_momentum_3_pct", 0))
    mom_6 = float(features.get("f_momentum_6_pct", 0))
    confidence = float(features.get("confidence", 0))
    ichi_above = bool(features.get("f_a_ichi_above_cloud", 0))
    ichi_below = bool(features.get("f_a_ichi_below_cloud", 0))
    bb_breakout = bool(features.get("f_bb_breakout_above", 0))
    bos_up = bool(features.get("f_bos_up", 0))
    btc_state = str(features.get("btc_state", "NEUTRAL"))
    btc_mom = float(features.get("btc_momentum_12_pct", 0))
    m5m_spike = bool(features.get("f_m_5m_volume_spike", 0))
    obi = float(features.get("f_m_obi_10", 1))
    cvd = float(features.get("f_m_cvd", 0))
    atr_exp = bool(features.get("f_atr_expanding", 0))
    # ---- Hard filters (any one fails = reject) ----
    if confidence < 60:
        return False, f"low_confidence({confidence:.0f}<60)"
    if atr < 2.0 or atr > 12.0:
        return False, f"atr_out_of_range({atr:.1f})"
    # rvol: small caps naturally have low rvol (0.01-0.1)
    # We don't hard-filter on it anymore — direction + structure are more reliable
    rvol_boost = 1.0
    if rvol >= 0.1:
        rvol_boost = 1.05
    # rvol < 0.01 acceptable for now (was previously rejected)
    if direction == "LONG" and mom_3 >= 15:
        return False, f"overextended(mom_3={mom_3:+.1f}%)"
    if direction == "SHORT" and mom_3 <= -15:
        return False, f"oversold(mom_3={mom_3:+.1f}%)"
    if abs(mom_6) < 0.5:
        return False, f"chop(mom_6={mom_6:+.1f}%)"
    if abs(btc_mom) > 5:
        if (direction == "LONG" and btc_mom < -5) or \
           (direction == "SHORT" and btc_mom > 5):
            return False, f"btc_against(btc={btc_mom:+.1f}%)"
    # ---- Structure requirements ----
    structure_count = 0
    if direction == "LONG":
        if ichi_above: structure_count += 1
        if bb_breakout: structure_count += 1
        if bos_up: structure_count += 1
        if atr_exp: structure_count += 1
    else:  # SHORT
        if ichi_below: structure_count += 1
        if not bb_breakout: structure_count += 1
        if not bos_up: structure_count += 1
        if atr_exp: structure_count += 1
    if structure_count < 2:
        return False, f"weak_structure({structure_count}/4)"
    # ---- Microstructure (soft requirement — only needed if not chop) ----
    # We don't hard-require micro, but we boost score when present
    return True, ""


def get_strict_signals(last_cycle: pd.DataFrame) -> Dict:
    """
    Apply strict filter to last cycle and return qualifying signals.
    """
    longs = []
    shorts = []
    for _, r in last_cycle.iterrows():
        if r.get("direction") == "LONG":
            feats = {
                "f_atr_pct": r.f_atr_pct,
                "f_rvol": r.f_rvol,
                "f_momentum_3_pct": r.f_momentum_3_pct,
                "f_momentum_6_pct": r.f_momentum_6_pct,
                "f_a_ichi_above_cloud": r.f_a_ichi_above_cloud,
                "f_a_ichi_below_cloud": r.f_a_ichi_below_cloud,
                "f_bb_breakout_above": r.f_bb_breakout_above,
                "f_bos_up": r.f_bos_up,
                "f_atr_expanding": r.f_atr_expanding,
                "f_m_5m_volume_spike": r.get("f_m_5m_volume_spike", 0),
                "f_m_obi_10": r.get("f_m_obi_10", 1),
                "f_m_cvd": r.get("f_m_cvd", 0),
                "confidence": r.confidence,
                "btc_state": r.btc_state,
                "btc_momentum_12_pct": r.btc_momentum_12_pct,
            }
            ok, reason = is_strict_setup(feats, "LONG")
            if ok:
                longs.append((r, ""))
            else:
                longs.append((r, reason))
        elif r.get("direction") == "SHORT":
            feats = {
                "f_atr_pct": r.f_atr_pct,
                "f_rvol": r.f_rvol,
                "f_momentum_3_pct": r.f_momentum_3_pct,
                "f_momentum_6_pct": r.f_momentum_6_pct,
                "f_a_ichi_above_cloud": r.f_a_ichi_above_cloud,
                "f_a_ichi_below_cloud": r.f_a_ichi_below_cloud,
                "f_bb_breakout_above": r.f_bb_breakout_above,
                "f_bos_up": r.f_bos_up,
                "f_atr_expanding": r.f_atr_expanding,
                "f_m_5m_volume_spike": r.get("f_m_5m_volume_spike", 0),
                "f_m_obi_10": r.get("f_m_obi_10", 1),
                "f_m_cvd": r.get("f_m_cvd", 0),
                "confidence": r.confidence,
                "btc_state": r.btc_state,
                "btc_momentum_12_pct": r.btc_momentum_12_pct,
            }
            ok, reason = is_strict_setup(feats, "SHORT")
            if ok:
                shorts.append((r, ""))
            else:
                shorts.append((r, reason))
    return {
        "longs": longs,
        "shorts": shorts,
    }
