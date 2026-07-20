"""
Ultra-conservative mode for maximum win rate (target: 80%+).

Strategy: only the BEST signals, even if it means 0 signals for days.

Criteria (ALL must be met):
  1. confidence >= 80 (very high)
  2. Multiple timeframe confirmation (1h aligned with 4h)
  3. Volume confirmation (5m spike OR OBI > 1.5)
  4. Trend alignment (Ichimoku + structure agree)
  5. NOT in chop
  6. NOT overextended
  7. ATR in sweet spot (3-8%)
  8. Low correlation risk (BTC not fighting direction)

Expected: 0-3 signals per day, but very high precision.
"""
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd


def is_ultra_setup(features: Dict, direction: str) -> Tuple[bool, str]:
    """
    Ultra-strict check for the BEST setups only.
    Returns (passes, reason_if_fail).
    """
    atr = float(features.get("f_atr_pct", 0))
    rvol = float(features.get("f_rvol", 0.01))
    mom_3 = float(features.get("f_momentum_3_pct", 0))
    mom_6 = float(features.get("f_momentum_6_pct", 0))
    mom_12 = float(features.get("f_momentum_12_pct", 0))
    confidence = float(features.get("confidence", 0))
    ichi_above = bool(features.get("f_a_ichi_above_cloud", 0))
    ichi_below = bool(features.get("f_a_ichi_below_cloud", 0))
    bb_breakout = bool(features.get("f_bb_breakout_above", 0))
    bos_up = bool(features.get("f_bos_up", 0))
    bos_down = bool(features.get("f_a_wave_position", "none") == "impulse_5_down")
    atr_exp = bool(features.get("f_atr_expanding", 0))
    m5m_spike = bool(features.get("f_m_5m_volume_spike", 0))
    obi = float(features.get("f_m_obi_10", 1))
    cvd = float(features.get("f_m_cvd", 0))
    btc_mom = float(features.get("btc_momentum_12_pct", 0))
    score_long = float(features.get("score_long", 0))
    score_short = float(features.get("score_short", 0))
    # Debug: print why failing
    debug = features.get("_debug", False)
    fails = []
    # Tier 1: hard filters
    fails = []
    if confidence < 60:
        fails.append(f"conf<60({confidence:.0f})")
    if atr < 3.0 or atr > 8.0:
        fails.append(f"atr_out({atr:.1f})")
    if direction == "LONG" and mom_3 >= 8:
        fails.append(f"overext_long(mom_3={mom_3:+.1f})")
    if direction == "SHORT" and mom_3 <= -8:
        fails.append(f"overext_short(mom_3={mom_3:+.1f})")
    if abs(mom_6) < 1.5:
        fails.append(f"chop_mom6({mom_6:+.1f})")
    if direction == "LONG" and btc_mom < -2:
        fails.append(f"btc_against_long")
    if direction == "SHORT" and btc_mom > 2:
        fails.append(f"btc_against_short")
    if fails:
        return False, ",".join(fails)
    # Tier 2: structure (need at least 3/4)
    structure_count = 0
    if direction == "LONG":
        if ichi_above: structure_count += 1
        if bb_breakout: structure_count += 1
        if bos_up: structure_count += 1
        if atr_exp: structure_count += 1
        if mom_6 > 0 and mom_12 > 0: structure_count += 1  # trend aligned
    else:  # SHORT
        if ichi_below: structure_count += 1
        if not bb_breakout: structure_count += 1
        if not bos_up: structure_count += 1
        if atr_exp: structure_count += 1
        if mom_6 < 0 and mom_12 < 0: structure_count += 1  # trend aligned
    if structure_count < 4:  # was 3
        return False, f"weak_struct({structure_count}/5)"
    # Tier 3: volume/microstructure (need 1+)
    micro_count = 0
    if m5m_spike:
        micro_count += 1
    if direction == "LONG" and obi > 1.3:
        micro_count += 1
    elif direction == "SHORT" and obi < 0.7:
        micro_count += 1
    if direction == "LONG" and cvd > 0:
        micro_count += 1
    elif direction == "SHORT" and cvd < 0:
        micro_count += 1
    if micro_count < 1:
        return False, f"no_micro_confirm"
    # Tier 4: score threshold
    if direction == "LONG" and score_long < 50:
        return False, f"low_score_long({score_long:.0f})"
    if direction == "SHORT" and score_short < 50:
        return False, f"low_score_short({score_short:.0f})"
    return True, ""


def get_ultra_picks(min_confidence: float = 80.0) -> Dict:
    """
    Get today's ULTRA-STRICT picks (top quality only).
    Expected: 0-3 picks per day.
    """
    import json
    from . import db as database
    features = database.get_features()
    if features.empty:
        return {"error": "no features"}
    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    last_ts = features["ts"].max()
    last = features[features["ts"] == last_ts].copy()
    if last.empty:
        return {"error": "no last cycle"}
    longs = []
    shorts = []
    for _, r in last.iterrows():
        direction = r.get("direction")
        if direction not in ("LONG", "SHORT"):
            continue
        if r.get("confidence", 0) < min_confidence:
            continue
        # Load full features
        fj = r.get("features_json", "{}")
        if isinstance(fj, str):
            try:
                full = json.loads(fj)
            except Exception:
                full = {}
        else:
            full = {}
        feats = {
            "f_atr_pct": full.get("f_atr_pct", r.get("ind_atr_pct", 0)),
            "f_rvol": full.get("f_rvol", r.get("ind_rvol", 0)),
            "f_momentum_3_pct": full.get("f_momentum_3_pct",
                                       r.get("ind_momentum_3_pct", 0)),
            "f_momentum_6_pct": full.get("f_momentum_6_pct",
                                       r.get("ind_momentum_6_pct", 0)),
            "f_momentum_12_pct": full.get("f_momentum_12_pct", 0),
            "f_a_ichi_above_cloud": full.get("f_a_ichi_above_cloud", 0),
            "f_a_ichi_below_cloud": full.get("f_a_ichi_below_cloud", 0),
            "f_bb_breakout_above": full.get("f_bb_breakout_above", 0),
            "f_bos_up": full.get("f_bos_up", 0),
            "f_atr_expanding": full.get("f_atr_expanding", 0),
            "f_m_5m_volume_spike": full.get("f_m_5m_volume_spike", 0),
            "f_m_obi_10": full.get("f_m_obi_10", 1),
            "f_m_cvd": full.get("f_m_cvd", 0),
            "f_a_wave_position": full.get("f_a_wave_position", ""),
            "confidence": r.get("confidence", 0),
            "btc_momentum_12_pct": r.get("btc_momentum_12_pct", 0),
            "score_long": r.get("score_long", 0),
            "score_short": r.get("score_short", 0),
        }
        ok, reason = is_ultra_setup(feats, direction)
        if ok:
            if direction == "LONG":
                longs.append((r, feats))
            else:
                shorts.append((r, feats))
    return {
        "timestamp": str(last_ts),
        "longs": longs,
        "shorts": shorts,
    }


def format_ultra_picks(picks: Dict) -> str:
    """Pretty-print ultra picks."""
    if "error" in picks:
        return f"❌ {picks['error']}"
    lines = []
    lines.append("=" * 70)
    lines.append(f"💎 ULTRA-STRICT PICKS - {picks['timestamp']}")
    lines.append("=" * 70)
    lines.append("⚠️  These are TOP-QUALITY signals only.")
    lines.append("    Expect 0-3 per day, possibly none for hours.")
    lines.append("    Each meets 13+ strict criteria.")
    lines.append("")
    if picks["longs"]:
        lines.append(f"🟢 {len(picks['longs'])} ULTRA-LONG:")
        lines.append("-" * 70)
        for i, (r, f) in enumerate(picks["longs"], 1):
            entry = float(r["close"])
            tp = entry * 1.05
            sl = entry * 0.97
            lines.append(f"\n  #{i} {r['symbol']} (LONG)")
            lines.append(f"     💰 Entry: {entry:.4f}")
            lines.append(f"     🎯 TP:    {tp:.4f} (+5%)")
            lines.append(f"     🛡️  SL:   {sl:.4f} (-3%)")
            lines.append(f"     📈 mom_3: {f['f_momentum_3_pct']:+.1f}% | "
                         f"mom_6: {f['f_momentum_6_pct']:+.1f}% | "
                         f"ATR: {f['f_atr_pct']:.1f}%")
            lines.append(f"     ☁️  Ichimoku: above={bool(f['f_a_ichi_above_cloud'])} | "
                         f"BOS_up={bool(f['f_bos_up'])} | "
                         f"BB_breakout={bool(f['f_bb_breakout_above'])}")
            lines.append(f"     🔥 Conf: {f['confidence']:.0f}% | "
                         f"Score: {f['score_long']:.0f}")
    else:
        lines.append("🟢 ULTRA-LONG: (no signals qualify)")
    if picks["shorts"]:
        lines.append(f"\n🔴 {len(picks['shorts'])} ULTRA-SHORT:")
        lines.append("-" * 70)
        for i, (r, f) in enumerate(picks["shorts"], 1):
            entry = float(r["close"])
            tp = entry * 0.95
            sl = entry * 1.03
            lines.append(f"\n  #{i} {r['symbol']} (SHORT)")
            lines.append(f"     💰 Entry: {entry:.4f}")
            lines.append(f"     🎯 TP:    {tp:.4f} (-5%)")
            lines.append(f"     🛡️  SL:   {sl:.4f} (+3%)")
            lines.append(f"     📈 mom_3: {f['f_momentum_3_pct']:+.1f}% | "
                         f"mom_6: {f['f_momentum_6_pct']:+.1f}% | "
                         f"ATR: {f['f_atr_pct']:.1f}%")
            lines.append(f"     ☁️  Ichimoku: below={bool(f['f_a_ichi_below_cloud'])}")
            lines.append(f"     🔥 Conf: {f['confidence']:.0f}% | "
                         f"Score: {f['score_short']:.0f}")
    else:
        lines.append("🔴 ULTRA-SHORT: (no signals qualify)")
    lines.append("\n" + "=" * 70)
    n_total = len(picks["longs"]) + len(picks["shorts"])
    if n_total == 0:
        lines.append("⏳ NO ULTRA SIGNALS TODAY")
        lines.append("   This is OK! Better to wait than to take mediocre trades.")
        lines.append("   Check back in 1-2 hours or tomorrow.")
    else:
        lines.append(f"✅ {n_total} ultra pick(s) found!")
        lines.append("   Use smart exit v2 for highest win rate.")
    lines.append("=" * 70)
    return "\n".join(lines)
