"""
Daily picks generator with ultra-strict filtering and smart exits.

Outputs the best LONG and SHORT picks for today, with:
  - Entry price
  - TP (target)
  - SL (stop)
  - Breakeven trigger
  - Lock-in triggers
  - Trailing stop distance
  - Risk/reward ratio
  - Confidence score

Usage:
  python -m src.picks
"""
from __future__ import annotations
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

from .adaptive_tp_sl import get_signal_tp_sl
from .strict_mode import is_strict_setup
from .smart_exit import smart_sl_logic, adjust_tp_sl_based_on_history
from . import db as database


def get_today_picks(min_confidence: float = 75.0) -> Dict:
    """
    Get today's strict picks from the latest cycle.
    """
    # Get the latest features
    features = database.get_features()
    if features.empty:
        return {"error": "no features in database"}
    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    last_ts = features["ts"].max()
    if pd.isna(last_ts):
        return {"error": "no valid timestamps"}
    last = features[features["ts"] == last_ts].copy()
    if last.empty:
        return {"error": "no data in last cycle"}
    # Apply strict filter
    import json
    longs_pass = []
    shorts_pass = []
    for _, r in last.iterrows():
        # Load full features from JSON
        features_json = r.get("features_json", "{}")
        if isinstance(features_json, str):
            try:
                full_feats = json.loads(features_json)
            except Exception:
                full_feats = {}
        else:
            full_feats = {}
        feats = {
            "f_atr_pct": full_feats.get("f_atr_pct", r.get("ind_atr_pct", 0)),
            "f_rvol": full_feats.get("f_rvol", r.get("ind_rvol", 1)),
            "f_momentum_3_pct": full_feats.get("f_momentum_3_pct",
                                              r.get("ind_momentum_3_pct", 0)),
            "f_momentum_6_pct": full_feats.get("f_momentum_6_pct",
                                              r.get("ind_momentum_6_pct", 0)),
            "f_a_ichi_above_cloud": full_feats.get("f_a_ichi_above_cloud", 0),
            "f_a_ichi_below_cloud": full_feats.get("f_a_ichi_below_cloud", 0),
            "f_bb_breakout_above": full_feats.get("f_bb_breakout_above", 0),
            "f_bos_up": full_feats.get("f_bos_up", 0),
            "f_atr_expanding": full_feats.get("f_atr_expanding", 0),
            "f_m_5m_volume_spike": full_feats.get("f_m_5m_volume_spike", 0),
            "f_m_obi_10": full_feats.get("f_m_obi_10", 1),
            "f_m_cvd": full_feats.get("f_m_cvd", 0),
            "confidence": r.get("confidence", 0),
            "btc_state": r.get("btc_state", "NEUTRAL"),
            "btc_momentum_12_pct": r.get("btc_momentum_12_pct", 0),
        }
        if r.get("direction") == "LONG" and r.get("confidence", 0) >= min_confidence:
            ok, reason = is_strict_setup(feats, "LONG")
            if ok:
                tp_sl = get_signal_tp_sl(feats)
                longs_pass.append((r, tp_sl))
        elif r.get("direction") == "SHORT" and r.get("confidence", 0) >= min_confidence:
            ok, reason = is_strict_setup(feats, "SHORT")
            if ok:
                tp_sl = get_signal_tp_sl(feats)
                shorts_pass.append((r, tp_sl))
    # Sort by score
    longs_pass.sort(key=lambda x: x[0].get("score_long", 0), reverse=True)
    shorts_pass.sort(key=lambda x: x[0].get("score_short", 0), reverse=True)
    # Get recent win rate
    resolved = database.get_resolved_signals()
    n_resolved = len(resolved)
    win_rate = 0.5
    if n_resolved >= 5:
        n_tp = (resolved["status"] == "TP_HIT").sum()
        n_sl = (resolved["status"] == "SL_HIT").sum()
        decided = n_tp + n_sl
        if decided > 0:
            win_rate = float(n_tp) / decided
    # Build picks
    def build_pick(r, tp_sl, direction):
        entry = float(r["close"])
        tp_pct = tp_sl["tp_pct"]
        sl_pct = tp_sl["sl_pct"]
        tp_price = entry * (1 + tp_pct / 100) if direction == "LONG" else entry * (1 - tp_pct / 100)
        sl_price = entry * (1 - sl_pct / 100) if direction == "LONG" else entry * (1 + sl_pct / 100)
        rr = tp_pct / sl_pct if sl_pct else 0
        atr = float(r.get("ind_atr_pct", 3))
        rvol = float(r.get("ind_rvol", 1))
        mom_3 = float(r.get("ind_momentum_3_pct", 0))
        return {
            "symbol": r["symbol"],
            "direction": direction,
            "entry_price": round(entry, 6),
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_price": round(tp_price, 6),
            "sl_price": round(sl_price, 6),
            "rr_ratio": round(rr, 2),
            "confidence": float(r.get("confidence", 0)),
            "score": float(r.get("score_long" if direction == "LONG" else "score_short", 0)),
            "atr_pct": atr,
            "rvol": rvol,
            "mom_3_pct": mom_3,
            # Smart exit triggers
            "breakeven_at": 1.5,  # % gain
            "lock_25_at": 2.5,
            "lock_50_at": 4.0,
            "trail_after": 3.0,
            "trail_distance": 1.5,
            "tp_reasoning": tp_sl.get("reasoning", ""),
        }
    return {
        "timestamp": str(last_ts),
        "win_rate_history": win_rate,
        "n_resolved": n_resolved,
        "longs": [build_pick(r, t, "LONG") for r, t in longs_pass[:5]],
        "shorts": [build_pick(r, t, "SHORT") for r, t in shorts_pass[:5]],
    }


def format_picks(picks: Dict) -> str:
    """Pretty-print picks."""
    if "error" in picks:
        return f"❌ Error: {picks['error']}"
    lines = []
    lines.append("=" * 70)
    lines.append(f"🎯 DAILY PICKS - {picks['timestamp']}")
    lines.append("=" * 70)
    lines.append(f"📊 Historical win rate: {picks['win_rate_history']*100:.1f}% "
                 f"(n={picks['n_resolved']})")
    lines.append("")
    if picks["longs"]:
        lines.append("🟢 LONG PICKS:")
        lines.append("-" * 70)
        for i, p in enumerate(picks["longs"], 1):
            lines.append(f"\n{i}. {p['symbol']} (LONG)")
            lines.append(f"   💰 Entry: {p['entry_price']}")
            lines.append(f"   🎯 TP: +{p['tp_pct']}% → {p['tp_price']}")
            lines.append(f"   🛡️ SL: -{p['sl_pct']}% → {p['sl_price']}")
            lines.append(f"   📊 R/R: 1:{p['rr_ratio']:.2f}")
            lines.append(f"   ✅ Confidence: {p['confidence']:.0f}% | "
                         f"Score: {p['score']:.0f}")
            lines.append(f"   📈 mom_3: {p['mom_3_pct']:+.1f}% | "
                         f"ATR: {p['atr_pct']:.1f}% | rvol: {p['rvol']:.2f}")
            lines.append(f"   🔒 Smart exits:")
            lines.append(f"      - At +{p['breakeven_at']}%: SL → entry "
                         f"(breakeven lock)")
            lines.append(f"      - At +{p['lock_25_at']}%: SL → entry+0.5% "
                         f"(lock small profit)")
            lines.append(f"      - At +{p['lock_50_at']}%: SL → entry+2% "
                         f"(lock bigger profit)")
            lines.append(f"      - After +{p['trail_after']}%: trail SL at "
                         f"{p['trail_distance']}% below highest")
    else:
        lines.append("🟢 LONG PICKS: (none qualify under strict filter)")
    lines.append("")
    if picks["shorts"]:
        lines.append("\n🔴 SHORT PICKS:")
        lines.append("-" * 70)
        for i, p in enumerate(picks["shorts"], 1):
            lines.append(f"\n{i}. {p['symbol']} (SHORT)")
            lines.append(f"   💰 Entry: {p['entry_price']}")
            lines.append(f"   🎯 TP: -{p['tp_pct']}% → {p['tp_price']}")
            lines.append(f"   🛡️ SL: +{p['sl_pct']}% → {p['sl_price']}")
            lines.append(f"   📊 R/R: 1:{p['rr_ratio']:.2f}")
            lines.append(f"   ✅ Confidence: {p['confidence']:.0f}% | "
                         f"Score: {p['score']:.0f}")
            lines.append(f"   📈 mom_3: {p['mom_3_pct']:+.1f}% | "
                         f"ATR: {p['atr_pct']:.1f}% | rvol: {p['rvol']:.2f}")
            lines.append(f"   🔒 Smart exits (mirror):")
            lines.append(f"      - At +{p['breakeven_at']}%: SL → entry")
            lines.append(f"      - At +{p['lock_25_at']}%: SL → entry-0.5%")
            lines.append(f"      - At +{p['lock_50_at']}%: SL → entry-2%")
            lines.append(f"      - After +{p['trail_after']}%: trail SL at "
                         f"{p['trail_distance']}% above")
    else:
        lines.append("🔴 SHORT PICKS: (none qualify under strict filter)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("⚠️ RISK WARNING:")
    lines.append("   - 80%+ win rate requires 100+ validated signals")
    lines.append("   - Current data is insufficient for such claims")
    lines.append("   - Always use stop losses, never risk more than you can lose")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    picks = get_today_picks(min_confidence=75.0)
    print(format_picks(picks))


if __name__ == "__main__":
    main()
