"""
Dual-direction picks: ALWAYS outputs best LONG and SHORT together.
Scans entire 77-symbol watchlist (12 primary repeaters + 65 secondary).
Uses:
  - Repeater pattern (pre-pump setup)
  - ML filter (81.3% accuracy: atr_pct>1.09 AND flat_hours<34)
  - Direction scores (long/short)
  - Smart exits (breakeven, locks, trail)
  - Per-cap TP/SL (low/mid/high)

Usage:
  python3 -m src.dual_picks
  python3 -m src.dual_picks --min-conf 60
"""
from __future__ import annotations
import sys
import os
import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repeater_config import REPEATERS, SECONDARY_WATCHLIST
from src import db as database
from src.pump_filter import is_pump_setup_atr_flat, is_pump_setup_combined
from src.adaptive_tp_sl import get_signal_tp_sl
from src.strict_mode import is_strict_setup


# ============================================================================
# Per-cap TP/SL profiles (from watchlist research)
# ============================================================================
LOW_CAP_TPSL = {  # <$50M
    "tp_pct": 30.0, "sl_pct": 3.0,
    "trail_pct": 7.0, "trail_activate_pct": 15.0,
    "max_hold_hours": 12.0,
    "breakeven_at": 2.0, "lock_25_at": 5.0, "lock_50_at": 10.0,
    "reentry_on_dip": True, "reentry_rsi_max": 35.0, "reentry_size": 0.30,
}
MID_CAP_TPSL = {  # $50M-$500M
    "tp_pct": 20.0, "sl_pct": 2.5,
    "trail_pct": 5.0, "trail_activate_pct": 12.0,
    "max_hold_hours": 10.0,
    "breakeven_at": 1.5, "lock_25_at": 3.0, "lock_50_at": 6.0,
    "reentry_on_dip": True, "reentry_rsi_max": 35.0, "reentry_size": 0.30,
}
PRIMARY_REPEATER_TPSL = {  # 12 main repeaters
    "tp_pct": 20.0, "sl_pct": 2.5,
    "trail_pct": 5.0, "trail_activate_pct": 15.0,
    "max_hold_hours": 12.0,
    "breakeven_at": 1.5, "lock_25_at": 3.0, "lock_50_at": 6.0,
    "reentry_on_dip": True, "reentry_rsi_max": 35.0, "reentry_size": 0.30,
}


def get_tpsl_for_symbol(symbol: str) -> Dict:
    """Return TP/SL profile based on cap category."""
    if symbol in REPEATERS:
        # Use repeater config first if it has overrides
        cfg = REPEATERS[symbol]
        if "tp_pct" in cfg:
            return {
                "tp_pct": cfg.get("tp_pct", 20.0),
                "sl_pct": cfg.get("sl_pct", 2.5),
                "trail_pct": cfg.get("trail_pct", 5.0),
                "trail_activate_pct": cfg.get("trail_activate_pct", 15.0),
                "max_hold_hours": cfg.get("max_hold_hours", 12.0),
                "breakeven_at": 1.5, "lock_25_at": 3.0, "lock_50_at": 6.0,
                "reentry_on_dip": cfg.get("reentry_on_dip", True),
                "reentry_rsi_max": cfg.get("reentry_rsi_max", 35.0),
                "reentry_size": cfg.get("reentry_size", 0.30),
            }
        return PRIMARY_REPEATER_TPSL

    # Check cap-based CSV categorization
    try:
        from src.cap_watchlist import LOW_CAP, MID_CAP
        if symbol in LOW_CAP:
            return LOW_CAP_TPSL
        if symbol in MID_CAP:
            return MID_CAP_TPSL
    except Exception:
        pass

    # Default to mid-cap profile
    return MID_CAP_TPSL


def compute_dual_scores(row) -> Dict:
    """Compute long and short score for a feature row."""
    rvol = float(row.get("ind_rvol", 1.0) or 1.0)
    m1 = float(row.get("ind_momentum_3_pct", 0) or 0)  # 3-period momentum
    m3 = float(row.get("ind_momentum_6_pct", 0) or 0)
    atr = float(row.get("ind_atr_pct", 0) or 0)
    vwap_dist = float(row.get("ind_vwap_distance_pct", 0) or 0)
    bb_squeeze = bool(row.get("ind_bb_squeeze", False))
    long_score = float(row.get("score_long", 0) or 0)
    short_score = float(row.get("score_short", 0) or 0)
    return {
        "long_score": long_score,
        "short_score": short_score,
        "rvol": rvol, "m1": m1, "m3": m3,
        "atr": atr, "vwap_dist": vwap_dist, "bb_squeeze": bb_squeeze,
    }


def ml_filter_passes(row, symbol: str) -> bool:
    """Apply ML filter (81.3% accuracy) - extracted from features_json."""
    fjson = row.get("features_json", "{}")
    if isinstance(fjson, str):
        try:
            feats = json.loads(fjson)
        except Exception:
            feats = {}
    else:
        feats = fjson or {}
    atr_pct = float(feats.get("f_atr_pct", row.get("ind_atr_pct", 0)) or 0)
    flat_hours = float(feats.get("f_flat_hours", 0) or 0)
    # Rule: atr_pct > 1.09 AND flat_hours < 34
    return atr_pct > 1.09 and flat_hours < 34


def build_pick(row, direction: str, symbol: str) -> Dict:
    """Build a complete pick with TP/SL/smart-exit."""
    entry = float(row["close"])
    tpsl = get_tpsl_for_symbol(symbol)
    tp_pct = tpsl["tp_pct"]
    sl_pct = tpsl["sl_pct"]
    if direction == "LONG":
        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 - sl_pct / 100)
    else:
        tp_price = entry * (1 - tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)
    rr = tp_pct / sl_pct if sl_pct else 0
    score = float(row.get(f"score_{direction.lower()}", 0) or 0)
    feats = compute_dual_scores(row)
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry, 6),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "rr_ratio": round(rr, 2),
        "confidence": float(row.get("confidence", 0) or 0),
        "score": round(score, 1),
        "rvol": round(feats["rvol"], 2),
        "mom_3_pct": round(feats["m1"], 2),
        "mom_6_pct": round(feats["m3"], 2),
        "atr_pct": round(feats["atr"], 2),
        "vwap_dist": round(feats["vwap_dist"], 2),
        "bb_squeeze": feats["bb_squeeze"],
        "ml_filter": ml_filter_passes(row, symbol),
        "smart_exit": {
            "breakeven_at": tpsl["breakeven_at"],
            "lock_25_at": tpsl["lock_25_at"],
            "lock_50_at": tpsl["lock_50_at"],
            "trail_pct": tpsl["trail_pct"],
            "trail_activate_pct": tpsl["trail_activate_pct"],
            "max_hold_hours": tpsl["max_hold_hours"],
            "reentry_on_dip": tpsl["reentry_on_dip"],
            "reentry_rsi_max": tpsl["reentry_rsi_max"],
        },
    }


def get_dual_picks(min_confidence: float = 50.0, top_n: int = 10) -> Dict:
    """
    Get the best LONG and SHORT picks from latest features.
    Scans all 77 symbols in watchlist (REPEATERS + SECONDARY).
    """
    features = database.get_features()
    if features.empty:
        return {"error": "no features in database", "longs": [], "shorts": []}

    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    features = features.dropna(subset=["ts"])
    last_ts = features["ts"].max()
    if pd.isna(last_ts):
        return {"error": "no valid timestamps", "longs": [], "shorts": []}
    last = features[features["ts"] == last_ts].copy()
    if last.empty:
        return {"error": "no data in last cycle", "longs": [], "shorts": []}

    # All watchlist symbols
    all_symbols = set(REPEATERS.keys()) | set(SECONDARY_WATCHLIST)

    longs_all = []
    shorts_all = []

    for _, r in last.iterrows():
        sym = r["symbol"]
        if sym not in all_symbols:
            continue
        conf = float(r.get("confidence", 0) or 0)
        if conf < min_confidence:
            continue
        long_s = float(r.get("score_long", 0) or 0)
        short_s = float(r.get("score_short", 0) or 0)
        if long_s > 0:
            longs_all.append((r, build_pick(r, "LONG", sym)))
        if short_s > 0:
            shorts_all.append((r, build_pick(r, "SHORT", sym)))

    # Sort: prioritize ML filter pass, then score
    longs_all.sort(
        key=lambda x: (x[1]["ml_filter"], x[1]["score"], x[1]["confidence"]),
        reverse=True,
    )
    shorts_all.sort(
        key=lambda x: (x[1]["ml_filter"], x[1]["score"], x[1]["confidence"]),
        reverse=True,
    )

    # Recent win rate
    try:
        resolved = database.get_resolved_signals()
        n_resolved = len(resolved)
        win_rate = 0.5
        if n_resolved >= 3:
            n_tp = (resolved["status"] == "TP_HIT").sum()
            n_sl = (resolved["status"] == "SL_HIT").sum()
            decided = n_tp + n_sl
            if decided > 0:
                win_rate = float(n_tp) / decided
    except Exception:
        win_rate, n_resolved = 0.5, 0

    return {
        "timestamp": str(last_ts),
        "win_rate_history": win_rate,
        "n_resolved": n_resolved,
        "n_symbols_scanned": len(all_symbols),
        "n_features_last_cycle": len(last),
        "longs": [p for _, p in longs_all[:top_n]],
        "shorts": [p for _, p in shorts_all[:top_n]],
        # Counts for the report
        "n_longs_total": len(longs_all),
        "n_shorts_total": len(shorts_all),
    }


def format_dual_picks(picks: Dict) -> str:
    """Pretty-print dual-direction picks."""
    if "error" in picks:
        return f"❌ {picks['error']}"

    L = []
    L.append("=" * 78)
    L.append(f"🎯 PUMPHUNTER-AI  |  DUAL PICKS (LONG + SHORT)  |  {picks['timestamp']}")
    L.append("=" * 78)
    L.append(
        f"📊 Win Rate: {picks['win_rate_history']*100:.1f}% "
        f"(n={picks['n_resolved']} resolved)  |  "
        f"Scanned: {picks['n_symbols_scanned']} symbols  |  "
        f"Active: {picks['n_features_last_cycle']}"
    )
    L.append(
        f"🟢 Longs: {picks['n_longs_total']} qualified  |  "
        f"🔴 Shorts: {picks['n_shorts_total']} qualified"
    )
    L.append("")

    def render_pick(p, i):
        sym = p["symbol"]
        ml = "🧠ML" if p["ml_filter"] else "  "
        arrow = "🟢" if p["direction"] == "LONG" else "🔴"
        side = p["direction"]
        E = p["entry_price"]
        T = p["tp_price"]
        S = p["sl_price"]
        rr = p["rr_ratio"]
        sc = p["score"]
        cf = p["confidence"]
        rv = p["rvol"]
        m3 = p["mom_3_pct"]
        at = p["atr_pct"]
        se = p["smart_exit"]
        L.append(f"{i}. {arrow} {ml} {side:5s} {sym:12s} | conf {cf:.0f}% | score {sc:.0f}")
        L.append(
            f"   💰 E: {E}  🎯 TP: {T} (+{p['tp_pct']}%)  "
            f"🛡️ SL: {S} (-{p['sl_pct']}%)  R/R 1:{rr:.1f}"
        )
        L.append(
            f"   📈 mom3: {m3:+.1f}%  rvol: {rv:.2f}  ATR: {at:.1f}%  "
            f"BBsqueeze: {p['bb_squeeze']}"
        )
        L.append(
            f"   🔒 Smart: BE@{se['breakeven_at']}%  "
            f"L25@{se['lock_25_at']}%  L50@{se['lock_50_at']}%  "
            f"Trail{se['trail_pct']}%@{se['trail_activate_pct']}%  "
            f"MaxHold:{se['max_hold_hours']}h"
        )
        if se["reentry_on_dip"]:
            L.append(
                f"   ♻️ Re-entry: RSI<{se['reentry_rsi_max']} → add 30%"
            )
        L.append("")

    if picks["longs"]:
        L.append("🟢 LONG PICKS (best setups for upside):")
        L.append("-" * 78)
        for i, p in enumerate(picks["longs"], 1):
            render_pick(p, i)
    else:
        L.append("🟢 LONG PICKS: (none qualify under filter)")
        L.append("")

    if picks["shorts"]:
        L.append("🔴 SHORT PICKS (best setups for downside):")
        L.append("-" * 78)
        for i, p in enumerate(picks["shorts"], 1):
            render_pick(p, i)
    else:
        L.append("🔴 SHORT PICKS: (none qualify under filter)")
        L.append("")

    L.append("=" * 78)
    L.append("🧠 ML = pump_filter atr+flat (81.3% backtest accuracy)")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-conf", type=float, default=50.0)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    picks = get_dual_picks(args.min_conf, args.top)
    if args.json:
        print(json.dumps(picks, indent=2, default=str))
    else:
        print(format_dual_picks(picks))
