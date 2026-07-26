"""
Backtest ultra_strict v2 with extended indicators.

Compare:
  V1: ultra_strict (current, no extended)
  V2: ultra_strict v2 (with extended indicators)

On same dataset, same symbols, same TP/SL.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from data_quality import validate_ohlcv
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, volume_continuity, momentum_features, mtf_alignment,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from features import build_features
from btc_filter import BTCFilter
from scoring import rule_based_score
from extended_indicators import compute_all_extended

from ultra_strict_v2 import is_ultra_setup_v2, compute_extended_score
from anti_late import check_anti_late


def ultra_check_v1(pack):
    """Simplified ultra check matching backtest_v4 logic (no structure)."""
    feats = pack["feats_v1"]
    direction = pack["direction"]
    atr = float(feats.get("f_atr_pct", 0))
    mom_3 = float(feats.get("f_momentum_3_pct", 0))
    mom_6 = float(feats.get("f_momentum_6_pct", 0))
    btc_mom = float(feats.get("btc_momentum_12_pct", 0))
    confidence = float(feats.get("confidence", 0))
    if confidence < 60: return False
    if atr < 3 or atr > 8: return False
    if direction == "LONG" and mom_3 >= 8: return False
    if direction == "SHORT" and mom_3 <= -8: return False
    if abs(mom_6) < 1.5: return False
    if direction == "LONG" and btc_mom < -2: return False
    if direction == "SHORT" and btc_mom > 2: return False
    return True


def ultra_check_v2(pack):
    """V2: v1 + extended indicators."""
    if not ultra_check_v1(pack):
        return False
    feats = pack["feats_v2"]
    direction = pack["direction"]
    ex_delta, _ = compute_extended_score(feats, direction)
    confidence = float(feats.get("confidence", 0))
    adj_conf = confidence + ex_delta
    threshold = 60
    if ex_delta >= 20:
        threshold = 55
    if adj_conf < threshold:
        return False
    return True


def snapshot_full(df_4h, df_1h, df_15m, df_5m, idx, btc_state):
    """Build full features (v1 + extended) for one snapshot."""
    if idx < 60 or idx + 1 >= len(df_4h):
        return {}
    sub_4h = df_4h.iloc[: idx + 1].copy().reset_index(drop=True)
    sub_1h = df_1h.iloc[: min(len(df_1h), (idx + 1) * 4)].copy().reset_index(drop=True)
    if len(sub_4h) < 60:
        return {}
    q = validate_ohlcv(sub_4h, min_candles=60, interval_hours=4.0,
                       max_age_hours=1_000_000)
    if not q.ok or q.cleaned is None or q.cleaned.empty:
        return {}
    sub_4h = q.cleaned
    tech = technical_analysis(sub_4h)
    ind = {}
    ind.update(vwap_features(sub_4h))
    ind.update(atr_features(sub_4h))
    ind.update(bollinger_features(sub_4h))
    ind.update(relative_volume(sub_4h))
    ind.update(volume_continuity(sub_4h))
    ind.update(momentum_features(sub_4h))
    struct = structure_features(sub_4h)
    candle = candle_quality_features(sub_4h)
    mtf = mtf_alignment(sub_1h, sub_4h) if len(sub_1h) >= 30 else {
        "alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
        "aligned": False, "same_sign": False,
    }
    feats = build_features(tech, ind, struct, candle, mtf, btc_state)
    rb = rule_based_score(tech, ind, struct, candle, mtf, btc_state,
                          {"technical": 12, "momentum": 12, "volume": 18,
                           "vwap": 8, "atr_bb": 6, "structure": 10,
                           "candle": 8, "mtf": 8, "pattern": 8})
    mom_3 = float(ind.get("momentum_3_pct", 0))
    mom_6 = float(ind.get("momentum_6_pct", 0))
    # Use ABS value to test both directions
    if mom_3 > 1.0 and mom_6 > 0.5:
        direction = "LONG"
    elif mom_3 < -1.0 and mom_6 < -0.5:
        direction = "SHORT"
    else:
        # Use absolute mom_3 to decide
        if mom_3 > 0:
            direction = "LONG"
        else:
            direction = "SHORT"
    # Get extended features
    try:
        ex = compute_all_extended(sub_4h)
    except Exception:
        ex = {}
    # Build v1 features dict (legacy, no extended)
    feats_v1 = {
        "f_atr_pct": ind.get("atr_pct", 0),
        "f_rvol": ind.get("rvol", 1),
        "f_momentum_1_pct": ind.get("momentum_1_pct", 0),
        "f_momentum_3_pct": mom_3,
        "f_momentum_6_pct": mom_6,
        "f_momentum_12_pct": ind.get("momentum_12_pct", 0),
        "f_momentum_acceleration": ind.get("momentum_acceleration", 0),
        "f_a_ichi_above_cloud": feats.get("price_above_vwap", 0),  # proxy
        "f_a_ichi_below_cloud": 0,
        "f_bb_breakout_above": int(bool(ind.get("bb_breakout_above", False))),
        "f_bos_up": int(bool(struct.get("bos_up", False))),
        "f_atr_expanding": int(bool(ind.get("atr_expanding", False))),
        # Use 5m-derived proxy: rvol_15m as proxy
        "f_m_5m_volume_spike": int(volume_continuity(df_4h).get("sustained", False)),
        "f_m_obi_10": 1.0,  # default neutral
        "f_m_cvd": 0,
        "f_a_wave_position": "none",
        "f_candle_strength": candle.get("candle_strength", 0.5),
        "f_big_wick_top": int(bool(candle.get("big_wick_top", False))),
        "f_wick_body_ratio": 0.0,
        "confidence": rb["composite_score"],
        "btc_momentum_12_pct": btc_state.get("btc_momentum_12_pct", 0),
        "score_long": rb["composite_score"],
        "score_short": 100 - rb["composite_score"],
    }
    # v2: add extended
    feats_v2 = dict(feats_v1)
    for k, v in ex.items():
        feats_v2[f"f_{k}"] = v
    return {
        "direction": direction,
        "feats_v1": feats_v1,
        "feats_v2": feats_v2,
        "sub_4h": sub_4h,
    }


def simulate_trade(df_4h, idx, direction, *, tp_pct=2.5, sl_pct=2.0,
                   max_hold=24, trailing_pct=0.5):
    """Quick trade simulator (smart v2)."""
    if df_4h is None or df_4h.empty or idx + 1 >= len(df_4h):
        return {"win": False, "pnl": 0, "reason": "no_data"}
    entry = float(df_4h["close"].iloc[idx])
    sign = 1 if direction == "LONG" else -1
    tp_price = entry * (1 + sign * tp_pct / 100)
    sl_price = entry * (1 - sign * sl_pct / 100)
    best = entry
    current_sl = sl_price
    for i in range(idx + 1, min(idx + 1 + max_hold, len(df_4h))):
        bar = df_4h.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        if direction == "LONG":
            best = max(best, high)
            new_sl = best * (1 - trailing_pct / 100)
            if new_sl > current_sl:
                current_sl = new_sl
            # Lock at +2%
            lock = entry * 1.02
            if best >= lock and lock > current_sl:
                current_sl = lock
        else:
            best = min(best, low)
            new_sl = best * (1 + trailing_pct / 100)
            if new_sl < current_sl:
                current_sl = new_sl
            lock = entry * 0.98
            if best <= lock and lock < current_sl:
                current_sl = lock
        if direction == "LONG":
            if low <= current_sl:
                return {"win": current_sl > entry, "pnl": (current_sl - entry) / entry * 100,
                        "reason": "be_lock" if current_sl >= entry else "sl"}
            if high >= tp_price:
                return {"win": True, "pnl": tp_pct, "reason": "tp"}
        else:
            if high >= current_sl:
                return {"win": current_sl < entry, "pnl": (entry - current_sl) / entry * 100,
                        "reason": "be_lock" if current_sl <= entry else "sl"}
            if low <= tp_price:
                return {"win": True, "pnl": tp_pct, "reason": "tp"}
    return {"win": False, "pnl": 0, "reason": "max_hold"}


DEFAULT_SYMBOLS = [
    "ALICEUSDT", "TLMUSDT", "HYPERUSDT", "WCTUSDT", "RECALLUSDT",
    "GROVEUSDT", "RESOLVUSDT", "TOWNSUSDT", "OPNUSDT", "TUTUSDT",
    "FIGHTUSDT", "PARTIUSDT", "USATUSDT", "BREVUSDT", "EVAAUSDT",
    "BANKUSDT", "ACEUSDT",
]


def run_compare(symbols, days=60, snap_every=6, tp=2.5, sl=2.0, trail=0.5,
                out_dir="backtest_ultra_v2"):
    os.makedirs(out_dir, exist_ok=True)
    toobit = ToobitClient()
    btc_state = BTCFilter(toobit).evaluate()
    print(f"[compare] BTC: {btc_state['state']}")
    v1_trades = []
    v2_trades = []
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[compare] ({s_idx}/{len(symbols)}) {sym}...", end=" ", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", days * 24 + 50)
            df_15m = toobit.get_klines(sym, "15m", days * 96 + 50)
            df_5m = toobit.get_klines(sym, "5m", days * 288 + 50)
        except Exception as e:
            print(f"err: {e}")
            continue
        if df_4h.empty or len(df_4h) < 80:
            print("too few")
            continue
        n_v1 = 0
        n_v2 = 0
        for idx in range(60, len(df_4h) - 24, snap_every):
            pack = snapshot_full(df_4h, df_1h, df_15m, df_5m, idx, btc_state)
            if not pack:
                continue
            direction = pack["direction"]
            # V1: simplified ultra check
            ok_v1 = ultra_check_v1(pack)
            if ok_v1:
                t = simulate_trade(df_4h, idx, direction,
                                   tp_pct=tp, sl_pct=sl, trailing_pct=trail)
                v1_trades.append({
                    "symbol": sym, "idx": idx, "direction": direction,
                    "win": t["win"], "pnl": t["pnl"], "reason": t["reason"],
                })
                n_v1 += 1
            # V2: with extended
            ok_v2 = ultra_check_v2(pack)
            if ok_v2:
                adj_conf = float(pack["feats_v2"].get("confidence", 0))
                # Compute ex_delta for logging
                ex_delta, _ = compute_extended_score(
                    pack["feats_v2"], direction
                )
                adj_conf = adj_conf + ex_delta
                t = simulate_trade(df_4h, idx, direction,
                                   tp_pct=tp, sl_pct=sl, trailing_pct=trail)
                v2_trades.append({
                    "symbol": sym, "idx": idx, "direction": direction,
                    "win": t["win"], "pnl": t["pnl"], "reason": t["reason"],
                    "adj_conf": adj_conf,
                })
                n_v2 += 1
        print(f"v1={n_v1} v2={n_v2}")
        import time
        time.sleep(0.4)
    # Stats
    def stats(trades, name):
        if not trades:
            return {"name": name, "n": 0}
        df = pd.DataFrame(trades)
        n = len(df)
        wins = (df["win"] == True).sum()  # noqa
        wr = wins / n if n > 0 else 0
        pnl = df["pnl"].sum()
        avg = df["pnl"].mean()
        gp = df.loc[df["pnl"] > 0, "pnl"].sum()
        gl = abs(df.loc[df["pnl"] < 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else float('inf')
        return {
            "name": name, "n": int(n), "wins": int(wins),
            "win_rate": float(wr), "total_pnl": float(pnl),
            "avg_pnl": float(avg), "profit_factor": float(pf),
        }
    s1 = stats(v1_trades, "V1 (no extended)")
    s2 = stats(v2_trades, "V2 (with extended)")
    print("\n" + "="*100)
    print(f"{'Version':<25} {'N':>5} {'Wins':>5} {'WR%':>6} {'Total':>9} {'Avg':>7} {'PF':>7}")
    print("="*100)
    for s in [s1, s2]:
        print(f"{s['name']:<25} {s['n']:>5} {s['wins']:>5} "
              f"{s['win_rate']*100:>6.1f} {s['total_pnl']:>+8.1f}% "
              f"{s['avg_pnl']:>+6.2f}% {s['profit_factor']:>7.2f}")
    print("="*100)
    if s1['n'] > 0 and s2['n'] > 0:
        wr_diff = s2['win_rate'] - s1['win_rate']
        pnl_diff = s2['total_pnl'] - s1['total_pnl']
        print(f"\nV2 vs V1: WR {wr_diff*100:+.1f}pp, P&L {pnl_diff:+.1f}%, "
              f"n {s2['n'] - s1['n']:+d}")
    # Save
    with open(os.path.join(out_dir, "compare.json"), "w") as f:
        json.dump({
            "v1": s1, "v2": s2,
            "params": {"days": days, "tp": tp, "sl": sl, "trail": trail,
                       "snap_every": snap_every},
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)
    if v1_trades:
        pd.DataFrame(v1_trades).to_csv(
            os.path.join(out_dir, "v1_trades.csv"), index=False)
    if v2_trades:
        pd.DataFrame(v2_trades).to_csv(
            os.path.join(out_dir, "v2_trades.csv"), index=False)
    return {"v1": s1, "v2": s2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--top", type=int, default=17)
    p.add_argument("--snap", type=int, default=6)
    p.add_argument("--tp", type=float, default=2.5)
    p.add_argument("--sl", type=float, default=2.0)
    p.add_argument("--trail", type=float, default=0.5)
    p.add_argument("--out", type=str, default="backtest_ultra_v2")
    args = p.parse_args()
    symbols = DEFAULT_SYMBOLS[: args.top]
    run_compare(symbols, days=args.days, snap_every=args.snap,
                tp=args.tp, sl=args.sl, trail=args.trail, out_dir=args.out)


if __name__ == "__main__":
    main()
