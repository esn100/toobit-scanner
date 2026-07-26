"""
Strategy comparison backtest for PumpHunter-AI.

Tests MULTIPLE strategies on the same dataset to find the best one:
  S1: ULTRA-STRICT (current: conf>=60, ATR 3-8, mom3<8, mom6>1.5)
  S2: LOOSE (more signals, lower threshold)
  S3: MOMENTUM-ONLY (only mom_3 + rvol)
  S4: VOLUME-EXPLOSION (only rvol + 5m spike)
  S5: ANTI-LATE-ONLY (momentum anti-late, no ultra filter)
  S6: BREAKOUT (BB breakout + rvol)
  S7: PULLBACK (mom_3 between -3 and +1, buy dip)
  S8: TREND-FOLLOWING (mom_6 > 2, rvol > 1.2, in_range=False)

Each strategy is tested on same dataset with same TP/SL grid.
Output: data/backtest_strategies.csv + report
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from data_quality import validate_ohlcv
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, volume_continuity, momentum_features,
    mtf_alignment,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from features import build_features
from btc_filter import BTCFilter
from scoring import rule_based_score

from anti_late import check_anti_late


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
STRATEGIES = {
    "S1_ULTRA_STRICT": {
        "name": "Ultra Strict (current)",
        "description": "conf>=60, ATR 3-8, mom3<8, mom6>1.5, anti-late pass",
    },
    "S2_LOOSE": {
        "name": "Loose (more signals)",
        "description": "conf>=40, ATR 2-12, no mom filter",
    },
    "S3_MOMENTUM": {
        "name": "Momentum only",
        "description": "mom_3 > 2%, rvol > 1.3, no other filters",
    },
    "S4_VOLUME": {
        "name": "Volume explosion",
        "description": "rvol_4h > 1.5 AND rvol_15m > 2.0",
    },
    "S5_ANTI_LATE": {
        "name": "Anti-late only",
        "description": "no ultra filter, just anti-late (mom1 < 4, mom3 < 10)",
    },
    "S6_BREAKOUT": {
        "name": "BB Breakout",
        "description": "bb_breakout + rvol > 1.5 + mom6 > 0",
    },
    "S7_PULLBACK": {
        "name": "Pullback (buy dip)",
        "description": "mom_3 in [-3, +1], rvol > 1.0, atr > 3",
    },
    "S8_TREND_FOLLOW": {
        "name": "Trend following",
        "description": "mom_6 > 2, rvol > 1.2, NOT in range",
    },
    "S9_CONF_HIGH": {
        "name": "High confidence only",
        "description": "conf >= 70, no other filter",
    },
    "S10_MULTI": {
        "name": "Multi-factor",
        "description": "rvol > 1.3 AND mom6 > 1 AND atr in 3-8",
    },
}


def apply_strategy(row: pd.Series, strategy: str) -> Tuple[bool, str]:
    """Return (passes, reason_if_fail) for a given row and strategy."""
    conf = float(row.get("confidence", 0))
    atr = float(row.get("f_atr_pct", row.get("ind_atr_pct", 0)))
    mom_3 = float(row.get("f_momentum_3_pct", row.get("ind_momentum_3_pct", 0)))
    mom_6 = float(row.get("f_momentum_6_pct", row.get("ind_momentum_6_pct", 0)))
    mom_12 = float(row.get("f_momentum_12_pct", row.get("ind_momentum_12_pct", 0)))
    rvol = float(row.get("f_rvol", row.get("ind_rvol", 1)))
    bb_break = float(row.get("f_bb_breakout_above", 0))
    bos_up = float(row.get("f_bos_up", 0))
    in_range = float(row.get("f_in_range", 1))
    direction = row.get("direction", "LONG")
    if strategy == "S1_ULTRA_STRICT":
        if conf < 60: return False, "conf<60"
        if atr < 3 or atr > 8: return False, f"atr({atr:.1f})"
        if direction == "LONG" and mom_3 >= 8: return False, f"mom3_ext({mom_3:.1f})"
        if direction == "SHORT" and mom_3 <= -8: return False, f"mom3_ext({mom_3:.1f})"
        if abs(mom_6) < 1.5: return False, f"chop({mom_6:.1f})"
        # anti-late
        anti_feats = {
            "momentum_1_pct": float(row.get("f_momentum_1_pct", row.get("ind_momentum_1_pct", 0))),
            "momentum_3_pct": mom_3,
            "candle_strength": float(row.get("f_candle_strength", 0.5)),
            "big_wick_top": bool(row.get("f_big_wick_top", 0)),
        }
        ok, pen, why = check_anti_late(anti_feats, direction)
        if not ok: return False, f"anti_late_{why}"
        return True, ""
    if strategy == "S2_LOOSE":
        if conf < 40: return False, "conf<40"
        if atr < 2 or atr > 12: return False, f"atr({atr:.1f})"
        if direction == "LONG" and mom_3 >= 15: return False, "too_ext"
        if direction == "SHORT" and mom_3 <= -15: return False, "too_ext"
        return True, ""
    if strategy == "S3_MOMENTUM":
        if direction == "LONG" and mom_3 <= 2: return False, "no_mom"
        if direction == "SHORT" and mom_3 >= -2: return False, "no_mom"
        if rvol <= 1.3: return False, "low_rvol"
        return True, ""
    if strategy == "S4_VOLUME":
        rvol_15 = float(row.get("f_rvol", 1))
        if rvol <= 1.5: return False, "rvol_4h_low"
        if rvol_15 <= 2.0: return False, "rvol_15m_low"
        return True, ""
    if strategy == "S5_ANTI_LATE":
        anti_feats = {
            "momentum_1_pct": float(row.get("f_momentum_1_pct", row.get("ind_momentum_1_pct", 0))),
            "momentum_3_pct": mom_3,
            "candle_strength": float(row.get("f_candle_strength", 0.5)),
            "big_wick_top": bool(row.get("f_big_wick_top", 0)),
        }
        ok, pen, why = check_anti_late(anti_feats, direction)
        if not ok: return False, f"anti_late_{why}"
        return True, ""
    if strategy == "S6_BREAKOUT":
        if not bb_break: return False, "no_bb_break"
        if rvol <= 1.5: return False, "low_rvol"
        if mom_6 <= 0: return False, "no_mom"
        return True, ""
    if strategy == "S7_PULLBACK":
        # Buy the dip
        if mom_3 < -3 or mom_3 > 1: return False, f"mom3({mom_3:.1f})"
        if rvol < 1.0: return False, "low_rvol"
        if atr < 3: return False, "low_atr"
        if direction == "SHORT": return False, "wrong_dir"  # pullback is for LONG
        return True, ""
    if strategy == "S8_TREND_FOLLOW":
        if abs(mom_6) < 2: return False, "no_trend"
        if rvol < 1.2: return False, "low_rvol"
        if in_range: return False, "in_range"
        # direction aligned with mom
        if direction == "LONG" and mom_6 < 0: return False, "wrong_dir"
        if direction == "SHORT" and mom_6 > 0: return False, "wrong_dir"
        return True, ""
    if strategy == "S9_CONF_HIGH":
        if conf < 70: return False, "conf<70"
        return True, ""
    if strategy == "S10_MULTI":
        if rvol < 1.3: return False, "rvol"
        if abs(mom_6) < 1: return False, "mom6"
        if atr < 3 or atr > 8: return False, "atr"
        return True, ""
    return False, "unknown_strategy"


# ---------------------------------------------------------------------------
# Simulate TP/SL with smart v2 logic
# ---------------------------------------------------------------------------
def simulate_trade(
    df: pd.DataFrame, idx: int, direction: str, *,
    tp_pct: float, sl_pct: float, max_hold: int = 24,
    trailing_pct: float = 0.5,
    use_breakeven: bool = True,
    be_trigger: float = 1.0,
    lock_trigger: float = 2.5,
    lock_pct: float = 1.0,
) -> Dict:
    """Simulate trade with smart exit v2 logic."""
    empty = {"peak": 0, "dd": 0, "exit_pct": 0, "exit_reason": "no_data",
             "exit_bars": 0, "ok": False}
    if df is None or df.empty or idx + 1 >= len(df):
        return empty
    entry = float(df["close"].iloc[idx])
    if entry <= 0:
        return empty
    sign = 1 if direction == "LONG" else -1
    tp_price = entry * (1 + sign * tp_pct / 100.0)
    sl_price = entry * (1 - sign * sl_pct / 100.0)
    be_price = entry * (1 + sign * be_trigger / 100.0)
    lock_price = entry * (1 + sign * lock_pct / 100.0)
    best = entry
    current_sl = sl_price
    exit_pct = 0
    exit_reason = "max_hold"
    exit_bars = max_hold
    for i in range(idx + 1, min(idx + 1 + max_hold, len(df))):
        bar = df.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        if direction == "LONG":
            best = max(best, high)
        else:
            best = min(best, low)
        # Trailing SL
        if direction == "LONG":
            new_sl = best * (1 - trailing_pct / 100.0)
            if new_sl > current_sl:
                current_sl = new_sl
        else:
            new_sl = best * (1 + trailing_pct / 100.0)
            if new_sl < current_sl:
                current_sl = new_sl
        # Breakeven lock
        if use_breakeven:
            if direction == "LONG" and best >= be_price and be_price > current_sl:
                current_sl = be_price
            elif direction == "SHORT" and best <= be_price and be_price < current_sl:
                current_sl = be_price
        # Lock profit
        if direction == "LONG" and best >= entry * (1 + lock_trigger/100) and lock_price > current_sl:
            current_sl = lock_price
        elif direction == "SHORT" and best <= entry * (1 - lock_trigger/100) and lock_price < current_sl:
            current_sl = lock_price
        # Check SL
        if direction == "LONG" and low <= current_sl:
            exit_pct = (current_sl - entry) / entry * 100
            exit_reason = "breakeven_lock" if current_sl >= entry else "stop_loss"
            exit_bars = i - idx
            break
        elif direction == "SHORT" and high >= current_sl:
            exit_pct = (entry - current_sl) / entry * 100
            exit_reason = "breakeven_lock" if current_sl <= entry else "stop_loss"
            exit_bars = i - idx
            break
        # Check TP
        if direction == "LONG" and high >= tp_price:
            exit_pct = tp_pct
            exit_reason = "take_profit"
            exit_bars = i - idx
            break
        elif direction == "SHORT" and low <= tp_price:
            exit_pct = tp_pct
            exit_reason = "take_profit"
            exit_bars = i - idx
            break
    else:
        last = min(idx + max_hold, len(df) - 1)
        close = float(df["close"].iloc[last])
        exit_pct = (close - entry) / entry * 100 * sign
        exit_reason = "max_hold"
        exit_bars = last - idx
    # Peak and DD
    forward = df.iloc[idx + 1: idx + 1 + exit_bars]
    if forward.empty:
        peak = exit_pct
        dd = exit_pct
    else:
        highs = forward["high"].astype(float)
        lows = forward["low"].astype(float)
        if direction == "LONG":
            peak = float(((highs - entry) / entry * 100).max())
            dd = float(((lows - entry) / entry * 100).min())
        else:
            peak = float(((entry - lows) / entry * 100).max())
            dd = float(((entry - highs) / entry * 100).max())
    return {"peak": peak, "dd": dd, "exit_pct": exit_pct,
            "exit_reason": exit_reason, "exit_bars": exit_bars, "ok": True}


# ---------------------------------------------------------------------------
# Build features for a snapshot (same as backtest_v4)
# ---------------------------------------------------------------------------
def snapshot_features(df_4h, df_1h, df_15m, df_5m, idx, btc_state):
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
    if mom_3 > 1.0 and mom_6 > 0.5:
        direction = "LONG"
    elif mom_3 < -1.0 and mom_6 < -0.5:
        direction = "SHORT"
    else:
        direction = "LONG"
    return {
        "direction": direction,
        "composite_score": rb["composite_score"],
        "features": feats,
        "indicators": ind,
        "structure": struct,
        "candle": candle,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_strategy_backtest(
    symbols: List[str],
    days: int = 60,
    snap_every: int = 6,
    max_hold: int = 24,
    out_dir: str = "backtest_strategies",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    toobit = ToobitClient()
    btc_state = BTCFilter(toobit).evaluate()
    print(f"[strat] BTC: {btc_state['state']}, symbols={len(symbols)}, days={days}")
    # Collect features for each snapshot
    all_snaps = []
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[strat] ({s_idx}/{len(symbols)}) {sym}...", end=" ", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", days * 24 + 50)
            df_15m = toobit.get_klines(sym, "15m", days * 96 + 50)
            df_5m = toobit.get_klines(sym, "5m", days * 288 + 50)
        except Exception as e:
            print(f"err: {e}")
            continue
        if df_4h.empty or len(df_4h) < 80:
            print(f"too few ({len(df_4h)})")
            continue
        max_idx = len(df_4h) - 1 - max_hold
        n_snap = 0
        for idx in range(60, max_idx + 1, snap_every):
            pack = snapshot_features(df_4h, df_1h, df_15m, df_5m, idx, btc_state)
            if not pack:
                continue
            n_snap += 1
            rec = {
                "symbol": sym,
                "idx": idx,
                "direction": pack["direction"],
                "composite_score": pack["composite_score"],
                "confidence": pack["composite_score"],
            }
            for k, v in pack["features"].items():
                rec[k] = v
            for k, v in pack["indicators"].items():
                rec[f"ind_{k}"] = v if not isinstance(v, bool) else int(v)
            for k, v in pack["candle"].items():
                rec[f"candle_{k}"] = v if not isinstance(v, bool) else int(v)
            for k, v in pack["structure"].items():
                rec[f"struct_{k}"] = v if not isinstance(v, bool) else int(v)
            all_snaps.append((rec, df_4h, idx))
        print(f"{n_snap} snaps")
        time.sleep(0.4)
    if not all_snaps:
        print("[strat] no data")
        return {}
    print(f"\n[strat] {len(all_snaps)} snapshots total")
    # For each strategy, test multiple TP/SL combinations
    results = []
    tp_sl_grid = [
        (2.0, 1.5, 0.5),   # TP, SL, trail
        (2.5, 2.0, 0.5),
        (3.0, 2.0, 0.5),
        (5.0, 3.0, 1.5),
        (7.0, 3.0, 1.5),
    ]
    for strat in STRATEGIES.keys():
        for tp, sl, tr in tp_sl_grid:
            wins, losses, breakeven = 0, 0, 0
            pnl_list = []
            signals = 0
            for row, df_4h, idx in all_snaps:
                ok, why = apply_strategy(row, strat)
                if not ok:
                    continue
                signals += 1
                result = simulate_trade(
                    df_4h, idx, row["direction"],
                    tp_pct=tp, sl_pct=sl, trailing_pct=tr,
                    max_hold=max_hold,
                )
                ep = result["exit_pct"]
                pnl_list.append(ep)
                if ep > 0.05:
                    wins += 1
                elif ep < -0.05:
                    losses += 1
                else:
                    breakeven += 1
            if signals == 0:
                continue
            wr = wins / signals
            total_pnl = sum(pnl_list)
            avg_pnl = total_pnl / signals
            gp = sum(p for p in pnl_list if p > 0)
            gl = abs(sum(p for p in pnl_list if p < 0))
            pf = gp / gl if gl > 0 else 0
            results.append({
                "strategy": strat,
                "strategy_name": STRATEGIES[strat]["name"],
                "tp_pct": tp, "sl_pct": sl, "trail_pct": tr,
                "n_signals": signals,
                "n_wins": wins, "n_losses": losses, "n_breakeven": breakeven,
                "win_rate": wr,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "profit_factor": pf,
                "signals_per_day": signals / days / len(symbols) * 24 / snap_every,
            })
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_rate", ascending=False)
    df_results.to_csv(os.path.join(out_dir, "strategy_results.csv"), index=False)
    # Best per strategy
    best_per = (df_results.sort_values("win_rate", ascending=False)
                .drop_duplicates(["strategy"]).sort_values("win_rate", ascending=False))
    best_per.to_csv(os.path.join(out_dir, "best_per_strategy.csv"), index=False)
    # Print top
    print("\n" + "="*100)
    print(f"{'Strategy':<25} {'TP':>5} {'SL':>5} {'Tr':>4} {'Signals':>8} {'WR%':>6} {'Total':>8} {'Avg':>7} {'PF':>7}")
    print("="*100)
    for _, r in best_per.iterrows():
        print(f"{r['strategy']:<25} {r['tp_pct']:>5.1f} {r['sl_pct']:>5.1f} "
              f"{r['trail_pct']:>4.1f} {r['n_signals']:>8.0f} {r['win_rate']*100:>6.1f} "
              f"{r['total_pnl']:>+7.1f}% {r['avg_pnl']:>+6.2f}% {r['profit_factor']:>7.2f}")
    print("="*100)
    # Overall best (top 5)
    print("\n## TOP 5 OVERALL:")
    print(df_results.head(5).to_string(index=False))
    return {"df": df_results, "out_dir": out_dir}


DEFAULT_SYMBOLS = [
    "ALICEUSDT", "TLMUSDT", "HYPERUSDT", "WCTUSDT", "RECALLUSDT",
    "GROVEUSDT", "RESOLVUSDT", "TOWNSUSDT", "OPNUSDT", "TUTUSDT",
    "FIGHTUSDT", "PARTIUSDT", "USATUSDT", "BREVUSDT", "EVAAUSDT",
    "BANKUSDT", "ACEUSDT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--top", type=int, default=17)
    parser.add_argument("--snap", type=int, default=6)
    parser.add_argument("--out", type=str, default="backtest_strategies")
    args = parser.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS[: args.top]
    print(f"[strat] {len(symbols)} symbols, {args.days} days, snap every {args.snap}")
    res = run_strategy_backtest(symbols, days=args.days, snap_every=args.snap, out_dir=args.out)
    if res:
        print(f"\n[strat] saved to {args.out}/strategy_results.csv")


if __name__ == "__main__":
    main()
