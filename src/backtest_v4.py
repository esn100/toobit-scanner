"""
Backtest v4: pump hunter with the improved universe and prefilter.

This is the main validation tool — runs the FULL pipeline:
  universe (improved) -> prefilter_v2 -> anti_late -> ultra_strict -> win rate

Walks forward through 4h klines for each symbol, runs the entire filter
chain, simulates TP/SL hits, and reports win rate + P&L.

Output:
  - data/backtest_v4_dataset.csv   (raw rows with all features + label)
  - data/backtest_v4_results.json  (metrics summary)
  - data/backtest_v4_report.md     (human-readable)

Usage:
    python -m src.backtest_v4 --days 60 --top 20
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
from features import build_features, FEATURE_NAMES
from btc_filter import BTCFilter
from scoring import rule_based_score

# NEW: Improved modules
from universe import (
    discover_small_caps, MarketCapResolver, MAJORS_BLACKLIST,
)
from prefilter_v2 import prefilter_score_v2, passes_prefilter_v2
from anti_late import check_anti_late


# ---------------------------------------------------------------------------
# TP/SL simulation
# ---------------------------------------------------------------------------
def simulate_tp_sl(
    df: pd.DataFrame, snapshot_idx: int, *,
    tp_pct: float = 5.0, sl_pct: float = 3.0,
    max_hold_bars: int = 24,  # 4h * 24 = 96h
    trailing_pct: float = 1.5,
    use_trailing: bool = True,
    use_breakeven: bool = True,
    breakeven_trigger: float = 1.5,  # move to entry at +1.5%
    lock_pct: float = 2.0,           # lock at +2% after +5%
    lock_trigger: float = 5.0,
) -> Dict:
    """
    Walk forward from snapshot_idx and simulate trade.
    Returns exit info: peak, dd, exit_pct, exit_reason, exit_bars.
    """
    empty = {
        "peak_pct": 0.0, "dd_pct": 0.0, "exit_pct": 0.0,
        "exit_reason": "no_data", "exit_bars": 0, "ok": False,
    }
    if df is None or df.empty:
        return empty
    if snapshot_idx + 1 >= len(df):
        return empty
    entry = float(df["close"].iloc[snapshot_idx])
    if entry <= 0:
        return empty
    tp_price = entry * (1 + tp_pct / 100.0)
    sl_price = entry * (1 - sl_pct / 100.0)
    be_price = entry * (1 + breakeven_trigger / 100.0)
    lock_price = entry * (1 + lock_pct / 100.0)
    high_since = entry
    current_sl = sl_price
    exit_pct = 0.0
    exit_reason = "max_hold"
    exit_bars = max_hold_bars
    for i in range(snapshot_idx + 1, min(snapshot_idx + 1 + max_hold_bars, len(df))):
        bar = df.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        high_since = max(high_since, high)
        # Update trailing SL
        if use_trailing:
            new_sl = high_since * (1 - trailing_pct / 100.0)
            if new_sl > current_sl:
                current_sl = new_sl
        # Lock profit at +lock_pct if high hit +lock_trigger
        if use_breakeven and high_since >= lock_price and lock_price > current_sl:
            current_sl = lock_price
        # Check SL
        if low <= current_sl:
            exit_pct = (current_sl - entry) / entry * 100.0
            exit_reason = "breakeven_lock" if current_sl >= entry else "stop_loss"
            if current_sl >= entry and current_sl < be_price:
                exit_reason = "breakeven_lock"
            exit_bars = i - snapshot_idx
            break
        # Check TP
        if high >= tp_price:
            exit_pct = tp_pct
            exit_reason = "take_profit"
            exit_bars = i - snapshot_idx
            break
    else:
        # No exit: close at last available bar
        last = min(snapshot_idx + max_hold_bars, len(df) - 1)
        close = float(df["close"].iloc[last])
        exit_pct = (close - entry) / entry * 100.0
        exit_reason = "max_hold"
        exit_bars = last - snapshot_idx
    # Peak and DD over the whole window
    forward = df.iloc[snapshot_idx + 1: snapshot_idx + 1 + exit_bars]
    if forward.empty:
        peak = exit_pct
        dd = exit_pct
    else:
        highs = forward["high"].astype(float)
        lows = forward["low"].astype(float)
        peak = float(((highs - entry) / entry * 100.0).max())
        dd = float(((lows - entry) / entry * 100.0).min())
    return {
        "peak_pct": peak, "dd_pct": dd, "exit_pct": exit_pct,
        "exit_reason": exit_reason, "exit_bars": exit_bars, "ok": True,
    }


# ---------------------------------------------------------------------------
# Per-snapshot feature build (with full filter chain)
# ---------------------------------------------------------------------------
def snapshot_full_pipeline(
    sym: str,
    df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m: pd.DataFrame,
    df_5m: pd.DataFrame, btc_df_1h: pd.DataFrame, idx: int,
    btc_state: dict,
) -> Dict:
    """
    Build features + apply ALL filters for one snapshot.
    Returns dict with all features and a 'passed_ultra' flag.
    """
    if idx < 60 or idx + 1 >= len(df_4h):
        return {}
    sub_4h = df_4h.iloc[: idx + 1].copy().reset_index(drop=True)
    sub_1h = df_1h.iloc[: min(len(df_1h), (idx + 1) * 4)].copy().reset_index(drop=True)
    if len(sub_4h) < 60:
        return {}
    # Quality
    q = validate_ohlcv(sub_4h, min_candles=60, interval_hours=4.0,
                       max_age_hours=1_000_000)
    if not q.ok or q.cleaned is None or q.cleaned.empty:
        return {}
    sub_4h = q.cleaned
    # 1) Prefilter v2 (fast)
    try:
        pf = prefilter_score_v2(df_5m.iloc[: 12 * (idx + 1)] if not df_5m.empty else df_5m,
                                 sub_1h, sub_4h,
                                 df_15m.iloc[: 4 * (idx + 1)] if not df_15m.empty else df_15m,
                                 btc_df_1h.iloc[: 24 * (idx + 1)] if not btc_df_1h.empty else btc_df_1h)
        pf_pass = passes_prefilter_v2(pf, min_score=25.0)
    except Exception:
        pf_pass = True
        pf = {"prefilter_v2": 0.0, "flags": {}, "rejects": []}
    # 2) Indicators + features
    tech = technical_analysis(sub_4h)
    ind: Dict = {}
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
    # 3) Anti-late
    anti_feats = {
        "momentum_1_pct": float(ind.get("momentum_1_pct", 0)),
        "momentum_3_pct": float(ind.get("momentum_3_pct", 0)),
        "momentum_6_pct": float(ind.get("momentum_6_pct", 0)),
        "momentum_acceleration": float(ind.get("momentum_acceleration", 0)),
        "candle_strength": float(candle.get("candle_strength", 0.5)),
        "big_wick_top": bool(candle.get("big_wick_top", False)),
        "wick_body_ratio": 0.0,
    }
    # Direction from features
    mom_3 = float(ind.get("momentum_3_pct", 0))
    mom_6 = float(ind.get("momentum_6_pct", 0))
    if mom_3 > 1.0 and mom_6 > 0.5:
        direction = "LONG"
    elif mom_3 < -1.0 and mom_6 < -0.5:
        direction = "SHORT"
    else:
        direction = "LONG"  # default to LONG in backtest
    anti_late_pass, anti_late_pen, anti_late_reason = check_anti_late(anti_feats, direction)
    # 4) Ultra-strict simplified
    confidence = rb["composite_score"]
    score_long = confidence
    score_short = 100 - confidence
    # For simplicity, ultra_strict test
    atr_v = float(ind.get("atr_pct", 0))
    mom_3_v = float(ind.get("momentum_3_pct", 0))
    mom_6_v = float(ind.get("momentum_6_pct", 0))
    rvol_v = float(ind.get("rvol", 0))
    btc_mom = float(btc_state.get("btc_momentum_12_pct", 0))
    ichi_above = float(feats.get("price_above_vwap", 0)) > 0.5  # proxy
    bb_breakout = bool(ind.get("bb_breakout_above", False))
    atr_exp = bool(ind.get("atr_expanding", False))
    ultra_pass = True
    ultra_reasons = []
    if not pf_pass:
        ultra_pass = False
        ultra_reasons.append("prefilter")
    if not anti_late_pass:
        ultra_pass = False
        ultra_reasons.append(f"anti_late_{anti_late_reason}")
    if confidence < 60:
        ultra_pass = False
        ultra_reasons.append(f"low_conf({confidence:.0f})")
    if atr_v < 3 or atr_v > 8:
        ultra_pass = False
        ultra_reasons.append(f"atr({atr_v:.1f})")
    if direction == "LONG" and mom_3_v >= 8:
        ultra_pass = False
        ultra_reasons.append(f"mom3_ext({mom_3_v:.1f})")
    if abs(mom_6_v) < 1.5:
        ultra_pass = False
        ultra_reasons.append(f"chop_mom6({mom_6_v:.1f})")
    if direction == "LONG" and btc_mom < -2:
        ultra_pass = False
        ultra_reasons.append("btc_bear")
    return {
        "passed_prefilter": pf_pass,
        "passed_anti_late": anti_late_pass,
        "passed_ultra": ultra_pass,
        "ultra_reasons": ",".join(ultra_reasons),
        "direction": direction,
        "prefilter_score": pf.get("prefilter_v2", 0.0),
        "prefilter_rejects": ",".join(pf.get("rejects", [])),
        "anti_late_pen": anti_late_pen,
        "anti_late_reason": anti_late_reason,
        "composite_score": rb["composite_score"],
        "sub_scores": rb["sub_scores"],
        "features": feats,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_backtest_v4(
    symbols: List[str],
    days: int = 60,
    snapshot_every_bars: int = 6,  # every 24h on 4h
    max_hold_bars: int = 24,       # 96h max hold
    tp_pct: float = 5.0,
    sl_pct: float = 3.0,
    trailing_pct: float = 1.5,
    out_dir: str = "backtest_results_v4",
) -> Dict:
    """Run the full pipeline backtest."""
    os.makedirs(out_dir, exist_ok=True)
    toobit = ToobitClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[v4] BTC state: {btc_state['state']}", flush=True)
    # Get BTC 1h
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", days * 24 + 50)
    rows: List[Dict] = []
    trade_rows: List[Dict] = []
    total = len(symbols)
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[v4] ({s_idx}/{total}) {sym} ...", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", days * 24 + 50)
            df_15m = toobit.get_klines(sym, "15m", days * 96 + 50)
            df_5m = toobit.get_klines(sym, "5m", days * 288 + 50)
        except Exception as e:
            print(f"[v4]   fetch error: {e}", flush=True)
            continue
        if df_4h.empty or len(df_4h) < 80:
            print(f"[v4]   too few candles ({len(df_4h)})", flush=True)
            continue
        max_idx = len(df_4h) - 1 - max_hold_bars
        n_snap = 0
        n_passed_ultra = 0
        for idx in range(60, max_idx + 1, snapshot_every_bars):
            pack = snapshot_full_pipeline(
                sym, df_4h, df_1h, df_15m, df_5m, btc_df_1h, idx, btc_state
            )
            if not pack:
                continue
            n_snap += 1
            # Compute trade outcome (if passed ultra)
            if pack["passed_ultra"]:
                outcome = simulate_tp_sl(
                    df_4h, idx,
                    tp_pct=tp_pct, sl_pct=sl_pct,
                    max_hold_bars=max_hold_bars,
                    trailing_pct=trailing_pct,
                )
                n_passed_ultra += 1
            else:
                outcome = {
                    "peak_pct": 0.0, "dd_pct": 0.0, "exit_pct": 0.0,
                    "exit_reason": "not_taken", "exit_bars": 0, "ok": False,
                }
            rec = {
                "symbol": sym,
                "idx": idx,
                "timestamp": df_4h["open_time"].iloc[idx].isoformat()
                if hasattr(df_4h["open_time"].iloc[idx], "isoformat")
                else str(df_4h["open_time"].iloc[idx]),
                "passed_prefilter": int(pack["passed_prefilter"]),
                "passed_anti_late": int(pack["passed_anti_late"]),
                "passed_ultra": int(pack["passed_ultra"]),
                "ultra_reasons": pack["ultra_reasons"],
                "direction": pack["direction"],
                "prefilter_score": pack["prefilter_score"],
                "composite_score": pack["composite_score"],
                "peak_pct": outcome["peak_pct"],
                "dd_pct": outcome["dd_pct"],
                "exit_pct": outcome["exit_pct"],
                "exit_reason": outcome["exit_reason"],
                "exit_bars": outcome["exit_bars"],
            }
            for k, v in pack["features"].items():
                rec[k] = v
            rows.append(rec)
            if pack["passed_ultra"]:
                trade_rows.append({
                    "symbol": sym,
                    "timestamp": rec["timestamp"],
                    "direction": pack["direction"],
                    "composite_score": pack["composite_score"],
                    "exit_pct": outcome["exit_pct"],
                    "exit_reason": outcome["exit_reason"],
                    "exit_bars": outcome["exit_bars"],
                    "peak_pct": outcome["peak_pct"],
                    "dd_pct": outcome["dd_pct"],
                })
        print(f"[v4]   {sym}: {n_snap} snaps, {n_passed_ultra} ultra signals",
              flush=True)
        time.sleep(0.5)
    if not rows:
        print("[v4] no data", flush=True)
        return {}
    # Save raw
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "v4_dataset.csv"), index=False)
    print(f"[v4] saved {len(df)} snapshots", flush=True)
    # ===== Win rate calculation =====
    if not trade_rows:
        print("[v4] no trades", flush=True)
        return {"df": df, "out_dir": out_dir, "n_rows": len(df)}
    trades = pd.DataFrame(trade_rows)
    trades.to_csv(os.path.join(out_dir, "v4_trades.csv"), index=False)
    # Win = exit_pct > 0
    n_trades = len(trades)
    n_wins = (trades["exit_pct"] > 0).sum()
    n_losses = (trades["exit_pct"] < 0).sum()
    n_breakeven = (trades["exit_pct"] == 0).sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    total_pnl = trades["exit_pct"].sum()
    avg_pnl = trades["exit_pct"].mean()
    # Profit factor
    gross_profit = trades.loc[trades["exit_pct"] > 0, "exit_pct"].sum()
    gross_loss = abs(trades.loc[trades["exit_pct"] < 0, "exit_pct"].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    # By direction
    by_dir = {}
    for d in ["LONG", "SHORT"]:
        sub = trades[trades["direction"] == d]
        if not sub.empty:
            by_dir[d] = {
                "n": int(len(sub)),
                "wins": int((sub["exit_pct"] > 0).sum()),
                "win_rate": float((sub["exit_pct"] > 0).mean()),
                "avg_pnl": float(sub["exit_pct"].mean()),
                "total_pnl": float(sub["exit_pct"].sum()),
            }
    # By exit reason
    by_reason = {}
    for r in trades["exit_reason"].unique():
        sub = trades[trades["exit_reason"] == r]
        by_reason[r] = {
            "n": int(len(sub)),
            "wins": int((sub["exit_pct"] > 0).sum()),
            "avg_pnl": float(sub["exit_pct"].mean()),
        }
    # Filter pass rates
    pass_pre = df["passed_prefilter"].sum()
    pass_anti = (df["passed_prefilter"] & df["passed_anti_late"]).sum()
    pass_ultra = df["passed_ultra"].sum()
    summary = {
        "n_snapshots": int(len(df)),
        "n_trades": int(n_trades),
        "n_wins": int(n_wins),
        "n_losses": int(n_losses),
        "n_breakeven": int(n_breakeven),
        "win_rate": float(win_rate),
        "avg_pnl_pct": float(avg_pnl),
        "total_pnl_pct": float(total_pnl),
        "profit_factor": float(pf),
        "by_direction": by_dir,
        "by_exit_reason": by_reason,
        "filter_pass": {
            "prefilter": int(pass_pre),
            "prefilter+anti_late": int(pass_anti),
            "ultra": int(pass_ultra),
            "from_total": int(len(df)),
        },
        "params": {
            "tp_pct": tp_pct, "sl_pct": sl_pct,
            "trailing_pct": trailing_pct, "max_hold_bars": max_hold_bars,
            "days": days, "snapshot_every_bars": snapshot_every_bars,
        },
        "btc_state": btc_state["state"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "v4_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # Print summary
    print("\n" + "=" * 60)
    print(f"[v4] BACKTEST RESULTS — {summary['n_snapshots']} snapshots, "
          f"{summary['n_trades']} ultra signals")
    print("=" * 60)
    print(f"  Win rate:      {win_rate*100:.1f}%  ({n_wins}/{n_trades})")
    print(f"  Total P&L:     {total_pnl:+.2f}%")
    print(f"  Avg P&L:       {avg_pnl:+.2f}%")
    print(f"  Profit factor: {pf:.2f}")
    print()
    print(f"  Prefilter pass:      {pass_pre} ({pass_pre/len(df)*100:.1f}%)")
    print(f"  +Anti-late pass:     {pass_anti} ({pass_anti/len(df)*100:.1f}%)")
    print(f"  +Ultra-strict pass:  {pass_ultra} ({pass_ultra/len(df)*100:.1f}%)")
    print()
    if by_dir:
        print("  By direction:")
        for d, s in by_dir.items():
            print(f"    {d}: {s['n']} trades, "
                  f"win_rate={s['win_rate']*100:.1f}%, "
                  f"avg_pnl={s['avg_pnl']:+.2f}%")
    if by_reason:
        print("  By exit reason:")
        for r, s in by_reason.items():
            print(f"    {r}: {s['n']}, wins={s['wins']}, "
                  f"avg_pnl={s['avg_pnl']:+.2f}%")
    return {"df": df, "trades": trades, "summary": summary,
            "out_dir": out_dir}


# ---------------------------------------------------------------------------
# Default symbols (current small-cap universe + a few mid-caps for sanity)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    # Small caps we found via universe_v2
    "ALICEUSDT", "TLMUSDT", "HYPERUSDT", "WCTUSDT", "RECALLUSDT",
    "GROVEUSDT", "RESOLVUSDT", "TOWNSUSDT", "OPNUSDT", "TUTUSDT",
    "FIGHTUSDT", "PARTIUSDT", "USATUSDT", "BREVUSDT", "EVAAUSDT",
    # Some mid-caps for comparison
    "BANKUSDT", "ACEUSDT", "AFCUSDT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--tp", type=float, default=5.0)
    parser.add_argument("--sl", type=float, default=3.0)
    parser.add_argument("--trail", type=float, default=1.5)
    parser.add_argument("--max-hold", type=int, default=24)
    parser.add_argument("--snap", type=int, default=6)
    parser.add_argument("--out", type=str, default="backtest_results_v4")
    args = parser.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS[: args.top]
    print(f"[v4] running on {len(symbols)} symbols, {args.days} days",
          flush=True)
    res = run_backtest_v4(
        symbols,
        days=args.days,
        snapshot_every_bars=args.snap,
        max_hold_bars=args.max_hold,
        tp_pct=args.tp, sl_pct=args.sl, trailing_pct=args.trail,
        out_dir=args.out,
    )
    if not res:
        return
    print(f"\n[v4] dataset -> {args.out}/v4_dataset.csv")
    print(f"[v4] trades  -> {args.out}/v4_trades.csv")
    print(f"[v4] summary -> {args.out}/v4_results.json")


if __name__ == "__main__":
    main()
