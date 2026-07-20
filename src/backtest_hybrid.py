"""
Backtest hybrid exit strategy vs smart exit vs fixed TP/SL.

Compares three strategies on the same data:
  1. Fixed TP/SL (TP=+5%, SL=-3%)
  2. Smart Exit (breakeven, locks, trail)
  3. Hybrid Exit (TP1=+6% / TP2=+10%, scaled, smart SL)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from . import db as database
from .backtest_smart_exit import (
    get_price_at_time, simulate_smart_exit
)
from .hybrid_exit import simulate_hybrid_exit
from .smart_exit_v2 import simulate_smart_v2


def run_backtest_strategy(
    strategy: str,
    min_confidence: float = 50.0,
    limit_symbols: int = 15,
) -> pd.DataFrame:
    """Run backtest with a specific strategy."""
    features = database.get_features()
    if features.empty:
        return pd.DataFrame()
    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    signals = features[
        (features["direction"].isin(["LONG", "SHORT"])) &
        (features["confidence"] >= min_confidence)
    ].copy()
    if signals.empty:
        return pd.DataFrame()
    unique_symbols = signals["symbol"].drop_duplicates().head(limit_symbols).tolist()
    signals = signals[signals["symbol"].isin(unique_symbols)]
    results = []
    for _, r in signals.iterrows():
        symbol = r["symbol"]
        entry_time = r["ts"]
        direction = r["direction"]
        entry_price = float(r["close"])
        if pd.isna(entry_time) or entry_price <= 0:
            continue
        price_series = get_price_at_time(symbol, entry_time, 24)
        if not price_series:
            continue
        if strategy == "fixed":
            outcome = simulate_smart_exit(
                entry_time, entry_price, direction, price_series,
                tp_pct=5.0, sl_pct=3.0, use_smart_exit=False,
            )
        elif strategy == "smart":
            outcome = simulate_smart_exit(
                entry_time, entry_price, direction, price_series,
                tp_pct=5.0, sl_pct=3.0, use_smart_exit=True,
            )
        elif strategy == "hybrid":
            # Tuned: smaller TP, tighter SL, faster time
            outcome = simulate_hybrid_exit(
                entry_time, entry_price, direction, price_series,
                tp1_pct=4.0, tp2_pct=7.0, sl_pct=2.0, max_hours=8.0,
            )
        elif strategy == "smart_v2":
            # Smart v2: tight SL + aggressive locks + wider TP
            outcome = simulate_smart_v2(
                entry_time, entry_price, direction, price_series,
                tp_pct=5.0, sl_pct=3.0, max_hours=8.0,
            )
        else:
            continue
        outcome["symbol"] = symbol
        outcome["entry_time"] = entry_time
        outcome["direction"] = direction
        outcome["entry_price"] = entry_price
        outcome["confidence"] = r.get("confidence", 0)
        outcome["atr_pct"] = r.get("ind_atr_pct", 0)
        outcome["mom_3_pct"] = r.get("ind_momentum_3_pct", 0)
        results.append(outcome)
    return pd.DataFrame(results)


def compute_stats(df: pd.DataFrame, label: str) -> Dict:
    if df.empty:
        return {"label": label, "n_total": 0}
    n = len(df)
    n_tp = (df["exit_reason"] == "TP_HIT").sum() + \
           (df["exit_reason"] == "TP1_HIT").sum() + \
           (df["exit_reason"] == "TP2_HIT").sum()
    n_sl = (df["exit_reason"] == "SL_HIT").sum()
    n_be = (df["exit_reason"] == "BREAKEVEN_LOCK").sum()
    n_pl = (df["exit_reason"] == "PROFIT_LOCK").sum()
    n_to = (df["exit_reason"] == "TIMEOUT").sum()
    n_win = n_tp + n_be + n_pl
    n_lose = n_sl
    n_decided = n_win + n_lose
    win_rate = (n_win / n_decided) if n_decided > 0 else 0
    wins_pnl = df[df["exit_pct"] > 0]["exit_pct"].sum()
    losses_pnl = abs(df[df["exit_pct"] < 0]["exit_pct"].sum())
    pf = (wins_pnl / losses_pnl) if losses_pnl > 0 else 0
    avg_win = df[df["exit_pct"] > 0]["exit_pct"].mean() if (df["exit_pct"] > 0).any() else 0
    avg_loss = df[df["exit_pct"] < 0]["exit_pct"].mean() if (df["exit_pct"] < 0).any() else 0
    avg_pnl = df["exit_pct"].mean()
    total_pnl = df["exit_pct"].sum()
    df_sorted = df.sort_values("entry_time")
    cum_pnl = df_sorted["exit_pct"].cumsum()
    running_max = cum_pnl.cummax()
    max_dd = (running_max - cum_pnl).max()
    return {
        "label": label,
        "n_total": n,
        "n_tp": int(n_tp),
        "n_sl": int(n_sl),
        "n_breakeven": int(n_be),
        "n_profit_lock": int(n_pl),
        "n_timeout": int(n_to),
        "n_wins": int(n_win),
        "n_losses": int(n_lose),
        "win_rate": round(win_rate, 3),
        "profit_factor": round(pf, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "avg_pnl": round(avg_pnl, 3),
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(max_dd, 2),
    }


def main():
    print("=" * 70)
    print("BACKTEST: FIXED vs SMART vs HYBRID")
    print("=" * 70)
    print()
    strategies = [
        ("fixed", "Fixed TP/SL (TP=+5%, SL=-3%)"),
        ("smart", "Smart Exit (breakeven, locks, trail)"),
        ("smart_v2", "Smart v2 (TP=+5%, SL=-3%, locks @1.5/3/5, trail, 8h)"),
    ]
    all_stats = []
    for strat_key, strat_label in strategies:
        print(f"\n[{strat_key.upper()}] Running backtest: {strat_label}")
        df = run_backtest_strategy(strat_key, min_confidence=50.0, limit_symbols=15)
        stats = compute_stats(df, strat_label)
        all_stats.append((strat_key, df, stats))
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'FIXED':<15} {'SMART':<15} {'HYBRID':<15}")
    print("-" * 70)
    for key in ["n_total", "n_wins", "n_losses", "win_rate", "profit_factor",
                "avg_win_pct", "avg_loss_pct", "avg_pnl", "total_pnl",
                "max_drawdown"]:
        row = []
        for _, _, s in all_stats:
            v = s.get(key, 0)
            row.append(v)
        if isinstance(row[0], float):
            print(f"  {key:<25} {row[0]:<15.3f} {row[1]:<15.3f} {row[2]:<15.3f}")
        else:
            print(f"  {key:<25} {row[0]:<15} {row[1]:<15} {row[2]:<15}")
    print()
    # Best strategy
    best_winrate = max(all_stats, key=lambda x: x[2]["win_rate"])
    best_pnl = max(all_stats, key=lambda x: x[2]["total_pnl"])
    print(f"🏆 Best win rate: {best_winrate[2]['label']} "
          f"({best_winrate[2]['win_rate']*100:.1f}%)")
    print(f"💰 Best total P&L: {best_pnl[2]['label']} "
          f"({best_pnl[2]['total_pnl']:+.2f}%)")
    # Save detailed
    for strat_key, df, _ in all_stats:
        if not df.empty:
            df["strategy"] = strat_key
            df.to_csv(f"data/backtest_{strat_key}.csv", index=False)
    print("\nDetailed results saved to data/backtest_*.csv")


if __name__ == "__main__":
    main()
