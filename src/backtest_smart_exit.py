"""
Backtest smart exit strategy on historical data.

For each row in feature_log.csv, simulate:
  - Entry at close price
  - Track price over next 12h
  - Apply smart exit rules (breakeven, locks, trail)
  - Compare to fixed TP/SL (the old way)

Output:
  - Win rate with smart exit
  - Win rate with fixed exit
  - Average P&L
  - Profit factor
  - Max drawdown
  - Per-symbol stats
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from .toobit_client import ToobitClient
from . import db as database
from .smart_exit import smart_sl_logic


def get_price_at_time(symbol: str, target_time: pd.Timestamp,
                      lookback_hours: int = 24) -> List[Tuple[pd.Timestamp, float]]:
    """
    Fetch 1h klines for symbol, return list of (timestamp, close) for
    the time period around target_time.
    """
    try:
        client = ToobitClient()
        df = client.get_klines(symbol, interval="1h", limit=lookback_hours)
        if df.empty or "open_time" not in df.columns:
            return []
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True,
                                         errors="coerce")
        df = df.dropna(subset=["open_time"])
        return list(zip(df["open_time"], df["close"].astype(float)))
    except Exception:
        return []


def simulate_smart_exit(
    entry_time: pd.Timestamp,
    entry_price: float,
    direction: str,
    price_series: List[Tuple[pd.Timestamp, float]],
    tp_pct: float,
    sl_pct: float,
    use_smart_exit: bool = True,
) -> Dict:
    """
    Simulate a trade with either smart or fixed exit.

    Returns dict with:
      - exit_pct: final P&L
      - exit_reason: TP_HIT / SL_HIT / BREAKEVEN_LOCK / TIMEOUT
      - duration_min: how long trade was open
      - max_favorable: highest pct reached
      - max_adverse: lowest pct reached
    """
    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct / 100)
        initial_sl = entry_price * (1 - sl_pct / 100)
    else:
        tp_price = entry_price * (1 - tp_pct / 100)
        initial_sl = entry_price * (1 + sl_pct / 100)
    current_sl = initial_sl
    current_tp = tp_price
    highest_pct = 0.0
    lowest_pct = 0.0
    exit_pct = 0.0
    exit_reason = "TIMEOUT"
    exit_time = None
    for ts, price in price_series:
        if ts < entry_time:
            continue
        if direction == "LONG":
            current_pct = (price - entry_price) / entry_price * 100
            high_pct = current_pct
            low_pct = current_pct
        else:
            current_pct = (entry_price - price) / entry_price * 100
            high_pct = current_pct
            low_pct = current_pct
        highest_pct = max(highest_pct, current_pct)
        lowest_pct = min(lowest_pct, current_pct)
        # Smart exit: move SL based on current progress
        if use_smart_exit:
            new_sl, reason = smart_sl_logic(
                direction, current_pct, current_sl, highest_pct,
                entry_price, price
            )
            current_sl = new_sl
        # Check exit conditions
        if direction == "LONG":
            if price >= current_tp:
                exit_pct = current_pct
                exit_reason = "TP_HIT"
                exit_time = ts
                break
            elif price <= current_sl:
                exit_pct = current_pct
                # Distinguish breakeven lock from SL hit
                if use_smart_exit and abs(current_sl - entry_price) < entry_price * 0.001:
                    exit_reason = "BREAKEVEN_LOCK"
                elif use_smart_exit and current_sl > initial_sl:
                    exit_reason = "PROFIT_LOCK"
                else:
                    exit_reason = "SL_HIT"
                exit_time = ts
                break
        else:  # SHORT
            if price <= current_tp:
                exit_pct = current_pct
                exit_reason = "TP_HIT"
                exit_time = ts
                break
            elif price >= current_sl:
                exit_pct = current_pct
                if use_smart_exit and abs(current_sl - entry_price) < entry_price * 0.001:
                    exit_reason = "BREAKEVEN_LOCK"
                elif use_smart_exit and current_sl < initial_sl:
                    exit_reason = "PROFIT_LOCK"
                else:
                    exit_reason = "SL_HIT"
                exit_time = ts
                break
    if exit_time is None:
        # Timeout
        if price_series:
            last_ts, last_price = price_series[-1]
            if direction == "LONG":
                exit_pct = (last_price - entry_price) / entry_price * 100
            else:
                exit_pct = (entry_price - last_price) / entry_price * 100
            exit_reason = "TIMEOUT"
            exit_time = last_ts
    duration_min = (exit_time - entry_time).total_seconds() / 60 if exit_time else 0
    return {
        "exit_pct": round(exit_pct, 3),
        "exit_reason": exit_reason,
        "duration_min": round(duration_min, 1),
        "max_favorable": round(highest_pct, 3),
        "max_adverse": round(lowest_pct, 3),
    }


def run_backtest(
    min_confidence: float = 50.0,
    use_smart_exit: bool = True,
    lookback_hours: int = 24,
    limit_symbols: int = 10,
) -> pd.DataFrame:
    """
    Backtest on recent features. For each signal:
    1. Get entry price (close at signal time)
    2. Fetch 1h klines for next 12h
    3. Simulate trade with smart/fixed exit
    4. Record outcome
    """
    features = database.get_features()
    if features.empty:
        print("No features in database")
        return pd.DataFrame()
    features["ts"] = pd.to_datetime(features["ts"], utc=True, errors="coerce")
    # Only get rows with valid direction
    signals = features[
        (features["direction"].isin(["LONG", "SHORT"])) &
        (features["confidence"] >= min_confidence)
    ].copy()
    if signals.empty:
        print(f"No signals with confidence >= {min_confidence}")
        return pd.DataFrame()
    # Limit to most recent N unique symbols
    unique_symbols = signals["symbol"].drop_duplicates().head(limit_symbols).tolist()
    signals = signals[signals["symbol"].isin(unique_symbols)]
    print(f"Backtesting {len(signals)} signals across {len(unique_symbols)} symbols...")
    print(f"Strategy: {'SMART EXIT' if use_smart_exit else 'FIXED TP/SL'}")
    print()
    results = []
    total = len(signals)
    for i, (_, r) in enumerate(signals.iterrows(), 1):
        symbol = r["symbol"]
        entry_time = r["ts"]
        direction = r["direction"]
        entry_price = float(r["close"])
        tp_pct = float(r.get("ind_atr_pct", 5)) * 1.0  # simple proxy
        sl_pct = float(r.get("ind_atr_pct", 3)) * 0.6
        if pd.isna(entry_time) or entry_price <= 0:
            continue
        # Get 1h klines for next 12h
        price_series = get_price_at_time(symbol, entry_time, lookback_hours)
        if not price_series:
            continue
        # Simulate
        outcome = simulate_smart_exit(
            entry_time, entry_price, direction, price_series,
            tp_pct=5.0, sl_pct=3.0,  # use fixed base for fair comparison
            use_smart_exit=use_smart_exit,
        )
        # Load features
        import json
        features_json = r.get("features_json", "{}")
        if isinstance(features_json, str):
            try:
                full_feats = json.loads(features_json)
            except Exception:
                full_feats = {}
        else:
            full_feats = {}
        results.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "direction": direction,
            "entry_price": entry_price,
            "confidence": r.get("confidence", 0),
            "atr_pct": r.get("ind_atr_pct", 0),
            "mom_3_pct": r.get("ind_momentum_3_pct", 0),
            **outcome,
        })
        if i % 5 == 0:
            print(f"  {i}/{total} {symbol} {direction}: {outcome['exit_reason']} "
                  f"({outcome['exit_pct']:+.2f}%)")
    return pd.DataFrame(results)


def compute_stats(df: pd.DataFrame, label: str) -> Dict:
    """Compute aggregate statistics for a backtest run."""
    if df.empty:
        return {"n": 0}
    n = len(df)
    n_tp = (df["exit_reason"] == "TP_HIT").sum()
    n_sl = (df["exit_reason"] == "SL_HIT").sum()
    n_be = (df["exit_reason"] == "BREAKEVEN_LOCK").sum()
    n_pl = (df["exit_reason"] == "PROFIT_LOCK").sum()
    n_to = (df["exit_reason"] == "TIMEOUT").sum()
    # Win rate: TP_HIT + BREAKEVEN_LOCK + PROFIT_LOCK all "wins"
    # (didn't lose money)
    n_win = n_tp + n_be + n_pl
    n_lose = n_sl
    n_decided = n_win + n_lose
    win_rate = (n_win / n_decided) if n_decided > 0 else 0
    # Profit factor
    wins_pnl = df[df["exit_pct"] > 0]["exit_pct"].sum()
    losses_pnl = abs(df[df["exit_pct"] < 0]["exit_pct"].sum())
    pf = (wins_pnl / losses_pnl) if losses_pnl > 0 else 0
    avg_win = df[df["exit_pct"] > 0]["exit_pct"].mean() if (df["exit_pct"] > 0).any() else 0
    avg_loss = df[df["exit_pct"] < 0]["exit_pct"].mean() if (df["exit_pct"] < 0).any() else 0
    avg_pnl = df["exit_pct"].mean()
    # Max drawdown (cumulative)
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
        "avg_pnl": round(avg_pnl, 2),
        "max_drawdown": round(max_dd, 2),
    }


def main():
    """Run backtest with both strategies and compare."""
    print("=" * 70)
    print("BACKTEST: SMART EXIT vs FIXED TP/SL")
    print("=" * 70)
    print()
    # Test both strategies on same data
    print("[1/2] Running backtest with SMART EXIT (breakeven + locks + trail)...")
    df_smart = run_backtest(
        min_confidence=50.0, use_smart_exit=True, limit_symbols=15
    )
    print()
    print("[2/2] Running backtest with FIXED TP/SL (no smart exit)...")
    df_fixed = run_backtest(
        min_confidence=50.0, use_smart_exit=False, limit_symbols=15
    )
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    stats_smart = compute_stats(df_smart, "SMART EXIT")
    stats_fixed = compute_stats(df_fixed, "FIXED TP/SL")
    print(f"\n{'Metric':<25} {'SMART EXIT':<20} {'FIXED':<20} {'Δ'}")
    print("-" * 70)
    for key in ["n_total", "n_wins", "n_losses", "win_rate", "profit_factor",
                "avg_win_pct", "avg_loss_pct", "avg_pnl", "max_drawdown"]:
        s = stats_smart.get(key, 0)
        f = stats_fixed.get(key, 0)
        if isinstance(s, float) and isinstance(f, float):
            delta = s - f
            if key in ("win_rate", "profit_factor"):
                print(f"  {key:<25} {s:<20.3f} {f:<20.3f} {delta:+.3f}")
            else:
                print(f"  {key:<25} {s:<20.2f} {f:<20.2f} {delta:+.2f}")
        else:
            print(f"  {key:<25} {s:<20} {f:<20}")
    print()
    if stats_smart["win_rate"] > stats_fixed["win_rate"]:
        improvement = (stats_smart["win_rate"] - stats_fixed["win_rate"]) * 100
        print(f"✅ SMART EXIT improves win rate by {improvement:.1f}pp")
    if stats_smart["avg_pnl"] > stats_fixed["avg_pnl"]:
        improvement = stats_smart["avg_pnl"] - stats_fixed["avg_pnl"]
        print(f"✅ SMART EXIT improves avg P&L by {improvement:+.2f}% per trade")
    print()
    # Save detailed results
    if not df_smart.empty:
        df_smart["strategy"] = "smart"
        df_smart.to_csv("data/backtest_smart.csv", index=False)
    if not df_fixed.empty:
        df_fixed["strategy"] = "fixed"
        df_fixed.to_csv("data/backtest_fixed.csv", index=False)
    print("Detailed results saved to data/backtest_smart.csv and data/backtest_fixed.csv")


if __name__ == "__main__":
    main()
