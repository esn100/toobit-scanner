"""
Signal tracker with TP/SL (حد سود/حد ضرر).

For each signal, we want to know:
  1. Entry price (at signal time)
  2. Take Profit (TP) — exit at +X% gain
  3. Stop Loss (SL) — exit at -Y% loss
  4. Trailing stop (optional)
  5. Time-based exit (max 12h)
  6. The actual exit price and reason

The tracker stores a row per active signal and resolves it when:
  - TP hit
  - SL hit
  - Max hold time exceeded
  - Manual close (rare)

Each cycle (every 10 min) we check current prices against active signals
and update exit status. The system learns which TP/SL combo works best.
"""
from __future__ import annotations
import os
import time
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


DATA_DIR = Path("data")
SIGNAL_LOG = DATA_DIR / "active_signals.csv"
RESOLVED_LOG = DATA_DIR / "resolved_signals.csv"

# Default TP/SL settings (will be tunable from analysis)
DEFAULT_TP_PCT = 5.0      # Take profit at +5%
DEFAULT_SL_PCT = 3.0      # Stop loss at -3%
DEFAULT_MAX_HOLD_HOURS = 12


# Required columns
SIGNAL_COLS = [
    "signal_id", "ts_entry", "symbol", "direction",
    "entry_price", "tp_price", "sl_price",
    "tp_pct", "sl_pct", "max_hold_hours",
    "score_long", "score_short", "confidence",
    "n_long_signals", "n_short_signals",
    "f_momentum_3_pct", "f_momentum_6_pct", "f_rvol", "f_atr_pct",
    "f_a_ichi_above_cloud", "f_a_ichi_below_cloud",
    "status",  # OPEN / TP_HIT / SL_HIT / TIMEOUT / MANUAL
    "current_price", "current_pct", "highest_pct", "lowest_pct",
    "ts_last_check", "ts_exit", "exit_price", "exit_pct", "exit_reason",
    "num_checks",
]

RESOLVED_COLS = SIGNAL_COLS + [
    "duration_hours", "max_favorable_pct", "max_adverse_pct",
]


def _empty_df(cols: list) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


def _load_active() -> pd.DataFrame:
    if SIGNAL_LOG.exists():
        try:
            df = pd.read_csv(SIGNAL_LOG)
            for c in SIGNAL_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df
        except Exception:
            return _empty_df(SIGNAL_COLS)
    return _empty_df(SIGNAL_COLS)


def _save_active(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(SIGNAL_LOG, index=False)


def _load_resolved() -> pd.DataFrame:
    if RESOLVED_LOG.exists():
        try:
            df = pd.read_csv(RESOLVED_LOG)
            for c in RESOLVED_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df
        except Exception:
            return _empty_df(RESOLVED_COLS)
    return _empty_df(RESOLVED_COLS)


def _save_resolved(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RESOLVED_LOG, index=False)


def open_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    score_long: float = 0,
    score_short: float = 0,
    confidence: float = 0,
    features: Optional[Dict] = None,
    tp_pct: float = DEFAULT_TP_PCT,
    sl_pct: float = DEFAULT_SL_PCT,
    max_hold_hours: float = DEFAULT_MAX_HOLD_HOURS,
) -> Optional[str]:
    """
    Open a new signal. Returns the signal_id (string) or None if invalid.
    """
    if direction not in ("LONG", "SHORT"):
        return None
    if entry_price <= 0:
        return None
    if not features:
        features = {}
    df = _load_active()
    # Don't open duplicate
    dup = df[(df["symbol"] == symbol)
             & (df["direction"] == direction)
             & (df["status"] == "OPEN")]
    if not dup.empty:
        return None
    # Compute TP/SL
    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
    signal_id = f"{symbol}_{direction}_{int(time.time())}"
    rec = {
        "signal_id": signal_id,
        "ts_entry": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "tp_price": round(tp_price, 8),
        "sl_price": round(sl_price, 8),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "max_hold_hours": max_hold_hours,
        "score_long": score_long,
        "score_short": score_short,
        "confidence": confidence,
        "n_long_signals": features.get("n_long_signals", 0),
        "n_short_signals": features.get("n_short_signals", 0),
        "f_momentum_3_pct": features.get("f_momentum_3_pct", 0),
        "f_momentum_6_pct": features.get("f_momentum_6_pct", 0),
        "f_rvol": features.get("f_rvol", 1),
        "f_atr_pct": features.get("f_atr_pct", 0),
        "f_a_ichi_above_cloud": features.get("f_a_ichi_above_cloud", 0),
        "f_a_ichi_below_cloud": features.get("f_a_ichi_below_cloud", 0),
        "status": "OPEN",
        "current_price": entry_price,
        "current_pct": 0.0,
        "highest_pct": 0.0,
        "lowest_pct": 0.0,
        "ts_last_check": datetime.now(timezone.utc).isoformat(),
        "ts_exit": pd.NA,
        "exit_price": pd.NA,
        "exit_pct": pd.NA,
        "exit_reason": pd.NA,
        "num_checks": 0,
    }
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    _save_active(df)
    return signal_id


def check_and_resolve(
    current_prices: Dict[str, float],
    min_score: float = 0.0,
) -> Tuple[int, int, int]:
    """
    Check all open signals against current prices and resolve any
    that hit TP, SL, or timeout.

    Args:
        current_prices: {symbol: current_price}
        min_score: only check signals with confidence >= this

    Returns:
        (num_resolved, num_tp_hit, num_sl_hit)
    """
    df = _load_active()
    if df.empty:
        return 0, 0, 0
    now = datetime.now(timezone.utc)
    resolved_count = 0
    tp_count = 0
    sl_count = 0
    timeout_count = 0
    rows_to_resolve = []
    for idx, row in df.iterrows():
        if row.get("status") != "OPEN":
            continue
        sym = row["symbol"]
        if sym not in current_prices:
            continue
        cur_price = float(current_prices[sym])
        if cur_price <= 0:
            continue
        entry = float(row["entry_price"])
        direction = row["direction"]
        tp_price = float(row["tp_price"])
        sl_price = float(row["sl_price"])
        # Compute current % (signed: + = good for LONG, - = good for SHORT)
        if direction == "LONG":
            cur_pct = (cur_price - entry) / entry * 100
            high_pct = cur_pct  # we only know current, no high/low
            low_pct = cur_pct
        else:  # SHORT
            cur_pct = (entry - cur_price) / entry * 100
            high_pct = cur_pct
            low_pct = cur_pct
        # Update tracking columns
        df.at[idx, "current_price"] = cur_price
        df.at[idx, "current_pct"] = round(cur_pct, 3)
        # Update highest/lowest
        prev_high = float(row.get("highest_pct", 0) or 0)
        prev_low = float(row.get("lowest_pct", 0) or 0)
        df.at[idx, "highest_pct"] = round(max(prev_high, cur_pct), 3)
        df.at[idx, "lowest_pct"] = round(min(prev_low, cur_pct), 3)
        df.at[idx, "ts_last_check"] = now.isoformat()
        df.at[idx, "num_checks"] = int(row.get("num_checks", 0) or 0) + 1
        # Check resolution conditions
        reason = None
        exit_price = cur_price
        if direction == "LONG":
            if cur_price >= tp_price:
                reason = "TP_HIT"
                tp_count += 1
            elif cur_price <= sl_price:
                reason = "SL_HIT"
                sl_count += 1
        else:  # SHORT
            if cur_price <= tp_price:
                reason = "TP_HIT"
                tp_count += 1
            elif cur_price >= sl_price:
                reason = "SL_HIT"
                sl_count += 1
        # Check timeout
        if reason is None:
            try:
                ts_entry = pd.Timestamp(row["ts_entry"])
                if ts_entry.tzinfo is None:
                    ts_entry = ts_entry.tz_localize("UTC")
                age_h = (now - ts_entry).total_seconds() / 3600
                if age_h >= float(row["max_hold_hours"]):
                    reason = "TIMEOUT"
                    timeout_count += 1
            except Exception:
                pass
        if reason:
            df.at[idx, "status"] = reason
            df.at[idx, "ts_exit"] = now.isoformat()
            df.at[idx, "exit_price"] = cur_price
            df.at[idx, "exit_pct"] = round(cur_pct, 3)
            df.at[idx, "exit_reason"] = reason
            rows_to_resolve.append(idx)
            resolved_count += 1
    # Save updated active log
    _save_active(df)
    # Move resolved rows to resolved log
    if rows_to_resolve:
        resolved_df = df.loc[rows_to_resolve].copy()
        # Add extra analysis columns
        for idx in rows_to_resolve:
            try:
                ts_entry = pd.Timestamp(df.at[idx, "ts_entry"])
                if ts_entry.tzinfo is None:
                    ts_entry = ts_entry.tz_localize("UTC")
                ts_exit = pd.Timestamp(df.at[idx, "ts_exit"])
                if ts_exit.tzinfo is None:
                    ts_exit = ts_exit.tz_localize("UTC")
                duration_h = (ts_exit - ts_entry).total_seconds() / 3600
            except Exception:
                duration_h = 0
            resolved_df.at[idx, "duration_hours"] = round(duration_h, 2)
            resolved_df.at[idx, "max_favorable_pct"] = df.at[idx, "highest_pct"]
            resolved_df.at[idx, "max_adverse_pct"] = df.at[idx, "lowest_pct"]
        # Ensure all RESOLVED_COLS exist
        for c in RESOLVED_COLS:
            if c not in resolved_df.columns:
                resolved_df[c] = pd.NA
        existing = _load_resolved()
        combined = pd.concat([existing, resolved_df[RESOLVED_COLS]],
                             ignore_index=True, sort=False)
        _save_resolved(combined)
        # Remove from active
        df = df.drop(rows_to_resolve).reset_index(drop=True)
        _save_active(df)
    return resolved_count, tp_count, sl_count


def get_open_signals() -> pd.DataFrame:
    df = _load_active()
    return df[df["status"] == "OPEN"]


def get_resolved_signals() -> pd.DataFrame:
    return _load_resolved()


def get_stats(tp_pct: Optional[float] = None,
              sl_pct: Optional[float] = None) -> Dict:
    """
    Compute hit rate, avg win, avg loss, etc. for resolved signals.
    If tp_pct/sl_pct are given, filter to those settings.
    """
    df = _load_resolved()
    if df.empty:
        return {
            "n_total": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0,
        }
    if tp_pct is not None:
        df = df[df["tp_pct"] == tp_pct]
    if sl_pct is not None:
        df = df[df["sl_pct"] == sl_pct]
    if df.empty:
        return {
            "n_total": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0,
        }
    n_tp = int((df["status"] == "TP_HIT").sum())
    n_sl = int((df["status"] == "SL_HIT").sum())
    n_to = int((df["status"] == "TIMEOUT").sum())
    n = len(df)
    tp_rows = df[df["status"] == "TP_HIT"]
    sl_rows = df[df["status"] == "SL_HIT"]
    to_rows = df[df["status"] == "TIMEOUT"]
    avg_win = float(tp_rows["exit_pct"].mean()) if len(tp_rows) else 0.0
    avg_loss = float(sl_rows["exit_pct"].mean()) if len(sl_rows) else 0.0
    avg_to = float(to_rows["exit_pct"].mean()) if len(to_rows) else 0.0
    # Win rate: TP_HIT / (TP_HIT + SL_HIT) — exclude TIMEOUT
    decided = n_tp + n_sl
    win_rate = (n_tp / decided) if decided > 0 else 0.0
    # Expectancy: avg outcome across all resolved
    expectancy = float(df["exit_pct"].mean())
    return {
        "n_total": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_timeout": n_to,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "avg_timeout": round(avg_to, 3),
        "expectancy": round(expectancy, 3),
    }


def suggest_tp_sl(min_signals: int = 10) -> Dict:
    """
    Find the best TP/SL combo by grid search over resolved signals.
    We can't go back and re-test different TP/SL on the same price path,
    but we can compute stats per (tp_pct, sl_pct) combo from data we've
    already collected.
    """
    df = _load_resolved()
    if len(df) < min_signals:
        return {"best": None, "all": {}, "n_total": len(df)}
    results = {}
    for tp in [3, 4, 5, 6, 8, 10]:
        for sl in [2, 3, 4, 5]:
            stats = get_stats(tp_pct=tp, sl_pct=sl)
            if stats["n_total"] >= 5:
                results[f"tp{tp}_sl{sl}"] = stats
    if not results:
        return {"best": None, "all": {}, "n_total": len(df)}
    # Best by expectancy
    best_key = max(results, key=lambda k: results[k]["expectancy"])
    return {"best": best_key, "best_stats": results[best_key],
            "all": results, "n_total": len(df)}
