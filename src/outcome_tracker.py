"""
Outcome tracker for the small-cap scanner.

Every time we issue an APPROVED signal, we:
  1. Store the snapshot in data/signal_history.csv
  2. Schedule an outcome check 12h later
  3. On next scan, resolve pending outcomes by fetching current price
  4. Label each signal: 1 (success = +10% peak), 0 (fail)
  5. Update rolling precision statistics
"""
from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

HISTORY_COLS = [
    "timestamp", "symbol", "composite_score", "decision",
    "entry_price", "peak_price_12h", "dd_price_12h",
    "peak_pct", "dd_pct", "label", "resolved_at",
    # Feature snapshot for ML
    "rvol", "momentum_1_pct", "momentum_3_pct", "rsi_1h",
    "atr_pct", "bb_squeeze", "higher_lows",
    "btc_corr_2d", "smart_money_score", "independent_mover",
    "consensus_count",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLS)


def _load(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            for c in HISTORY_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            return df
        except Exception:
            return _empty_df()
    return _empty_df()


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def record_signal(
    path: str, symbol: str, score: float, decision: str,
    entry_price: float, features: Dict,
) -> None:
    """Record a new signal in the history."""
    df = _load(path)
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "composite_score": score,
        "decision": decision,
        "entry_price": entry_price,
        "peak_price_12h": np.nan,
        "dd_price_12h": np.nan,
        "peak_pct": np.nan,
        "dd_pct": np.nan,
        "label": np.nan,
        "resolved_at": np.nan,
    }
    # Copy features
    for k in HISTORY_COLS:
        if k in features:
            rec[k] = features[k]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df = df.tail(10000)
    _save(df, path)


def resolve_pending(
    path: str, toobit, horizon_hours: int = 12,
) -> int:
    """
    For every signal older than `horizon_hours` that is unresolved,
    fetch current price, compute peak/drawdown, and label.
    """
    df = _load(path)
    if df.empty:
        return 0
    # Mask: unresolved and old enough
    now = pd.Timestamp.now(tz="UTC")
    def is_old_enough(ts_str):
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return (now - ts).total_seconds() >= horizon_hours * 3600
        except Exception:
            return False
    unresolved = df[df["label"].isna() & df["timestamp"].apply(is_old_enough)]
    if unresolved.empty:
        return 0
    resolved = 0
    for idx, row in unresolved.iterrows():
        sym = row["symbol"]
        try:
            df_k = toobit.get_klines(sym, "1h", 30)
        except Exception:
            continue
        if df_k.empty:
            continue
        entry = float(row["entry_price"])
        if entry <= 0:
            # Try to get entry from the timestamp
            try:
                # Find the first candle at or after the timestamp
                ts = pd.Timestamp(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                mask = df_k["open_time"] >= ts
                if mask.any():
                    entry = float(df_k.loc[mask, "open"].iloc[0])
                else:
                    continue
            except Exception:
                continue
        # Compute peak/dd in the next 12h
        try:
            ts = pd.Timestamp(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            future = df_k[df_k["open_time"] >= ts].head(horizon_hours)
            if future.empty:
                continue
            peak_pct = float(((future["high"].max() - entry) / entry * 100.0))
            dd_pct = float(((future["low"].min() - entry) / entry * 100.0))
            label = 1 if peak_pct >= 10.0 else 0
            df.at[idx, "peak_pct"] = round(peak_pct, 2)
            df.at[idx, "dd_pct"] = round(dd_pct, 2)
            df.at[idx, "peak_price_12h"] = float(future["high"].max())
            df.at[idx, "dd_price_12h"] = float(future["low"].min())
            df.at[idx, "label"] = int(label)
            df.at[idx, "resolved_at"] = now.isoformat()
            resolved += 1
        except Exception as e:
            print(f"[outcome] resolve error for {sym}: {e}")
            continue
    if resolved > 0:
        _save(df, path)
    return resolved


def get_rolling_precision(
    path: str, score_threshold: float = 80.0, window: int = 20
) -> Dict:
    """
    Compute rolling precision on the last N signals with score >= threshold.
    """
    df = _load(path)
    if df.empty:
        return {"rolling_precision": 0.0, "n_samples": 0,
                "n_success": 0, "threshold": score_threshold}
    high = df[(df["composite_score"] >= score_threshold)
              & df["label"].notna()]
    if high.empty:
        return {"rolling_precision": 0.0, "n_samples": 0,
                "n_success": 0, "threshold": score_threshold}
    last_n = high.tail(window)
    n = int(len(last_n))
    n_succ = int(last_n["label"].sum())
    prec = float(n_succ / n) if n > 0 else 0.0
    return {
        "rolling_precision": prec,
        "n_samples": n,
        "n_success": n_succ,
        "threshold": score_threshold,
        "window": window,
    }


def suggest_threshold(
    path: str, target_precision: float = 0.8,
    min_samples: int = 5,
) -> float:
    """
    Find the highest threshold that achieves >= target_precision
    on the last N labelled signals.
    """
    df = _load(path)
    if df.empty:
        return 80.0
    labeled = df[df["label"].notna()].copy()
    if len(labeled) < min_samples:
        return 80.0
    # Try thresholds from 50 to 95 in steps of 5
    for thr in range(95, 49, -5):
        sub = labeled[labeled["composite_score"] >= thr]
        if len(sub) >= 3:
            prec = sub["label"].mean()
            if prec >= target_precision:
                return float(thr)
    # If no threshold meets target, return the lowest with at least 3 samples
    for thr in range(95, 49, -5):
        sub = labeled[labeled["composite_score"] >= thr]
        if len(sub) >= 3:
            return float(thr)
    return 80.0
