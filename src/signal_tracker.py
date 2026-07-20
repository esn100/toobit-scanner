"""
Signal tracker v2 — advanced TP/SL with multiple exit strategies.

Exit strategies (configured per signal):
  1. FIXED: TP at +X%, SL at -Y%
  2. TRAILING: SL moves up with price (locks in profit)
  3. SCALED: exit 50% at TP1, 50% at TP2 (or SL hit)
  4. TIME-BASED: exit after max_hold regardless of P&L

Tracking:
  - Per-cycle: current price, current_pct, highest/lowest
  - Resolved: TP_HIT / SL_HIT / TIMEOUT / SCALED_TP1 / SCALED_TP2 / TRAILING
  - Analytics: win rate, profit factor, expectancy, max drawdown

Storage:
  - data/active_signals.csv: open positions
  - data/resolved_signals.csv: closed positions (full history)
  - data/signal_stats.json: rolling stats (win rate by confidence, etc.)

The system learns which TP/SL combo + min_confidence works best.
"""
from __future__ import annotations
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from . import db as database


DATA_DIR = Path("data")
SIGNAL_LOG = DATA_DIR / "active_signals.csv"
RESOLVED_LOG = DATA_DIR / "resolved_log.csv"  # historical resolved
STATS_FILE = DATA_DIR / "signal_stats.json"

# ---- Default settings ----
DEFAULT_TP_PCT = 5.0
DEFAULT_SL_PCT = 3.0
DEFAULT_MAX_HOLD_HOURS = 12
DEFAULT_TRAILING_PCT = 2.0   # trailing SL: 2% below highest
DEFAULT_TP1_PCT = 3.0        # scaled: 50% at +3%, rest at +5%
DEFAULT_TP2_PCT = 5.0


# ---- Columns ----
SIGNAL_COLS = [
    "signal_id", "ts_entry", "symbol", "direction",
    "entry_price", "tp_price", "sl_price", "initial_sl_price",
    "tp_pct", "sl_pct", "max_hold_hours",
    "trailing_pct", "use_trailing", "use_scaled",
    "tp1_price", "tp1_hit",
    "score_long", "score_short", "confidence",
    "n_long_signals", "n_short_signals",
    "f_momentum_3_pct", "f_momentum_6_pct", "f_rvol", "f_atr_pct",
    "f_a_ichi_above_cloud", "f_a_ichi_below_cloud",
    "f_a_fib_dist_0.618", "f_a_fib_distance_pct",
    "f_volume_spike", "f_m_5m_volume_spike",
    "btc_state", "btc_momentum_12_pct",
    "market_regime",  # RISK_ON / RISK_OFF / NEUTRAL
    "status",  # OPEN / TP_HIT / SL_HIT / TIMEOUT / SCALED_TP1 / SCALED_TP2 / TRAILING_HIT
    "current_price", "current_pct",
    "highest_pct", "lowest_pct",
    "current_trailing_sl",  # updated SL if trailing is on
    "ts_last_check", "ts_exit", "exit_price", "exit_pct", "exit_reason",
    "num_checks", "size_remaining",  # 1.0 = full, 0.5 = scaled out half
    "partial_pnl",  # realized P&L from scaled exits
    # Microstructure at entry
    "m_obi_10_at_entry", "m_cvd_at_entry", "m_5m_rvol_at_entry",
    "m_spread_pct_at_entry",
]

RESOLVED_COLS = SIGNAL_COLS + [
    "duration_hours", "max_favorable_pct", "max_adverse_pct",
    "max_drawdown_pct", "max_runup_pct",
    "pips_to_tp", "pips_to_sl",  # % away from each at any point
]


# ============================================================================
# Load / save
# ============================================================================
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


def _load_stats() -> Dict:
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_update": None,
        "by_confidence_bucket": {},  # 50-60, 60-70, etc.
        "by_tp_sl": {},
        "by_symbol": {},
        "by_direction": {},
        "by_hour_of_day": {},
        "by_market_regime": {},
        "rolling_24h": {},
        "consecutive_losses": 0,
        "consecutive_wins": 0,
    }


def _save_stats(stats: Dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=str)


# ============================================================================
# Open / resolve
# ============================================================================
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
    trailing_pct: float = DEFAULT_TRAILING_PCT,
    use_trailing: bool = True,
    use_scaled: bool = False,
    btc_state: str = "NEUTRAL",
    btc_momentum: float = 0.0,
    market_regime: str = "NEUTRAL",
) -> Optional[str]:
    """
    Open a new tracked signal. Returns signal_id or None.
    """
    if direction not in ("LONG", "SHORT"):
        return None
    if entry_price <= 0:
        return None
    if not features:
        features = {}
    df = _load_active()
    # No duplicate open position
    dup = df[(df["symbol"] == symbol)
             & (df["direction"] == direction)
             & (df["status"] == "OPEN")]
    if not dup.empty:
        return None
    # Compute TP/SL
    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
        tp1_price = entry_price * (1 + DEFAULT_TP1_PCT / 100)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        tp1_price = entry_price * (1 - DEFAULT_TP1_PCT / 100)
    signal_id = f"{symbol}_{direction}_{int(time.time())}"
    rec = {
        "signal_id": signal_id,
        "ts_entry": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "tp_price": round(tp_price, 8),
        "sl_price": round(sl_price, 8),
        "initial_sl_price": round(sl_price, 8),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "max_hold_hours": max_hold_hours,
        "trailing_pct": trailing_pct,
        "use_trailing": int(use_trailing),
        "use_scaled": int(use_scaled),
        "tp1_price": round(tp1_price, 8),
        "tp1_hit": 0,
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
        "f_a_fib_dist_0.618": features.get("f_a_fib_dist_0.618", 99),
        "f_a_fib_distance_pct": features.get("f_a_fib_distance_pct", 0),
        "f_volume_spike": features.get("f_volume_spike", 0),
        "f_m_5m_volume_spike": features.get("f_m_5m_volume_spike", 0),
        "btc_state": btc_state,
        "btc_momentum_12_pct": btc_momentum,
        "market_regime": market_regime,
        "status": "OPEN",
        "current_price": entry_price,
        "current_pct": 0.0,
        "highest_pct": 0.0,
        "lowest_pct": 0.0,
        "current_trailing_sl": round(sl_price, 8),
        "ts_last_check": datetime.now(timezone.utc).isoformat(),
        "ts_exit": pd.NA,
        "exit_price": pd.NA,
        "exit_pct": pd.NA,
        "exit_reason": pd.NA,
        "num_checks": 0,
        "size_remaining": 1.0,
        "partial_pnl": 0.0,
        # Microstructure at entry (if available)
        "m_obi_10_at_entry": features.get("f_m_obi_10", 0),
        "m_cvd_at_entry": features.get("f_m_cvd", 0),
        "m_5m_rvol_at_entry": features.get("f_m_5m_rvol", 0),
        "m_spread_pct_at_entry": features.get("f_m_spread_pct", 0),
    }
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    _save_active(df)
    # Also insert into SQLite
    try:
        database.init_db()
        database.upsert_signal(rec)
    except Exception as e:
        pass  # don't fail signal opening
    return signal_id


def check_and_resolve(
    current_prices: Dict[str, float],
    min_score: float = 0.0,
) -> Tuple[int, int, int, int, int]:
    """
    Check all open signals against current prices and resolve any
    that hit TP, SL, trailing SL, or timeout.

    Returns:
        (n_resolved, n_tp, n_sl, n_trailing, n_timeout)
    """
    df = _load_active()
    if df.empty:
        return 0, 0, 0, 0, 0
    now = datetime.now(timezone.utc)
    n_resolved = n_tp = n_sl = n_trail = n_to = 0
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
        tp1_price = float(row.get("tp1_price", tp_price))
        cur_sl = float(row.get("current_trailing_sl", row["sl_price"]))
        trailing_pct = float(row.get("trailing_pct", 0))
        use_trailing = bool(int(row.get("use_trailing", 0)))
        use_scaled = bool(int(row.get("use_scaled", 0)))
        size_remaining = float(row.get("size_remaining", 1.0))
        partial_pnl = float(row.get("partial_pnl", 0.0))
        tp1_hit = int(row.get("tp1_hit", 0))
        # Compute current pct
        if direction == "LONG":
            cur_pct = (cur_price - entry) / entry * 100
        else:
            cur_pct = (entry - cur_price) / entry * 100
        prev_high = float(row.get("highest_pct", 0) or 0)
        prev_low = float(row.get("lowest_pct", 0) or 0)
        new_high = max(prev_high, cur_pct)
        new_low = min(prev_low, cur_pct)
        # Update trailing SL
        if use_trailing and new_high > 0:
            if direction == "LONG":
                new_sl = entry * (1 + (new_high - trailing_pct) / 100)
                # Never lower the trailing SL (only ratchet up)
                if new_sl > cur_sl:
                    cur_sl = new_sl
            else:  # SHORT
                new_sl = entry * (1 - (new_high - trailing_pct) / 100)
                if new_sl < cur_sl:
                    cur_sl = new_sl
        df.at[idx, "current_price"] = cur_price
        df.at[idx, "current_pct"] = round(cur_pct, 3)
        df.at[idx, "highest_pct"] = round(new_high, 3)
        df.at[idx, "lowest_pct"] = round(new_low, 3)
        df.at[idx, "current_trailing_sl"] = round(cur_sl, 8)
        df.at[idx, "ts_last_check"] = now.isoformat()
        df.at[idx, "num_checks"] = int(row.get("num_checks", 0) or 0) + 1
        reason = None
        exit_price = cur_price
        exit_pct = cur_pct
        # Check TP1 (scaled exit at first target)
        if use_scaled and not tp1_hit and size_remaining == 1.0:
            tp1_hit_now = False
            if direction == "LONG" and cur_price >= tp1_price:
                tp1_hit_now = True
            elif direction == "SHORT" and cur_price <= tp1_price:
                tp1_hit_now = True
            if tp1_hit_now:
                # Close half
                partial_pnl += 0.5 * exit_pct
                size_remaining = 0.5
                tp1_hit = 1
                df.at[idx, "tp1_hit"] = 1
                df.at[idx, "size_remaining"] = 0.5
                df.at[idx, "partial_pnl"] = round(partial_pnl, 3)
        # Check TP/SL/Trailing
        if direction == "LONG":
            if cur_price >= tp_price:
                reason = "TP_HIT"
                n_tp += 1
            elif cur_price <= cur_sl:
                if use_trailing and new_high > trailing_pct:
                    reason = "TRAILING_HIT"
                    n_trail += 1
                else:
                    reason = "SL_HIT"
                    n_sl += 1
        else:  # SHORT
            if cur_price <= tp_price:
                reason = "TP_HIT"
                n_tp += 1
            elif cur_price >= cur_sl:
                if use_trailing and new_high > trailing_pct:
                    reason = "TRAILING_HIT"
                    n_trail += 1
                else:
                    reason = "SL_HIT"
                    n_sl += 1
        # Check timeout
        if reason is None:
            try:
                ts_entry = pd.Timestamp(row["ts_entry"])
                if ts_entry.tzinfo is None:
                    ts_entry = ts_entry.tz_localize("UTC")
                age_h = (now - ts_entry).total_seconds() / 3600
                if age_h >= float(row["max_hold_hours"]):
                    reason = "TIMEOUT"
                    n_to += 1
            except Exception:
                pass
        if reason:
            # If scaled, add the remaining portion's pnl
            if use_scaled and size_remaining < 1.0:
                final_pnl = partial_pnl + size_remaining * exit_pct
            else:
                final_pnl = exit_pct
            df.at[idx, "status"] = reason
            df.at[idx, "ts_exit"] = now.isoformat()
            df.at[idx, "exit_price"] = cur_price
            df.at[idx, "exit_pct"] = round(final_pnl, 3)
            df.at[idx, "exit_reason"] = reason
            rows_to_resolve.append(idx)
            n_resolved += 1
    # Sync to SQLite (faster than CSV)
    _save_active(df)
    if rows_to_resolve:
        resolved_df = df.loc[rows_to_resolve].copy()
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
            # max drawdown / runup
            try:
                hp = float(df.at[idx, "highest_pct"])
                lp = float(df.at[idx, "lowest_pct"])
                resolved_df.at[idx, "max_runup_pct"] = hp
                resolved_df.at[idx, "max_drawdown_pct"] = abs(min(0, lp))
            except Exception:
                pass
        for c in RESOLVED_COLS:
            if c not in resolved_df.columns:
                resolved_df[c] = pd.NA
        # Save to CSV (for backward compat)
        existing = _load_resolved()
        combined = pd.concat([existing, resolved_df[RESOLVED_COLS]],
                             ignore_index=True, sort=False)
        _save_resolved(combined)
        # Also save to SQLite
        try:
            for _, r in resolved_df.iterrows():
                sig_dict = r.to_dict()
                sig_dict["duration_hours"] = duration_h if 'duration_h' in dir() else 0
                database.move_to_resolved(sig_dict)
        except Exception as e:
            pass
        df = df.drop(rows_to_resolve).reset_index(drop=True)
        _save_active(df)
        # Delete from SQLite signals
        try:
            with database.get_conn() as conn:
                for idx in rows_to_resolve:
                    sig_id = df.iloc[idx]["signal_id"] if idx < len(df) else None
                    if sig_id:
                        conn.execute("DELETE FROM signals WHERE signal_id = ?",
                                     (sig_id,))
                conn.commit()
        except Exception:
            pass
        # Update stats
        try:
            update_stats(resolved_df)
        except Exception as e:
            print(f"  stats update error: {e}")
    return n_resolved, n_tp, n_sl, n_trail, n_to


# ============================================================================
# Analytics
# ============================================================================
def update_stats(new_resolved: pd.DataFrame) -> None:
    """Update stats file with new resolved signals."""
    stats = _load_stats()
    stats["last_update"] = datetime.now(timezone.utc).isoformat()
    # By confidence bucket
    for _, row in new_resolved.iterrows():
        try:
            conf = float(row.get("confidence", 0))
            bucket = f"{int(conf // 10) * 10}-{int(conf // 10) * 10 + 10}"
            if bucket not in stats["by_confidence_bucket"]:
                stats["by_confidence_bucket"][bucket] = {
                    "n": 0, "n_tp": 0, "n_sl": 0, "n_to": 0, "total_pnl": 0.0
                }
            b = stats["by_confidence_bucket"][bucket]
            b["n"] += 1
            b["total_pnl"] += float(row.get("exit_pct", 0))
            status = row.get("status", "")
            if status == "TP_HIT":
                b["n_tp"] += 1
            elif status == "SL_HIT":
                b["n_sl"] += 1
            elif status == "TIMEOUT":
                b["n_to"] += 1
            # By TP/SL combo
            tp = float(row.get("tp_pct", 0))
            sl = float(row.get("sl_pct", 0))
            key = f"tp{int(tp)}_sl{int(sl)}"
            if key not in stats["by_tp_sl"]:
                stats["by_tp_sl"][key] = {
                    "n": 0, "n_tp": 0, "n_sl": 0, "total_pnl": 0.0
                }
            t = stats["by_tp_sl"][key]
            t["n"] += 1
            t["total_pnl"] += float(row.get("exit_pct", 0))
            if status == "TP_HIT":
                t["n_tp"] += 1
            elif status == "SL_HIT":
                t["n_sl"] += 1
            # By symbol
            sym = str(row.get("symbol", ""))
            if sym not in stats["by_symbol"]:
                stats["by_symbol"][sym] = {
                    "n": 0, "n_tp": 0, "n_sl": 0, "total_pnl": 0.0
                }
            s = stats["by_symbol"][sym]
            s["n"] += 1
            s["total_pnl"] += float(row.get("exit_pct", 0))
            if status == "TP_HIT":
                s["n_tp"] += 1
            elif status == "SL_HIT":
                s["n_sl"] += 1
            # By direction
            d = str(row.get("direction", ""))
            if d not in stats["by_direction"]:
                stats["by_direction"][d] = {
                    "n": 0, "n_tp": 0, "n_sl": 0, "total_pnl": 0.0
                }
            dd = stats["by_direction"][d]
            dd["n"] += 1
            dd["total_pnl"] += float(row.get("exit_pct", 0))
            if status == "TP_HIT":
                dd["n_tp"] += 1
            elif status == "SL_HIT":
                dd["n_sl"] += 1
            # By market regime
            regime = str(row.get("market_regime", "NEUTRAL"))
            if regime not in stats["by_market_regime"]:
                stats["by_market_regime"][regime] = {
                    "n": 0, "n_tp": 0, "n_sl": 0, "total_pnl": 0.0
                }
            r = stats["by_market_regime"][regime]
            r["n"] += 1
            r["total_pnl"] += float(row.get("exit_pct", 0))
            if status == "TP_HIT":
                r["n_tp"] += 1
            elif status == "SL_HIT":
                r["n_sl"] += 1
            # By hour of day (entry time)
            try:
                ts = pd.Timestamp(row.get("ts_entry", ""))
                if pd.notna(ts):
                    h = f"h{int(ts.hour):02d}"
                    if h not in stats["by_hour_of_day"]:
                        stats["by_hour_of_day"][h] = {
                            "n": 0, "n_tp": 0, "n_sl": 0, "total_pnl": 0.0
                        }
                    hh = stats["by_hour_of_day"][h]
                    hh["n"] += 1
                    hh["total_pnl"] += float(row.get("exit_pct", 0))
                    if status == "TP_HIT":
                        hh["n_tp"] += 1
                    elif status == "SL_HIT":
                        hh["n_sl"] += 1
            except Exception:
                pass
        except Exception as e:
            continue
    # Compute consecutive W/L
    try:
        all_resolved = _load_resolved()
        if not all_resolved.empty:
            last_5 = all_resolved.tail(10)
            consec_l = 0
            consec_w = 0
            for _, r in last_5.iterrows():
                if r.get("status") == "TP_HIT":
                    consec_w += 1
                    consec_l = 0
                elif r.get("status") == "SL_HIT":
                    consec_l += 1
                    consec_w = 0
            stats["consecutive_wins"] = consec_w
            stats["consecutive_losses"] = consec_l
    except Exception:
        pass
    # 24h rolling
    try:
        all_resolved = _load_resolved()
        if not all_resolved.empty and "ts_exit" in all_resolved.columns:
            all_resolved["ts_exit"] = pd.to_datetime(
                all_resolved["ts_exit"], utc=True, errors="coerce"
            )
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
            last_24h = all_resolved[all_resolved["ts_exit"] >= cutoff]
            n_tp = int((last_24h["status"] == "TP_HIT").sum())
            n_sl = int((last_24h["status"] == "SL_HIT").sum())
            n_to = int((last_24h["status"] == "TIMEOUT").sum())
            n_tr = int((last_24h["status"] == "TRAILING_HIT").sum())
            total = len(last_24h)
            total_pnl = float(last_24h["exit_pct"].sum()) if total else 0
            stats["rolling_24h"] = {
                "n": total, "n_tp": n_tp, "n_sl": n_sl,
                "n_to": n_to, "n_trailing": n_tr,
                "win_rate": round(n_tp / (n_tp + n_sl) if (n_tp + n_sl) else 0, 3),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / total if total else 0, 3),
            }
    except Exception:
        pass
    _save_stats(stats)


def get_stats(tp_pct: Optional[float] = None,
              sl_pct: Optional[float] = None,
              direction: Optional[str] = None) -> Dict:
    """
    Compute comprehensive stats on resolved signals.
    """
    df = _load_resolved()
    if df.empty:
        return {
            "n_total": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "n_trailing": 0, "win_rate": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        }
    if tp_pct is not None:
        df = df[df["tp_pct"] == tp_pct]
    if sl_pct is not None:
        df = df[df["sl_pct"] == sl_pct]
    if direction is not None:
        df = df[df["direction"] == direction]
    if df.empty:
        return {
            "n_total": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "n_trailing": 0, "win_rate": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        }
    n_tp = int((df["status"] == "TP_HIT").sum())
    n_sl = int((df["status"] == "SL_HIT").sum())
    n_to = int((df["status"] == "TIMEOUT").sum())
    n_tr = int((df["status"] == "TRAILING_HIT").sum())
    n = len(df)
    tp_rows = df[df["status"] == "TP_HIT"]
    sl_rows = df[df["status"] == "SL_HIT"]
    to_rows = df[df["status"] == "TIMEOUT"]
    tr_rows = df[df["status"] == "TRAILING_HIT"]
    avg_win = float(tp_rows["exit_pct"].mean()) if len(tp_rows) else 0.0
    avg_loss = float(sl_rows["exit_pct"].mean()) if len(sl_rows) else 0.0
    avg_to = float(to_rows["exit_pct"].mean()) if len(to_rows) else 0.0
    avg_tr = float(tr_rows["exit_pct"].mean()) if len(tr_rows) else 0.0
    decided = n_tp + n_sl + n_tr
    win_rate = ((n_tp + n_tr) / decided) if decided > 0 else 0.0
    # Profit factor: total wins / abs(total losses)
    total_wins = (tp_rows["exit_pct"].sum() if len(tp_rows) else 0) + \
                 (tr_rows["exit_pct"].sum() if len(tr_rows) else 0)
    total_losses = abs(sl_rows["exit_pct"].sum()) if len(sl_rows) else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0
    expectancy = float(df["exit_pct"].mean())
    # Max consecutive wins/losses
    sorted_df = df.reset_index(drop=True)
    max_cw = 0
    max_cl = 0
    cur_cw = 0
    cur_cl = 0
    for _, r in sorted_df.iterrows():
        s = r.get("status", "")
        if s in ("TP_HIT", "TRAILING_HIT"):
            cur_cw += 1
            cur_cl = 0
            max_cw = max(max_cw, cur_cw)
        elif s == "SL_HIT":
            cur_cl += 1
            cur_cw = 0
            max_cl = max(max_cl, cur_cl)
        else:  # TIMEOUT
            cur_cw = 0
            cur_cl = 0
    # Max drawdown
    sorted_df["cum_pnl"] = sorted_df["exit_pct"].cumsum()
    sorted_df["running_max"] = sorted_df["cum_pnl"].cummax()
    sorted_df["drawdown"] = sorted_df["running_max"] - sorted_df["cum_pnl"]
    max_dd = float(sorted_df["drawdown"].max()) if len(sorted_df) else 0.0
    return {
        "n_total": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_timeout": n_to,
        "n_trailing": n_tr,
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "avg_trailing": round(avg_tr, 3),
        "avg_timeout": round(avg_to, 3),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 3),
        "max_consecutive_wins": max_cw,
        "max_consecutive_losses": max_cl,
        "max_drawdown_pct": round(max_dd, 2),
        "total_pnl": round(float(sorted_df["cum_pnl"].iloc[-1]) if len(sorted_df) else 0, 2),
    }


def get_open_signals() -> pd.DataFrame:
    df = _load_active()
    return df[df["status"] == "OPEN"]


def get_resolved_signals() -> pd.DataFrame:
    return _load_resolved()


def suggest_min_confidence(min_signals: int = 20) -> Dict:
    """
    Find the minimum confidence threshold that maximizes expectancy.
    """
    df = _load_resolved()
    if len(df) < min_signals:
        return {"best_confidence": None, "n_total": len(df)}
    results = {}
    for thresh in [40, 50, 60, 70, 80, 90]:
        sub = df[df["confidence"] >= thresh]
        if len(sub) >= 5:
            stats = get_stats()
            # Filter by recomputing
            n_tp = int((sub["status"] == "TP_HIT").sum())
            n_sl = int((sub["status"] == "SL_HIT").sum())
            n_tr = int((sub["status"] == "TRAILING_HIT").sum())
            expectancy = float(sub["exit_pct"].mean())
            win_rate = (n_tp + n_tr) / (n_tp + n_sl + n_tr) if (n_tp + n_sl + n_tr) else 0
            results[f">={thresh}"] = {
                "n": len(sub), "win_rate": round(win_rate, 3),
                "expectancy": round(expectancy, 3),
            }
    if not results:
        return {"best_confidence": None, "n_total": len(df)}
    best = max(results, key=lambda k: results[k]["expectancy"])
    return {"best_confidence": best, "results": results, "n_total": len(df)}


def suggest_tp_sl(min_signals: int = 10) -> Dict:
    """
    Grid search for best TP/SL combo.
    """
    df = _load_resolved()
    if len(df) < min_signals:
        return {"best": None, "all": {}, "n_total": len(df)}
    results = {}
    for tp in [3, 4, 5, 6, 8, 10]:
        for sl in [2, 3, 4, 5]:
            sub = df[(df["tp_pct"] == tp) & (df["sl_pct"] == sl)]
            if len(sub) >= 5:
                n_tp = int((sub["status"] == "TP_HIT").sum())
                n_sl = int((sub["status"] == "SL_HIT").sum())
                n_tr = int((sub["status"] == "TRAILING_HIT").sum())
                expectancy = float(sub["exit_pct"].mean())
                win_rate = (n_tp + n_tr) / (n_tp + n_sl + n_tr) if (n_tp + n_sl + n_tr) else 0
                results[f"tp{tp}_sl{sl}"] = {
                    "n": len(sub), "win_rate": round(win_rate, 3),
                    "expectancy": round(expectancy, 3),
                    "n_tp": n_tp, "n_sl": n_sl, "n_tr": n_tr,
                }
    if not results:
        return {"best": None, "all": {}, "n_total": len(df)}
    best = max(results, key=lambda k: results[k]["expectancy"])
    return {"best": best, "best_stats": results[best],
            "all": results, "n_total": len(df)}


def print_full_report() -> None:
    """Print a comprehensive analytics report."""
    print("=" * 70)
    print("📊 SIGNAL TRACKER FULL REPORT")
    print("=" * 70)
    # Overall stats
    stats = get_stats()
    print(f"\n[Overall] {stats['n_total']} resolved signals")
    print(f"  Win rate: {stats['win_rate']*100:.1f}% "
          f"(TP={stats['n_tp']}, SL={stats['n_sl']}, "
          f"Trailing={stats['n_trailing']}, Timeout={stats['n_timeout']})")
    print(f"  Avg win: {stats['avg_win']:+.2f}% | "
          f"Avg loss: {stats['avg_loss']:+.2f}%")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Expectancy: {stats['expectancy']:+.2f}% per trade")
    print(f"  Max consecutive W/L: {stats['max_consecutive_wins']}/"
          f"{stats['max_consecutive_losses']}")
    print(f"  Max drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"  Total P&L: {stats['total_pnl']:+.2f}%")
    # Open signals
    open_df = get_open_signals()
    print(f"\n[Open] {len(open_df)} active signals")
    if not open_df.empty:
        for _, r in open_df.iterrows():
            print(f"  {r['symbol']:<14} {r['direction']:<6} "
                  f"entry={r['entry_price']:.4f} "
                  f"TP={r['tp_price']:.4f} SL={r['sl_price']:.4f} "
                  f"cur={r['current_pct']:+.1f}%")
    # By direction
    print("\n[By Direction]")
    for d in ["LONG", "SHORT"]:
        s = get_stats(direction=d)
        if s["n_total"] > 0:
            print(f"  {d}: n={s['n_total']}, win={s['win_rate']*100:.1f}%, "
                  f"avg={s['expectancy']:+.2f}%")
    # 24h rolling
    file_stats = _load_stats()
    if "rolling_24h" in file_stats and file_stats["rolling_24h"].get("n", 0) > 0:
        r24 = file_stats["rolling_24h"]
        print(f"\n[24h Rolling] n={r24['n']}, win={r24['win_rate']*100:.1f}%, "
              f"total_pnl={r24['total_pnl']:+.2f}%")
    # Suggestions
    print("\n[Suggestions]")
    sug_conf = suggest_min_confidence()
    if sug_conf.get("best_confidence"):
        print(f"  Best min confidence: {sug_conf['best_confidence']}")
        for k, v in sug_conf.get("results", {}).items():
            print(f"    {k}: n={v['n']}, win={v['win_rate']*100:.1f}%, "
                  f"exp={v['expectancy']:+.2f}%")
    sug_tp = suggest_tp_sl()
    if sug_tp.get("best"):
        print(f"  Best TP/SL: {sug_tp['best']}")
        b = sug_tp["best_stats"]
        print(f"    n={b['n']}, win={b['win_rate']*100:.1f}%, "
              f"exp={b['expectancy']:+.2f}%")
    print("=" * 70)
