"""
Auto-trader: only opens signals that pass ultra-strict filter.

Each cycle:
  1. Get last cycle features
  2. Apply ultra-strict filter
  3. For each approved signal:
     - Open position via signal_tracker
     - Use Smart v2 exit (breakeven + locks + trail)
     - TP/SL from self-trained params (default +5%/-3%)
     - 8 hour max hold
  4. Check existing positions (smart exit v2 logic)

Quality over quantity: 0-3 signals per day.

NEW: Reads TP/SL/trail from data/self_trained_params.json if available.
     Self-training loop updates this file every hour.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import pandas as pd

from . import db as database
from .ultra_strict import is_ultra_setup, get_ultra_picks
from .signal_tracker import (
    open_signal, check_and_resolve, get_open_signals, get_stats,
)
from .smart_exit_v2 import smart_exit_v2_logic


def load_self_trained_params() -> dict:
    """Load self-trained params from data/self_trained_params.json."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "self_trained_params.json")
    if not os.path.exists(path):
        return {
            "tp_pct": 5.0,
            "sl_pct": 3.0,
            "trail_pct": 1.5,
            "conf_threshold": 60.0,
        }
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {
            "tp_pct": 5.0,
            "sl_pct": 3.0,
            "trail_pct": 1.5,
            "conf_threshold": 60.0,
        }


def open_ultra_signals(
    min_confidence: float = 60.0,
    tp_pct: float = None,
    sl_pct: float = None,
    max_hours: float = 8.0,
) -> List[str]:
    """
    Open signals for all ultra-approved setups.
    Uses smart v2 exit strategy.
    If tp_pct/sl_pct not provided, reads from self_trained_params.json.
    Returns list of signal_ids opened.
    """
    # Load params (from self-training if available)
    params = load_self_trained_params()
    if tp_pct is None:
        tp_pct = params.get("tp_pct", 5.0)
    if sl_pct is None:
        sl_pct = params.get("sl_pct", 3.0)
    # Use conf threshold from params
    conf_threshold = params.get("conf_threshold", min_confidence)
    min_conf = min(conf_threshold, min_confidence)  # take the more permissive
    picks = get_ultra_picks(min_confidence=min_conf)
    if "error" in picks:
        return []
    signal_ids = []
    for r, feats in picks.get("longs", []):
        signal_id = _open_one(r, "LONG", feats,
                              tp_pct=tp_pct, sl_pct=sl_pct,
                              max_hours=max_hours,
                              trail_pct=params.get("trail_pct", 1.5))
        if signal_id:
            signal_ids.append(signal_id)
    for r, feats in picks.get("shorts", []):
        signal_id = _open_one(r, "SHORT", feats,
                              tp_pct=tp_pct, sl_pct=sl_pct,
                              max_hours=max_hours,
                              trail_pct=params.get("trail_pct", 1.5))
        if signal_id:
            signal_ids.append(signal_id)
    return signal_ids


def _open_one(
    r: pd.Series,
    direction: str,
    feats: Dict,
    tp_pct: float = 5.0,
    sl_pct: float = 3.0,
    max_hours: float = 8.0,
    trail_pct: float = 1.5,
) -> str:
    """Open a single signal with smart v2 exit."""
    symbol = r["symbol"]
    entry_price = float(r["close"])
    if entry_price <= 0:
        return ""
    # Build features dict for signal_tracker
    features = {
        "n_long_signals": 0,
        "n_short_signals": 0,
        "f_momentum_3_pct": feats.get("f_momentum_3_pct", 0),
        "f_momentum_6_pct": feats.get("f_momentum_6_pct", 0),
        "f_rvol": feats.get("f_rvol", 0),
        "f_atr_pct": feats.get("f_atr_pct", 0),
        "f_a_ichi_above_cloud": feats.get("f_a_ichi_above_cloud", 0),
        "f_a_ichi_below_cloud": feats.get("f_a_ichi_below_cloud", 0),
        "f_a_fib_dist_0.618": feats.get("f_a_fib_dist_0.618", 99),
        "f_volume_spike": feats.get("f_volume_spike", 0),
        "f_m_5m_volume_spike": feats.get("f_m_5m_volume_spike", 0),
        "f_m_obi_10": feats.get("f_m_obi_10", 1),
        "f_m_cvd": feats.get("f_m_cvd", 0),
        "f_m_5m_rvol": feats.get("f_m_5m_rvol", 0),
        "f_bb_breakout_above": feats.get("f_bb_breakout_above", 0),
        "f_bos_up": feats.get("f_bos_up", 0),
        "f_atr_expanding": feats.get("f_atr_expanding", 0),
    }
    try:
        signal_id = open_signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            score_long=float(r.get("score_long", 0)),
            score_short=float(r.get("score_short", 0)),
            confidence=float(feats.get("confidence", 0)),
            features=features,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            max_hold_hours=max_hours,
            trailing_pct=trail_pct,  # from self-trained params
            use_trailing=True,
            use_scaled=False,  # no scaled exit (was hurting)
            btc_state=r.get("btc_state", "NEUTRAL"),
            btc_momentum=float(r.get("btc_momentum_12_pct", 0)),
            market_regime="NEUTRAL",
        )
        return signal_id or ""
    except Exception as e:
        print(f"  open error for {symbol}: {e}")
        return ""


def check_signals_smart_v2(current_prices: Dict[str, float]) -> Tuple[int, int, int, int]:
    """
    Check existing signals with smart v2 exit logic.

    For each open signal:
      1. Calculate current_pct and highest_pct
      2. Apply smart v2 SL logic (breakeven, locks, trail)
      3. Update SL in DB
      4. If SL or TP hit, resolve

    Returns (n_resolved, n_tp, n_sl, n_breakeven_lock).
    """
    open_df = get_open_signals()
    if open_df.empty:
        return 0, 0, 0, 0
    n_resolved = n_tp = n_sl = n_be = 0
    for idx, row in open_df.iterrows():
        sym = row["symbol"]
        if sym not in current_prices:
            continue
        cur_price = float(current_prices[sym])
        if cur_price <= 0:
            continue
        entry = float(row["entry_price"])
        direction = row["direction"]
        # Calculate current pct
        if direction == "LONG":
            cur_pct = (cur_price - entry) / entry * 100
        else:
            cur_pct = (entry - cur_price) / entry * 100
        # Get current sl
        current_sl = float(row.get("current_trailing_sl", row["sl_price"]))
        highest_pct = float(row.get("highest_pct", 0) or 0)
        # Update highest pct
        new_high = max(highest_pct, cur_pct)
        # Apply smart v2 logic
        new_sl, reason = smart_exit_v2_logic(
            direction, cur_pct, current_sl, new_high, entry, cur_price
        )
        # Update DB
        updates = {
            "current_price": cur_price,
            "current_pct": round(cur_pct, 3),
            "highest_pct": round(new_high, 3),
            "current_trailing_sl": round(new_sl, 6),
            "ts_last_check": datetime.now(timezone.utc).isoformat(),
            "num_checks": int(row.get("num_checks", 0) or 0) + 1,
        }
        from . import db as database
        database.update_signal(row["signal_id"], updates)
        # Check resolution
        tp_price = float(row["tp_price"])
        if direction == "LONG":
            if cur_price >= tp_price:
                n_tp += 1
                n_resolved += 1
            elif cur_price <= new_sl:
                if abs(new_sl - entry) < entry * 0.001:
                    n_be += 1
                else:
                    n_sl += 1
                n_resolved += 1
        else:  # SHORT
            if cur_price <= tp_price:
                n_tp += 1
                n_resolved += 1
            elif cur_price >= new_sl:
                if abs(new_sl - entry) < entry * 0.001:
                    n_be += 1
                else:
                    n_sl += 1
                n_resolved += 1
    return n_resolved, n_tp, n_sl, n_be


def run_auto_trader_cycle(verbose: bool = True) -> Dict:
    """
    Main auto-trader cycle. Run this every 10 minutes.

    Steps:
      1. Run REPEATER scanner (priority - 6 known pump symbols, 24/7)
      2. Open new signals (ultra-approved + repeater)
      3. Check existing signals against current prices
      4. Report stats

    Returns dict with cycle stats.
    """
    from .toobit_client import ToobitClient
    from .live_collector import _get_btc_df
    from .repeater_scanner import run_repeater_cycle as run_repeaters
    client = ToobitClient()
    # Get latest prices
    features_df = database.get_features()
    if features_df.empty:
        return {"error": "no features"}
    features_df["ts"] = pd.to_datetime(features_df["ts"], utc=True,
                                       errors="coerce")
    last_ts = features_df["ts"].max()
    last = features_df[features_df["ts"] == last_ts].copy()
    if last.empty:
        return {"error": "no last cycle"}
    current_prices = {r["symbol"]: float(r["close"])
                      for _, r in last.iterrows()
                      if r.get("close", 0) > 0}
    if not current_prices:
        return {"error": "no prices"}
    # Step 0: REPEATER scanner (priority: catches 6 known pump symbols 24/7)
    repeater_summary = run_repeaters(verbose=verbose)
    if verbose:
        n_repeater = len(repeater_summary.get("pre_pumps", [])) + len(repeater_summary.get("confirmed", []))
        if n_repeater > 0:
            print(f"  Repeater scanner: {n_repeater} signals opened")
    # Step 1: open new ultra signals
    opened = open_ultra_signals(min_confidence=60.0)
    if verbose:
        print(f"Opened {len(opened)} ultra signals")
    # Step 2: check existing
    n_resolved, n_tp, n_sl, n_be = check_signals_smart_v2(current_prices)
    if verbose and n_resolved > 0:
        print(f"Resolved {n_resolved}: TP={n_tp}, SL={n_sl}, "
              f"Breakeven={n_be}")
    # Step 3: report
    stats = get_stats()
    return {
        "opened": len(opened),
        "resolved": n_resolved,
        "tp": n_tp,
        "sl": n_sl,
        "breakeven_lock": n_be,
        "win_rate": stats.get("win_rate", 0),
        "total_resolved": stats.get("n_total", 0),
        "total_pnl": stats.get("total_pnl", 0),
        "repeater_pre": len(repeater_summary.get("pre_pumps", [])),
        "repeater_confirmed": len(repeater_summary.get("confirmed", [])),
        "repeater_cluster_active": repeater_summary.get("cluster_active", False),
    }
