"""
Repeater Scanner - hunts 6 known pump repeaters 24/7.

Three modes:
  1. REPEATER_WATCH: scan 6 repeaters every 10min, look for pre-pump pattern
  2. CLUSTER_FOLLOW: when any repeater pumps, increase watch on others for 24h
  3. PATTERN_TRIGGER: 3-stage detection (pre-pump -> confirm -> entry)

Two-stage entry:
  - t-1h (pre-pump): enter with 30% size if pattern matches
  - t+0 (confirm):   if volume spike + mom_3 > +5%, add 70%

Aggressive TP/SL:
  - TP=8%, SL=2% (configurable per repeater)
  - Breakeven @ +2% (tighter than default 1.5%)
  - Trail @ 3% (looser, lets pumps run)
  - Max hold: 4-8h depending on repeater

Output: signals written to signal_tracker (DB) and active_signals.csv
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import math

from . import db as database
from .repeater_config import (
    REPEATERS, SECONDARY_WATCHLIST, CLUSTER_FOLLOW_HOURS,
    PRE_PUMP_SIZE_FRACTION, CONFIRM_SIZE_FRACTION,
    PRE_PUMP_CONFIDENCE_MIN, CONFIRM_CONFIDENCE_MIN,
)
from .signal_tracker import open_signal, get_open_signals


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PRE_PUMP_STATE_FILE = os.path.join(DATA_DIR, "repeater_state.json")


def _toobit_get(path: str) -> Optional[dict]:
    """GET request to Toobit public API"""
    base = "https://api.toobit.com"
    url = f"{base}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None


def _gate_get(path: str) -> Optional[list]:
    """GET request to Gate.io (fallback for history)"""
    url = f"https://api.gateio.ws/api/v4{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None


def _fetch_klines_1h(symbol: str, limit: int = 100) -> List[dict]:
    """Fetch 1h klines. Try Toobit first, fallback to Gate.io."""
    base = symbol.replace("USDT", "")
    # Try Toobit
    data = _toobit_get(f"/quote/v1/klines?symbol={symbol}&interval=60&limit={limit}")
    if data and isinstance(data, list):
        rows = []
        for r in data:
            rows.append({
                "ts": int(r["t"]),
                "o": float(r["o"]),
                "h": float(r["h"]),
                "l": float(r["l"]),
                "c": float(r["c"]),
                "v": float(r["v"]),
            })
        rows.sort(key=lambda x: x["ts"])
        if rows:
            return rows
    # Fallback Gate.io
    data = _gate_get(f"/spot/candlesticks?currency_pair={base}_USDT&interval=1h&limit={limit}")
    if data and isinstance(data, list):
        rows = []
        for r in data:
            rows.append({
                "ts": int(r[0]),
                "o": float(r[5]),
                "h": float(r[3]),
                "l": float(r[4]),
                "c": float(r[2]),
                "v": float(r[1]),
            })
        rows.sort(key=lambda x: x["ts"])
        return rows
    return []


def _rsi(closes: List[float], n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, n + 1):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    avg_g = sum(gains) / n
    avg_l = sum(losses) / n
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        avg_g = (avg_g * (n - 1) + max(ch, 0)) / n
        avg_l = (avg_l * (n - 1) + max(-ch, 0)) / n
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def _atr_pct(rows: List[dict], n: int = 14) -> float:
    if len(rows) < n + 1:
        return 0.0
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]["h"], rows[i]["l"], rows[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[-n:]) / min(n, len(trs))
    last = rows[-1]["c"]
    return atr / last * 100 if last > 0 else 0


def _flat_hours(rows: List[dict], threshold_pct: float = 5.0, max_lookback: int = 168) -> float:
    """How many hours has price been within threshold_pct of current price?"""
    if len(rows) < 5:
        return 0.0
    last = rows[-1]["c"]
    flat_start = len(rows) - 1
    for j in range(len(rows) - 2, max(0, len(rows) - max_lookback - 1), -1):
        if rows[j]["c"] <= 0:
            break
        change = abs(last - rows[j]["c"]) / rows[j]["c"] * 100
        if change <= threshold_pct:
            flat_start = j
        else:
            break
    return (len(rows) - 1 - flat_start)


def _compute_features(rows: List[dict]) -> Dict:
    """Compute scanner features for a 1h row series."""
    if len(rows) < 30:
        return {}
    cur = rows[-1]
    closes = [r["c"] for r in rows]
    vols = [r["v"] for r in rows]

    # Momentum windows
    def _mom(k: int) -> float:
        if len(rows) <= k:
            return 0.0
        prev = rows[-1 - k]["c"]
        return (cur["c"] - prev) / prev * 100 if prev > 0 else 0.0

    mom_1 = _mom(1)
    mom_3 = _mom(3)
    mom_6 = _mom(6)
    mom_12 = _mom(12)
    mom_24 = _mom(24)

    # Volume
    avg_v = sum(vols[:-1]) / max(len(vols) - 1, 1) if len(vols) > 1 else vols[0]
    rvol = cur["v"] / avg_v if avg_v > 0 else 1.0

    # Max rvol in last 4 candles
    recent = vols[-5:-1] if len(vols) > 5 else vols[:-1]
    max_rvol_4h = max((v / avg_v for v in recent), default=1.0) if avg_v > 0 else 1.0

    # Body ratio
    body = abs(cur["c"] - cur["o"]) / (cur["h"] - cur["l"]) if cur["h"] > cur["l"] else 0.0

    # RSI
    rsi_v = _rsi(closes)

    # ATR
    atr = _atr_pct(rows)

    # Flat hours
    flat_h = _flat_hours(rows)

    return {
        "close": cur["c"],
        "ts": cur["ts"],
        "mom_1": mom_1,
        "mom_3": mom_3,
        "mom_6": mom_6,
        "mom_12": mom_12,
        "mom_24": mom_24,
        "rvol": rvol,
        "max_rvol_4h": max_rvol_4h,
        "body": body,
        "rsi": rsi_v,
        "atr_pct": atr,
        "flat_hours": flat_h,
    }


def _load_state() -> Dict:
    """Load repeater state (last pump times, pending pre-pump signals)."""
    if os.path.exists(PRE_PUMP_STATE_FILE):
        try:
            with open(PRE_PUMP_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_pump_time": {},  # sym -> ISO time
        "pending_pre_pump": {},  # sym -> {entry_price, ts, size_fraction}
        "cluster_window_active_until": None,  # ISO time
    }


def _save_state(state: Dict) -> None:
    with open(PRE_PUMP_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _check_pre_pump_pattern(features: Dict, repeater: Dict) -> Tuple[bool, float]:
    """
    Check if features match a pre-pump pattern.
    Returns (is_match, confidence_0_100).
    """
    if not features:
        return False, 0.0

    score = 0.0
    max_score = 0.0

    # Rule 1: rvol above minimum
    max_score += 20
    if features["rvol"] >= repeater["pre_pump_rvol_min"]:
        score += 20

    # Rule 2: max_rvol_4h above minimum (volume build-up signature)
    max_score += 25
    if features["max_rvol_4h"] >= repeater["pre_pump_max_rvol_4h_min"]:
        score += 25
    elif features["max_rvol_4h"] >= repeater["pre_pump_max_rvol_4h_min"] * 0.7:
        score += 12  # partial credit

    # Rule 3: mom_3 in expected range (not overextended, not flat)
    max_score += 20
    if repeater["pre_pump_mom_3_min"] <= features["mom_3"] <= repeater["pre_pump_mom_3_max"]:
        score += 20
    elif abs(features["mom_3"]) <= repeater["pre_pump_mom_3_max"] * 1.5:
        score += 10

    # Rule 4: price was flat for at least N hours (consolidation)
    max_score += 20
    if features["flat_hours"] >= repeater["pre_pump_flat_min_hours"]:
        score += 20
    elif features["flat_hours"] >= repeater["pre_pump_flat_min_hours"] * 0.5:
        score += 10

    # Rule 5: RSI not overbought
    max_score += 15
    if features["rsi"] < 80:
        score += 15
    elif features["rsi"] < 90:
        score += 7

    confidence = (score / max_score * 100) if max_score > 0 else 0
    is_match = confidence >= 60
    return is_match, confidence


def _check_confirm_signal(features: Dict, repeater: Dict) -> Tuple[bool, float]:
    """
    Check if a pre-pump signal is being CONFIRMED (t+0).
    Volume spike + positive momentum.
    """
    if not features:
        return False, 0.0

    score = 0.0
    max_score = 100.0

    # Strong volume spike
    if features["rvol"] >= 3.0:
        score += 35
    elif features["rvol"] >= 2.0:
        score += 25
    elif features["rvol"] >= 1.5:
        score += 15

    # Positive momentum
    if features["mom_1"] >= 2.0:
        score += 25
    elif features["mom_1"] >= 1.0:
        score += 18
    elif features["mom_1"] >= 0.5:
        score += 10

    # Mom_3 confirming
    if features["mom_3"] >= 5.0:
        score += 20
    elif features["mom_3"] >= 2.0:
        score += 15
    elif features["mom_3"] >= 0.5:
        score += 8

    # Body ratio (real buying, not wicks)
    if features["body"] >= 0.6:
        score += 10
    elif features["body"] >= 0.4:
        score += 5

    # RSI not too overbought
    if features["rsi"] < 85:
        score += 10
    elif features["rsi"] < 95:
        score += 5

    confidence = score
    is_match = confidence >= 60
    return is_match, confidence


def scan_repeater(symbol: str) -> Dict:
    """
    Scan a single repeater symbol.
    Returns action: {action: 'enter_pre'|'confirm'|'none', confidence, features, reason}
    """
    repeater = REPEATERS.get(symbol, {})
    if not repeater:
        return {"action": "none", "reason": f"{symbol} not a repeater"}

    # Already have open position?
    open_df = get_open_signals()
    if not open_df.empty:
        dup = open_df[(open_df["symbol"] == symbol) & (open_df["status"] == "OPEN")]
        if not dup.empty:
            return {"action": "none", "reason": "already have open position"}

    # Fetch klines
    rows = _fetch_klines_1h(symbol, limit=200)
    if len(rows) < 50:
        return {"action": "none", "reason": f"insufficient data ({len(rows)} rows)"}

    features = _compute_features(rows)
    if not features:
        return {"action": "none", "reason": "feature computation failed"}

    state = _load_state()
    pending = state.get("pending_pre_pump", {}).get(symbol)
    now_ts = features["ts"]
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()

    # === STAGE 1: Check for pre-pump pattern (t-1h) ===
    # If we have a pending pre-pump, check if it's now confirming or expiring
    if pending:
        age_h = (now_ts - pending["ts"]) / 3600
        # Pre-pump signal expires after 6h without confirmation
        if age_h > 6:
            del state["pending_pre_pump"][symbol]
            _save_state(state)
        else:
            # Check if confirmation signal is now active
            confirmed, conf_conf = _check_confirm_signal(features, repeater)
            if confirmed and conf_conf >= CONFIRM_CONFIDENCE_MIN:
                # Pre-pump CONFIRMED - return confirm action
                return {
                    "action": "confirm",
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry_price": features["close"],
                    "ts": now_ts,
                    "iso_time": now_iso,
                    "pre_pump_confidence": pending["confidence"],
                    "confirm_confidence": conf_conf,
                    "features": features,
                    "repeater": repeater,
                    "pending": pending,
                    "age_hours": age_h,
                }
            else:
                # Still waiting; return minimal status
                return {
                    "action": "watch",
                    "symbol": symbol,
                    "pre_pump_pending": True,
                    "pre_pump_age_h": age_h,
                    "features": features,
                }

    # === STAGE 2: Look for new pre-pump signal ===
    is_match, confidence = _check_pre_pump_pattern(features, repeater)
    if is_match and confidence >= PRE_PUMP_CONFIDENCE_MIN:
        # Save as pending
        if "pending_pre_pump" not in state:
            state["pending_pre_pump"] = {}
        state["pending_pre_pump"][symbol] = {
            "ts": now_ts,
            "iso_time": now_iso,
            "entry_price": features["close"],
            "confidence": confidence,
            "size_fraction": PRE_PUMP_SIZE_FRACTION,
            "features": features,
        }
        _save_state(state)
        return {
            "action": "enter_pre",
            "symbol": symbol,
            "direction": "LONG",
            "entry_price": features["close"],
            "ts": now_ts,
            "iso_time": now_iso,
            "confidence": confidence,
            "size_fraction": PRE_PUMP_SIZE_FRACTION,
            "features": features,
            "repeater": repeater,
        }

    return {
        "action": "none",
        "symbol": symbol,
        "features": features,
        "pattern_confidence": confidence,
    }


def scan_all_repeaters() -> List[Dict]:
    """Scan all repeaters. Returns list of action dicts."""
    results = []
    for symbol in REPEATERS.keys():
        try:
            r = scan_repeater(symbol)
            results.append(r)
        except Exception as e:
            results.append({"action": "error", "symbol": symbol, "error": str(e)})
        time.sleep(0.5)  # rate limit
    return results


def execute_entry(action: Dict) -> Optional[str]:
    """
    Execute a pre-pump or confirm entry via signal_tracker.
    Returns signal_id if opened.
    """
    if action.get("action") not in ("enter_pre", "confirm"):
        return None

    symbol = action["symbol"]
    direction = action["direction"]
    entry_price = action["entry_price"]
    features = action["features"]
    repeater = action["repeater"]

    # TP/SL from repeater config
    tp_pct = repeater["tp_pct"]
    sl_pct = repeater["sl_pct"]
    trail_pct = repeater["trail_pct"]
    max_hold = repeater["max_hold_hours"]

    # For pre-pump entry: use looser SL (we expect volatility)
    # For confirm entry: standard SL
    if action["action"] == "enter_pre":
        # Slightly wider SL for early entry
        sl_pct_adj = sl_pct * 1.3
        confidence = action["confidence"]
        entry_mode = "PRE_PUMP_30"
        score_long = action["confidence"] * 0.5  # lower score for early
        position_size = PRE_PUMP_SIZE_FRACTION  # 30%
    else:
        sl_pct_adj = sl_pct
        confidence = action["confirm_confidence"]
        entry_mode = "CONFIRM_100"
        score_long = action["confirm_confidence"]
        position_size = CONFIRM_SIZE_FRACTION  # 70% (the remaining 30% was pre-pump)

    feats_for_tracker = {
        "n_long_signals": 1,
        "n_short_signals": 0,
        "f_momentum_3_pct": features.get("mom_3", 0),
        "f_momentum_6_pct": features.get("mom_6", 0),
        "f_rvol": features.get("rvol", 1),
        "f_atr_pct": features.get("atr_pct", 0),
        "f_volume_spike": 1 if features.get("rvol", 0) > 2 else 0,
        "f_m_5m_volume_spike": 1 if features.get("rvol", 0) > 3 else 0,
        "f_bb_breakout_above": 1 if features.get("mom_3", 0) > 0 else 0,
    }

    try:
        signal_id = open_signal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            score_long=score_long,
            score_short=0,
            confidence=confidence,
            features=feats_for_tracker,
            tp_pct=tp_pct,
            sl_pct=sl_pct_adj,
            max_hold_hours=max_hold,
            trailing_pct=trail_pct,
            use_trailing=True,
            use_scaled=False,
            btc_state="NEUTRAL",
            btc_momentum=0.0,
            market_regime=entry_mode,  # tag with mode for analytics
            position_size=position_size,
            entry_mode=entry_mode,
        )
        return signal_id
    except Exception as e:
        print(f"  repeater entry error for {symbol}: {e}")
        return None


def mark_pump_detected(symbol: str) -> None:
    """Mark that a pump was detected on this symbol (used for cluster follow)."""
    state = _load_state()
    state["last_pump_time"][symbol] = datetime.now(timezone.utc).isoformat()
    state["cluster_window_active_until"] = (
        datetime.now(timezone.utc) + timedelta(hours=CLUSTER_FOLLOW_HOURS)
    ).isoformat()
    # Clear pending pre-pump for this symbol
    if symbol in state.get("pending_pre_pump", {}):
        del state["pending_pre_pump"][symbol]
    _save_state(state)


def cluster_follow_active() -> bool:
    """Are we in an active cluster follow window?"""
    state = _load_state()
    until = state.get("cluster_window_active_until")
    if not until:
        return False
    try:
        until_dt = datetime.fromisoformat(until)
        return datetime.now(timezone.utc) < until_dt
    except Exception:
        return False


def run_repeater_cycle(verbose: bool = True) -> Dict:
    """
    Main cycle: scan all repeaters, take action on signals.
    Returns summary dict.
    """
    if verbose:
        print(f"\n[REPEATER] Scanning {len(REPEATERS)} repeaters...")

    cluster_active = cluster_follow_active()
    results = scan_all_repeaters()

    summary = {
        "n_scanned": len(results),
        "pre_pumps": [],
        "confirmed": [],
        "errors": [],
        "cluster_active": cluster_active,
    }

    for r in results:
        action = r.get("action")
        sym = r.get("symbol", "?")
        if action == "enter_pre":
            if verbose:
                print(f"  [PRE]  {sym:<14} conf={r['confidence']:.0f}%  size={r['size_fraction']*100:.0f}%  rvol={r['features']['rvol']:.2f}  mom_3={r['features']['mom_3']:+.2f}%  flat={r['features']['flat_hours']:.0f}h")
            signal_id = execute_entry(r)
            if signal_id:
                summary["pre_pumps"].append({"symbol": sym, "signal_id": signal_id, "confidence": r["confidence"]})
                mark_pump_detected(sym)
        elif action == "confirm":
            if verbose:
                print(f"  [CONFIRM] {sym:<14} pre_conf={r['pre_pump_confidence']:.0f}%  conf_conf={r['confirm_confidence']:.0f}%  rvol={r['features']['rvol']:.2f}  mom_3={r['features']['mom_3']:+.2f}%")
            signal_id = execute_entry(r)
            if signal_id:
                summary["confirmed"].append({"symbol": sym, "signal_id": signal_id, "confirm_confidence": r["confirm_confidence"]})
        elif action == "watch":
            if verbose:
                print(f"  [WAIT]  {sym:<14} pre-pump pending ({r['pre_pump_age_h']:.1f}h ago), waiting for confirm")
        elif action == "error":
            if verbose:
                print(f"  [ERR]   {sym:<14} {r.get('error', '?')}")
            summary["errors"].append({"symbol": sym, "error": r.get("error")})

    return summary


if __name__ == "__main__":
    # Manual test
    summary = run_repeater_cycle(verbose=True)
    print(f"\nSummary: {summary}")
