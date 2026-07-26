"""
LIVE dual-direction picks - scans all 77 watchlist symbols in real time.
Fetches from Toobit directly (no DB dependency), applies ML filter + cap-based TP/SL.

Usage:
  python3 -m src.live_dual_picks
  python3 -m src.live_dual_picks --top 5 --min-conf 50
"""
from __future__ import annotations
import sys
import os
import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repeater_config import REPEATERS, SECONDARY_WATCHLIST
# self-contained feature computation (no external indicators needed)


# ============================================================================
# Toobit public API helpers
# ============================================================================
TOOBIT_BASE = "https://api.toobit.com"


def fetch_ticker(symbol: str, timeout: int = 8) -> Optional[Dict]:
    """24h ticker from Toobit."""
    try:
        url = f"{TOOBIT_BASE}/quote/v1/ticker/24hr?symbol={symbol}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        if data and len(data) > 0:
            t = data[0]
            return {
                "symbol": symbol,
                "price": float(t.get("c", 0)),
                "change_24h": float(t.get("pcp", 0)),
                "vol_24h": float(t.get("qv", 0)),
                "high_24h": float(t.get("h", 0)),
                "low_24h": float(t.get("l", 0)),
            }
    except Exception:
        return None
    return None


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 100,
                 timeout: int = 10) -> Optional[list]:
    """Klines from Toobit public API."""
    try:
        url = f"{TOOBIT_BASE}/quote/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        # Toobit returns: [[openTime, open, high, low, close, vol, ...], ...]
        if data and isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def parse_klines(raw: list) -> Optional[Dict]:
    """Convert raw klines to numpy arrays."""
    if not raw:
        return None
    try:
        opens = np.array([float(r[1]) for r in raw])
        highs = np.array([float(r[2]) for r in raw])
        lows = np.array([float(r[3]) for r in raw])
        closes = np.array([float(r[4]) for r in raw])
        vols = np.array([float(r[5]) for r in raw])
        return {"open": opens, "high": highs, "low": lows,
                "close": closes, "volume": vols}
    except Exception:
        return None


def compute_features(klines: Dict) -> Dict:
    """Compute minimal feature set on raw arrays."""
    closes = klines["close"]
    highs = klines["high"]
    lows = klines["low"]
    vols = klines["volume"]
    n = len(closes)
    if n < 30:
        return {}

    # Momentum
    mom_1 = (closes[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] else 0
    mom_3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if n > 3 and closes[-4] else 0
    mom_6 = (closes[-1] - closes[-7]) / closes[-7] * 100 if n > 6 and closes[-7] else 0
    mom_12 = (closes[-1] - closes[-13]) / closes[-13] * 100 if n > 12 and closes[-13] else 0

    # RSI(14)
    diffs = np.diff(closes[-15:])
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = gains.mean() if len(gains) else 0
    avg_loss = losses.mean() if len(losses) else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # ATR(14) as percent of price
    trs = []
    for i in range(-14, 0):
        if i == -14:
            continue
        prev_c = closes[i - 1]
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - prev_c), abs(lows[i] - prev_c))
        trs.append(tr)
    atr = np.mean(trs) if trs else 0
    atr_pct = (atr / closes[-1]) * 100 if closes[-1] else 0

    # Relative volume: last 4h avg vs 20-period avg
    if n > 20:
        recent_vol = vols[-4:].mean()
        base_vol = vols[-24:].mean() if n > 24 else vols[:-4].mean()
        rvol = recent_vol / base_vol if base_vol > 0 else 1.0
    else:
        rvol = 1.0

    # Bollinger squeeze: bb width < 0.05
    if n >= 20:
        sma20 = closes[-20:].mean()
        std20 = closes[-20:].std()
        bb_width = (2 * std20) / sma20 if sma20 > 0 else 0
        bb_squeeze = bb_width < 0.05
        bb_pct_b = (closes[-1] - (sma20 - 2 * std20)) / (4 * std20) if std20 > 0 else 0.5
    else:
        bb_squeeze = False
        bb_pct_b = 0.5

    # VWAP distance (using recent 24h)
    if n >= 24:
        vwap = np.sum(closes[-24:] * vols[-24:]) / np.sum(vols[-24:]) if np.sum(vols[-24:]) > 0 else closes[-1]
    else:
        vwap = closes[-1]
    vwap_dist = (closes[-1] - vwap) / vwap * 100 if vwap > 0 else 0

    # Higher highs / higher lows
    hh = (highs[-1] > highs[-2] > highs[-3]) if n > 3 else False
    hl = (lows[-1] > lows[-2] > lows[-3]) if n > 3 else False
    lh = (highs[-1] < highs[-2] < highs[-3]) if n > 3 else False
    ll = (lows[-1] < lows[-2] < lows[-3]) if n > 3 else False

    # Flat hours: count of candles with abs(change) < 1%
    flat_hours = 0
    for i in range(-1, -min(48, n), -1):
        if i == -1 or i == 0:
            continue
        c1 = closes[i]
        c2 = closes[i - 1]
        if c2 > 0 and abs(c1 - c2) / c2 < 0.01:
            flat_hours += 1
        else:
            break

    return {
        "price": float(closes[-1]),
        "mom_1": float(mom_1), "mom_3": float(mom_3),
        "mom_6": float(mom_6), "mom_12": float(mom_12),
        "rsi": float(rsi), "atr_pct": float(atr_pct),
        "rvol": float(rvol), "bb_squeeze": bool(bb_squeeze),
        "bb_pct_b": float(bb_pct_b), "vwap_dist": float(vwap_dist),
        "higher_highs": bool(hh), "higher_lows": bool(hl),
        "lower_highs": bool(lh), "lower_lows": bool(ll),
        "flat_hours": int(flat_hours),
    }


def compute_long_score(f: Dict) -> float:
    """Long score 0-100."""
    s = 0.0
    if f["rvol"] >= 5.0: s += 35
    elif f["rvol"] >= 3.0: s += 25
    elif f["rvol"] >= 2.0: s += 15
    elif f["rvol"] >= 1.0: s += 5
    else: s -= 5
    if f["mom_3"] > 5: s += 15
    elif f["mom_3"] > 2: s += 10
    elif f["mom_3"] > 0: s += 5
    if f["mom_1"] > 1: s += 8
    elif f["mom_1"] > 0: s += 4
    if 30 <= f["rsi"] <= 65: s += 10
    elif f["rsi"] > 75: s -= 12
    if f["bb_squeeze"]: s += 5
    if f["higher_lows"]: s += 5
    if f["vwap_dist"] > 0 and f["vwap_dist"] < 5: s += 5
    if f["flat_hours"] >= 3: s += 5
    return max(0, min(100, s))


def compute_short_score(f: Dict) -> float:
    """Short score 0-100."""
    s = 0.0
    if f["higher_highs"]: s += 12
    if f["higher_lows"]: s += 8
    if f["mom_3"] < -5: s += 15
    elif f["mom_3"] < -2: s += 10
    elif f["mom_3"] < 0: s += 5
    if f["mom_1"] < -1: s += 8
    elif f["mom_1"] < 0: s += 4
    if f["rsi"] > 75: s += 18
    elif f["rsi"] > 65: s += 10
    if f["rvol"] >= 3.0 and f["mom_1"] < 0: s += 12
    elif f["rvol"] >= 2.0 and f["mom_1"] < 0: s += 8
    if f["lower_highs"] and f["lower_lows"]: s += 10
    if f["vwap_dist"] < -3: s += 8
    return max(0, min(100, s))


def ml_pump_setup(f: Dict) -> bool:
    """ML filter: atr_pct > 1.09 AND flat_hours < 34."""
    return f.get("atr_pct", 0) > 1.09 and f.get("flat_hours", 99) < 34


def get_tpsl_for_symbol(symbol: str) -> Dict:
    """Per-symbol TP/SL profile."""
    if symbol in REPEATERS:
        cfg = REPEATERS[symbol]
        return {
            "tp_pct": cfg.get("tp_pct", 20.0),
            "sl_pct": cfg.get("sl_pct", 2.5),
            "trail_pct": cfg.get("trail_pct", 5.0),
            "trail_activate_pct": cfg.get("trail_activate_pct", 15.0),
            "max_hold_hours": cfg.get("max_hold_hours", 12.0),
            "breakeven_at": 1.5, "lock_25_at": 3.0, "lock_50_at": 6.0,
            "reentry_on_dip": cfg.get("reentry_on_dip", True),
            "reentry_rsi_max": cfg.get("reentry_rsi_max", 35.0),
        }
    return {  # default mid-cap
        "tp_pct": 20.0, "sl_pct": 2.5,
        "trail_pct": 5.0, "trail_activate_pct": 12.0,
        "max_hold_hours": 10.0,
        "breakeven_at": 1.5, "lock_25_at": 3.0, "lock_50_at": 6.0,
        "reentry_on_dip": True, "reentry_rsi_max": 35.0,
    }


def build_pick(f: Dict, direction: str, symbol: str) -> Dict:
    entry = f["price"]
    tpsl = get_tpsl_for_symbol(symbol)
    tp_pct = tpsl["tp_pct"]
    sl_pct = tpsl["sl_pct"]
    if direction == "LONG":
        tp_price = entry * (1 + tp_pct / 100)
        sl_price = entry * (1 - sl_pct / 100)
        score = compute_long_score(f)
    else:
        tp_price = entry * (1 - tp_pct / 100)
        sl_price = entry * (1 + sl_pct / 100)
        score = compute_short_score(f)
    rr = tp_pct / sl_pct if sl_pct else 0
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry, 6),
        "tp_pct": tp_pct, "sl_pct": sl_pct,
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "rr_ratio": round(rr, 2),
        "score": round(score, 1),
        "rvol": round(f["rvol"], 2),
        "rsi": round(f["rsi"], 1),
        "mom_3_pct": round(f["mom_3"], 2),
        "atr_pct": round(f["atr_pct"], 2),
        "vwap_dist": round(f["vwap_dist"], 2),
        "bb_squeeze": f["bb_squeeze"],
        "flat_hours": f["flat_hours"],
        "ml_filter": ml_pump_setup(f),
        "is_repeater": symbol in REPEATERS,
        "smart_exit": {
            "breakeven_at": tpsl["breakeven_at"],
            "lock_25_at": tpsl["lock_25_at"],
            "lock_50_at": tpsl["lock_50_at"],
            "trail_pct": tpsl["trail_pct"],
            "trail_activate_pct": tpsl["trail_activate_pct"],
            "max_hold_hours": tpsl["max_hold_hours"],
            "reentry_on_dip": tpsl["reentry_on_dip"],
            "reentry_rsi_max": tpsl["reentry_rsi_max"],
        },
    }


def scan_live(top_n: int = 8, min_score: int = 30, max_workers: int = 8) -> Dict:
    """Scan all 77 symbols live and return dual picks."""
    all_symbols = sorted(set(REPEATERS.keys()) | set(SECONDARY_WATCHLIST))
    results = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"🔍 Scanning {len(all_symbols)} symbols live from Toobit...")

    # Sequential scan with progress (parallel = blocked by Toobit)
    for i, sym in enumerate(all_symbols, 1):
        if i % 10 == 0:
            print(f"  [{i}/{len(all_symbols)}] {sym}...")
        klines = fetch_klines(sym, "1h", 100)
        if not klines:
            continue
        kd = parse_klines(klines)
        if not kd:
            continue
        f = compute_features(kd)
        if not f:
            continue
        results.append((sym, f))

    print(f"  ✓ Got data for {len(results)} symbols\n")
    longs, shorts = [], []
    for sym, f in results:
        ls = compute_long_score(f)
        ss = compute_short_score(f)
        if ls >= min_score:
            longs.append(build_pick(f, "LONG", sym))
        if ss >= min_score:
            shorts.append(build_pick(f, "SHORT", sym))

    # Sort: ML filter pass first, then score
    longs.sort(key=lambda x: (x["ml_filter"], x["score"], x["rvol"]), reverse=True)
    shorts.sort(key=lambda x: (x["ml_filter"], x["score"], x["rvol"]), reverse=True)

    return {
        "timestamp": now,
        "n_scanned": len(all_symbols),
        "n_with_data": len(results),
        "longs": longs[:top_n],
        "shorts": shorts[:top_n],
        "n_longs_qualified": len(longs),
        "n_shorts_qualified": len(shorts),
    }


def format_live_picks(picks: Dict) -> str:
    L = []
    L.append("=" * 80)
    L.append(f"🎯 PUMPHUNTER-AI LIVE  |  DUAL PICKS  |  {picks['timestamp']}")
    L.append("=" * 80)
    L.append(
        f"📡 Scanned: {picks['n_scanned']} symbols  |  "
        f"Data: {picks['n_with_data']}  |  "
        f"🟢 Longs: {picks['n_longs_qualified']}  |  "
        f"🔴 Shorts: {picks['n_shorts_qualified']}"
    )
    L.append("")

    def render(p, i, side_emoji):
        sym = p["symbol"]
        is_rep = "⭐" if p["is_repeater"] else "  "
        ml = "🧠" if p["ml_filter"] else "  "
        L.append(
            f"{i}. {side_emoji} {is_rep} {ml} {p['direction']:5s} {sym:12s} | "
            f"score {p['score']:.0f}"
        )
        L.append(
            f"   💰 E: {p['entry_price']}  🎯 TP: {p['tp_price']} "
            f"(+{p['tp_pct']}%)  🛡️ SL: {p['sl_price']} (-{p['sl_pct']}%)  "
            f"R/R 1:{p['rr_ratio']:.1f}"
        )
        L.append(
            f"   📈 mom3: {p['mom_3_pct']:+.1f}%  RSI: {p['rsi']:.0f}  "
            f"rvol: {p['rvol']:.2f}  ATR: {p['atr_pct']:.1f}%  "
            f"BBsqueeze: {p['bb_squeeze']}  flat: {p['flat_hours']}h"
        )
        se = p["smart_exit"]
        L.append(
            f"   🔒 BE@{se['breakeven_at']}% L25@{se['lock_25_at']}% "
            f"L50@{se['lock_50_at']}% Trail{se['trail_pct']}%@{se['trail_activate_pct']}% "
            f"MaxHold:{se['max_hold_hours']}h"
        )
        if se["reentry_on_dip"]:
            L.append(f"   ♻️ Re-entry: RSI<{se['reentry_rsi_max']} → add 30%")
        L.append("")

    if picks["longs"]:
        L.append("🟢 TOP LONG PICKS:")
        L.append("-" * 80)
        for i, p in enumerate(picks["longs"], 1):
            render(p, i, "🟢")
    else:
        L.append("🟢 TOP LONG PICKS: (none qualify)")
        L.append("")
    if picks["shorts"]:
        L.append("🔴 TOP SHORT PICKS:")
        L.append("-" * 80)
        for i, p in enumerate(picks["shorts"], 1):
            render(p, i, "🔴")
    else:
        L.append("🔴 TOP SHORT PICKS: (none qualify)")
        L.append("")
    L.append("=" * 80)
    L.append("⭐ = Primary repeater  |  🧠 = ML filter (81.3% accuracy)  |  R/R 1:N risk:reward")
    L.append("=" * 80)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-score", type=int, default=30)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    picks = scan_live(top_n=args.top, min_score=args.min_score)
    if args.json:
        print(json.dumps(picks, indent=2, default=str))
    else:
        print(format_live_picks(picks))
