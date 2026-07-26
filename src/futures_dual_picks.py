"""
LIVE dual-direction picks - FUTURES ONLY (Toobit perps).
Filters watchlist to symbols actually listed in Toobit futures.
Uses -SWAP-USDT format for klines endpoint.

Usage:
  python3 -m src.futures_dual_picks
  python3 -m src.futures_dual_picks --top 5
"""
from __future__ import annotations
import sys
import os
import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repeater_config import REPEATERS, SECONDARY_WATCHLIST


# ============================================================================
# Toobit futures helpers
# ============================================================================
TOOBIT_BASE = "https://api.toobit.com"
SPOT_TICKER = "/quote/v1/ticker/24hr"
FUTURES_KLINES = "/quote/v1/klines"  # works for perps too with -SWAP-USDT


def base_to_perp(symbol: str) -> str:
    """BTCUSDT -> BTC-SWAP-USDT for futures klines."""
    if "-SWAP-" in symbol:
        return symbol
    base = symbol.replace("USDT", "")
    return f"{base}-SWAP-USDT"


def fetch_futures_perp_list() -> Set[str]:
    """Get all TRADING perpetual symbols from Toobit."""
    try:
        url = f"{TOOBIT_BASE}/api/v1/exchangeInfo"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        contracts = data.get("contracts", [])
        out = set()
        for c in contracts:
            if c.get("status") == "TRADING":
                full = c.get("symbol", "")
                base = full.replace("-SWAP-USDT", "").replace("USDT", "")
                out.add(f"{base}USDT")
        return out
    except Exception as e:
        print(f"⚠️ Could not fetch perp list: {e}")
        return set()


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 100,
                 timeout: int = 10) -> Optional[list]:
    """Fetch klines for FUTURES (uses -SWAP-USDT format)."""
    perp = base_to_perp(symbol)
    try:
        url = f"{TOOBIT_BASE}{FUTURES_KLINES}?symbol={perp}&interval={interval}&limit={limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        if data and isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def parse_klines(raw: list) -> Optional[Dict]:
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
    closes = klines["close"]
    highs = klines["high"]
    lows = klines["low"]
    vols = klines["volume"]
    n = len(closes)
    if n < 30:
        return {}

    mom_1 = (closes[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] else 0
    mom_3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if n > 3 and closes[-4] else 0
    mom_6 = (closes[-1] - closes[-7]) / closes[-7] * 100 if n > 6 and closes[-7] else 0

    diffs = np.diff(closes[-15:])
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = gains.mean() if len(gains) else 0
    avg_loss = losses.mean() if len(losses) else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

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

    if n > 24:
        recent_vol = vols[-4:].mean()
        base_vol = vols[-24:].mean()
        rvol = recent_vol / base_vol if base_vol > 0 else 1.0
    else:
        rvol = 1.0

    if n >= 20:
        sma20 = closes[-20:].mean()
        std20 = closes[-20:].std()
        bb_width = (2 * std20) / sma20 if sma20 > 0 else 0
        bb_squeeze = bb_width < 0.05
    else:
        bb_squeeze = False

    if n >= 24:
        vwap = np.sum(closes[-24:] * vols[-24:]) / np.sum(vols[-24:]) if np.sum(vols[-24:]) > 0 else closes[-1]
    else:
        vwap = closes[-1]
    vwap_dist = (closes[-1] - vwap) / vwap * 100 if vwap > 0 else 0

    flat_hours = 0
    for i in range(-1, -min(48, n), -1):
        if i == 0 or -i >= n:
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
        "mom_6": float(mom_6), "rsi": float(rsi),
        "atr_pct": float(atr_pct), "rvol": float(rvol),
        "bb_squeeze": bool(bb_squeeze), "vwap_dist": float(vwap_dist),
        "flat_hours": int(flat_hours),
    }


def compute_long_score(f: Dict) -> float:
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
    if f["vwap_dist"] > 0 and f["vwap_dist"] < 5: s += 5
    if f["flat_hours"] >= 3: s += 5
    return max(0, min(100, s))


def compute_short_score(f: Dict) -> float:
    s = 0.0
    if f["mom_3"] < -5: s += 15
    elif f["mom_3"] < -2: s += 10
    elif f["mom_3"] < 0: s += 5
    if f["mom_1"] < -1: s += 8
    elif f["mom_1"] < 0: s += 4
    if f["rsi"] > 75: s += 18
    elif f["rsi"] > 65: s += 10
    if f["rvol"] >= 3.0 and f["mom_1"] < 0: s += 12
    elif f["rvol"] >= 2.0 and f["mom_1"] < 0: s += 8
    if f["vwap_dist"] < -3: s += 8
    return max(0, min(100, s))


def ml_pump_setup(f: Dict) -> bool:
    return f.get("atr_pct", 0) > 1.09 and f.get("flat_hours", 99) < 34


def get_tpsl_for_symbol(symbol: str) -> Dict:
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
    return {
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
        "perp_symbol": base_to_perp(symbol),
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


def scan_futures(top_n: int = 8, min_score: int = 30) -> Dict:
    """Scan watchlist filtered to Toobit futures only."""
    all_symbols = sorted(set(REPEATERS.keys()) | set(SECONDARY_WATCHLIST))
    print(f"📡 Fetching Toobit futures perp list...")
    perp_list = fetch_futures_perp_list()
    if not perp_list:
        return {"error": "Could not fetch perp list"}
    print(f"  ✓ Toobit has {len(perp_list)} USDT perps")

    # Filter watchlist to perps only
    watchlist = [s for s in all_symbols if s in perp_list]
    excluded = [s for s in all_symbols if s not in perp_list]
    print(f"  ✓ Watchlist: {len(watchlist)} in futures, {len(excluded)} excluded")
    if excluded:
        print(f"  ❌ Excluded: {', '.join(excluded[:5])}{'...' if len(excluded) > 5 else ''}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    results = []
    for i, sym in enumerate(watchlist, 1):
        if i % 10 == 0:
            print(f"  [{i}/{len(watchlist)}] scanning...")
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

    print(f"  ✓ Got valid data for {len(results)} symbols\n")
    longs, shorts = [], []
    for sym, f in results:
        ls = compute_long_score(f)
        ss = compute_short_score(f)
        if ls >= min_score:
            longs.append(build_pick(f, "LONG", sym))
        if ss >= min_score:
            shorts.append(build_pick(f, "SHORT", sym))

    longs.sort(key=lambda x: (x["ml_filter"], x["score"], x["rvol"]), reverse=True)
    shorts.sort(key=lambda x: (x["ml_filter"], x["score"], x["rvol"]), reverse=True)

    return {
        "timestamp": now,
        "n_watchlist": len(all_symbols),
        "n_in_futures": len(watchlist),
        "n_excluded": len(excluded),
        "n_with_data": len(results),
        "longs": longs[:top_n],
        "shorts": shorts[:top_n],
        "n_longs_qualified": len(longs),
        "n_shorts_qualified": len(shorts),
        "excluded": excluded,
    }


def format_picks(picks: Dict) -> str:
    if "error" in picks:
        return f"❌ {picks['error']}"
    L = []
    L.append("=" * 82)
    L.append(f"🎯 PUMPHUNTER-AI FUTURES (Toobit Perps)  |  {picks['timestamp']}")
    L.append("=" * 82)
    L.append(
        f"📊 Watchlist: {picks['n_watchlist']}  |  "
        f"In Futures: {picks['n_in_futures']}  |  "
        f"Data: {picks['n_with_data']}  |  "
        f"🟢 Longs: {picks['n_longs_qualified']}  |  "
        f"🔴 Shorts: {picks['n_shorts_qualified']}"
    )
    L.append("")

    def render(p, i, emoji):
        sym = p["symbol"]
        perp = p["perp_symbol"]
        is_rep = "⭐" if p["is_repeater"] else "  "
        ml = "🧠" if p["ml_filter"] else "  "
        L.append(
            f"{i}. {emoji} {is_rep} {ml} {p['direction']:5s} {sym:12s} | "
            f"score {p['score']:.0f} | {perp}"
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
        L.append("-" * 82)
        for i, p in enumerate(picks["longs"], 1):
            render(p, i, "🟢")
    else:
        L.append("🟢 TOP LONG PICKS: (none qualify)")
        L.append("")
    if picks["shorts"]:
        L.append("🔴 TOP SHORT PICKS:")
        L.append("-" * 82)
        for i, p in enumerate(picks["shorts"], 1):
            render(p, i, "🔴")
    else:
        L.append("🔴 TOP SHORT PICKS: (none qualify)")
        L.append("")
    L.append("=" * 82)
    L.append("⭐ = Primary repeater  |  🧠 = ML filter (81.3%)  |  R/R 1:N = risk:reward")
    L.append("=" * 82)
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-score", type=int, default=30)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    picks = scan_futures(top_n=args.top, min_score=args.min_score)
    if args.json:
        print(json.dumps(picks, indent=2, default=str))
    else:
        print(format_picks(picks))
