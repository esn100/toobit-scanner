"""
Direction-aware scanner: detects BOTH pumps and dumps.

For each small cap, computes:
  - Long score: probability of +5%/+10% in 12h
  - Short score: probability of -5%/-10% in 12h

Output signals:
  - APPROVED_LONG: high long score, expect pump
  - APPROVED_SHORT: high short score, expect dump
  - WATCHLIST: borderline
  - REJECTED: no clear signal
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from coinpaprika import CoinPaprikaClient
from data_quality import validate_ohlcv
from prefilter import prefilter_score, passes_prefilter
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, momentum_features,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from chart_patterns import detect_all_patterns
from features import build_features
from btc_filter import BTCFilter
from btc_correlation import btc_correlation_features
from cooldown import CooldownGuard
from telegram_notifier import TelegramNotifier
from okx_history import OKXHistory
from binance_history import BinanceHistory
from elliott_wave import detect_elliott_waves, elliott_score
from fibonacci import compute_fib_levels, fib_score
from ichimoku import ichimoku_features, ichimoku_score


def compute_long_score(pack: Dict) -> float:
    """Score 0-100 for LONG (pump) signal."""
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
    candle_1h = pack.get("candle_1h", {})
    btc_corr = pack.get("btc_corr", {})
    rvol = ind_1h.get("rvol", 1.0)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    m6 = ind_1h.get("momentum_6_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    rsi_1h = tech_1h.get("rsi_value", 50)
    higher_lows = struct_1h.get("higher_lows", False)
    bb_squeeze = ind_1h.get("bb_squeeze", False)
    volume_spike = ind_1h.get("volume_spike", False)
    score = 0.0
    # Volume: king for pumps
    if rvol >= 5.0: score += 35
    elif rvol >= 3.0: score += 25
    elif rvol >= 2.0: score += 15
    elif rvol < 0.5: score -= 10
    if volume_spike: score += 8
    # Positive momentum
    if m3 > 5.0: score += 15
    elif m3 > 2.0: score += 10
    elif m3 > 0: score += 5
    if m1 > 1.0: score += 8
    elif m1 > 0: score += 4
    if mom_acc > 0: score += min(10, mom_acc * 5)
    # RSI sweet spot (not overbought)
    if 30 <= rsi_1h <= 65: score += 10
    elif rsi_1h > 75: score -= 12
    # BB squeeze (pre-breakout)
    if bb_squeeze: score += 5
    # Big lower wick = rejection of lows (bullish)
    if candle_1h.get("big_wick_top") is False:
        score += 3
    # BTC independent
    if btc_corr.get("independent_mover"):
        score += 5
    return max(0, min(100, score))


def compute_short_score(pack: Dict) -> float:
    """Score 0-100 for SHORT (dump) signal."""
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
    candle_1h = pack.get("candle_1h", {})
    btc_corr = pack.get("btc_corr", {})
    rvol = ind_1h.get("rvol", 1.0)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    m6 = ind_1h.get("momentum_6_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    rsi_1h = tech_1h.get("rsi_value", 50)
    higher_highs = struct_1h.get("higher_highs", False)
    higher_lows = struct_1h.get("higher_lows", False)
    big_wick = candle_1h.get("big_wick_top", False)
    score = 0.0
    # Distribution pattern (smart money selling into strength)
    if higher_highs: score += 15
    if higher_lows: score += 10
    # Negative momentum
    if m3 < -5.0: score += 15
    elif m3 < -2.0: score += 10
    elif m3 < 0: score += 5
    if m1 < -1.0: score += 8
    elif m1 < 0: score += 4
    # Momentum deceleration (acceleration turning negative)
    if mom_acc < 0: score += min(10, abs(mom_acc) * 5)
    # RSI overbought
    if rsi_1h > 75: score += 15
    elif rsi_1h > 65: score += 8
    # High volume on negative move = strong dump
    if rvol >= 3.0 and m1 < 0: score += 12
    elif rvol >= 2.0 and m1 < 0: score += 8
    # Big upper wick = rejection
    if big_wick: score += 12
    # BTC correlation (dumping with BTC)
    btc_corr_2d = btc_corr.get("btc_corr_2d", 0)
    if btc_corr_2d > 0.6:
        score += 5
    return max(0, min(100, score))


def deep_analyze(symbol: str, toobit: ToobitClient,
                 btc_state: dict, btc_df_1h: pd.DataFrame,
                 okx: Optional[OKXHistory] = None,
                 binance: Optional[BinanceHistory] = None) -> Dict:
    """Full deep analysis on a symbol."""
    out = {"pass2_ok": False, "long_score": 0.0, "short_score": 0.0,
           "max_direction": "NEUTRAL", "max_score": 0.0}
    try:
        df_15m = toobit.get_klines(symbol, "15m", 200)
        df_1h = toobit.get_klines(symbol, "1h", 200)
        df_4h = toobit.get_klines(symbol, "4h", 50)
    except Exception:
        return out
    if df_15m.empty or len(df_15m) < 30:
        return out
    pf = prefilter_score(df_15m, df_1h, df_4h, btc_df_1h)
    if not passes_prefilter(pf, min_score=20.0):
        return out
    if df_1h.empty or len(df_1h) < 30:
        return out
    tech_1h = technical_analysis(df_1h)
    ind_1h = {}
    ind_1h.update(vwap_features(df_1h))
    ind_1h.update(atr_features(df_1h))
    ind_1h.update(bollinger_features(df_1h))
    ind_1h.update(relative_volume(df_1h))
    ind_1h.update(momentum_features(df_1h))
    struct_1h = structure_features(df_1h)
    candle_1h = candle_quality_features(df_1h)
    patterns_1h = detect_all_patterns(df_1h)
    btc_corr = btc_correlation_features(df_1h, btc_df_1h)
    # ----- NEW: Elliott Wave, Fibonacci, Ichimoku on OKX history -----
    elliott = {"wave": "none", "score": 50.0, "details": {}}
    fib = {"levels": {}, "direction": "none", "current_price": 0,
           "closest_level": None, "distance_to_closest": 100.0}
    ichi = {"current_price": 0, "price_vs_cloud": "neutral",
            "tk_cross": "neutral", "cloud_color": "neutral",
            "future_cloud_color": "neutral", "cloud_thickness_pct": 0}
    # Try OKX first (works from Iran), then Binance
    history_provider = None
    if okx is not None:
        try:
            df_hist = okx.get_history_for_toobit_symbol(symbol, "1H", 300)
            if not df_hist.empty and len(df_hist) >= 60:
                history_provider = "okx"
        except Exception as e:
            print(f"[dir] okx err for {symbol}: {e}")
    if history_provider is None and binance is not None:
        try:
            df_hist = binance.get_binance_history_for_toobit_symbol(
                symbol, "1h", days=30
            )
            if not df_hist.empty and len(df_hist) >= 60:
                history_provider = "binance"
        except Exception as e:
            print(f"[dir] binance err for {symbol}: {e}")
    if history_provider is not None:
        try:
            elliott = detect_elliott_waves(df_hist, threshold=0.04)
            fib = compute_fib_levels(df_hist, lookback=60)
            ichi = ichimoku_features(df_hist)
        except Exception as e:
            print(f"[dir] indicator err for {symbol}: {e}")
    # ----- end new -----
    out["pass2_ok"] = True
    out["ind_1h"] = ind_1h
    out["tech_1h"] = tech_1h
    out["struct_1h"] = struct_1h
    out["candle_1h"] = candle_1h
    out["patterns_1h"] = patterns_1h
    out["btc_corr"] = btc_corr
    out["elliott"] = elliott
    out["fib"] = fib
    out["ichimoku"] = ichi
    out["long_score"] = compute_long_score(out)
    out["short_score"] = compute_short_score(out)
    # Apply Elliott/Fib/Ichimoku boost
    from advanced_indicators import advanced_score_boost
    long_boost = advanced_score_boost(out, "LONG")
    short_boost = advanced_score_boost(out, "SHORT")
    out["long_score"] = max(0, min(100, out["long_score"] + long_boost))
    out["short_score"] = max(0, min(100, out["short_score"] + short_boost))
    if out["long_score"] > out["short_score"]:
        out["max_direction"] = "LONG"
        out["max_score"] = out["long_score"]
    else:
        out["max_direction"] = "SHORT"
        out["max_score"] = out["short_score"]
    if not df_1h.empty:
        out["last_price"] = float(df_1h["close"].iloc[-1])
        out["volume_24h_usd"] = float(
            df_1h["quote_volume"].tail(24).sum()
        )
    return out


def run_direction_scan(
    project_root: str = ".",
    max_symbols: int = 60,
    score_threshold: float = 60.0,
) -> Dict:
    """Run direction-aware scan with adaptive threshold."""
    toobit = ToobitClient()
    cp = CoinPaprikaClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[dir] BTC state: {btc_state['state']}", flush=True)
    if btc_state["freeze"]:
        return {"alerts_long": [], "alerts_short": [], "watchlist": []}
    # Binance history for Elliott/Fib/Ichimoku
    try:
        binance = BinanceHistory()
    except Exception:
        binance = None
    # OKX history (more reliable from Iran)
    try:
        okx = OKXHistory()
    except Exception:
        okx = None
    # Adaptive thresholds based on BTC regime
    long_thr = score_threshold
    short_thr = score_threshold
    if btc_state["state"] == "BULLISH":
        long_thr = score_threshold * 0.9  # easier to long
        short_thr = score_threshold * 1.2  # harder to short
    elif btc_state["state"] == "BEARISH":
        long_thr = score_threshold * 1.2  # harder to long
        short_thr = score_threshold * 0.9  # easier to short
    print(f"[dir] Adaptive thresholds: LONG>={long_thr:.1f}, SHORT>={short_thr:.1f}",
          flush=True)
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        return {"alerts_long": [], "alerts_short": [], "watchlist": []}
    try:
        mc_map = cp.get_market_caps_for_symbols(tickers["base"].tolist())
    except Exception:
        mc_map = {}
    if not mc_map:
        return {"alerts_long": [], "alerts_short": [], "watchlist": []}
    t2 = tickers.copy()
    t2["market_cap_usd"] = t2["base"].map(mc_map).fillna(0.0)
    small = t2[
        (t2["market_cap_usd"] > 0)
        & (t2["market_cap_usd"] <= 20_000_000)
        & (t2["quote_volume_24h"] >= 500_000)
    ].sort_values("quote_volume_24h", ascending=False).head(max_symbols)
    universe = small["symbol"].tolist()
    print(f"[dir] Universe: {len(universe)} small caps", flush=True)
    cooldown = CooldownGuard(
        os.path.join(project_root, "data", "cooldown.json"),
        default_hours=2.0,
    )
    alerts_long = []
    alerts_short = []
    watchlist = []
    all_results = []
    for i, sym in enumerate(universe, 1):
        if i % 10 == 0 or i == len(universe):
            print(f"[dir] {i}/{len(universe)}", flush=True)
        try:
            pack = deep_analyze(sym, toobit, btc_state, btc_df_1h, okx, binance)
        except Exception as e:
            print(f"[dir] error {sym}: {e}")
            continue
        if not pack.get("pass2_ok"):
            time.sleep(0.1)
            continue
        long_s = pack["long_score"]
        short_s = pack["short_score"]
        direction = pack["max_direction"]
        max_s = pack["max_score"]
        # Decide with adaptive thresholds
        if (direction == "LONG" and long_s >= long_thr
                and cooldown.is_cool(f"toobit:{sym}:LONG")):
            decision = "APPROVED_LONG"
            alerts_long.append({"symbol": sym, "score": long_s, "pack": pack})
            cooldown.mark(f"toobit:{sym}:LONG")
        elif (direction == "SHORT" and short_s >= short_thr
                and cooldown.is_cool(f"toobit:{sym}:SHORT")):
            decision = "APPROVED_SHORT"
            alerts_short.append({"symbol": sym, "score": short_s, "pack": pack})
            cooldown.mark(f"toobit:{sym}:SHORT")
        elif max_s >= score_threshold * 0.8:
            decision = "WATCHLIST"
            watchlist.append({"symbol": sym, "score": max_s, "pack": pack,
                              "direction": direction})
        else:
            decision = "REJECTED"
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "long_score": round(long_s, 2),
            "short_score": round(short_s, 2),
            "direction": direction,
            "max_score": round(max_s, 2),
            "decision": decision,
            "last_price": pack.get("last_price", 0),
            "volume_24h_usd": pack.get("volume_24h_usd", 0),
            "rsi_1h": pack["tech_1h"].get("rsi_value", 50),
            "rvol_1h": pack["ind_1h"].get("rvol", 1.0),
            "momentum_3_pct": pack["ind_1h"].get("momentum_3_pct", 0),
            "momentum_6_pct": pack["ind_1h"].get("momentum_6_pct", 0),
            "atr_pct": pack["ind_1h"].get("atr_pct", 0),
            "higher_highs": pack["struct_1h"].get("higher_highs", False),
            "higher_lows": pack["struct_1h"].get("higher_lows", False),
            "big_wick_top": pack["candle_1h"].get("big_wick_top", False),
            "btc_corr_2d": pack["btc_corr"].get("btc_corr_2d", 0),
            "patterns": pack["patterns_1h"].get("patterns", []),
        }
        all_results.append(result)
        time.sleep(0.2)
    # Sort
    alerts_long.sort(key=lambda x: x["score"], reverse=True)
    alerts_short.sort(key=lambda x: x["score"], reverse=True)
    all_results.sort(key=lambda x: x["max_score"], reverse=True)
    print(f"[dir] LONG: {len(alerts_long)}, SHORT: {len(alerts_short)}, "
          f"WATCH: {len(watchlist)}", flush=True)
    # Telegram
    notifier = TelegramNotifier.from_env()
    if notifier:
        try:
            for a in alerts_long[:3]:
                notifier.send_alert(_format_for_tg(a, "LONG"))
                time.sleep(0.5)
            for a in alerts_short[:3]:
                notifier.send_alert(_format_for_tg(a, "SHORT"))
                time.sleep(0.5)
        except Exception as e:
            print(f"[dir] TG send failed: {e}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_state": btc_state["state"],
        "alerts_long": [{"symbol": a["symbol"], "score": a["score"]} for a in alerts_long],
        "alerts_short": [{"symbol": a["symbol"], "score": a["score"]} for a in alerts_short],
        "watchlist": [{"symbol": w["symbol"], "score": w["score"]} for w in watchlist],
        "all": all_results[:30],
    }


def _format_for_tg(a: Dict, direction: str) -> Dict:
    return {
        "timestamp": a.get("timestamp", ""),
        "symbol": a["symbol"],
        "score": a["score"],
        "market_cap_usd": 0,
        "quote_volume_24h": a.get("pack", {}).get("volume_24h_usd", 0),
        "rsi_value": a.get("pack", {}).get("tech_1h", {}).get("rsi_value", 50),
        "rsi_divergence": "none",
        "macd_hist": 0,
        "ema_alignment": "bullish" if direction == "LONG" else "bearish",
        "patterns": a.get("pack", {}).get("patterns_1h", {}).get("patterns", []),
        "social_score": 0,
        "liq_bias": 0,
    }


if __name__ == "__main__":
    res = run_direction_scan(project_root="..", max_symbols=50,
                              score_threshold=60.0)
    print(json.dumps({
        "btc": res.get("btc_state"),
        "alerts_long": res["alerts_long"][:10],
        "alerts_short": res["alerts_short"][:10],
        "watchlist": res["watchlist"][:10],
    }, indent=2))
