"""
Regime-aware direction scanner.

Integrates the 5-source sentiment module to dynamically bias signals
based on market regime:
  - BEARISH regime: only SHORT signals, higher threshold for LONG
  - BULLISH regime: only LONG signals, higher threshold for SHORT
  - SIDEWAYS: both directions, standard thresholds
  - HIGH_VOLATILITY: +10 to all thresholds

The signal selection also uses bias multipliers from sentiment to
favour the appropriate direction.
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
from elliott_wave import detect_elliott_waves
from fibonacci import compute_fib_levels
from ichimoku import ichimoku_features
from advanced_indicators import advanced_score_boost
from sentiment_v2 import build_sentiment_v2, SentimentSnapshot


def detect_market_regime(sentiment: SentimentSnapshot, btc_change_24h: float) -> str:
    """
    Determine the overall market regime from sentiment + BTC action.
    Returns: BULLISH / BEARISH / SIDEWAYS / VOLATILE
    """
    # Strong bearish: panic OR strong BTC drop
    if sentiment.risk_regime == "PANIC":
        return "BEARISH"
    if btc_change_24h <= -5.0:
        return "BEARISH"
    if sentiment.btc_drawdown_30d_pct <= -15:
        return "BEARISH"
    # Strong bullish
    if sentiment.risk_regime == "RISK_ON" and btc_change_24h >= 3.0:
        return "BULLISH"
    if sentiment.aggregate >= 70 and btc_change_24h >= 2.0:
        return "BULLISH"
    # Volatile
    if sentiment.btc_volatility_24h_pct >= 5.0:
        return "VOLATILE"
    if abs(btc_change_24h) >= 4.0:
        return "VOLATILE"
    return "SIDEWAYS"


def compute_long_score(pack: Dict) -> float:
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
    candle_1h = pack.get("candle_1h", {})
    btc_corr = pack.get("btc_corr", {})
    rvol = ind_1h.get("rvol", 1.0)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    rsi_1h = tech_1h.get("rsi_value", 50)
    bb_squeeze = ind_1h.get("bb_squeeze", False)
    volume_spike = ind_1h.get("volume_spike", False)
    score = 0.0
    if rvol >= 5.0: score += 35
    elif rvol >= 3.0: score += 25
    elif rvol >= 2.0: score += 15
    elif rvol < 0.5: score -= 10
    if volume_spike: score += 8
    if m3 > 5.0: score += 15
    elif m3 > 2.0: score += 10
    elif m3 > 0: score += 5
    if m1 > 1.0: score += 8
    elif m1 > 0: score += 4
    if mom_acc > 0: score += min(10, mom_acc * 5)
    if 30 <= rsi_1h <= 65: score += 10
    elif rsi_1h > 75: score -= 12
    if bb_squeeze: score += 5
    if candle_1h.get("big_wick_top") is False:
        score += 3
    if btc_corr.get("independent_mover"):
        score += 5
    return max(0, min(100, score))


def compute_short_score(pack: Dict) -> float:
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
    candle_1h = pack.get("candle_1h", {})
    btc_corr = pack.get("btc_corr", {})
    rvol = ind_1h.get("rvol", 1.0)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    rsi_1h = tech_1h.get("rsi_value", 50)
    higher_highs = struct_1h.get("higher_highs", False)
    higher_lows = struct_1h.get("higher_lows", False)
    big_wick = candle_1h.get("big_wick_top", False)
    score = 0.0
    if higher_highs: score += 15
    if higher_lows: score += 10
    if m3 < -5.0: score += 15
    elif m3 < -2.0: score += 10
    elif m3 < 0: score += 5
    if m1 < -1.0: score += 8
    elif m1 < 0: score += 4
    if mom_acc < 0: score += min(10, abs(mom_acc) * 5)
    if rsi_1h > 75: score += 15
    elif rsi_1h > 65: score += 8
    if rvol >= 3.0 and m1 < 0: score += 12
    elif rvol >= 2.0 and m1 < 0: score += 8
    if big_wick: score += 12
    btc_corr_2d = btc_corr.get("btc_corr_2d", 0)
    if btc_corr_2d > 0.6:
        score += 5
    return max(0, min(100, score))


def deep_analyze(symbol: str, toobit: ToobitClient,
                 btc_state: dict, btc_df_1h: pd.DataFrame,
                 okx: Optional[OKXHistory] = None) -> Dict:
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
    # OKX history
    elliott = {"wave": "none", "score": 50.0, "details": {}}
    fib = {"levels": {}, "direction": "none", "current_price": 0,
           "closest_level": None, "distance_to_closest": 100.0}
    ichi = {"current_price": 0, "price_vs_cloud": "neutral"}
    if okx is not None:
        try:
            df_hist = okx.get_history_for_toobit_symbol(symbol, "1H", 1440)
            if not df_hist.empty and len(df_hist) >= 60:
                elliott = detect_elliott_waves(df_hist, threshold=0.04)
                fib = compute_fib_levels(df_hist, lookback=60)
                ichi = ichimoku_features(df_hist)
        except Exception:
            pass
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
        out["volume_24h_usd"] = float(df_1h["quote_volume"].tail(24).sum())
    return out


def run_regime_scan(
    project_root: str = ".",
    max_symbols: int = 60,
    base_threshold: float = 50.0,
) -> Dict:
    """
    Run regime-aware scan with sentiment integration.
    """
    toobit = ToobitClient()
    cp = CoinPaprikaClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[regime] BTC state: {btc_state['state']}", flush=True)
    if btc_state["freeze"]:
        return {"alerts_long": [], "alerts_short": [], "watchlist": []}
    # Build sentiment (5 sources)
    sentiment = build_sentiment_v2(
        toobit,
        cryptopanic_token=os.environ.get("CRYPTOPANIC_API_KEY"),
        cache_dir=os.path.join(project_root, "data"),
    )
    btc_change_24h = sentiment.btc_change_24h_pct
    regime = detect_market_regime(sentiment, btc_change_24h)
    print(f"[regime] Sentiment: {sentiment.aggregate:.1f} "
          f"({sentiment.risk_regime})", flush=True)
    print(f"[regime] BTC 24h: {btc_change_24h:+.2f}%, 7d: {sentiment.btc_change_7d_pct:+.2f}%, "
          f"drawdown 30d: {sentiment.btc_drawdown_30d_pct:+.2f}%", flush=True)
    print(f"[regime] Volatility 24h: {sentiment.btc_volatility_24h_pct:.2f}%", flush=True)
    print(f"[regime] Detected regime: {regime}", flush=True)
    print(f"[regime] Bias: long={sentiment.long_bias:.2f}, "
          f"short={sentiment.short_bias:.2f}", flush=True)

    # Adjust thresholds based on regime
    long_thr = base_threshold
    short_thr = base_threshold
    vol_adj = 0.0
    if regime == "BEARISH":
        long_thr = base_threshold * 1.5
        short_thr = base_threshold * 0.8
        print(f"[regime] BEARISH: long_thr={long_thr:.1f}, short_thr={short_thr:.1f}",
              flush=True)
    elif regime == "BULLISH":
        long_thr = base_threshold * 0.8
        short_thr = base_threshold * 1.5
        print(f"[regime] BULLISH: long_thr={long_thr:.1f}, short_thr={short_thr:.1f}",
              flush=True)
    elif regime == "VOLATILE":
        vol_adj = 10.0
        long_thr = base_threshold + vol_adj
        short_thr = base_threshold + vol_adj
        print(f"[regime] VOLATILE: thresholds raised by {vol_adj}", flush=True)
    else:  # SIDEWAYS
        long_thr = base_threshold
        short_thr = base_threshold
        print(f"[regime] SIDEWAYS: standard thresholds", flush=True)

    # Hard block in extreme regimes
    block_long = (regime == "BEARISH" and sentiment.risk_regime == "PANIC")
    block_short = (regime == "BULLISH" and sentiment.risk_regime == "RISK_ON"
                   and sentiment.aggregate >= 75)
    if block_long:
        print("[regime] HARD BLOCK: LONG signals (PANIC regime)", flush=True)
    if block_short:
        print("[regime] HARD BLOCK: SHORT signals (strong RISK_ON)", flush=True)

    # Continue with scanning
    try:
        okx = OKXHistory()
    except Exception:
        okx = None
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
    print(f"[regime] Universe: {len(universe)} small caps", flush=True)
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
            print(f"[regime] {i}/{len(universe)}", flush=True)
        try:
            pack = deep_analyze(sym, toobit, btc_state, btc_df_1h, okx)
        except Exception as e:
            continue
        if not pack.get("pass2_ok"):
            time.sleep(0.1)
            continue
        long_s = pack["long_score"]
        short_s = pack["short_score"]
        # Apply sentiment bias multipliers
        long_s_adj = long_s * sentiment.long_bias
        short_s_adj = short_s * sentiment.short_bias
        direction = "LONG" if long_s_adj > short_s_adj else "SHORT"
        max_s = max(long_s_adj, short_s_adj)
        # Decide with regime-aware thresholds and hard blocks
        if direction == "LONG" and not block_long \
                and long_s_adj >= long_thr \
                and cooldown.is_cool(f"toobit:{sym}:LONG"):
            decision = "APPROVED_LONG"
            alerts_long.append({"symbol": sym, "score": long_s_adj,
                                "raw_score": long_s, "pack": pack})
            cooldown.mark(f"toobit:{sym}:LONG")
        elif direction == "SHORT" and not block_short \
                and short_s_adj >= short_thr \
                and cooldown.is_cool(f"toobit:{sym}:SHORT"):
            decision = "APPROVED_SHORT"
            alerts_short.append({"symbol": sym, "score": short_s_adj,
                                 "raw_score": short_s, "pack": pack})
            cooldown.mark(f"toobit:{sym}:SHORT")
        elif max_s >= base_threshold * 0.8:
            decision = "WATCHLIST"
            watchlist.append({"symbol": sym, "score": max_s,
                              "direction": direction, "pack": pack})
        else:
            decision = "REJECTED"
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "long_score_raw": round(long_s, 2),
            "short_score_raw": round(short_s, 2),
            "long_score_adj": round(long_s_adj, 2),
            "short_score_adj": round(short_s_adj, 2),
            "direction": direction,
            "decision": decision,
            "regime": regime,
            "last_price": pack.get("last_price", 0),
            "volume_24h_usd": pack.get("volume_24h_usd", 0),
            "rsi_1h": pack["tech_1h"].get("rsi_value", 50),
            "rvol_1h": pack["ind_1h"].get("rvol", 1.0),
            "momentum_3_pct": pack["ind_1h"].get("momentum_3_pct", 0),
            "higher_highs": pack["struct_1h"].get("higher_highs", False),
            "big_wick_top": pack["candle_1h"].get("big_wick_top", False),
            "btc_corr_2d": pack["btc_corr"].get("btc_corr_2d", 0),
        }
        all_results.append(result)
        time.sleep(0.2)
    alerts_long.sort(key=lambda x: x["score"], reverse=True)
    alerts_short.sort(key=lambda x: x["score"], reverse=True)
    all_results.sort(key=lambda x: max(x["long_score_adj"], x["short_score_adj"]),
                     reverse=True)
    print(f"[regime] APPROVED_LONG: {len(alerts_long)}, "
          f"APPROVED_SHORT: {len(alerts_short)}, "
          f"WATCHLIST: {len(watchlist)}", flush=True)
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
            print(f"[regime] TG send failed: {e}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_state": btc_state["state"],
        "regime": regime,
        "sentiment_aggregate": sentiment.aggregate,
        "risk_regime": sentiment.risk_regime,
        "long_bias": sentiment.long_bias,
        "short_bias": sentiment.short_bias,
        "block_long": block_long,
        "block_short": block_short,
        "long_threshold": long_thr,
        "short_threshold": short_thr,
        "alerts_long": [{"symbol": a["symbol"], "score": round(a["score"], 1),
                        "raw": round(a["raw_score"], 1)}
                       for a in alerts_long],
        "alerts_short": [{"symbol": a["symbol"], "score": round(a["score"], 1),
                         "raw": round(a["raw_score"], 1)}
                        for a in alerts_short],
        "watchlist": [{"symbol": w["symbol"], "score": round(w["score"], 1),
                       "direction": w["direction"]} for w in watchlist],
        "all": all_results[:30],
        "sentiment": {
            "fear_greed": sentiment.fear_greed,
            "fear_greed_label": sentiment.fear_greed_label,
            "btc_change_24h": round(sentiment.btc_change_24h_pct, 2),
            "btc_change_7d": round(sentiment.btc_change_7d_pct, 2),
            "btc_drawdown_30d": round(sentiment.btc_drawdown_30d_pct, 2),
            "volatility_24h": round(sentiment.btc_volatility_24h_pct, 2),
            "stablecoin_7d": round(sentiment.stablecoin_supply_change_7d_pct, 2),
            "btc_dominance": round(sentiment.btc_dominance, 2),
            "news_sentiment": round(sentiment.news_sentiment, 1),
        },
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
    res = run_regime_scan(project_root="..", max_symbols=50,
                          base_threshold=50.0)
    print(json.dumps({
        "regime": res.get("regime"),
        "risk_regime": res.get("risk_regime"),
        "sentiment_aggregate": res.get("sentiment_aggregate"),
        "long_threshold": res.get("long_threshold"),
        "short_threshold": res.get("short_threshold"),
        "block_long": res.get("block_long"),
        "block_short": res.get("block_short"),
        "alerts_long": res["alerts_long"][:10],
        "alerts_short": res["alerts_short"][:10],
        "watchlist": res["watchlist"][:10],
        "sentiment": res.get("sentiment"),
    }, indent=2))
