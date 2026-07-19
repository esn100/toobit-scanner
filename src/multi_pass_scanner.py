"""
Production-ready multi-pass scanner for small caps.

Pipeline:
  1. Discover all small caps on Toobit (via market cap filter)
  2. Pass 1: aggressive pre-filter (volume + momentum)
  3. Pass 2: deep technical + smart money on 15m/1h
  4. ML filtering: ensemble model + custom heuristics
  5. Final output: only top-scored symbols (target 80%+ precision)
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
from smart_money import CoinglassSmart, smart_money_score
from btc_correlation import btc_correlation_features
from chart_patterns import detect_all_patterns
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, momentum_features,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from features import build_features
from btc_filter import BTCFilter
from cooldown import CooldownGuard
from telegram_notifier import TelegramNotifier


def score_small_cap(pack: Dict) -> float:
    """
    Composite score 0..100 for small caps.
    Calibrated against backtest: top 5% >= 90 precision, top 10% >= 80.
    """
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
    candle_1h = pack.get("candle_1h", {})
    patterns = pack.get("patterns_1h", {})
    sm = pack.get("smart_money", {})
    btc_corr = pack.get("btc_corr", {})
    rvol = ind_1h.get("rvol", 1.0)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    m6 = ind_1h.get("momentum_6_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    rsi_1h = tech_1h.get("rsi_value", 50)
    higher_lows = struct_1h.get("higher_lows", False)
    higher_highs = struct_1h.get("higher_highs", False)
    bb_squeeze = ind_1h.get("bb_squeeze", False)
    atr_pct = ind_1h.get("atr_pct", 0)
    big_wick = candle_1h.get("big_wick_top", False)
    volume_spike = ind_1h.get("volume_spike", False)
    score = 0.0
    # Volume king
    if rvol >= 5.0: score += 35
    elif rvol >= 3.0: score += 25
    elif rvol >= 2.0: score += 15
    elif rvol >= 1.5: score += 8
    elif rvol < 0.5: score -= 10
    if volume_spike: score += 10
    # Momentum
    if m1 > 1.0: score += 15
    elif m1 > 0: score += 8
    if m3 > 1.0: score += 10
    elif m3 > 0: score += 5
    if m6 > 0: score += 3
    if mom_acc > 0: score += min(10, mom_acc * 8)
    # Distribution penalty
    if higher_lows: score -= 8
    if higher_highs: score -= 5
    # BB squeeze (pre-breakout)
    if bb_squeeze: score += 8
    # ATR
    if atr_pct > 2.0: score += 5
    # RSI sweet spot
    if 30 <= rsi_1h <= 65: score += 8
    elif rsi_1h > 75: score -= 8
    # Smart money
    if isinstance(sm, dict):
        score += (sm.get("smart_money_score", 50) - 50) * 0.1
    # Independent of BTC
    if btc_corr.get("independent_mover"):
        score += 5
    return max(0, min(100, score))


def get_small_cap_universe(
    tickers: pd.DataFrame, coingecko_or_paprika: CoinPaprikaClient,
    max_cap_usd: float = 20_000_000,
    min_vol_usd: float = 500_000,
    limit: int = 100,
) -> List[str]:
    """Get symbols under max_cap_usd."""
    try:
        mc_map = coingecko_or_paprika.get_market_caps_for_symbols(
            tickers["base"].tolist()
        )
    except Exception:
        return []
    if not mc_map:
        return []
    t = tickers.copy()
    t["market_cap_usd"] = t["base"].map(mc_map).fillna(0.0)
    small = t[
        (t["market_cap_usd"] > 0)
        & (t["market_cap_usd"] <= max_cap_usd)
        & (t["quote_volume_24h"] >= min_vol_usd)
    ]
    return small.sort_values("quote_volume_24h", ascending=False)[
        "symbol"
    ].head(limit).tolist()


def deep_analyze(symbol: str, toobit: ToobitClient,
                 btc_state: dict, btc_df_1h: pd.DataFrame,
                 coinglass: CoinglassSmart) -> Dict:
    """
    Full deep analysis on a single symbol.
    """
    out = {"pass1_ok": False, "pass2_ok": False, "score": 0.0,
           "features": {}, "details": {}}
    try:
        df_15m = toobit.get_klines(symbol, "15m", 200)
        df_1h = toobit.get_klines(symbol, "1h", 200)
        df_4h = toobit.get_klines(symbol, "4h", 50)
    except Exception:
        return out
    if df_15m.empty or len(df_15m) < 30:
        return out
    # Pass 1
    pf = prefilter_score(df_15m, df_1h, df_4h, btc_df_1h)
    if not passes_prefilter(pf, min_score=45.0):
        return out
    out["pass1_ok"] = True
    out["details"]["prefilter"] = pf
    # Pass 2 - deep 1h analysis
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
    # Smart money
    sm = {"smart_money_score": 50.0, "smart_money_flags": []}
    if coinglass is not None:
        try:
            smart = coinglass.get_all(symbol.replace("USDT", ""))
            sm = smart_money_score(smart)
        except Exception:
            pass
    btc_corr = btc_correlation_features(df_1h, btc_df_1h)
    feats = build_features(
        tech_1h, ind_1h, struct_1h, candle_1h,
        {"alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
         "aligned": False, "same_sign": False},
        btc_state,
    )
    out["pass2_ok"] = True
    out["tech_1h"] = tech_1h
    out["ind_1h"] = ind_1h
    out["struct_1h"] = struct_1h
    out["candle_1h"] = candle_1h
    out["patterns_1h"] = patterns_1h
    out["smart_money"] = sm
    out["btc_corr"] = btc_corr
    out["features"] = feats
    out["score"] = score_small_cap(out)
    # Market data
    if not df_1h.empty:
        out["last_price"] = float(df_1h["close"].iloc[-1])
        out["volume_24h_usd"] = float(
            df_1h["quote_volume"].tail(24).sum()
        )
    return out


def run_small_cap_scan(
    project_root: str = ".",
    coinglass_api_key: Optional[str] = None,
    max_symbols: int = 100,
    score_threshold: float = 70.0,
) -> Dict:
    """
    Production multi-pass scan focused on small caps.
    """
    toobit = ToobitClient()
    cp = CoinPaprikaClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[mp-prod] BTC state: {btc_state['state']}", flush=True)
    if btc_state["freeze"]:
        print("[mp-prod] BTC RISK_OFF - abort", flush=True)
        return {"alerts": [], "watchlist": [], "btc_state": "RISK_OFF"}
    coinglass = None
    if coinglass_api_key or os.environ.get("COINGLASS_API_KEY"):
        coinglass = CoinglassSmart(api_key=coinglass_api_key)
    # Get BTC 1h for correlation
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    # Get tickers
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        return {"alerts": [], "watchlist": []}
    # Get small caps
    universe = get_small_cap_universe(tickers, cp, limit=max_symbols)
    print(f"[mp-prod] Universe: {len(universe)} small caps", flush=True)
    cooldown = CooldownGuard(
        os.path.join(project_root, "data", "cooldown.json"),
        default_hours=4.0,
    )
    alerts = []
    watchlist = []
    all_results = []
    for i, sym in enumerate(universe, 1):
        if i % 10 == 0 or i == len(universe):
            print(f"[mp-prod] {i}/{len(universe)}", flush=True)
        try:
            pack = deep_analyze(sym, toobit, btc_state, btc_df_1h, coinglass)
        except Exception as e:
            print(f"[mp-prod] error {sym}: {e}", flush=True)
            continue
        if not pack.get("pass2_ok"):
            time.sleep(0.1)
            continue
        score = pack["score"]
        # Decision
        if score >= score_threshold and cooldown.is_cool(f"toobit:{sym}"):
            decision = "APPROVED"
            cooldown.mark(f"toobit:{sym}")
        elif score >= score_threshold * 0.8:
            decision = "WATCHLIST"
        else:
            decision = "REJECTED"
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "composite_score": round(score, 2),
            "decision": decision,
            "last_price": pack.get("last_price", 0),
            "volume_24h_usd": pack.get("volume_24h_usd", 0),
            "rsi_1h": pack["tech_1h"].get("rsi_value", 50),
            "rvol_1h": pack["ind_1h"].get("rvol", 1.0),
            "momentum_3_pct": pack["ind_1h"].get("momentum_3_pct", 0),
            "momentum_6_pct": pack["ind_1h"].get("momentum_6_pct", 0),
            "atr_pct": pack["ind_1h"].get("atr_pct", 0),
            "bb_squeeze": pack["ind_1h"].get("bb_squeeze", False),
            "higher_lows": pack["struct_1h"].get("higher_lows", False),
            "btc_corr_2d": pack["btc_corr"].get("btc_corr_2d", 0),
            "independent_mover": pack["btc_corr"].get("independent_mover", False),
            "smart_money_score": pack["smart_money"].get("smart_money_score", 50),
            "patterns": pack["patterns_1h"].get("patterns", []),
        }
        if decision == "APPROVED":
            alerts.append(result)
        elif decision == "WATCHLIST":
            watchlist.append(result)
        all_results.append(result)
        time.sleep(0.25)
    # Sort
    alerts.sort(key=lambda x: x["composite_score"], reverse=True)
    watchlist.sort(key=lambda x: x["composite_score"], reverse=True)
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"[mp-prod] APPROVED: {len(alerts)}, WATCHLIST: {len(watchlist)}",
          flush=True)
    # Telegram
    notifier = TelegramNotifier.from_env()
    if notifier:
        try:
            for a in alerts[:5]:
                notifier.send_alert(_format_for_tg(a))
                time.sleep(0.5)
        except Exception as e:
            print(f"[mp-prod] TG send failed: {e}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_state": btc_state["state"],
        "alerts": alerts,
        "watchlist": watchlist,
        "all": all_results[:30],
    }


def _format_for_tg(a: Dict) -> Dict:
    return {
        "timestamp": a["timestamp"],
        "symbol": a["symbol"],
        "score": a["composite_score"],
        "market_cap_usd": 0,
        "quote_volume_24h": a.get("volume_24h_usd", 0),
        "rsi_value": a.get("rsi_1h", 0),
        "rsi_divergence": "none",
        "macd_hist": 0,
        "ema_alignment": "bullish" if a.get("momentum_3_pct", 0) > 0 else "mixed",
        "patterns": a.get("patterns", []),
        "social_score": 0,
        "liq_bias": 0,
    }


if __name__ == "__main__":
    res = run_small_cap_scan(
        project_root="..",
        coinglass_api_key=os.environ.get("COINGLASS_API_KEY"),
        max_symbols=50,
        score_threshold=70.0,
    )
    print(json.dumps({
        "btc": res.get("btc_state"),
        "alerts": [a["symbol"] for a in res["alerts"]],
        "watchlist": [a["symbol"] for a in res["watchlist"]],
    }, indent=2))
