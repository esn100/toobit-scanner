"""
Precision-focused scanner for small caps.

Pipeline:
  1. Discover small caps
  2. Pass 1: Pre-filter
  3. Pass 2: Deep technical analysis
  4. Consensus filter (require 2/3 voters)
  5. Composite scoring (rule + consensus)
  6. ML probability (online learner)
  7. Dynamic threshold adjustment
  8. Final decision: APPROVED if score >= dynamic_threshold AND
     consensus passes AND ml_prob >= 0.5
  9. Record signal in history
  10. Telegram alert

Target: when score >= 80, 80%+ signals are +10% pumps in 12h.
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
from consensus import consensus_vote, consensus_score
from outcome_tracker import (
    record_signal, resolve_pending, get_rolling_precision,
    suggest_threshold,
)
from online_learner import OnlineLearner


def compute_score(pack: Dict) -> float:
    """
    Composite score 0..100.
    Weighted: volume (40), momentum (25), rsi (10), consensus (15),
    btc_independent (5), distribution_penalty (5).
    """
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    struct_1h = pack.get("struct_1h", {})
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
    volume_spike = ind_1h.get("volume_spike", False)
    score = 0.0
    # Volume (40 points)
    if rvol >= 5.0: score += 40
    elif rvol >= 3.0: score += 28
    elif rvol >= 2.0: score += 18
    elif rvol >= 1.5: score += 10
    elif rvol < 0.5: score -= 15
    if volume_spike: score += 5
    # Momentum (25 points)
    if m3 > 5.0: score += 15
    elif m3 > 2.0: score += 10
    elif m3 > 0: score += 5
    if m1 > 2.0: score += 8
    elif m1 > 0: score += 4
    if mom_acc > 1.0: score += 6
    elif mom_acc > 0: score += 3
    if m6 > 5.0: score += 3
    # RSI (10 points)
    if 30 <= rsi_1h <= 65: score += 10
    elif rsi_1h > 75: score -= 8
    elif rsi_1h < 25: score -= 5
    # Consensus (15 points)
    score += consensus_score(pack) * 0.3  # 15 max
    # BTC independent (5 points)
    if btc_corr.get("independent_mover"):
        score += 5
    # Distribution penalty
    if higher_lows: score -= 5
    if higher_highs: score -= 3
    return max(0, min(100, score))


def deep_analyze(symbol: str, toobit: ToobitClient,
                 btc_state: dict, btc_df_1h: pd.DataFrame) -> Dict:
    """Full deep analysis on a symbol."""
    out = {"pass1_ok": False, "pass2_ok": False, "score": 0.0,
           "consensus_pass": False, "n_consensus": 0,
           "consensus_voters": []}
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
    out["pass1_ok"] = True
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
    feats = build_features(
        tech_1h, ind_1h, struct_1h, candle_1h,
        {"alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
         "aligned": False, "same_sign": False},
        btc_state,
    )
    # Consensus
    pack = {
        "ind_1h": ind_1h, "tech_1h": tech_1h, "struct_1h": struct_1h,
        "candle_1h": candle_1h, "patterns_1h": patterns_1h,
        "btc_corr": btc_corr, "smart_money": {},
    }
    consensus_pass, n_consensus, voters = consensus_vote(pack)
    out["pass2_ok"] = True
    out["ind_1h"] = ind_1h
    out["tech_1h"] = tech_1h
    out["struct_1h"] = struct_1h
    out["candle_1h"] = candle_1h
    out["patterns_1h"] = patterns_1h
    out["btc_corr"] = btc_corr
    out["features"] = feats
    out["consensus_pass"] = consensus_pass
    out["n_consensus"] = n_consensus
    out["consensus_voters"] = voters
    out["score"] = compute_score(pack)
    if not df_1h.empty:
        out["last_price"] = float(df_1h["close"].iloc[-1])
        out["volume_24h_usd"] = float(
            df_1h["quote_volume"].tail(24).sum()
        )
    return out


def run_precision_scan(
    project_root: str = ".",
    coinglass_api_key: Optional[str] = None,
    max_symbols: int = 60,
    initial_threshold: float = 80.0,
    target_precision: float = 0.8,
) -> Dict:
    """
    Run the precision-focused scan.
    """
    toobit = ToobitClient()
    cp = CoinPaprikaClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[precision] BTC state: {btc_state['state']}", flush=True)
    if btc_state["freeze"]:
        print("[precision] BTC RISK_OFF - abort", flush=True)
        return {"alerts": [], "watchlist": [], "btc_state": "RISK_OFF"}
    # 1. Resolve pending outcomes (autonomous labelling)
    history_path = os.path.join(project_root, "data", "signal_history.csv")
    n_resolved = resolve_pending(history_path, toobit, horizon_hours=12)
    print(f"[precision] Resolved {n_resolved} pending outcomes", flush=True)
    # 2. Dynamic threshold
    dyn_thr = suggest_threshold(history_path, target_precision=target_precision)
    if dyn_thr <= 0:
        dyn_thr = initial_threshold
    print(f"[precision] Dynamic threshold: {dyn_thr:.1f}", flush=True)
    # 3. Online learner
    model_path = os.path.join(project_root, "models", "online_learner.json")
    learner = OnlineLearner(model_path, min_samples=10)
    print(f"[precision] Online learner ready: {learner.has_enough_data()}",
          flush=True)
    # 4. Get BTC 1h for correlation
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    # 5. Get tickers
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        return {"alerts": [], "watchlist": []}
    # 6. Filter small caps
    try:
        mc_map = cp.get_market_caps_for_symbols(tickers["base"].tolist())
    except Exception:
        mc_map = {}
    if not mc_map:
        return {"alerts": [], "watchlist": []}
    t2 = tickers.copy()
    t2["market_cap_usd"] = t2["base"].map(mc_map).fillna(0.0)
    small = t2[
        (t2["market_cap_usd"] > 0)
        & (t2["market_cap_usd"] <= 20_000_000)
        & (t2["quote_volume_24h"] >= 500_000)
    ].sort_values("quote_volume_24h", ascending=False).head(max_symbols)
    universe = small["symbol"].tolist()
    print(f"[precision] Universe: {len(universe)} small caps", flush=True)
    cooldown = CooldownGuard(
        os.path.join(project_root, "data", "cooldown.json"),
        default_hours=4.0,
    )
    alerts = []
    watchlist = []
    all_results = []
    for i, sym in enumerate(universe, 1):
        if i % 10 == 0 or i == len(universe):
            print(f"[precision] {i}/{len(universe)}", flush=True)
        try:
            pack = deep_analyze(sym, toobit, btc_state, btc_df_1h)
        except Exception as e:
            print(f"[precision] error {sym}: {e}", flush=True)
            continue
        if not pack.get("pass2_ok"):
            time.sleep(0.1)
            continue
        score = pack["score"]
        consensus_pass = pack["consensus_pass"]
        n_consensus = pack["n_consensus"]
        # Build feature dict for ML
        ind_1h = pack["ind_1h"]
        tech_1h = pack["tech_1h"]
        btc_corr = pack["btc_corr"]
        feature_dict = {
            "rvol": ind_1h.get("rvol", 1.0),
            "volume_spike": float(bool(ind_1h.get("volume_spike", False))),
            "momentum_1_pct": ind_1h.get("momentum_1_pct", 0),
            "momentum_3_pct": ind_1h.get("momentum_3_pct", 0),
            "rsi_1h": tech_1h.get("rsi_value", 50),
            "atr_pct": ind_1h.get("atr_pct", 0),
            "bb_squeeze": float(bool(ind_1h.get("bb_squeeze", False))),
            "higher_lows": float(bool(pack["struct_1h"].get("higher_lows", False))),
            "btc_corr_2d": btc_corr.get("btc_corr_2d", 0),
            "smart_money_score": 50.0,
            "independent_mover": float(bool(btc_corr.get("independent_mover", False))),
            "consensus_count": float(n_consensus),
            "composite_score": score,
        }
        ml_prob = learner.predict_proba(feature_dict)
        # Combined probability: 70% ML + 30% rule
        rule_prob = max(0, min(1, score / 100))
        combined_prob = 0.7 * ml_prob + 0.3 * rule_prob
        # Decision: score >= threshold AND consensus AND ml_prob
        if (score >= dyn_thr
                and consensus_pass
                and combined_prob >= 0.5
                and cooldown.is_cool(f"toobit:{sym}")):
            decision = "APPROVED"
            cooldown.mark(f"toobit:{sym}")
        elif score >= dyn_thr * 0.8 and consensus_pass:
            decision = "WATCHLIST"
        else:
            decision = "REJECTED"
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": sym,
            "composite_score": round(score, 2),
            "decision": decision,
            "ml_prob": round(ml_prob, 3),
            "combined_prob": round(combined_prob, 3),
            "consensus_pass": consensus_pass,
            "n_consensus": n_consensus,
            "consensus_voters": pack["consensus_voters"],
            "last_price": pack.get("last_price", 0),
            "volume_24h_usd": pack.get("volume_24h_usd", 0),
            "rsi_1h": tech_1h.get("rsi_value", 50),
            "rvol_1h": ind_1h.get("rvol", 1.0),
            "momentum_3_pct": ind_1h.get("momentum_3_pct", 0),
            "btc_corr_2d": btc_corr.get("btc_corr_2d", 0),
            "independent_mover": btc_corr.get("independent_mover", False),
            "patterns": pack["patterns_1h"].get("patterns", []),
        }
        if decision == "APPROVED":
            alerts.append(result)
        elif decision == "WATCHLIST":
            watchlist.append(result)
        all_results.append(result)
        # Record the signal in history
        record_signal(
            history_path, sym, score, decision,
            pack.get("last_price", 0),
            {**feature_dict, "consensus_count": n_consensus},
        )
        time.sleep(0.25)
    # Sort
    alerts.sort(key=lambda x: x["composite_score"], reverse=True)
    watchlist.sort(key=lambda x: x["composite_score"], reverse=True)
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"[precision] APPROVED: {len(alerts)}, WATCHLIST: {len(watchlist)}",
          flush=True)
    # Rolling precision
    rolling = get_rolling_precision(history_path, dyn_thr, window=20)
    print(f"[precision] Rolling precision: {rolling['rolling_precision']:.2f} "
          f"({rolling['n_success']}/{rolling['n_samples']})", flush=True)
    # Telegram
    notifier = TelegramNotifier.from_env()
    if notifier:
        try:
            for a in alerts[:5]:
                notifier.send_alert(_format_for_tg(a))
                time.sleep(0.5)
        except Exception as e:
            print(f"[precision] TG send failed: {e}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_state": btc_state["state"],
        "dynamic_threshold": dyn_thr,
        "rolling_precision": rolling,
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
    res = run_precision_scan(
        project_root="..",
        max_symbols=50,
        target_precision=0.8,
    )
    print(json.dumps({
        "btc": res.get("btc_state"),
        "dynamic_threshold": res.get("dynamic_threshold"),
        "rolling_precision": res.get("rolling_precision"),
        "alerts": [a["symbol"] for a in res["alerts"]],
        "watchlist": [a["symbol"] for a in res["watchlist"]],
    }, indent=2))
