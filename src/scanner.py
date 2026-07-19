"""
PumpHunter-AI: Toobit scanner main pipeline.

For each scan:
  1. Evaluate BTC regime (btc_filter)
  2. Pull Toobit tickers
  3. Filter small caps via CoinPaprika (+ CoinGecko fallback)
  4. For each symbol:
       a. Validate data (data_quality)
       b. Pull 1h + 4h klines
       c. Compute indicators (RSI/MACD/EMA + VWAP/ATR/BB/Vol/Momentum)
       d. Compute market structure + candle quality
       e. Multi-timeframe alignment
       f. Build feature vector
       g. Rule-based score
       h. ML probability
       i. Final decision (APPROVED / WATCHLIST / REJECTED)
       j. Send Telegram alert for APPROVED + WATCHLIST borderline
  5. Evaluate previous outcomes (outcome.evaluate_pending_signals)
  6. Retrain ML if enough labeled data
  7. Adapt weights softly based on logistic regression
  8. Persist everything
"""
from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

from toobit_client import ToobitClient
from market_filter import CoinGeckoClient, filter_small_cap_symbols
from coinpaprika import CoinPaprikaClient
from lunarcrush import LunarCrushClient
from google_trends import GoogleTrendsClient
from tradingview_scraper import TradingViewScraper
from whale_data import CoinGlassClient, get_whale_features

from data_quality import validate_ohlcv
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, volume_continuity, momentum_features,
    mtf_alignment,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from features import build_features, feature_vector, FEATURE_NAMES
from scoring import rule_based_score
from btc_filter import BTCFilter
from cooldown import CooldownGuard
from ml_engine import (
    PumpHunterML, append_signal_history, soft_adapt_weights,
)
from outcome import evaluate_pending_signals
from decision import decide
from telegram_notifier import TelegramNotifier
from sentiment import build_sentiment
from chart_patterns import detect_all_patterns
from ensemble import EnsembleModel


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
def load_config(path: str | None = None) -> dict:
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        path = os.path.join(project_root, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_PATH_KEYS = {
    "ml": ["model_path", "history_path"],
    "cooldown": ["state_path"],
}


def _resolve_paths(cfg: dict, project_root: str) -> dict:
    for section, keys in _PATH_KEYS.items():
        sec = cfg.get(section, {})
        for k in keys:
            v = sec.get(k)
            if v and not os.path.isabs(v):
                sec[k] = os.path.join(project_root, v)
    return cfg


# ----------------------------------------------------------------------------
# Per-symbol scan
# ----------------------------------------------------------------------------
def scan_symbol(
    symbol: str,
    market_cap: float,
    quote_volume: float,
    cfg: dict,
    trends: GoogleTrendsClient,
    lunar: LunarCrushClient,
    tv: TradingViewScraper,
    coinglass: CoinGlassClient | None,
) -> Dict:
    toobit = ToobitClient()
    interval = cfg["scanner"]["timeframe"]
    limit = cfg["scanner"]["candles_limit"]

    # 4h
    df_4h = toobit.get_klines(symbol, interval, limit)
    # 1h for multi-timeframe alignment
    df_1h = toobit.get_klines(symbol, "1h", min(limit, 200))

    # Validate
    q4 = validate_ohlcv(df_4h, min_candles=60, interval_hours=4.0)
    if not q4.ok or q4.cleaned is None or q4.cleaned.empty:
        return {"_rejected": "data quality", "_quality": q4}
    df_4h = q4.cleaned

    # Technical (RSI/MACD/EMA/patterns)
    tech = technical_analysis(df_4h)

    # New indicators
    ind: Dict = {}
    ind.update(vwap_features(df_4h))
    ind.update(atr_features(df_4h))
    ind.update(bollinger_features(df_4h))
    ind.update(relative_volume(df_4h))
    ind.update(volume_continuity(df_4h))
    ind.update(momentum_features(df_4h))

    # Structure + candle quality
    struct = structure_features(df_4h)
    candle = candle_quality_features(df_4h)

    # Advanced chart patterns
    patterns = detect_all_patterns(df_4h)

    # Multi-timeframe (1h vs 4h)
    mtf = mtf_alignment(df_1h if not df_1h.empty else df_4h,
                        df_4h) if not df_1h.empty else {
        "fast_bias": 0.0, "slow_bias": 0.0,
        "aligned": False, "same_sign": False, "alignment_score": 50.0,
    }

    # Social + whale (as before)
    clean_sym = symbol.replace("USDT", "")
    lc = lunar.get_coin_metrics(clean_sym)
    gt = trends.get_interest(clean_sym)
    tv_data = tv.get_idea_sentiment(clean_sym)
    whale = get_whale_features(clean_sym, coinglass)

    from ml_weights import social_score_from_metrics, whale_score_from_features
    social = social_score_from_metrics(lc, gt, tv_data)
    whale_s = whale_score_from_features(whale)

    return {
        "symbol": symbol,
        "market_cap_usd": float(market_cap),
        "quote_volume_24h": float(quote_volume),
        "quality": q4,
        "technical": tech,
        "indicators": ind,
        "structure": struct,
        "candle": candle,
        "patterns": patterns,
        "mtf": mtf,
        "whale": whale,
        "social_score_raw": social,
        "whale_score_raw": whale_s,
        "lunarcrush": lc,
        "trends": gt,
        "tradingview": tv_data,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_scan(cfg: dict | None = None, verbose: bool = True) -> dict:
    if cfg is None:
        cfg = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = _resolve_paths(cfg, project_root)
    print(f"[scanner] Starting scan at "
          f"{datetime.now(timezone.utc).isoformat()}", flush=True)

    # ----------------- data sources -----------------
    cache_dir = os.path.join(project_root, "data")
    toobit = ToobitClient()
    coingecko = CoinGeckoClient(cache_dir=cache_dir)
    coingpaprika = CoinPaprikaClient()
    lunar = LunarCrushClient()
    trends = GoogleTrendsClient()
    tv = TradingViewScraper()
    coinglass = CoinGlassClient() if cfg["data_sources"].get(
        "coinglass", {}).get("enabled", True) else None

    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[scanner] BTC state: {btc_state['state']} "
          f"(modifier={btc_state['score_modifier']})", flush=True)

    # Multi-source sentiment snapshot (Fear&Greed, BTC dominance,
    # stablecoin supply, CryptoPanic news)
    sentiment_snap = None
    try:
        sentiment_snap = build_sentiment(
            symbol=None,
            cryptopanic_token=os.environ.get("CRYPTOPANIC_API_KEY"),
            cache_dir=os.path.join(project_root, "data"),
        )
        print(f"[scanner] Sentiment: {sentiment_snap.aggregate:.1f} "
              f"({sentiment_snap.risk_regime})", flush=True)
    except Exception as e:
        print(f"[scanner] Sentiment fetch failed: {e}", flush=True)

    if btc_state["freeze"]:
        print("[scanner] BTC RISK_OFF - aborting scan.", flush=True)
        return {"alerts": [], "results": [], "btc": btc_state,
                "watchlist": []}

    cooldown = CooldownGuard(
        cfg.get("cooldown", {}).get(
            "state_path", os.path.join(project_root, "data", "cooldown.json")
        ),
        default_hours=cfg.get("cooldown", {}).get("hours", 6.0),
    )

    # ----------------- ML bootstrap -----------------
    history_path = cfg["ml"]["history_path"]
    model_path = cfg["ml"]["model_path"]
    ml = PumpHunterML(model_path, min_train=cfg["ml"].get("min_history_to_train", 30))
    # Ensemble model (stacking) - tries to use if data is sufficient
    ens_path = model_path.replace("logreg_model.joblib",
                                  "ensemble_model.joblib")
    ens = EnsembleModel(ens_path, min_train=cfg["ml"].get("min_history_to_train", 30))

    # Evaluate previous pending signals first (if not first run)
    if cfg["ml"].get("evaluate_previous", True):
        try:
            n = evaluate_pending_signals(history_path, toobit,
                                        forward_bars=3, max_eval=30)
            if verbose:
                print(f"[scanner] Resolved {n} previous outcomes.", flush=True)
        except Exception as e:
            print(f"[scanner] Outcome evaluation failed: {e}", flush=True)

    # Train ensemble (preferred) if we have enough data
    use_ensemble = False
    if ens.has_enough_data(history_path):
        try:
            ok = ens.train(history_path)
            if ok:
                use_ensemble = True
                m = ens.ensemble_metrics
                print(f"[scanner] Ensemble retrained. F1={m.get('f1', 0):.3f} "
                      f"Acc={m.get('accuracy', 0):.3f}", flush=True)
        except Exception as e:
            print(f"[scanner] Ensemble train failed: {e}", flush=True)
    # Fallback to single logistic
    if not use_ensemble and ml.has_enough_data(history_path):
        try:
            ml.train(history_path)
            if verbose:
                print("[scanner] ML (single) model retrained.", flush=True)
        except Exception as e:
            print(f"[scanner] ML train failed: {e}", flush=True)

    # Soft weight adaptation
    base_weights = cfg.get("rule_weights", cfg.get("weights", {}))
    adapted_weights = soft_adapt_weights(base_weights, ml, blend=0.3)
    print(f"[scanner] Active rule weights: {adapted_weights}", flush=True)

    # ----------------- tickers + market-cap filter -----------------
    print("[scanner] Fetching Toobit tickers...", flush=True)
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        print("[scanner] No tickers.", flush=True)
        return {"alerts": [], "results": [], "watchlist": [],
                "btc": btc_state}
    print(f"[scanner] Got {len(tickers)} tickers from Toobit.", flush=True)

    print("[scanner] Filtering by market cap via CoinPaprika...", flush=True)
    filtered = pd.DataFrame()
    try:
        mc_map = coingpaprika.get_market_caps_for_symbols(
            tickers["base"].tolist()
        )
        if mc_map:
            t2 = tickers.copy()
            t2["market_cap_usd"] = t2["base"].map(mc_map).fillna(0.0)
            t2 = t2[
                (t2["market_cap_usd"] > 0)
                & (t2["market_cap_usd"] <= cfg["scanner"]["max_market_cap_usd"])
                & (t2["quote_volume_24h"] >= cfg["scanner"]["min_24h_volume_usd"])
            ]
            filtered = t2.sort_values(
                "quote_volume_24h", ascending=False
            ).head(cfg["scanner"]["max_symbols_per_run"]).reset_index(drop=True)
            print(f"[scanner] CoinPaprika produced {len(filtered)} matches.",
                  flush=True)
    except Exception as e:
        print(f"[scanner] CoinPaprika failed: {e}", flush=True)
    if filtered.empty:
        try:
            filtered = filter_small_cap_symbols(
                tickers, coingecko,
                max_market_cap_usd=cfg["scanner"]["max_market_cap_usd"],
                min_volume_usd=cfg["scanner"]["min_24h_volume_usd"],
                max_symbols=cfg["scanner"]["max_symbols_per_run"],
            )
        except Exception as e:
            print(f"[scanner] CoinGecko fallback failed: {e}", flush=True)
    if filtered.empty:
        print("[scanner] No symbols passed the filter.", flush=True)
        return {"alerts": [], "results": [], "watchlist": [],
                "btc": btc_state}

    # ----------------- per-symbol scan -----------------
    results: List[Dict] = []
    for _, row in filtered.iterrows():
        sym = row["symbol"]
        cooldown_key = f"toobit:{sym}"
        cooldown_ok = cooldown.is_cool(cooldown_key)
        try:
            pack = scan_symbol(
                sym, row["market_cap_usd"], row["quote_volume_24h"],
                cfg, trends, lunar, tv, coinglass,
            )
        except Exception as e:
            print(f"[scanner] Error scanning {sym}: {e}", flush=True)
            time.sleep(0.5)
            continue
        if not pack or pack.get("_rejected"):
            time.sleep(0.3)
            continue

        # Build features
        feats = build_features(
            pack["technical"],
            pack["indicators"],
            pack["structure"],
            pack["candle"],
            pack["mtf"],
            btc_state,
        )
        # Rule-based score
        rb = rule_based_score(
            pack["technical"],
            pack["indicators"],
            pack["structure"],
            pack["candle"],
            pack["mtf"],
            btc_state,
            adapted_weights,
        )
        composite = rb["composite_score"]
        # ML probability: prefer ensemble, fall back to single model
        if use_ensemble:
            ml_prob = ens.predict_proba(feats)
            ind_probs = ens.predict_individual(feats)
        else:
            ml_prob = ml.predict_proba(feats)
            ind_probs = {}
        # Boost composite with advanced chart patterns
        patterns = pack.get("patterns", {})
        if patterns.get("bullish_score", 0) > patterns.get("bearish_score", 0):
            composite += min(8.0, patterns["bullish_score"] * 0.1)
        elif patterns.get("bearish_score", 0) > 0:
            composite -= min(8.0, patterns["bearish_score"] * 0.1)
        # Sentiment modifier
        if sentiment_snap is not None:
            # -5 to +5 based on aggregate
            sent_mod = (sentiment_snap.aggregate - 50.0) / 10.0
            composite += max(-5.0, min(5.0, sent_mod))
        composite = max(0.0, min(100.0, composite))
        # Decision
        dec = decide(
            composite=composite,
            ml_prob=ml_prob,
            btc=btc_state,
            quality_ok=True,
            cooldown_ok=cooldown_ok,
        )
        ts = datetime.now(timezone.utc).isoformat()
        result = {
            "timestamp": ts,
            "symbol": sym,
            "market_cap_usd": pack["market_cap_usd"],
            "quote_volume_24h": pack["quote_volume_24h"],
            # technical breakdown
            "rsi_value": pack["technical"]["rsi_value"],
            "rsi_divergence": pack["technical"]["rsi_divergence"],
            "macd_hist": pack["technical"]["macd_hist"],
            "macd_divergence": pack["technical"]["macd_divergence"],
            "ema_alignment": pack["technical"]["ema_alignment"],
            "patterns": pack["technical"]["patterns"],
            # new indicators
            "rvol": pack["indicators"].get("rvol", 1.0),
            "volume_spike": pack["indicators"].get("volume_spike", False),
            "vwap_distance_pct": pack["indicators"].get("vwap_distance_pct", 0.0),
            "price_above_vwap": pack["indicators"].get("price_above_vwap", False),
            "atr_pct": pack["indicators"].get("atr_pct", 0.0),
            "atr_expanding": pack["indicators"].get("atr_expanding", False),
            "bb_squeeze": pack["indicators"].get("bb_squeeze", False),
            "bb_breakout_above": pack["indicators"].get("bb_breakout_above", False),
            "momentum_3_pct": pack["indicators"].get("momentum_3_pct", 0.0),
            "momentum_6_pct": pack["indicators"].get("momentum_6_pct", 0.0),
            "momentum_acceleration": pack["indicators"].get("momentum_acceleration", 0.0),
            # structure + candle
            "structure_score": pack["structure"]["structure_score"],
            "higher_highs": pack["structure"]["higher_highs"],
            "higher_lows": pack["structure"]["higher_lows"],
            "bos_up": pack["structure"]["bos_up"],
            "candle_strength": pack["candle"]["candle_strength"],
            "big_wick_top": pack["candle"]["big_wick_top"],
            "power_streak": pack["candle"]["power_streak"],
            "candle_score": pack["candle"]["candle_score"],
            # advanced chart patterns
            "advanced_patterns": pack["patterns"]["patterns"],
            "advanced_pattern_score": round(
                pack["patterns"]["bullish_score"]
                - pack["patterns"]["bearish_score"], 2
            ),
            "pattern_target": pack["patterns"].get("target", 0.0),
            # multi-tf
            "mtf_alignment": pack["mtf"]["alignment_score"],
            # social
            "social_score": pack["social_score_raw"],
            "whale_score": pack["whale_score_raw"],
            "liq_bias": pack["whale"].get("liq_bias", 0.0),
            # scoring
            "composite_score": round(composite, 2),
            "ml_prob": round(ml_prob, 3),
            "ml_individual": {k: round(v, 3) for k, v in ind_probs.items()},
            "decision": dec["decision"],
            "decision_reasons": dec["reasons"],
            "confidence": dec["confidence"],
            "sub_scores": rb["sub_scores"],
            "penalties": rb["penalties"],
            "btc_state": btc_state["state"],
            # sentiment
            "sentiment_aggregate": (
                sentiment_snap.aggregate if sentiment_snap else 50.0
            ),
            "sentiment_regime": (
                sentiment_snap.risk_regime if sentiment_snap else "NEUTRAL"
            ),
            "fear_greed": (
                sentiment_snap.fear_greed if sentiment_snap else 50.0
            ),
            # raw entry price for later outcome evaluation
            "entry_price": float(pack["quality"].cleaned["close"].iloc[-1]),
        }
        # Persist to history (with features)
        try:
            history_row = {
                "timestamp": ts,
                "symbol": sym,
                "label": np.nan,            # filled in by outcome evaluation
                "score": round(composite, 2),
                "ml_prob": round(ml_prob, 3),
                "outcome_score": np.nan,
                "composite_score": round(composite, 2),
                "entry_price": result["entry_price"],
            }
            history_row.update({k: feats.get(k, 0.0) for k in FEATURE_NAMES})
            append_signal_history(history_path, history_row)
        except Exception as e:
            print(f"[scanner] history append failed for {sym}: {e}",
                  flush=True)

        # Cooldown: only mark if APPROVED
        if dec["decision"] == "APPROVED":
            cooldown.mark(cooldown_key)
        results.append(result)
        time.sleep(0.4)  # politeness

    # ----------------- separate alerts vs watchlist -----------------
    alerts = [r for r in results if r["decision"] == "APPROVED"]
    watchlist = [r for r in results if r["decision"] == "WATCHLIST"]
    alerts.sort(key=lambda x: x["composite_score"], reverse=True)
    watchlist.sort(key=lambda x: x["composite_score"], reverse=True)
    threshold = cfg["alerting"]["notify_threshold"]
    # Only send Telegram for scores above the user-defined threshold AND
    # the decision APPROVED
    alerts = [a for a in alerts if a["composite_score"] >= threshold]
    alerts = alerts[: cfg["alerting"]["max_alerts_per_run"]]

    # ----------------- persist + telegram -----------------
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": adapted_weights,
        "threshold": threshold,
        "btc": btc_state,
        "sentiment": (
            sentiment_snap.__dict__ if sentiment_snap else None
        ),
        "ml_model": "ensemble" if use_ensemble else "logistic",
        "scanned": len(results),
        "alerts_count": len(alerts),
        "watchlist_count": len(watchlist),
        "alerts": alerts,
        "watchlist": watchlist,
        "results": results,
    }
    out_path = os.path.join(
        project_root, "reports",
        f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    latest_path = os.path.join(project_root, "data", "last_scan.json")
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[scanner] Wrote report: {out_path}", flush=True)

    notifier = TelegramNotifier.from_env()
    if notifier:
        try:
            notifier.send_digest(len(results), alerts, adapted_weights)
            for a in alerts:
                notifier.send_alert(_format_for_tg(a))
                time.sleep(0.5)
        except Exception as e:
            print(f"[scanner] Telegram send failed: {e}", flush=True)
    else:
        print("[scanner] Telegram not configured.", flush=True)

    return report


def _format_for_tg(a: Dict) -> Dict:
    """Format a result row for the existing Telegram notifier."""
    return {
        "timestamp": a["timestamp"],
        "symbol": a["symbol"],
        "score": a["composite_score"],
        "market_cap_usd": a["market_cap_usd"],
        "quote_volume_24h": a["quote_volume_24h"],
        "rsi_value": a["rsi_value"],
        "rsi_divergence": a["rsi_divergence"],
        "macd_hist": a["macd_hist"],
        "ema_alignment": a["ema_alignment"],
        "patterns": a.get("patterns", []),
        "social_score": a.get("social_score", 0),
        "liq_bias": a.get("liq_bias", 0.0),
        "ml_prob": a.get("ml_prob", 0.0),
        "decision": a["decision"],
    }


if __name__ == "__main__":
    run_scan()
