"""
Toobit Scanner — main pipeline.

For each scan:
  1. Pull Toobit tickers
  2. Filter by market cap (<=20M) via CoinGecko
  3. For each remaining symbol:
       - Fetch 4h klines
       - Compute technical analysis
       - Fetch LunarCrush / Google Trends / TradingView
       - Fetch whale data (CoinGlass)
       - Compute final score with current (ML-tuned) weights
  4. Persist to data/last_scan.json
  5. Send Telegram alerts for any score > threshold
  6. Update ML history
"""
from __future__ import annotations
import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
import yaml

from toobit_client import ToobitClient
from market_filter import CoinGeckoClient, filter_small_cap_symbols
from lunarcrush import LunarCrushClient
from google_trends import GoogleTrendsClient
from tradingview_scraper import TradingViewScraper
from whale_data import CoinGlassClient, get_whale_features
from technical import technical_analysis
from ml_weights import (
    WeightTuner, compute_final_score, social_score_from_metrics,
    whale_score_from_features, append_history, resolve_previous_labels,
)
from telegram_notifier import TelegramNotifier


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    df = toobit.get_klines(
        symbol,
        interval=cfg["scanner"]["timeframe"],
        limit=cfg["scanner"]["candles_limit"],
    )
    if df.empty or len(df) < 60:
        return {}

    tech = technical_analysis(df)
    clean_sym = symbol.replace("USDT", "")
    lc = lunar.get_coin_metrics(clean_sym)
    gt = trends.get_interest(clean_sym)
    tv_data = tv.get_idea_sentiment(clean_sym)
    whale = get_whale_features(clean_sym, coinglass)

    social = social_score_from_metrics(lc, gt, tv_data)
    whale_s = whale_score_from_features(whale)

    # Final composite (placeholder weights; real ones applied after ML step)
    final = compute_final_score(
        cfg["weights"],
        tech_score=tech["technical_score"],
        pattern_score=tech["pattern_score"],
        social_score=social,
        whale_score=whale_s,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "market_cap_usd": float(market_cap),
        "quote_volume_24h": float(quote_volume),
        "rsi_value": tech["rsi_value"],
        "rsi_divergence": tech["rsi_divergence"],
        "macd_value": tech["macd_value"],
        "macd_signal": tech["macd_signal"],
        "macd_hist": tech["macd_hist"],
        "macd_divergence": tech["macd_divergence"],
        "ema20": tech["ema20"],
        "ema50": tech["ema50"],
        "ema100": tech["ema100"],
        "ema200": tech["ema200"],
        "ema_alignment": tech["ema_alignment"],
        "patterns": tech["patterns"],
        "technical_score": tech["technical_score"],
        "pattern_score": tech["pattern_score"],
        "social_score": social,
        "whale_score": whale_s,
        "liq_bias": whale.get("liq_bias", 0.0),
        "long_liq_usd": whale.get("long_liq_usd", 0.0),
        "short_liq_usd": whale.get("short_liq_usd", 0.0),
        "price_change_pct_24h": float(lc.get("percent_change_24h") or 0),
        "social_subscore_galaxy": lc.get("galaxy_score", 0),
        "social_subscore_sentiment": lc.get("sentiment", 0),
        "trends_rising": bool(gt.get("rising", False)),
        "tv_buy_ratio": tv_data.get("buy_ratio", 0.5),
        "score": round(final, 2),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_scan(cfg: dict | None = None, verbose: bool = True) -> dict:
    if cfg is None:
        cfg = load_config()
    print(f"[scanner] Starting scan at {datetime.now(timezone.utc).isoformat()}", flush=True)

    # 1. Resolve previous labels (use latest 24h prices as a proxy)
    history_path = cfg["ml"]["history_path"]
    coingecko = CoinGeckoClient()
    cg_simple = coingecko._get("/simple/price", {
        "vs_currencies": "usd",
        "ids": "",  # we'll use market caps for major coins
    })
    # Resolve labels with 0 placeholders (we will populate 'next_price' from
    # the tickers in this run)
    current_prices: Dict[str, float] = {}

    # 2. Load data sources
    toobit = ToobitClient()
    lunar = LunarCrushClient()
    trends = GoogleTrendsClient()
    tv = TradingViewScraper()
    coinglass = CoinGlassClient() if cfg["data_sources"].get("coinglass", {}).get("enabled", True) else None

    # 3. Toobit tickers
    print("[scanner] Fetching Toobit tickers...", flush=True)
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        print("[scanner] No tickers returned from Toobit.", flush=True)
        return {"alerts": [], "results": []}
    print(f"[scanner] Got {len(tickers)} tickers from Toobit.", flush=True)

    # 4. Filter by market cap
    print("[scanner] Filtering by market cap via CoinGecko...", flush=True)
    filtered = filter_small_cap_symbols(
        tickers, coingecko,
        max_market_cap_usd=cfg["scanner"]["max_market_cap_usd"],
        min_volume_usd=cfg["scanner"]["min_24h_volume_usd"],
        max_symbols=cfg["scanner"]["max_symbols_per_run"],
    )
    if filtered.empty:
        print("[scanner] No symbols passed the filter.", flush=True)
        return {"alerts": [], "results": []}
    print(f"[scanner] {len(filtered)} symbols passed the filter.", flush=True)

    # Fill current_prices for label resolution
    for _, r in filtered.iterrows():
        current_prices[r["symbol"]] = r["last_price"]

    # 5. ML tuner
    tuner = WeightTuner(
        model_path=cfg["ml"]["model_path"],
        history_path=history_path,
        min_train=cfg["ml"]["min_history_to_train"],
    )
    # Resolve old labels
    resolved = resolve_previous_labels(history_path, current_prices)
    if verbose:
        print(f"[scanner] Resolved {resolved} previous labels.", flush=True)
    # Try to retrain
    if tuner.has_enough_data():
        ok = tuner.train()
        if verbose:
            print(f"[scanner] Retrain {'OK' if ok else 'failed'}.", flush=True)
    # Suggested weights
    weights = tuner.suggest_weights(cfg["weights"])
    print(f"[scanner] Active weights: {weights}", flush=True)

    # 6. Scan each symbol
    results: List[Dict] = []
    for _, row in filtered.iterrows():
        try:
            r = scan_symbol(
                row["symbol"], row["market_cap_usd"], row["quote_volume_24h"],
                cfg, trends, lunar, tv, coinglass,
            )
            if r:
                # Re-score with active (possibly ML-tuned) weights
                r["score"] = round(compute_final_score(
                    weights,
                    r["technical_score"],
                    r["pattern_score"],
                    r["social_score"],
                    r["whale_score"],
                ), 2)
                # Tag weights used (for history)
                r["w_technical"] = weights["technical"]
                r["w_pattern"] = weights["pattern"]
                r["w_social"] = weights["social"]
                r["w_whale"] = weights["whale"]
                # Persist to history
                append_history(history_path, {
                    "timestamp": r["timestamp"],
                    "symbol": r["symbol"],
                    "market_cap_usd": r["market_cap_usd"],
                    "technical": r["technical_score"],
                    "pattern": r["pattern_score"],
                    "social": r["social_score"],
                    "whale": r["whale_score"],
                    "rsi_value": r["rsi_value"],
                    "macd_hist": r["macd_hist"],
                    "social_score": r["social_subscore_galaxy"],
                    "liq_bias": r["liq_bias"],
                    "price_change_pct_24h": r["price_change_pct_24h"],
                    "score": r["score"],
                    "w_technical": r["w_technical"],
                    "w_pattern": r["w_pattern"],
                    "w_social": r["w_social"],
                    "w_whale": r["w_whale"],
                })
                results.append(r)
        except Exception as e:
            print(f"[scanner] Error scanning {row['symbol']}: {e}", flush=True)
        time.sleep(0.3)  # politeness

    # 7. Sort by score, pick alerts above threshold
    results.sort(key=lambda x: x["score"], reverse=True)
    threshold = cfg["alerting"]["notify_threshold"]
    max_alerts = cfg["alerting"]["max_alerts_per_run"]
    alerts = [r for r in results if r["score"] >= threshold][:max_alerts]

    # 8. Persist JSON report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": weights,
        "threshold": threshold,
        "scanned": len(results),
        "alerts_count": len(alerts),
        "results": results,
        "alerts": alerts,
    }
    out_path = os.path.join("reports", f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # Always update a 'latest' file for the dashboard
    with open("data/last_scan.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[scanner] Wrote report: {out_path}", flush=True)

    # 9. Telegram
    notifier = TelegramNotifier.from_env()
    if notifier:
        notifier.send_digest(len(results), alerts, weights)
        for a in alerts:
            notifier.send_alert(a)
            time.sleep(0.5)
    else:
        print("[scanner] Telegram not configured.", flush=True)

    return report


if __name__ == "__main__":
    run_scan()
