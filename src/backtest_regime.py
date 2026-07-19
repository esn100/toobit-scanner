"""
Regime-aware backtest.

Tests the regime_scanner against historical data. Includes:
  - Market regime detection at each snapshot
  - Sentiment-adjusted scores
  - Direction bias multipliers
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from coinpaprika import CoinPaprikaClient
from okx_history import OKXHistory
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
from regime_scanner import (
    compute_long_score, compute_short_score, detect_market_regime
)
from sentiment_v2 import build_sentiment_v2, SentimentSnapshot
from advanced_indicators import advanced_score_boost
from elliott_wave import detect_elliott_waves
from fibonacci import compute_fib_levels
from ichimoku import ichimoku_features


def get_small_caps(toobit, cp, limit=20):
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        return []
    try:
        mc_map = cp.get_market_caps_for_symbols(tickers["base"].tolist())
    except Exception:
        return []
    t = tickers.copy()
    t["mc"] = t["base"].map(mc_map).fillna(0.0)
    small = t[(t["mc"] > 0) & (t["mc"] <= 20_000_000)
              & (t["quote_volume_24h"] >= 500_000)]
    return small.sort_values("quote_volume_24h", ascending=False)[
        "symbol"].head(limit).tolist()


def snapshot(symbol, toobit, idx_1h, sentiment, okx, btc_df_1h):
    out = {"pass2_ok": False, "long_score": 0.0, "short_score": 0.0}
    try:
        df_1h = toobit.get_klines(symbol, "1h", 200)
    except Exception:
        return out
    if df_1h.empty or idx_1h + 1 > len(df_1h) or idx_1h < 30:
        return out
    sub_1h = df_1h.iloc[: idx_1h + 1].copy().reset_index(drop=True)
    if len(sub_1h) < 30:
        return out
    tech_1h = technical_analysis(sub_1h)
    ind_1h = {}
    ind_1h.update(vwap_features(sub_1h))
    ind_1h.update(atr_features(sub_1h))
    ind_1h.update(bollinger_features(sub_1h))
    ind_1h.update(relative_volume(sub_1h))
    ind_1h.update(momentum_features(sub_1h))
    struct_1h = structure_features(sub_1h)
    candle_1h = candle_quality_features(sub_1h)
    patterns_1h = detect_all_patterns(sub_1h)
    btc_corr = btc_correlation_features(sub_1h, btc_df_1h)
    # OKX history
    elliott = {"wave": "none", "score": 50.0, "details": {}}
    fib = {"levels": {}, "direction": "none", "current_price": 0,
           "closest_level": None, "distance_to_closest": 100.0}
    ichi = {"current_price": 0, "price_vs_cloud": "neutral"}
    try:
        df_hist = okx.get_history_for_toobit_symbol(symbol, "1H", 1440)
        if not df_hist.empty and len(df_hist) >= 60:
            cutoff = df_1h["open_time"].iloc[idx_1h]
            df_hist = df_hist[df_hist["open_time"] <= cutoff].tail(200)
            if len(df_hist) >= 60:
                elliott = detect_elliott_waves(df_hist, threshold=0.04)
                fib = compute_fib_levels(df_hist, lookback=60)
                ichi = ichimoku_features(df_hist)
    except Exception:
        pass
    pack = {
        "ind_1h": ind_1h, "tech_1h": tech_1h, "struct_1h": struct_1h,
        "candle_1h": candle_1h, "patterns_1h": patterns_1h,
        "btc_corr": btc_corr, "elliott": elliott, "fib": fib,
        "ichimoku": ichi,
    }
    out["pass2_ok"] = True
    long_s = compute_long_score(pack)
    short_s = compute_short_score(pack)
    long_boost = advanced_score_boost(pack, "LONG")
    short_boost = advanced_score_boost(pack, "SHORT")
    out["long_score"] = max(0, min(100, long_s + long_boost))
    out["short_score"] = max(0, min(100, short_s + short_boost))
    # Apply sentiment bias
    out["long_score_adj"] = out["long_score"] * sentiment.long_bias
    out["short_score_adj"] = out["short_score"] * sentiment.short_bias
    out["regime"] = detect_market_regime(
        sentiment, sentiment.btc_change_24h_pct
    )
    return out


def forward_12h(df_1h, idx_1h):
    if idx_1h + 1 + 12 > len(df_1h):
        return None
    entry = float(df_1h["close"].iloc[idx_1h])
    forward = df_1h.iloc[idx_1h + 1: idx_1h + 13]
    peak = float(((forward["high"].astype(float) - entry) / entry * 100.0).max())
    dd = float(((forward["low"].astype(float) - entry) / entry * 100.0).min())
    return {"peak": peak, "dd": dd}


def main():
    toobit = ToobitClient()
    cp = CoinPaprikaClient()
    okx = OKXHistory()
    universe = get_small_caps(toobit, cp, limit=20)
    print(f"[regime-bt] universe: {len(universe)}", flush=True)
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    # Build sentiment once (uses current state)
    sentiment = build_sentiment_v2(
        toobit, cache_dir=os.path.join("..", "data")
    )
    print(f"[regime-bt] Sentiment: {sentiment.aggregate:.1f} "
          f"({sentiment.risk_regime})", flush=True)
    print(f"[regime-bt] Bias: long={sentiment.long_bias:.2f}, "
          f"short={sentiment.short_bias:.2f}", flush=True)
    rows = []
    print(f"[regime-bt] scanning, 14 days, every 12h", flush=True)
    for s_idx, sym in enumerate(universe, 1):
        print(f"[regime-bt] ({s_idx}/{len(universe)}) {sym}", flush=True)
        try:
            df_1h = toobit.get_klines(sym, "1h", 14 * 24 + 50)
        except Exception:
            continue
        if df_1h.empty or len(df_1h) < 50:
            continue
        for idx in range(30, len(df_1h) - 12, 12):
            pack = snapshot(sym, toobit, idx, sentiment, okx, btc_df_1h)
            if not pack.get("pass2_ok"):
                continue
            out = forward_12h(df_1h, idx)
            if not out:
                continue
            rec = {
                "symbol": sym,
                "long_score_adj": pack["long_score_adj"],
                "short_score_adj": pack["short_score_adj"],
                "regime": pack["regime"],
                "peak": out["peak"],
                "dd": out["dd"],
                "pump_5": 1 if out["peak"] >= 5.0 else 0,
                "pump_10": 1 if out["peak"] >= 10.0 else 0,
                "dump_5": 1 if out["dd"] <= -5.0 else 0,
                "dump_10": 1 if out["dd"] <= -10.0 else 0,
            }
            rows.append(rec)
        time.sleep(0.3)
    if not rows:
        print("[regime-bt] no data")
        return
    df = pd.DataFrame(rows)
    os.makedirs("backtest_results_regime", exist_ok=True)
    df.to_csv("backtest_results_regime/dataset.csv", index=False)
    print(f"[regime-bt] saved {len(df)} snapshots")
    # Distribution of regimes
    if "regime" in df.columns:
        print(f"[regime-bt] regime distribution: {df['regime'].value_counts().to_dict()}")
    print(f"[regime-bt] base: pump_5={df['pump_5'].mean():.3f}, "
          f"dump_5={df['dump_5'].mean():.3f}")
    # Best of long/short with adjusted scores
    df["best_score"] = df[["long_score_adj", "short_score_adj"]].max(axis=1)
    df["best_dir"] = np.where(
        df["long_score_adj"] >= df["short_score_adj"], "LONG", "SHORT"
    )
    df["success_5"] = np.where(
        df["best_dir"] == "LONG", df["pump_5"], df["dump_5"]
    )
    df["success_10"] = np.where(
        df["best_dir"] == "LONG", df["pump_10"], df["dump_10"]
    )
    print()
    print("[regime-bt] === REGIME-AWARE PRECISION ===")
    for thr in [30, 40, 50, 60, 70]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        n_long = (sub["best_dir"] == "LONG").sum()
        s5 = sub["success_5"].mean()
        s10 = sub["success_10"].mean()
        print(f"  best >= {thr:3d}: n={n:3d} (long={n_long})  "
              f"5%= {s5:.3f}  10%= {s10:.3f}")
    # By regime
    print()
    print("[regime-bt] === BY REGIME ===")
    if "regime" in df.columns:
        for reg in df["regime"].unique():
            sub_reg = df[df["regime"] == reg]
            if sub_reg.empty:
                continue
            n = len(sub_reg)
            s5 = sub_reg["success_5"].mean()
            n_long = (sub_reg["best_dir"] == "LONG").sum()
            print(f"  {reg}: n={n} (long={n_long})  5%= {s5:.3f}")
    summary = {
        "n_snapshots": int(len(df)),
        "base_pump_5": float(df["pump_5"].mean()),
        "base_dump_5": float(df["dump_5"].mean()),
        "regime_distribution": (df["regime"].value_counts().to_dict()
                                if "regime" in df.columns else {}),
    }
    for thr in [30, 40, 50, 60, 70]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        summary[f"best_{thr}"] = {
            "n": int(len(sub)),
            "long_count": int((sub["best_dir"] == "LONG").sum()),
            "success_5pct": float(sub["success_5"].mean()),
            "success_10pct": float(sub["success_10"].mean()),
        }
    with open("backtest_results_regime/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[regime-bt] summary saved")


if __name__ == "__main__":
    main()
