"""
Large-scale regime-aware backtest.
- 30 days, 30 symbols (top 15 Toobit small caps)
- Uses OKX history for 60+ days of Elliott/Fib/Ichimoku
- Stratifies by regime for honest precision estimates
- Identifies redundant/noisy features
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
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, momentum_features,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from chart_patterns import detect_all_patterns
from btc_filter import BTCFilter
from btc_correlation import btc_correlation_features
from regime_scanner import compute_long_score, compute_short_score
from sentiment_v2 import build_sentiment_v2
from advanced_indicators import advanced_score_boost
from elliott_wave import detect_elliott_waves
from fibonacci import compute_fib_levels
from ichimoku import ichimoku_features


def get_top_small_caps(toobit, cp, limit=30):
    tickers = toobit.get_24h_tickers()
    if tickers.empty:
        return []
    try:
        mc_map = cp.get_market_caps_for_symbols(tickers["base"].tolist())
    except Exception:
        return []
    t = tickers.copy()
    t["mc"] = t["base"].map(mc_map).fillna(0.0)
    small = t[(t["mc"] > 0) & (t["mc"] <= 30_000_000)
              & (t["quote_volume_24h"] >= 300_000)]
    return small.sort_values("quote_volume_24h", ascending=False)[
        "symbol"].head(limit).tolist()


def snapshot_full(symbol, toobit, idx_1h, sentiment, okx, btc_df_1h):
    """
    Build a complete feature snapshot for ML analysis.
    Returns: dict of features + sub-scores
    """
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
    # OKX history for Elliott/Fib/Ichimoku
    elliott = {"wave": "none", "score": 50.0, "details": {}}
    fib = {"levels": {}, "direction": "none", "current_price": 0,
           "closest_level": None, "distance_to_closest": 100.0}
    ichi = {"current_price": 0, "price_vs_cloud": "neutral",
            "tk_cross": "neutral", "cloud_color": "neutral"}
    okx_available = False
    if okx is not None:
        try:
            df_hist = okx.get_history_for_toobit_symbol(symbol, "1H", 1440)
            if not df_hist.empty and len(df_hist) >= 60:
                cutoff = df_1h["open_time"].iloc[idx_1h]
                df_hist = df_hist[df_hist["open_time"] <= cutoff].tail(200)
                if len(df_hist) >= 60:
                    elliott = detect_elliott_waves(df_hist, threshold=0.04)
                    fib = compute_fib_levels(df_hist, lookback=60)
                    ichi = ichimoku_features(df_hist)
                    okx_available = True
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
    out["okx_available"] = okx_available
    # Apply sentiment bias
    out["long_score_adj"] = out["long_score"] * sentiment.long_bias
    out["short_score_adj"] = out["short_score"] * sentiment.short_bias
    # Save all features for redundancy analysis
    out["features"] = {
        "rvol": ind_1h.get("rvol", 1.0),
        "volume_spike": float(bool(ind_1h.get("volume_spike", False))),
        "momentum_1_pct": ind_1h.get("momentum_1_pct", 0),
        "momentum_3_pct": ind_1h.get("momentum_3_pct", 0),
        "momentum_6_pct": ind_1h.get("momentum_6_pct", 0),
        "momentum_acceleration": ind_1h.get("momentum_acceleration", 0),
        "rsi_1h": tech_1h.get("rsi_value", 50),
        "rsi_divergence": tech_1h.get("rsi_divergence", "none"),
        "macd_hist": tech_1h.get("macd_hist", 0),
        "macd_divergence": tech_1h.get("macd_divergence", "none"),
        "ema_alignment": tech_1h.get("ema_alignment", "mixed"),
        "atr_pct": ind_1h.get("atr_pct", 0),
        "atr_expanding": float(bool(ind_1h.get("atr_expanding", False))),
        "bb_squeeze": float(bool(ind_1h.get("bb_squeeze", False))),
        "vwap_distance_pct": ind_1h.get("vwap_distance_pct", 0),
        "price_above_vwap": float(bool(ind_1h.get("price_above_vwap", False))),
        "higher_highs": float(bool(struct_1h.get("higher_highs", False))),
        "higher_lows": float(bool(struct_1h.get("higher_lows", False))),
        "bos_up": float(bool(struct_1h.get("bos_up", False))),
        "candle_strength": candle_1h.get("candle_strength", 0.5),
        "big_wick_top": float(bool(candle_1h.get("big_wick_top", False))),
        "btc_corr_2d": btc_corr.get("btc_corr_2d", 0),
        "independent_mover": float(bool(btc_corr.get("independent_mover", False))),
        "elliott_score": elliott.get("score", 50),
        "fib_distance": fib.get("distance_to_closest", 100),
        "ichimoku_score": ichi.get("price_vs_cloud", "neutral") != "neutral" and 60 or 50,
        "long_score_raw": long_s,
        "short_score_raw": short_s,
    }
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
    universe = get_top_small_caps(toobit, cp, limit=20)
    print(f"[large-bt] universe: {len(universe)} symbols", flush=True)
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    sentiment = build_sentiment_v2(
        toobit, cache_dir=os.path.join("..", "data")
    )
    print(f"[large-bt] sentiment: {sentiment.aggregate:.1f} "
          f"({sentiment.risk_regime})", flush=True)
    rows = []
    n_okx = 0
    print(f"[large-bt] scanning, 30 days, every 6h", flush=True)
    for s_idx, sym in enumerate(universe, 1):
        if s_idx % 5 == 0 or s_idx == len(universe):
            print(f"[large-bt] {s_idx}/{len(universe)} {sym}", flush=True)
        try:
            df_1h = toobit.get_klines(sym, "1h", 30 * 24 + 50)
        except Exception:
            continue
        if df_1h.empty or len(df_1h) < 50:
            continue
        # Sample every 12h (12 bars)
        for idx in range(30, len(df_1h) - 12, 12):
            pack = snapshot_full(sym, toobit, idx, sentiment, okx, btc_df_1h)
            if not pack.get("pass2_ok"):
                continue
            if pack.get("okx_available"):
                n_okx += 1
            out = forward_12h(df_1h, idx)
            if not out:
                continue
            rec = {
                "symbol": sym,
                "long_score": pack["long_score_adj"],
                "short_score": pack["short_score_adj"],
                "peak": out["peak"],
                "dd": out["dd"],
                "pump_5": 1 if out["peak"] >= 5.0 else 0,
                "pump_10": 1 if out["peak"] >= 10.0 else 0,
                "dump_5": 1 if out["dd"] <= -5.0 else 0,
                "dump_10": 1 if out["dd"] <= -10.0 else 0,
            }
            # Add all features
            for k, v in pack["features"].items():
                rec[k] = v
            rows.append(rec)
        time.sleep(0.2)
    if not rows:
        print("[large-bt] no data")
        return
    df = pd.DataFrame(rows)
    os.makedirs("backtest_results_large", exist_ok=True)
    df.to_csv("backtest_results_large/dataset.csv", index=False)
    print(f"\n[large-bt] saved {len(df)} snapshots "
          f"({n_okx} with OKX history)")
    print(f"[large-bt] base: pump_5={df['pump_5'].mean():.3f}, "
          f"dump_5={df['dump_5'].mean():.3f}")
    # Direction-aware
    df["best_score"] = df[["long_score", "short_score"]].max(axis=1)
    df["best_dir"] = np.where(
        df["long_score"] >= df["short_score"], "LONG", "SHORT"
    )
    df["success_5"] = np.where(
        df["best_dir"] == "LONG", df["pump_5"], df["dump_5"]
    )
    df["success_10"] = np.where(
        df["best_dir"] == "LONG", df["pump_10"], df["dump_10"]
    )
    print()
    print("[large-bt] === DIRECTION-AWARE (30d, large sample) ===")
    for thr in [20, 30, 40, 50, 60, 70, 80]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        n_long = (sub["best_dir"] == "LONG").sum()
        s5 = sub["success_5"].mean()
        s10 = sub["success_10"].mean()
        # 95% CI for precision
        if n > 0:
            se = np.sqrt(s5 * (1 - s5) / n) if s5 > 0 else 0
            ci_low = max(0, s5 - 1.96 * se)
            ci_high = min(1, s5 + 1.96 * se)
        else:
            ci_low, ci_high = 0, 0
        print(f"  best >= {thr:3d}: n={n:3d} (long={n_long})  "
              f"5%= {s5:.3f} [{ci_low:.2f}-{ci_high:.2f}]  10%= {s10:.3f}")
    # By direction
    print()
    print("[large-bt] === BY DIRECTION ===")
    for direction in ["LONG", "SHORT"]:
        sub_dir = df[df["best_dir"] == direction]
        if sub_dir.empty:
            continue
        for thr in [30, 50, 70]:
            sub = sub_dir[sub_dir["best_score"] >= thr]
            if sub.empty:
                continue
            n = len(sub)
            label = "pump" if direction == "LONG" else "dump"
            s5 = sub[f"{label}_5"].mean()
            print(f"  {direction} >= {thr}: n={n}  {label}_5= {s5:.3f}")
    # Redundancy analysis: features highly correlated with each other
    print()
    print("[large-bt] === FEATURE REDUNDANCY ANALYSIS ===")
    feature_cols = [c for c in df.columns
                    if c not in ("symbol", "pump_5", "pump_10",
                                 "dump_5", "dump_10", "best_dir",
                                 "best_score", "success_5", "success_10",
                                 "long_score", "short_score", "peak", "dd")
                    and df[c].dtype in (float, int)]
    # Correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = corr_matrix.iloc[i, j]
            if c > 0.7:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], c))
    if high_corr_pairs:
        print(f"  High correlation pairs (|r| > 0.7):")
        for f1, f2, c in sorted(high_corr_pairs, key=lambda x: -x[2])[:15]:
            print(f"    {f1:25s} <-> {f2:25s}  r={c:.3f}")
    # Low-information features (near-zero variance or zero correlation with success)
    print()
    print(f"  Low-information features (|r with success_5| < 0.03):")
    if "success_5" in df.columns:
        low_info = []
        for c in feature_cols:
            if c in df.columns:
                try:
                    r = float(np.corrcoef(df[c].fillna(0), df["success_5"])[0, 1])
                    if abs(r) < 0.03:
                        low_info.append((c, r))
                except Exception:
                    pass
        for c, r in sorted(low_info, key=lambda x: x[1])[:15]:
            print(f"    {c:25s}  r={r:+.4f}")
    # Save
    summary = {
        "n_snapshots": int(len(df)),
        "n_okx": n_okx,
        "base_pump_5": float(df["pump_5"].mean()),
        "base_dump_5": float(df["dump_5"].mean()),
        "direction_aware": {},
        "by_direction": {},
        "redundant_pairs": [
            {"f1": f1, "f2": f2, "r": round(c, 3)}
            for f1, f2, c in high_corr_pairs
        ],
    }
    for thr in [30, 40, 50, 60, 70]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        summary["direction_aware"][f"thr_{thr}"] = {
            "n": int(len(sub)),
            "long_count": int((sub["best_dir"] == "LONG").sum()),
            "success_5pct": float(sub["success_5"].mean()),
            "success_10pct": float(sub["success_10"].mean()),
        }
    for d in ["LONG", "SHORT"]:
        sub_dir = df[df["best_dir"] == d]
        if sub_dir.empty:
            continue
        for thr in [50, 70]:
            sub = sub_dir[sub_dir["best_score"] >= thr]
            if sub.empty:
                continue
            label = "pump" if d == "LONG" else "dump"
            summary["by_direction"][f"{d}_{thr}"] = {
                "n": int(len(sub)),
                "success_5pct": float(sub[f"{label}_5"].mean()),
            }
    with open("backtest_results_large/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[large-bt] summary -> backtest_results_large/summary.json")


if __name__ == "__main__":
    main()
