"""
Direction-aware backtest: long + short precision.
Tests whether high long_score predicts +5%/10% pumps and
high short_score predicts -5%/-10% dumps.
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
from direction_scanner import compute_long_score, compute_short_score


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


def snapshot(symbol, toobit, idx_1h, btc_state, btc_df_1h):
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
    pack = {
        "ind_1h": ind_1h, "tech_1h": tech_1h, "struct_1h": struct_1h,
        "candle_1h": candle_1h, "patterns_1h": patterns_1h,
        "btc_corr": btc_corr,
    }
    out["pass2_ok"] = True
    out["long_score"] = compute_long_score(pack)
    out["short_score"] = compute_short_score(pack)
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
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    universe = get_small_caps(toobit, cp, limit=20)
    print(f"[dir-bt] universe: {len(universe)}", flush=True)
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    rows = []
    print(f"[dir-bt] scanning {len(universe)} symbols, 30 days, every 12h",
          flush=True)
    for s_idx, sym in enumerate(universe, 1):
        print(f"[dir-bt] ({s_idx}/{len(universe)}) {sym}", flush=True)
        try:
            df_1h = toobit.get_klines(sym, "1h", 30 * 24 + 50)
        except Exception:
            continue
        if df_1h.empty or len(df_1h) < 50:
            continue
        for idx in range(30, len(df_1h) - 12, 12):
            pack = snapshot(sym, toobit, idx, btc_state, btc_df_1h)
            if not pack.get("pass2_ok"):
                continue
            out = forward_12h(df_1h, idx)
            if not out:
                continue
            rec = {
                "symbol": sym,
                "long_score": pack["long_score"],
                "short_score": pack["short_score"],
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
        print("[dir-bt] no data")
        return
    df = pd.DataFrame(rows)
    os.makedirs("backtest_results_direction", exist_ok=True)
    df.to_csv("backtest_results_direction/dataset.csv", index=False)
    print(f"[dir-bt] saved {len(df)} snapshots")
    print(f"[dir-bt] base rates: pump_5={df['pump_5'].mean():.3f}, "
          f"pump_10={df['pump_10'].mean():.3f}, "
          f"dump_5={df['dump_5'].mean():.3f}, "
          f"dump_10={df['dump_10'].mean():.3f}")
    # Long score precision
    print()
    print("[dir-bt] === LONG SCORE (PUMP) ===")
    for thr in [40, 50, 60, 70, 80]:
        sub = df[df["long_score"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        p5 = sub["pump_5"].mean()
        p10 = sub["pump_10"].mean()
        print(f"  long >= {thr:3d}: n={n:3d}  pump_5={p5:.3f}  pump_10={p10:.3f}")
    # Short score precision
    print()
    print("[dir-bt] === SHORT SCORE (DUMP) ===")
    for thr in [40, 50, 60, 70, 80]:
        sub = df[df["short_score"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        d5 = sub["dump_5"].mean()
        d10 = sub["dump_10"].mean()
        print(f"  short >= {thr:3d}: n={n:3d}  dump_5={d5:.3f}  dump_10={d10:.3f}")
    # Combined: take the better of long/short
    print()
    print("[dir-bt] === DIRECTION-AWARE (best of long/short) ===")
    df["best_score"] = df[["long_score", "short_score"]].max(axis=1)
    df["best_direction"] = np.where(
        df["long_score"] >= df["short_score"], "LONG", "SHORT"
    )
    df["success"] = np.where(
        df["best_direction"] == "LONG", df["pump_5"], df["dump_5"]
    )
    df["success_10"] = np.where(
        df["best_direction"] == "LONG", df["pump_10"], df["dump_10"]
    )
    for thr in [40, 50, 60, 70, 80]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        n = len(sub)
        n_long = (sub["best_direction"] == "LONG").sum()
        s5 = sub["success"].mean()
        s10 = sub["success_10"].mean()
        print(f"  best >= {thr:3d}: n={n:3d} (long={n_long})  "
              f"5%= {s5:.3f}  10%= {s10:.3f}")
    # Summary
    summary = {
        "n_snapshots": int(len(df)),
        "base_pump_5": float(df["pump_5"].mean()),
        "base_pump_10": float(df["pump_10"].mean()),
        "base_dump_5": float(df["dump_5"].mean()),
        "base_dump_10": float(df["dump_10"].mean()),
    }
    for thr in [50, 60, 70, 80]:
        sub = df[df["best_score"] >= thr]
        if sub.empty:
            continue
        summary[f"best_{thr}"] = {
            "n": int(len(sub)),
            "long_count": int((sub["best_direction"] == "LONG").sum()),
            "short_count": int((sub["best_direction"] == "SHORT").sum()),
            "success_5pct": float(sub["success"].mean()),
            "success_10pct": float(sub["success_10"].mean()),
        }
    with open("backtest_results_direction/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[dir-bt] summary saved")


if __name__ == "__main__":
    main()
