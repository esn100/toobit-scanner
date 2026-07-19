"""
Backtest v2: re-runs the backtester but with the new
  - Advanced chart patterns (bull_flag, cup_handle, etc.)
  - Multi-source sentiment (Fear&Greed, BTC dominance, etc.)
added to the score. Compares to v1.
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from data_quality import validate_ohlcv
from technical import technical_analysis
from indicators import (
    vwap_features, atr_features, bollinger_features,
    relative_volume, volume_continuity, momentum_features,
    mtf_alignment,
)
from market_structure import structure_features
from candle_quality import candle_quality_features
from features import build_features, FEATURE_NAMES
from btc_filter import BTCFilter
from scoring import rule_based_score
from chart_patterns import detect_all_patterns
from sentiment import build_sentiment


def snapshot_v2(df_4h, df_1h, idx, btc_state, sentiment_snap):
    sub_df = df_4h.iloc[: idx + 1].copy().reset_index(drop=True)
    if len(sub_df) < 60:
        return None
    q = validate_ohlcv(sub_df, min_candles=60, interval_hours=4.0,
                       max_age_hours=1_000_000)
    if not q.ok or q.cleaned is None:
        return None
    sub_df = q.cleaned
    tech = technical_analysis(sub_df)
    ind = {}
    ind.update(vwap_features(sub_df))
    ind.update(atr_features(sub_df))
    ind.update(bollinger_features(sub_df))
    ind.update(relative_volume(sub_df))
    ind.update(volume_continuity(sub_df))
    ind.update(momentum_features(sub_df))
    struct = structure_features(sub_df)
    candle = candle_quality_features(sub_df)
    patterns = detect_all_patterns(sub_df)
    if not df_1h.empty:
        one_h_idx = min(len(df_1h) - 1, (idx + 1) * 4)
        sub_1h = df_1h.iloc[: one_h_idx + 1].copy().reset_index(drop=True)
        mtf = mtf_alignment(sub_1h, sub_df) if len(sub_1h) >= 30 else {
            "alignment_score": 50.0,
        }
    else:
        mtf = {"alignment_score": 50.0}
    feats = build_features(tech, ind, struct, candle, mtf, btc_state)
    rb = rule_based_score(
        tech, ind, struct, candle, mtf, btc_state,
        {"technical": 9.9, "momentum": 9.9, "volume": 14.8, "vwap": 4.9,
         "atr_bb": 12.3, "structure": 18.5, "candle": 14.8, "mtf": 4.9,
         "pattern": 9.9},
    )
    composite = rb["composite_score"]
    # Add advanced patterns
    p_bull = patterns["bullish_score"]
    p_bear = patterns["bearish_score"]
    if p_bull > p_bear:
        composite += min(8.0, p_bull * 0.1)
    elif p_bear > 0:
        composite -= min(8.0, p_bear * 0.1)
    # Add sentiment modifier
    if sentiment_snap is not None:
        sent_mod = (sentiment_snap["aggregate"] - 50.0) / 10.0
        composite += max(-5.0, min(5.0, sent_mod))
    composite = max(0.0, min(100.0, composite))
    return {
        "composite_v2": composite,
        "patterns": patterns["patterns"],
        "pattern_score": p_bull - p_bear,
        "sub_scores": rb["sub_scores"],
        "features": feats,
    }


def forward_outcome(df, idx, fwd=3):
    if idx + 1 + fwd > len(df):
        return None
    entry = float(df["close"].iloc[idx])
    forward = df.iloc[idx + 1: idx + 1 + fwd]
    highs = forward["high"].astype(float)
    lows = forward["low"].astype(float)
    closes = forward["close"].astype(float)
    peak = float(((highs - entry) / entry * 100.0).max())
    dd = float(((lows - entry) / entry * 100.0).min())
    final = float(((closes.iloc[-1] - entry) / entry * 100.0))
    label = 1 if (peak >= 3.0 and dd >= -2.0) else 0
    return {"label": label, "peak": peak, "dd": dd, "final": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=str, default="backtest_results_v2")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Fetch sentiment snapshot ONCE
    sentiment_snap = None
    try:
        s = build_sentiment()
        sentiment_snap = s.__dict__
        print(f"[bt2] sentiment aggregate={s.aggregate:.1f} ({s.risk_regime})")
    except Exception as e:
        print(f"[bt2] sentiment fetch failed: {e}")

    toobit = ToobitClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    print(f"[bt2] BTC state: {btc_state['state']}")

    DEFAULT_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        "HYPERUSDT", "WCTUSDT", "RECALLUSDT", "TLMUSDT",
        "OPNUSDT", "ACEUSDT", "TUTUSDT", "SATSUSDT", "BREVUSDT",
    ]
    symbols = DEFAULT_SYMBOLS[: args.top]
    rows = []
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[bt2] ({s_idx}/{len(symbols)}) {sym} ...", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", args.days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", args.days * 24 + 50)
        except Exception as e:
            print(f"  fetch err: {e}")
            continue
        if df_4h.empty or len(df_4h) < 80:
            continue
        max_idx = len(df_4h) - 1 - 3
        n_snap = 0
        for idx in range(60, max_idx + 1, 2):
            pack = snapshot_v2(df_4h, df_1h, idx, btc_state, sentiment_snap)
            if not pack:
                continue
            out = forward_outcome(df_4h, idx, 3)
            if not out:
                continue
            rec = {
                "symbol": sym,
                "label": out["label"],
                "composite_v1": pack["composite_v2"],
                "peak": out["peak"],
                "dd": out["dd"],
                "final": out["final"],
                "pattern_score": pack["pattern_score"],
                "patterns": pack["patterns"],
            }
            for k, v in pack["features"].items():
                rec[k] = v
            rows.append(rec)
            n_snap += 1
        print(f"  {n_snap} snapshots", flush=True)
        time.sleep(0.3)
    if not rows:
        print("[bt2] no data")
        return
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "backtest_v2.csv")
    df.to_csv(path, index=False)
    print(f"[bt2] saved {len(df)} rows -> {path}")
    print(f"[bt2] success rate: {df['label'].mean():.3f}")

    # Compute top-10% precision (with patterns + sentiment)
    base_scores = df["composite_v1"].values
    y = df["label"].values
    base_top = base_scores >= np.percentile(base_scores, 90)
    base_prec = float(y[base_top].mean()) if base_top.any() else 0.0
    print(f"[bt2] Top-10% precision (v1 weights + patterns + sentiment): "
          f"{base_prec:.3f}")

    # Score separation
    if y.sum() > 0:
        sep = float(base_scores[y == 1].mean() - base_scores[y == 0].mean())
    else:
        sep = 0.0
    print(f"[bt2] Score separation: {sep:.2f}")

    # Pattern stats
    if "patterns" in df.columns:
        any_pat = df["patterns"].apply(lambda p: len(p) > 0 if isinstance(p, list) else False)
        if any_pat.any():
            pat_prec = float(df[any_pat]["label"].mean())
            print(f"[bt2] Symbols with detected patterns: {any_pat.sum()} "
                  f"({pat_prec:.3f} success rate)")
        else:
            print("[bt2] no patterns detected in any snapshot")


if __name__ == "__main__":
    main()
