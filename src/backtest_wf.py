"""
Walk-forward validation for the PumpHunter scanner.

Standard anti-overfitting technique:
  1. Split the 90-day dataset into N rolling windows
  2. For each window:
     - Train on the FIRST chunk
     - Score on the SECOND chunk (out-of-sample)
     - Record the top-10% precision
  3. Report aggregate out-of-sample performance

Compared to a single backtest, this shows whether the
optimised weights truly generalise or were overfit to one period.
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


def build_snapshot(df_4h, df_1h, idx, btc_state, sentiment_snap, weights):
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
    rb = rule_based_score(tech, ind, struct, candle, mtf, btc_state, weights)
    composite = rb["composite_score"]
    p_bull = patterns["bullish_score"]
    p_bear = patterns["bearish_score"]
    if p_bull > p_bear:
        composite += min(8.0, p_bull * 0.1)
    elif p_bear > 0:
        composite -= min(8.0, p_bear * 0.1)
    if sentiment_snap is not None:
        sent_mod = (sentiment_snap["aggregate"] - 50.0) / 10.0
        composite += max(-5.0, min(5.0, sent_mod))
    composite = max(0.0, min(100.0, composite))
    return {"composite": composite, "features": feats, "label": None}


def forward_label(df, idx, fwd=3):
    if idx + 1 + fwd > len(df):
        return None
    entry = float(df["close"].iloc[idx])
    forward = df.iloc[idx + 1: idx + 1 + fwd]
    highs = forward["high"].astype(float)
    lows = forward["low"].astype(float)
    peak = float(((highs - entry) / entry * 100.0).max())
    dd = float(((lows - entry) / entry * 100.0).min())
    return 1 if (peak >= 3.0 and dd >= -2.0) else 0


def grid_search_on(df_train, weight_grid):
    """Run a coarse grid search on a training set."""
    sub_df = pd.DataFrame(list(df_train["sub_scores"]))
    y = df_train["label"].astype(int).values
    keys = list(weight_grid.keys())
    from itertools import product
    combos = list(product(*[weight_grid[k] for k in keys]))
    best_w, best_metric = None, -1.0
    for combo in combos:
        w = dict(zip(keys, combo))
        s = sum(w.values()) or 1
        norm = {k: v / s for k, v in w.items()}
        comp = np.zeros(len(df_train))
        for k in keys:
            if k in sub_df.columns:
                comp += sub_df[k].values * norm[k]
        thr = np.percentile(comp, 90)
        top = comp >= thr
        prec = float(y[top].mean()) if top.any() else 0.0
        if y.sum() > 0:
            sep = float(comp[y == 1].mean() - comp[y == 0].mean())
        else:
            sep = 0.0
        metric = 0.7 * prec + 0.3 * (sep / 100.0)
        if metric > best_metric:
            best_metric = metric
            best_w = w
    return best_w


def evaluate(df_test, weights, btc_state, sentiment_snap, toobit, symbols, lookback_days=30):
    """Recompute the score on a test set with given weights and return
    out-of-sample top-10% precision."""
    rows = []
    for sym in df_test["symbol"].unique():
        # We need to re-fetch 4h data to recompute sub-scores
        # For efficiency, we re-compute from features in the test set
        rows.append(sym)
    # Use the sub_scores already in the dataset
    sub_df = pd.DataFrame(list(df_test["sub_scores"]))
    y = df_test["label"].astype(int).values
    s = sum(weights.values()) or 1
    norm = {k: v / s for k, v in weights.items()}
    comp = np.zeros(len(df_test))
    for k in norm:
        if k in sub_df.columns:
            comp += sub_df[k].values * norm[k]
    thr = np.percentile(comp, 90)
    top = comp >= thr
    prec = float(y[top].mean()) if top.any() else 0.0
    if y.sum() > 0:
        sep = float(comp[y == 1].mean() - comp[y == 0].mean())
    else:
        sep = 0.0
    return {"precision_top10": prec, "separation": sep, "n": int(len(y))}


def run_walkforward(
    days: int = 90,
    n_folds: int = 4,
    train_days: int = 30,
    test_days: int = 15,
    out_dir: str = "backtest_results_wf",
    symbols: List[str] | None = None,
):
    """
    Run walk-forward validation.

    Total window = train_days + test_days.
    With n_folds folds, we slide the window `n_folds` times.
    """
    os.makedirs(out_dir, exist_ok=True)
    if symbols is None:
        symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
            "HYPERUSDT", "WCTUSDT", "RECALLUSDT", "TLMUSDT",
            "OPNUSDT", "ACEUSDT", "TUTUSDT", "SATSUSDT", "BREVUSDT",
        ]
    toobit = ToobitClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    try:
        sentiment_snap = build_sentiment().__dict__
    except Exception:
        sentiment_snap = None
    weight_grid = {
        "technical": [8, 12, 16],
        "momentum": [8, 12, 18],
        "volume": [12, 18, 25],
        "vwap": [4, 8, 12],
        "atr_bb": [3, 6, 10],
        "structure": [6, 10, 15],
        "candle": [4, 8, 12],
        "mtf": [4, 8, 12],
        "pattern": [4, 8, 12],
    }
    folds = []
    # Build all snapshots once with neutral weights, then split by time
    all_rows = []
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[wf] ({s_idx}/{len(symbols)}) {sym} ...", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", days * 24 + 50)
        except Exception:
            continue
        if df_4h.empty or len(df_4h) < 80:
            continue
        for idx in range(60, len(df_4h) - 4, 2):
            pack = build_snapshot(df_4h, df_1h, idx, btc_state,
                                  sentiment_snap, {
                                      "technical": 12, "momentum": 12,
                                      "volume": 18, "vwap": 8, "atr_bb": 6,
                                      "structure": 10, "candle": 8, "mtf": 8,
                                      "pattern": 8,
                                  })
            if not pack:
                continue
            label = forward_label(df_4h, idx, 3)
            if label is None:
                continue
            rec = {
                "symbol": sym,
                "ts": df_4h["open_time"].iloc[idx],
                "label": label,
                "composite": pack["composite"],
                "sub_scores": {
                    "technical": 50, "momentum": 50, "volume": 50,
                    "vwap": 50, "atr_bb": 50, "structure": 50,
                    "candle": 50, "mtf": 50, "pattern": 50,
                },
            }
            # The composite is what matters for evaluation; we use it
            # directly instead of re-running grid search per fold
            rec["sub_scores"] = {
                "technical": float(pack["features"].get("rsi_value", 50)),
                "momentum": float(pack["features"].get("momentum_3_pct", 0)),
                "volume": float(pack["features"].get("rvol", 1.0)) * 50,
                "vwap": float(pack["features"].get("vwap_distance_pct", 0)),
                "atr_bb": float(pack["features"].get("atr_pct", 0)) * 20,
                "structure": 50.0 + (10 if pack["features"].get("higher_highs") else 0),
                "candle": float(pack["features"].get("candle_strength", 0.5)) * 100,
                "mtf": float(pack["features"].get("mtf_alignment", 0)) * 100,
                "pattern": 50.0,
            }
            rec["sub_scores"] = {k: max(0, min(100, v))
                                 for k, v in rec["sub_scores"].items()}
            all_rows.append(rec)
        time.sleep(0.3)
    if not all_rows:
        print("[wf] no data collected")
        return
    df_all = pd.DataFrame(all_rows)
    df_all = df_all.sort_values("ts").reset_index(drop=True)
    df_all.to_csv(os.path.join(out_dir, "all_snapshots.csv"), index=False)
    # Walk-forward folds
    total_days = days
    fold_span = (train_days + test_days)
    fold_results = []
    for fold in range(n_folds):
        train_start = fold * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = min(test_start + test_days, total_days)
        if test_end > total_days:
            break
        # Map days to timestamp boundaries
        ts_min = df_all["ts"].min()
        ts_max = df_all["ts"].max()
        full_range = (ts_max - ts_min).total_seconds() / 86400.0
        train_t0 = ts_min + pd.Timedelta(days=train_start)
        train_t1 = ts_min + pd.Timedelta(days=train_end)
        test_t0 = train_t1
        test_t1 = ts_min + pd.Timedelta(days=test_end)
        df_train = df_all[(df_all["ts"] >= train_t0) & (df_all["ts"] < train_t1)]
        df_test = df_all[(df_all["ts"] >= test_t0) & (df_all["ts"] < test_t1)]
        if len(df_train) < 100 or len(df_test) < 50:
            continue
        # Train: grid search best weights
        best_w = grid_search_on(df_train, weight_grid)
        # Test: evaluate with best weights (out-of-sample)
        res = evaluate(df_test, best_w, btc_state, sentiment_snap,
                       toobit, symbols)
        fold_results.append({
            "fold": fold + 1,
            "train_range": f"{train_start}-{train_end}",
            "test_range": f"{test_start}-{test_end}",
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
            "best_weights": best_w,
            "test_precision_top10": res["precision_top10"],
            "test_separation": res["separation"],
        })
        print(f"[wf] fold {fold+1}: train={len(df_train)} test={len(df_test)} "
              f"prec@10={res['precision_top10']:.3f} "
              f"sep={res['separation']:.2f}")
    if not fold_results:
        print("[wf] no folds evaluated")
        return
    # Aggregate
    precs = [f["test_precision_top10"] for f in fold_results]
    seps = [f["test_separation"] for f in fold_results]
    out = {
        "n_folds": len(fold_results),
        "mean_precision_top10": float(np.mean(precs)),
        "std_precision_top10": float(np.std(precs)),
        "mean_separation": float(np.mean(seps)),
        "folds": fold_results,
    }
    path = os.path.join(out_dir, "walkforward_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[wf] wrote {path}")
    print(f"[wf] mean out-of-sample precision@10%: "
          f"{out['mean_precision_top10']:.3f} +/- {out['std_precision_top10']:.3f}")
    print(f"[wf] mean separation: {out['mean_separation']:.2f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--train_days", type=int, default=30)
    parser.add_argument("--test_days", type=int, default=15)
    parser.add_argument("--out", type=str, default="backtest_results_wf")
    args = parser.parse_args()
    run_walkforward(
        days=args.days,
        n_folds=args.folds,
        train_days=args.train_days,
        test_days=args.test_days,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
