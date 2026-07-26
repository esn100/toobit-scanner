"""
Backtest: test which extended indicators actually help predict pumps.

Methodology:
  1. For each snapshot, compute all extended indicators
  2. Build label = 1 if forward 12h return > 3% (pump), else 0
  3. For EACH indicator, compute:
     - Correlation with label
     - Mutual information
     - Chi-square significance
     - Win rate improvement (top 25% vs bottom 25%)
  4. Rank indicators by combined score
  5. Output: which indicators are worth keeping
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
from data_quality import validate_ohlcv
from btc_filter import BTCFilter
from extended_indicators import compute_all_extended


FORWARD_BARS = 3   # 12h on 4h
PUMP_THRESHOLD = 3.0   # +3% in 12h = PUMP
DUMP_THRESHOLD = -3.0  # -3% in 12h = DUMP


def forward_label(df: pd.DataFrame, idx: int) -> int:
    """1 if pump (+3% in 12h), 0 otherwise."""
    if idx + 1 + FORWARD_BARS > len(df):
        return -1  # not enough data
    entry = float(df["close"].iloc[idx])
    forward = df.iloc[idx + 1: idx + 1 + FORWARD_BARS]
    if forward.empty:
        return -1
    peak = float(((forward["high"] - entry) / entry * 100).max())
    dd = float(((forward["low"] - entry) / entry * 100).min())
    if peak >= PUMP_THRESHOLD and dd >= DUMP_THRESHOLD:
        return 1
    return 0


def collect_features_and_labels(symbols: List[str], days: int = 60,
                                snap_every: int = 6,
                                out_dir: str = "backtest_extended"):
    """Walk through history, collect features + labels."""
    os.makedirs(out_dir, exist_ok=True)
    toobit = ToobitClient()
    btc_state = BTCFilter(toobit).evaluate()
    print(f"[ext] BTC: {btc_state['state']}, symbols={len(symbols)}")
    all_rows = []
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[ext] ({s_idx}/{len(symbols)}) {sym}...", end=" ", flush=True)
        try:
            df = toobit.get_klines(sym, "4h", days * 6 + 50)
        except Exception as e:
            print(f"err: {e}")
            continue
        if df.empty or len(df) < 80:
            print(f"too few ({len(df)})")
            continue
        n_snap = 0
        n_labeled = 0
        for idx in range(60, len(df) - FORWARD_BARS, snap_every):
            # Quality
            sub = df.iloc[: idx + 1].copy().reset_index(drop=True)
            if len(sub) < 60:
                continue
            q = validate_ohlcv(sub, min_candles=60, interval_hours=4.0,
                               max_age_hours=1_000_000)
            if not q.ok or q.cleaned is None or q.cleaned.empty:
                continue
            sub = q.cleaned
            # Features
            try:
                features = compute_all_extended(sub)
            except Exception as e:
                continue
            # Label
            label = forward_label(df, idx)
            if label < 0:
                continue
            row = dict(features)
            row["symbol"] = sym
            row["idx"] = idx
            row["label"] = label
            all_rows.append(row)
            n_snap += 1
            n_labeled += 1
        print(f"{n_snap} snaps, {n_labeled} labeled")
        # Save periodically
        if len(all_rows) % 500 < snap_every:
            pd.DataFrame(all_rows).to_csv(
                os.path.join(out_dir, "raw_features.csv"), index=False
            )
    if not all_rows:
        print("[ext] no data")
        return pd.DataFrame()
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_dir, "raw_features.csv"), index=False)
    print(f"\n[ext] {len(df_all)} rows, {df_all['label'].sum()} pumps "
          f"({df_all['label'].mean()*100:.1f}%)")
    return df_all


def analyze_indicators(df: pd.DataFrame, out_dir: str) -> Dict:
    """Analyze each indicator's predictive power."""
    os.makedirs(out_dir, exist_ok=True)
    feat_cols = [c for c in df.columns if c.startswith("ex_") and c not in ("ex_",)]
    feat_cols = [c for c in feat_cols if c in df.columns]
    y = df["label"].astype(int).values
    print(f"\n[ext] Analyzing {len(feat_cols)} indicators "
          f"(n={len(df)}, pump_rate={y.mean()*100:.1f}%)\n")
    results = []
    for f in feat_cols:
        col = df[f].fillna(0.0)
        if col.std() < 1e-9:
            continue
        try:
            x = col.values
            # Pearson
            if x.std() > 0 and y.std() > 0:
                pearson = float(np.corrcoef(x, y)[0, 1])
            else:
                pearson = 0.0
            # Chi-square (bin by median)
            thr = float(np.median(x))
            hi = x >= thr
            lo = ~hi
            table = np.array([
                [int(y[hi].sum()), int((1 - y[hi]).sum())],
                [int(y[lo].sum()), int((1 - y[lo]).sum())],
            ])
            chi_p = 1.0
            if table.min() > 0 and table.sum() > 0:
                _, chi_p, _, _ = chi2_contingency(table)
            # Win rate top 25% vs bottom 25%
            top_thr = np.percentile(x, 75)
            bot_thr = np.percentile(x, 25)
            top_mask = x >= top_thr
            bot_mask = x <= bot_thr
            top_wr = y[top_mask].mean() if top_mask.any() else 0
            bot_wr = y[bot_mask].mean() if bot_mask.any() else 0
            wr_diff = float(top_wr - bot_wr)
            # Mean top 25% value
            top_val = float(x[top_mask].mean()) if top_mask.any() else 0
            bot_val = float(x[bot_mask].mean()) if bot_mask.any() else 0
            # Combined score
            score = (
                abs(pearson) * 0.3 +
                (1 - chi_p) * 0.3 +
                abs(wr_diff) / 0.5 * 0.4  # normalize
            )
            results.append({
                "indicator": f,
                "pearson_r": pearson,
                "chi_p": chi_p,
                "chi_sig": chi_p < 0.05,
                "wr_top25": float(top_wr),
                "wr_bot25": float(bot_wr),
                "wr_diff": wr_diff,
                "combined_score": score,
                "top25_mean": top_val,
                "bot25_mean": bot_val,
            })
        except Exception as e:
            continue
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("combined_score", ascending=False)
    df_results.to_csv(os.path.join(out_dir, "indicator_importance.csv"),
                      index=False)
    return {"df": df_results}


def recommend_keep_drop(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """Recommend which indicators to keep based on analysis."""
    # Top N by combined score
    top = df.head(top_n)
    keep = top["indicator"].tolist()
    # Bottom N (likely noise)
    bot = df.tail(top_n)
    drop = bot["indicator"].tolist()
    return {"keep": keep, "drop": drop, "top": top, "bot": bot}


# Default symbols
DEFAULT_SYMBOLS = [
    "ALICEUSDT", "TLMUSDT", "HYPERUSDT", "WCTUSDT", "RECALLUSDT",
    "GROVEUSDT", "RESOLVUSDT", "TOWNSUSDT", "OPNUSDT", "TUTUSDT",
    "FIGHTUSDT", "PARTIUSDT", "USATUSDT", "BREVUSDT", "EVAAUSDT",
    "BANKUSDT", "ACEUSDT",
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--top", type=int, default=17)
    p.add_argument("--snap", type=int, default=6)
    p.add_argument("--out", type=str, default="backtest_extended")
    args = p.parse_args()
    symbols = DEFAULT_SYMBOLS[: args.top]
    df = collect_features_and_labels(
        symbols, days=args.days, snap_every=args.snap, out_dir=args.out
    )
    if df.empty:
        return
    res = analyze_indicators(df, args.out)
    rec = recommend_keep_drop(res["df"], top_n=10)
    print("\n" + "=" * 100)
    print("TOP 10 INDICATORS (keep these):")
    print("=" * 100)
    print(f"{'Indicator':<30} {'Pearson':>8} {'Chi-p':>8} {'WR-Top':>7} {'WR-Bot':>7} {'Diff':>7} {'Score':>7}")
    for _, r in rec["top"].iterrows():
        print(f"{r['indicator']:<30} {r['pearson_r']:>+8.3f} {r['chi_p']:>8.4f} "
              f"{r['wr_top25']:>7.1%} {r['wr_bot25']:>7.1%} {r['wr_diff']:>+7.1%} "
              f"{r['combined_score']:>7.3f}")
    print("\n" + "=" * 100)
    print("BOTTOM 10 (consider dropping):")
    print("=" * 100)
    for _, r in rec["bot"].iterrows():
        print(f"{r['indicator']:<30} {r['pearson_r']:>+8.3f} {r['chi_p']:>8.4f} "
              f"{r['wr_top25']:>7.1%} {r['wr_bot25']:>7.1%} {r['wr_diff']:>+7.1%} "
              f"{r['combined_score']:>7.3f}")
    # Save report
    with open(os.path.join(args.out, "recommendation.json"), "w") as f:
        json.dump({
            "keep": rec["keep"],
            "drop": rec["drop"],
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)
    print(f"\n[ext] saved to {args.out}/")


if __name__ == "__main__":
    main()
