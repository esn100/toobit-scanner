"""
PumpHunter backtester (Layer 11).

Walks forward through historical 4h candles. At each snapshot:
  - Builds features for every symbol
  - Computes the composite score under given weights
  - Computes the actual forward outcome (12h later)
  - Records (features, score, outcome, success_label)

After walking:
  - Computes feature importance via:
      * Chi-Square
      * Mutual Information
      * Pearson/Spearman correlation with success
      * Logistic Regression coefficients
      * Random Forest feature importance
  - Runs grid search on the 9 sub-score weights to maximise a
    backtest Sharpe-style score
  - Writes a report (backtest_report.json) and a markdown summary

Usage:
    python backtest.py --days 90 --top 20
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
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

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


# ----------------------------------------------------------------------------
# Success definition
# ----------------------------------------------------------------------------
SUCCESS_THRESHOLD_PCT = 3.0   # +3% in 12h
DD_LIMIT_PCT = -2.0           # no drawdown beyond -2% during the hold


def forward_outcome(
    df: pd.DataFrame, snapshot_idx: int, forward_bars: int = 3
) -> Dict:
    """
    Compute the forward outcome for a snapshot at index `snapshot_idx`
    in the 4h dataframe. forward_bars=3 means 12 hours ahead.
    """
    empty = {
        "peak_pct": 0.0, "dd_pct": 0.0, "final_pct": 0.0,
        "label": 0, "ok": False,
    }
    if df is None or df.empty:
        return empty
    if snapshot_idx + 1 + forward_bars > len(df):
        return empty
    entry = float(df["close"].iloc[snapshot_idx])
    if entry <= 0:
        return empty
    forward = df.iloc[snapshot_idx + 1: snapshot_idx + 1 + forward_bars]
    if forward.empty:
        return empty
    highs = forward["high"].astype(float)
    lows = forward["low"].astype(float)
    closes = forward["close"].astype(float)
    peak = float(((highs - entry) / entry * 100.0).max())
    dd = float(((lows - entry) / entry * 100.0).min())
    final = float(((closes.iloc[-1] - entry) / entry * 100.0))
    label = 1 if (peak >= SUCCESS_THRESHOLD_PCT and dd >= DD_LIMIT_PCT) else 0
    return {
        "peak_pct": peak, "dd_pct": dd, "final_pct": final,
        "label": label, "ok": True,
    }


# ----------------------------------------------------------------------------
# Per-symbol snapshot
# ----------------------------------------------------------------------------
def snapshot_features(df_4h: pd.DataFrame, df_1h: pd.DataFrame, idx: int,
                      btc_state: dict) -> Dict:
    """
    Build features + sub-scores for a single (symbol, time) snapshot.
    Uses only data up to bar `idx` (inclusive).
    """
    sub_df = df_4h.iloc[: idx + 1].copy().reset_index(drop=True)
    if len(sub_df) < 60:
        return {}
    # Quality
    q = validate_ohlcv(sub_df, min_candles=60, interval_hours=4.0,
                       max_age_hours=1_000_000)
    if not q.ok or q.cleaned is None or q.cleaned.empty:
        return {}
    sub_df = q.cleaned
    tech = technical_analysis(sub_df)
    ind: Dict = {}
    ind.update(vwap_features(sub_df))
    ind.update(atr_features(sub_df))
    ind.update(bollinger_features(sub_df))
    ind.update(relative_volume(sub_df))
    ind.update(volume_continuity(sub_df))
    ind.update(momentum_features(sub_df))
    struct = structure_features(sub_df)
    candle = candle_quality_features(sub_df)
    # 1h MTF: take a corresponding window if available
    if not df_1h.empty:
        # 4h snapshot idx -> 1h bar index ~ idx*4
        one_h_idx = min(len(df_1h) - 1, (idx + 1) * 4)
        sub_1h = df_1h.iloc[: one_h_idx + 1].copy().reset_index(drop=True)
        mtf = mtf_alignment(sub_1h, sub_df) if len(sub_1h) >= 30 else {
            "alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
            "aligned": False, "same_sign": False,
        }
    else:
        mtf = {"alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
               "aligned": False, "same_sign": False}
    feats = build_features(tech, ind, struct, candle, mtf, btc_state)
    rb = rule_based_score(
        tech, ind, struct, candle, mtf, btc_state,
        # neutral weights, will be re-scored after grid search
        {"technical": 12, "momentum": 12, "volume": 18, "vwap": 8,
         "atr_bb": 6, "structure": 10, "candle": 8, "mtf": 8, "pattern": 8},
    )
    return {
        "features": feats,
        "sub_scores": rb["sub_scores"],
        "composite_neutral": rb["composite_score"],
    }


# ----------------------------------------------------------------------------
# Main backtest loop
# ----------------------------------------------------------------------------
def run_backtest(
    symbols: List[str],
    days: int = 90,
    snapshot_every_bars: int = 2,  # every 8h
    forward_bars: int = 3,          # 12h lookahead
    out_dir: str = "backtest_results",
) -> Dict:
    """
    Walk forward through history, collect (features, score, outcome)
    for every symbol at every snapshot, and return a structured dataset.
    """
    os.makedirs(out_dir, exist_ok=True)
    toobit = ToobitClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()  # cached for the duration
    print(f"[bt] BTC state: {btc_state['state']}", flush=True)

    rows: List[Dict] = []
    raw_scores: List[Dict] = []
    total = len(symbols)
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[bt] ({s_idx}/{total}) {sym} ...", flush=True)
        try:
            df_4h = toobit.get_klines(sym, "4h", days * 6 + 50)
            df_1h = toobit.get_klines(sym, "1h", days * 24 + 50)
        except Exception as e:
            print(f"[bt]   fetch error: {e}", flush=True)
            continue
        if df_4h.empty or len(df_4h) < 80:
            print(f"[bt]   too few candles ({len(df_4h)})", flush=True)
            continue
        # Walk forward
        max_idx = len(df_4h) - 1 - forward_bars
        n_snap = 0
        for idx in range(60, max_idx + 1, snapshot_every_bars):
            ts = df_4h["open_time"].iloc[idx]
            pack = snapshot_features(df_4h, df_1h, idx, btc_state)
            if not pack:
                continue
            out = forward_outcome(df_4h, idx, forward_bars)
            if not out["ok"]:
                continue
            rec = {
                "symbol": sym,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "label": int(out["label"]),
                "peak_pct": out["peak_pct"],
                "dd_pct": out["dd_pct"],
                "final_pct": out["final_pct"],
                "composite_neutral": pack["composite_neutral"],
                "sub_scores": pack["sub_scores"],
            }
            for k, v in pack["features"].items():
                rec[k] = v
            rows.append(rec)
            n_snap += 1
        print(f"[bt]   {sym}: {n_snap} snapshots", flush=True)
        time.sleep(0.3)
    if not rows:
        print("[bt] no data collected", flush=True)
        return {}
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "backtest_dataset.csv")
    df.to_csv(path, index=False)
    print(f"[bt] saved {len(df)} rows -> {path}", flush=True)
    return {"df": df, "out_dir": out_dir, "n_rows": len(df)}


# ----------------------------------------------------------------------------
# Feature importance analysis
# ----------------------------------------------------------------------------
def analyse_importance(df: pd.DataFrame, out_dir: str) -> Dict:
    """
    Run every importance method on the backtest dataset.
    """
    feat_cols = FEATURE_NAMES
    # Some features may be missing in the dataframe
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(0.0).values
    y = df["label"].astype(int).values
    print(f"[bt] analyse: {X.shape[0]} samples, {X.shape[1]} features, "
          f"positive rate {y.mean():.3f}", flush=True)
    if y.sum() < 5 or (1 - y).sum() < 5:
        print("[bt] not enough positive/negative samples", flush=True)
        return {}

    results: Dict = {"n_samples": int(X.shape[0]),
                     "n_features": int(X.shape[1]),
                     "positive_rate": float(y.mean()),
                     "feature_names": feat_cols}

    # 1) Pearson correlation with success
    cor = {}
    for i, f in enumerate(feat_cols):
        try:
            cor[f] = float(np.corrcoef(X[:, i], y)[0, 1])
        except Exception:
            cor[f] = 0.0
    results["pearson"] = dict(
        sorted(cor.items(), key=lambda kv: -abs(kv[1]))
    )

    # 2) Chi-Square: bin each feature by median, run 2x2
    chi = {}
    for i, f in enumerate(feat_cols):
        s = X[:, i]
        if s.std() < 1e-9:
            continue
        thr = float(np.median(s))
        hi = s >= thr
        lo = ~hi
        table = np.array([
            [int(y[hi].sum()), int((1 - y[hi]).sum())],
            [int(y[lo].sum()), int((1 - y[lo]).sum())],
        ])
        if table.min() == 0 or table.sum() == 0:
            continue
        chi_stat, p, _, _ = chi2_contingency(table)
        chi[f] = {"chi2": float(chi_stat), "p_value": float(p),
                  "significant": bool(p < 0.05)}
    results["chi_square"] = dict(
        sorted(chi.items(), key=lambda kv: -kv[1]["chi2"])
    )

    # 3) Mutual Information
    try:
        mi = mutual_info_classif(X, y, random_state=42)
        results["mutual_info"] = dict(
            sorted({f: float(m) for f, m in zip(feat_cols, mi)}.items(),
                   key=lambda kv: -kv[1])
        )
    except Exception as e:
        results["mutual_info_error"] = str(e)

    # 4) Logistic Regression
    try:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # balanced for the rare positive class
        logreg = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    C=0.5, random_state=42)
        logreg.fit(Xs, y)
        coef = logreg.coef_[0]
        results["logreg_coef"] = dict(
            sorted({f: float(c) for f, c in zip(feat_cols, coef)}.items(),
                   key=lambda kv: -abs(kv[1]))
        )
        cv = cross_val_score(logreg, Xs, y, cv=5, scoring="f1")
        results["logreg_cv_f1_mean"] = float(cv.mean())
        results["logreg_cv_f1_std"] = float(cv.std())
    except Exception as e:
        results["logreg_error"] = str(e)

    # 5) Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight="balanced", random_state=42)
        rf.fit(X, y)
        imp = rf.feature_importances_
        results["rf_importance"] = dict(
            sorted({f: float(v) for f, v in zip(feat_cols, imp)}.items(),
                   key=lambda kv: -kv[1])
        )
        cv = cross_val_score(rf, X, y, cv=5, scoring="f1")
        results["rf_cv_f1_mean"] = float(cv.mean())
        results["rf_cv_f1_std"] = float(cv.std())
    except Exception as e:
        results["rf_error"] = str(e)

    # 6) Group features into sub-score buckets and assess each
    sub_to_feats = {
        "technical": ["rsi_value", "rsi_divergence", "macd_hist",
                      "macd_divergence", "ema_alignment"],
        "momentum": ["momentum_1_pct", "momentum_3_pct", "momentum_6_pct",
                     "momentum_12_pct", "momentum_acceleration"],
        "volume": ["rvol", "volume_spike"],
        "vwap": ["vwap_distance_pct", "price_above_vwap"],
        "atr_bb": ["atr_pct", "atr_expanding", "bb_squeeze", "bb_breakout_above"],
        "structure": ["higher_highs", "higher_lows", "bos_up", "in_range"],
        "candle": ["candle_strength", "big_wick_top", "power_streak"],
        "mtf": ["mtf_alignment"],
        "pattern": ["rsi_divergence", "macd_divergence", "candle_strength",
                    "power_streak"],
    }
    sub_score = {}
    for sub, fnames in sub_to_feats.items():
        cols = [c for c in fnames if c in df.columns]
        if not cols:
            sub_score[sub] = {"auc": 0.5, "n": 0}
            continue
        # Use mean of normalised features as a sub-score proxy
        sub_vals = df[cols].fillna(0.0).mean(axis=1).values
        # Correlation with success
        if sub_vals.std() > 1e-9:
            r = float(np.corrcoef(sub_vals, y)[0, 1])
        else:
            r = 0.0
        sub_score[sub] = {"pearson": r, "n_features": len(cols)}
    results["sub_score_pearson"] = sub_score

    # Save report
    report_path = os.path.join(out_dir, "importance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[bt] importance report -> {report_path}", flush=True)
    return results


# ----------------------------------------------------------------------------
# Grid search for optimal sub-score weights
# ----------------------------------------------------------------------------
def grid_search_weights(df: pd.DataFrame, out_dir: str) -> Dict:
    """
    Search a small grid of weights to maximise a backtest performance
    metric: precision@top10% AND win-rate above 1.2x baseline.
    """
    sub_keys = ["technical", "momentum", "volume", "vwap", "atr_bb",
                "structure", "candle", "mtf", "pattern"]
    # Read each sub-score at snapshot time
    sub_df = pd.DataFrame(list(df["sub_scores"]))
    for k in sub_keys:
        if k not in sub_df.columns:
            sub_df[k] = 50.0
    y = df["label"].astype(int).values

    # Baseline precision: rate of top 10% of neutral composite
    base_scores = df["composite_neutral"].values
    base_top = base_scores >= np.percentile(base_scores, 90)
    base_precision = float(y[base_top].mean()) if base_top.any() else 0.0

    # Coarse grid
    grid = {
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
    keys = list(grid.keys())
    # Build a flat list of weight combinations
    from itertools import product
    combos = list(product(*[grid[k] for k in keys]))
    print(f"[bt] grid search: {len(combos)} combinations", flush=True)

    best = None
    best_metric = -1.0
    best_weights = None
    results = []
    for combo in combos:
        w = dict(zip(keys, combo))
        s = sum(w.values()) or 1
        norm = {k: v / s for k, v in w.items()}
        # Recompute composite
        comp = sum(sub_df[k].values * norm[k] for k in keys)
        # Top 10% precision
        thr = np.percentile(comp, 90)
        top_mask = comp >= thr
        prec = float(y[top_mask].mean()) if top_mask.any() else 0.0
        # Coverage: average score of positives vs negatives
        if y.sum() > 0:
            avg_pos = float(comp[y == 1].mean())
            avg_neg = float(comp[y == 0].mean())
            sep = avg_pos - avg_neg
        else:
            sep = 0.0
        # Combined metric: weighted precision + separation
        metric = 0.7 * prec + 0.3 * (sep / 100.0)
        results.append({"weights": w, "norm": norm, "precision_top10": prec,
                        "separation": sep, "metric": metric})
        if metric > best_metric:
            best_metric = metric
            best_weights = w
            best = results[-1]
    # Save
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "grid_search_results.csv"), index=False
    )
    final = {
        "best_weights": best_weights,
        "best_normalized": {k: v / sum(best_weights.values()) for k, v in best_weights.items()} if best_weights else None,
        "best_metric": best_metric,
        "best_precision_top10": best["precision_top10"] if best else 0,
        "best_separation": best["separation"] if best else 0,
        "baseline_precision_top10": base_precision,
        "n_combinations": len(combos),
    }
    with open(os.path.join(out_dir, "grid_search_best.json"), "w",
              encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"[bt] grid best: {best_weights}", flush=True)
    print(f"[bt] best top-10% precision: {best['precision_top10']:.3f} "
          f"(baseline {base_precision:.3f})", flush=True)
    return final


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def write_markdown_report(imp: Dict, grid: Dict, out_dir: str) -> str:
    """
    Compose a human-readable summary.
    """
    lines = ["# PumpHunter Backtest Report", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    if not imp:
        lines.append("No importance data available.")
        return "\n".join(lines)
    lines.append(f"**Samples**: {imp['n_samples']}")
    lines.append(f"**Features**: {imp['n_features']}")
    lines.append(f"**Positive rate**: {imp['positive_rate']:.3f}")
    lines.append("")
    if "rf_cv_f1_mean" in imp:
        lines.append(f"**Random Forest CV F1**: "
                     f"{imp['rf_cv_f1_mean']:.3f} ± {imp['rf_cv_f1_std']:.3f}")
    if "logreg_cv_f1_mean" in imp:
        lines.append(f"**Logistic Regression CV F1**: "
                     f"{imp['logreg_cv_f1_mean']:.3f} ± {imp['logreg_cv_f1_std']:.3f}")
    lines.append("")
    # Pearson
    lines.append("## 1. Pearson correlation with success")
    lines.append("| Feature | r |")
    lines.append("|---|---:|")
    for f, v in list(imp.get("pearson", {}).items())[:15]:
        lines.append(f"| {f} | {v:+.4f} |")
    lines.append("")
    # Chi-Square
    lines.append("## 2. Chi-Square significance")
    lines.append("| Feature | chi2 | p | significant |")
    lines.append("|---|---:|---:|:---:|")
    for f, d in list(imp.get("chi_square", {}).items())[:15]:
        sig = "✓" if d["significant"] else "—"
        lines.append(f"| {f} | {d['chi2']:.2f} | {d['p_value']:.4f} | {sig} |")
    lines.append("")
    # MI
    lines.append("## 3. Mutual Information")
    lines.append("| Feature | MI |")
    lines.append("|---|---:|")
    for f, v in list(imp.get("mutual_info", {}).items())[:15]:
        lines.append(f"| {f} | {v:.4f} |")
    lines.append("")
    # LogReg
    lines.append("## 4. Logistic Regression coefficients")
    lines.append("| Feature | coef |")
    lines.append("|---|---:|")
    for f, v in list(imp.get("logreg_coef", {}).items())[:15]:
        lines.append(f"| {f} | {v:+.4f} |")
    lines.append("")
    # RF
    lines.append("## 5. Random Forest feature importance")
    lines.append("| Feature | importance |")
    lines.append("|---|---:|")
    for f, v in list(imp.get("rf_importance", {}).items())[:15]:
        lines.append(f"| {f} | {v:.4f} |")
    lines.append("")
    # Sub-score
    lines.append("## 6. Sub-score bucket correlation")
    lines.append("| Sub-score | Pearson r |")
    lines.append("|---|---:|")
    for k, d in imp.get("sub_score_pearson", {}).items():
        lines.append(f"| {k} | {d.get('pearson', 0):+.4f} |")
    lines.append("")
    # Grid
    lines.append("## 7. Grid-search optimal weights")
    if grid and grid.get("best_weights"):
        lines.append(f"**Top-10% precision**: "
                     f"{grid['best_precision_top10']:.3f} "
                     f"(baseline {grid['baseline_precision_top10']:.3f})")
        lines.append(f"**Score separation**: {grid['best_separation']:.2f}")
        lines.append("")
        lines.append("| Sub-score | Weight | Normalised |")
        lines.append("|---|---:|---:|")
        for k, v in grid["best_weights"].items():
            n = grid["best_normalized"][k]
            lines.append(f"| {k} | {v} | {n*100:.1f}% |")
    lines.append("")
    md = "\n".join(lines)
    p = os.path.join(out_dir, "backtest_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    return p


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    # 10 large caps
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    # 10 small caps (recently discovered on Toobit)
    "HYPERUSDT", "WCTUSDT", "RECALLUSDT", "GROVEUSDT", "TLMUSDT",
    "OPNUSDT", "ACEUSDT", "TUTUSDT", "SATSUSDT", "BREVUSDT",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--out", type=str, default="backtest_results")
    args = parser.parse_args()
    symbols = args.symbols or DEFAULT_SYMBOLS[: args.top]
    print(f"[bt] running backtest over {len(symbols)} symbols "
          f"for {args.days} days", flush=True)
    res = run_backtest(symbols, days=args.days, out_dir=args.out)
    if not res:
        return
    df = res["df"]
    print(f"[bt] success rate in dataset: {df['label'].mean():.3f} "
          f"({df['label'].sum()}/{len(df)})", flush=True)
    imp = analyse_importance(df, args.out)
    grid = grid_search_weights(df, args.out)
    md_path = write_markdown_report(imp, grid, args.out)
    print(f"[bt] markdown report -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
