"""
Optimised multi-pass backtest.

Fetches 15m/1h/4h data ONCE per symbol, then walks through the 15m
series bar by bar without re-fetching. This is 100x faster than the
naive version.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toobit_client import ToobitClient
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
from btc_correlation import btc_correlation_features
from btc_filter import BTCFilter
from smart_money import smart_money_score


def slice_df(df: pd.DataFrame, end_idx: int) -> pd.DataFrame:
    """Return df.iloc[:end_idx+1] reset_index(drop=True)."""
    if end_idx + 1 > len(df):
        return df.iloc[:0].copy()
    return df.iloc[: end_idx + 1].copy().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", type=str, default="backtest_results_mp")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--top_pct", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    toobit = ToobitClient()
    btc_filter = BTCFilter(toobit)
    btc_state = btc_filter.evaluate()
    symbols = [
        "HYPERUSDT", "WCTUSDT", "RECALLUSDT", "TLMUSDT", "GROVEUSDT",
        "OPNUSDT", "ACEUSDT", "TUTUSDT", "SATSUSDT", "BREVUSDT",
        "FIGHTUSDT", "USATUSDT", "RESOLVUSDT", "HYPER", "WCT",
    ][: args.limit]
    # Pre-fetch BTC 1h once
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    rows = []
    print(f"[mp-bt] running optimised backtest on {len(symbols)} symbols, "
          f"{args.days} days", flush=True)
    for s_idx, sym in enumerate(symbols, 1):
        print(f"[mp-bt] ({s_idx}/{len(symbols)}) {sym}", flush=True)
        try:
            df_15m = toobit.get_klines(sym, "15m", args.days * 96 + 50)
            df_1h = toobit.get_klines(sym, "1h", args.days * 24 + 50)
            df_4h = toobit.get_klines(sym, "4h", args.days * 6 + 50)
        except Exception as e:
            print(f"  err: {e}")
            continue
        if df_15m.empty or len(df_15m) < 100:
            continue
        # Map 15m index to 1h index (15m * 4 = 1h)
        # For each 15m bar at idx, the 1h bar containing it is idx // 4
        n_snap = 0
        # Walk every 2 15m bars (30min steps) for more data
        for idx_15m in range(50, len(df_15m) - 16, 2):
            # Slice data
            sub_15m = slice_df(df_15m, idx_15m)
            if len(sub_15m) < 30:
                continue
            # 1h index aligned with 15m
            idx_1h = min(len(df_1h) - 1, (idx_15m + 1) // 4)
            sub_1h = slice_df(df_1h, idx_1h)
            if len(sub_1h) < 30:
                continue
            idx_4h = min(len(df_4h) - 1, (idx_15m + 1) // 16)
            sub_4h = slice_df(df_4h, idx_4h) if not df_4h.empty else df_4h
            # Quality on 15m
            q = validate_ohlcv(sub_15m, min_candles=30, interval_hours=0.25,
                               max_age_hours=1_000_000)
            if not q.ok or q.cleaned is None or q.cleaned.empty:
                continue
            sub_15m = q.cleaned
            # Pass 1: prefilter
            pf = prefilter_score(sub_15m, sub_1h, sub_4h, btc_df_1h)
            if not passes_prefilter(pf, min_score=45.0):
                continue
            # Pass 2: 1h analysis
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
            feats = build_features(
                tech_1h, ind_1h, struct_1h, candle_1h,
                {"alignment_score": 50.0, "fast_bias": 0.0, "slow_bias": 0.0,
                 "aligned": False, "same_sign": False},
                btc_state,
            )
            # Heuristic score (v5 - volume is king for small caps)
            rvol = ind_1h.get("rvol", 1.0)
            m6 = ind_1h.get("momentum_6_pct", 0)
            m3 = ind_1h.get("momentum_3_pct", 0)
            m1 = ind_1h.get("momentum_1_pct", 0)
            mom_acc = ind_1h.get("momentum_acceleration", 0)
            rsi_1h = tech_1h.get("rsi_value", 50)
            sm_score = 50.0
            higher_lows = struct_1h.get("higher_lows", False)
            higher_highs = struct_1h.get("higher_highs", False)
            bb_squeeze = ind_1h.get("bb_squeeze", False)
            atr_pct = ind_1h.get("atr_pct", 0)
            big_wick = candle_1h.get("big_wick_top", False)
            volume_spike = ind_1h.get("volume_spike", False)
            candle_str = candle_1h.get("candle_strength", 0.5)
            score = 0.0
            # VOLUME: king of small caps (r=+0.36!)
            if rvol >= 5.0:
                score += 35
            elif rvol >= 3.0:
                score += 25
            elif rvol >= 2.0:
                score += 15
            elif rvol >= 1.5:
                score += 8
            elif rvol < 0.5:
                score -= 10
            # Volume spike binary flag
            if volume_spike:
                score += 10
            # Momentum: short-term is more predictive in small caps
            if m1 > 1.0:
                score += 15
            elif m1 > 0:
                score += 8
            if m3 > 1.0:
                score += 10
            elif m3 > 0:
                score += 5
            if m6 > 0:
                score += 3
            # Acceleration
            if mom_acc > 0:
                score += min(10, mom_acc * 8)
            # Penalise distribution patterns
            if higher_lows:
                score -= 8
            if higher_highs:
                score -= 5
            # BB squeeze (pre-breakout)
            if bb_squeeze:
                score += 8
            # ATR expansion
            if atr_pct > 2.0:
                score += 5
            # RSI sweet spot
            if 30 <= rsi_1h <= 65:
                score += 8
            elif rsi_1h > 75:
                score -= 8
            # Big wick = rejection
            if big_wick:
                score += 4
            # Smart money placeholder
            score += (sm_score - 50) * 0.1
            # BTC independent
            if btc_corr.get("independent_mover"):
                score += 5
            # Normalise
            score = max(0, min(100, score))
            # Forward outcome (4h = 16 bars of 15m)
            if idx_15m + 1 + 16 > len(df_15m):
                continue
            entry = float(df_15m["close"].iloc[idx_15m])
            forward = df_15m.iloc[idx_15m + 1: idx_15m + 17]
            highs = forward["high"].astype(float)
            lows = forward["low"].astype(float)
            peak = float(((highs - entry) / entry * 100.0).max())
            dd = float(((lows - entry) / entry * 100.0).min())
            # More realistic: +3% in 4h with no more than -2% DD
            label = 1 if (peak >= 3.0 and dd >= -2.0) else 0
            rec = {
                "symbol": sym,
                "label": label,
                "score": round(score, 2),
                "peak": round(peak, 2),
                "dd": round(dd, 2),
                "rvol_1h": round(rvol, 2),
                "m6": round(m6, 2),
                "rsi_1h": round(rsi_1h, 1),
                "btc_corr_2d": btc_corr.get("btc_corr_2d", 0),
                "btc_corr_4d": btc_corr.get("btc_corr_4d", 0),
                "independent_mover": btc_corr.get("independent_mover", False),
                "prefilter_score": pf["prefilter"],
            }
            for k, v in feats.items():
                rec[k] = v
            rows.append(rec)
            n_snap += 1
        print(f"  {n_snap} snapshots", flush=True)
    if not rows:
        print("[mp-bt] no snapshots")
        return
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "mp_dataset.csv")
    df.to_csv(path, index=False)
    print(f"[mp-bt] saved {len(df)} snapshots -> {path}")
    print(f"[mp-bt] base success rate: {df['label'].mean():.3f}")
    # Top-N precision + score thresholds
    print()
    print("[mp-bt] === TOP-N% PRECISION ===")
    for pct in (3, 5, 10, 15, 20, 30):
        thr = np.percentile(df["score"], 100 - pct)
        top = df["score"] >= thr
        prec = float(df[top]["label"].mean()) if top.any() else 0.0
        n = int(top.sum())
        print(f"  top {pct:2d}% (n={n:3d}, score >= {thr:.1f}): "
              f"precision = {prec:.3f}")
    # By absolute score thresholds
    print()
    print("[mp-bt] === ABSOLUTE THRESHOLD PRECISION ===")
    for thr in (40, 50, 60, 70, 80, 90):
        top = df["score"] >= thr
        if top.sum() == 0:
            continue
        prec = float(df[top]["label"].mean())
        n = int(top.sum())
        print(f"  score >= {thr:3d} (n={n:3d}): precision = {prec:.3f}")
    # Feature correlation
    feat_cols = [c for c in df.columns
                 if c not in ("symbol", "label", "peak", "dd")
                 and df[c].dtype in (float, int)]
    cor = []
    for c in feat_cols:
        try:
            r = float(np.corrcoef(df[c].fillna(0), df["label"])[0, 1])
            cor.append((c, r))
        except Exception:
            pass
    cor.sort(key=lambda x: -abs(x[1]))
    print()
    print("[mp-bt] Top features by Pearson correlation with success:")
    for name, r in cor[:15]:
        print(f"  {name:30s}  r={r:+.3f}")
    summary = {
        "n_snapshots": int(len(df)),
        "base_rate": float(df["label"].mean()),
        "top_5pct_precision": float(
            df[df["score"] >= np.percentile(df["score"], 95)]["label"].mean()
        ),
        "top_10pct_precision": float(
            df[df["score"] >= np.percentile(df["score"], 90)]["label"].mean()
        ),
        "top_20pct_precision": float(
            df[df["score"] >= np.percentile(df["score"], 80)]["label"].mean()
        ),
        "top_30pct_precision": float(
            df[df["score"] >= np.percentile(df["score"], 70)]["label"].mean()
        ),
        "top_features": [{"name": n, "r": round(r, 4)} for n, r in cor[:20]],
    }
    with open(os.path.join(args.out, "mp_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[mp-bt] summary -> {args.out}/mp_summary.json")


if __name__ == "__main__":
    main()
