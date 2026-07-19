"""
Precision backtest: walk-forward evaluation of the precision scanner.

Simulates running the precision scanner every 12h for N days.
For each scan:
  - Identifies high-score signals (score >= 80)
  - Records the actual 12h forward outcome
Computes:
  - Precision when score >= 80
  - Precision when score >= 70
  - Consensus-pass rate
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
from consensus import consensus_vote, consensus_score
from precision_scanner import compute_score


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


def snapshot_at(symbol, toobit, idx_1h, btc_state, btc_df_1h):
    """Take a snapshot at index idx_1h in 1h bars."""
    out = {"pass2_ok": False, "score": 0.0, "consensus_pass": False}
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
        "btc_corr": btc_corr, "smart_money": {},
    }
    consensus_pass, n, _ = consensus_vote(pack)
    out["pass2_ok"] = True
    out["score"] = compute_score(pack)
    out["consensus_pass"] = consensus_pass
    out["n_consensus"] = n
    out["rvol"] = ind_1h.get("rvol", 1.0)
    out["momentum_3_pct"] = ind_1h.get("momentum_3_pct", 0)
    out["rsi"] = tech_1h.get("rsi_value", 50)
    return out


def forward_12h_pct(df_1h, idx_1h):
    """Return peak % in the next 12 hours (12 bars)."""
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
    print(f"[prec-bt] universe: {len(universe)}", flush=True)
    btc_df_1h = toobit.get_klines("BTCUSDT", "1h", 200)
    rows = []
    print(f"[prec-bt] scanning {len(universe)} symbols, 30 days, every 12h",
          flush=True)
    for s_idx, sym in enumerate(universe, 1):
        print(f"[prec-bt] ({s_idx}/{len(universe)}) {sym}", flush=True)
        try:
            df_1h = toobit.get_klines(sym, "1h", 30 * 24 + 50)
        except Exception:
            continue
        if df_1h.empty or len(df_1h) < 50:
            continue
        # Scan every 12 bars (12h)
        for idx in range(30, len(df_1h) - 12, 12):
            pack = snapshot_at(sym, toobit, idx, btc_state, btc_df_1h)
            if not pack.get("pass2_ok"):
                continue
            out = forward_12h_pct(df_1h, idx)
            if not out:
                continue
            label = 1 if out["peak"] >= 5.0 else 0
            rec = {
                "symbol": sym,
                "label_5pct": label,
                "label_10pct": (1 if out["peak"] >= 10.0 else 0),
                "label_3pct": (1 if out["peak"] >= 3.0 else 0),
                "score": pack["score"],
                "consensus_pass": pack["consensus_pass"],
                "n_consensus": pack["n_consensus"],
                "peak": out["peak"],
                "dd": out["dd"],
                "rvol": pack.get("rvol", 1.0),
                "momentum_3_pct": pack.get("momentum_3_pct", 0),
                "rsi": pack.get("rsi", 50),
            }
            rows.append(rec)
        time.sleep(0.3)
    if not rows:
        print("[prec-bt] no data")
        return
    df = pd.DataFrame(rows)
    os.makedirs("backtest_results_precision", exist_ok=True)
    df.to_csv("backtest_results_precision/dataset.csv", index=False)
    print(f"[prec-bt] saved {len(df)} snapshots")
    print(f"[prec-bt] base rate (5%): {df['label_5pct'].mean():.3f}")
    print(f"[prec-bt] base rate (10%): {df['label_10pct'].mean():.3f}")
    print(f"[prec-bt] base rate (3%): {df['label_3pct'].mean():.3f}")
    # Precision at thresholds (using 5% pump target)
    print()
    print("[prec-bt] === PRECISION (5% PUMP TARGET) ===")
    for thr in [40, 50, 60, 70, 80]:
        sub = df[df["score"] >= thr]
        if sub.empty:
            continue
        prec = sub["label_5pct"].mean()
        n = len(sub)
        n_succ = int(sub["label_5pct"].sum())
        print(f"  score >= {thr:3d}: n={n:3d}  precision={prec:.3f}  ({n_succ}/{n})")
    print()
    print("[prec-bt] === PRECISION (10% PUMP TARGET) ===")
    for thr in [40, 50, 60, 70, 80]:
        sub = df[df["score"] >= thr]
        if sub.empty:
            continue
        prec = sub["label_10pct"].mean()
        n = len(sub)
        n_succ = int(sub["label_10pct"].sum())
        print(f"  score >= {thr:3d}: n={n:3d}  precision={prec:.3f}  ({n_succ}/{n})")
    # Consensus + score
    print()
    print("[prec-bt] === CONSENSUS + SCORE (10% TARGET) ===")
    for thr in [50, 60, 70, 80]:
        sub = df[(df["consensus_pass"]) & (df["score"] >= thr)]
        if sub.empty:
            continue
        prec = sub["label_10pct"].mean()
        n = len(sub)
        print(f"  consensus + score >= {thr}: n={n}  precision={prec:.3f}")


if __name__ == "__main__":
    main()
