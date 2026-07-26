"""
Multi-timeframe composite features for PumpHunter-AI.

Combines features from multiple timeframes (5m, 15m, 1h, 4h) into
a single signal. Key insight: pump on 4h often shows precursors on
5m and 15m BEFORE the 4h candle moves.

Features:
  - MTF momentum alignment (do all timeframes agree on direction?)
  - MTF volume surge (5m/15m volume vs 4h baseline)
  - MTF volatility compression
  - MTF RSI divergence across timeframes
  - MTF BB breakout alignment
  - Cross-timeframe correlation features
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import (vwap_features, atr_features, bollinger_features,
                            relative_volume, momentum_features, atr as atr_func)
from src.technical import technical_analysis
from src.candle_quality import candle_quality_features
from src.extended_indicators import compute_all_extended
from src.toobit_client import ToobitClient


def mtf_composite_features(df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                            df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Dict:
    """
    Compute multi-timeframe composite features.

    Returns a dict with `mtf_` prefixed features.
    """
    out = {}
    try:
        # 1) MTF momentum alignment
        # All timeframes should agree on direction for highest conviction
        m5 = float(df_5m["close"].iloc[-1] - df_5m["close"].iloc[-4]) / max(df_5m["close"].iloc[-4], 1e-12) * 100
        m15 = float(df_15m["close"].iloc[-1] - df_15m["close"].iloc[-5]) / max(df_15m["close"].iloc[-5], 1e-12) * 100
        m1h = float(df_1h["close"].iloc[-1] - df_1h["close"].iloc[-4]) / max(df_1h["close"].iloc[-4], 1e-12) * 100
        m4h = float(df_4h["close"].iloc[-1] - df_4h["close"].iloc[-4]) / max(df_4h["close"].iloc[-4], 1e-12) * 100
        out["mtf_mom_5m"] = m5
        out["mtf_mom_15m"] = m15
        out["mtf_mom_1h"] = m1h
        out["mtf_mom_4h"] = m4h
        # Alignment: all positive = 1, all negative = -1, mixed = 0
        all_pos = m5 > 0 and m15 > 0 and m1h > 0 and m4h > 0
        all_neg = m5 < 0 and m15 < 0 and m1h < 0 and m4h < 0
        if all_pos:
            out["mtf_alignment"] = 1
        elif all_neg:
            out["mtf_alignment"] = -1
        else:
            out["mtf_alignment"] = 0
        # Short-term dominance: 5m vs 4h
        out["mtf_mom_divergence"] = float(m5 - m4h)
        # Average momentum
        out["mtf_mom_avg"] = float(np.mean([m5, m15, m1h, m4h]))
    except Exception:
        out["mtf_mom_5m"] = 0
        out["mtf_mom_15m"] = 0
        out["mtf_mom_1h"] = 0
        out["mtf_mom_4h"] = 0
        out["mtf_alignment"] = 0
        out["mtf_mom_divergence"] = 0
        out["mtf_mom_avg"] = 0

    # 2) MTF volume surge
    try:
        v5 = float(df_5m["volume"].iloc[-1])
        v5_avg = float(df_5m["volume"].iloc[-20:].mean())
        v15 = float(df_15m["volume"].iloc[-1])
        v15_avg = float(df_15m["volume"].iloc[-15:].mean())
        v1h = float(df_1h["volume"].iloc[-1])
        v1h_avg = float(df_1h["volume"].iloc[-15:].mean())
        v4h = float(df_4h["volume"].iloc[-1])
        v4h_avg = float(df_4h["volume"].iloc[-10:].mean())
        out["mtf_vol_rvol_5m"] = float(v5 / max(v5_avg, 1e-9))
        out["mtf_vol_rvol_15m"] = float(v15 / max(v15_avg, 1e-9))
        out["mtf_vol_rvol_1h"] = float(v1h / max(v1h_avg, 1e-9))
        out["mtf_vol_rvol_4h"] = float(v4h / max(v4h_avg, 1e-9))
        # Volume alignment (5m and 15m both above average = strong)
        out["mtf_vol_surge"] = int(
            out["mtf_vol_rvol_5m"] > 1.5 and out["mtf_vol_rvol_15m"] > 1.3
        )
    except Exception:
        out["mtf_vol_rvol_5m"] = 1
        out["mtf_vol_rvol_15m"] = 1
        out["mtf_vol_rvol_1h"] = 1
        out["mtf_vol_rvol_4h"] = 1
        out["mtf_vol_surge"] = 0

    # 3) MTF volatility compression
    try:
        atr5 = float(atr_func(df_5m, 14).iloc[-1])
        atr15 = float(atr_func(df_15m, 14).iloc[-1])
        atr1h = float(atr_func(df_1h, 14).iloc[-1])
        atr4h = float(atr_func(df_4h, 14).iloc[-1])
        out["mtf_atr_5m"] = atr5 / max(df_5m["close"].iloc[-1], 1e-12) * 100
        out["mtf_atr_15m"] = atr15 / max(df_15m["close"].iloc[-1], 1e-12) * 100
        out["mtf_atr_1h"] = atr1h / max(df_1h["close"].iloc[-1], 1e-12) * 100
        out["mtf_atr_4h"] = atr4h / max(df_4h["close"].iloc[-1], 1e-12) * 100
        # Compression: short-term ATR much lower than long-term = squeeze
        out["mtf_vol_compression"] = float(out["mtf_atr_5m"] / max(out["mtf_atr_4h"], 1e-9))
    except Exception:
        out["mtf_atr_5m"] = 0
        out["mtf_atr_15m"] = 0
        out["mtf_atr_1h"] = 0
        out["mtf_atr_4h"] = 0
        out["mtf_vol_compression"] = 1

    # 4) MTF BB breakout alignment
    try:
        # 5m BB position
        bb5 = bollinger_features(df_5m, 20, 2.0)
        bb15 = bollinger_features(df_15m, 20, 2.0)
        bb1h = bollinger_features(df_1h, 20, 2.0)
        bb4h = bollinger_features(df_4h, 20, 2.0)
        # Use width as feature
        out["mtf_bb_width_5m"] = bb5.get("bb_width_pct", 0)
        out["mtf_bb_width_15m"] = bb15.get("bb_width_pct", 0)
        out["mtf_bb_width_1h"] = bb1h.get("bb_width_pct", 0)
        out["mtf_bb_width_4h"] = bb4h.get("bb_width_pct", 0)
        out["mtf_bb_breakout_5m"] = int(bb5.get("bb_breakout_above", False))
        out["mtf_bb_breakout_15m"] = int(bb15.get("bb_breakout_above", False))
        out["mtf_bb_breakout_1h"] = int(bb1h.get("bb_breakout_above", False))
        out["mtf_bb_breakout_4h"] = int(bb4h.get("bb_breakout_above", False))
        out["mtf_bb_breakout_aligned"] = int(
            out["mtf_bb_breakout_5m"] and out["mtf_bb_breakout_15m"]
        )
    except Exception:
        for k in ["mtf_bb_width_5m", "mtf_bb_width_15m", "mtf_bb_width_1h", "mtf_bb_width_4h",
                  "mtf_bb_breakout_5m", "mtf_bb_breakout_15m", "mtf_bb_breakout_1h", "mtf_bb_breakout_4h",
                  "mtf_bb_breakout_aligned"]:
            out[k] = 0

    # 5) MTF RSI divergence
    try:
        from src.technical import _rsi
        rsi5 = float(_rsi(df_5m["close"], 14).iloc[-1])
        rsi15 = float(_rsi(df_15m["close"], 14).iloc[-1])
        rsi1h = float(_rsi(df_1h["close"], 14).iloc[-1])
        rsi4h = float(_rsi(df_4h["close"], 14).iloc[-1])
        out["mtf_rsi_5m"] = rsi5
        out["mtf_rsi_15m"] = rsi15
        out["mtf_rsi_1h"] = rsi1h
        out["mtf_rsi_4h"] = rsi4h
        # All oversold/overbought?
        all_oversold = rsi5 < 30 and rsi15 < 30 and rsi1h < 35
        all_overbought = rsi5 > 70 and rsi15 > 70 and rsi1h > 65
        out["mtf_rsi_all_oversold"] = int(all_oversold)
        out["mtf_rsi_all_overbought"] = int(all_overbought)
    except Exception:
        for k in ["mtf_rsi_5m", "mtf_rsi_15m", "mtf_rsi_1h", "mtf_rsi_4h",
                  "mtf_rsi_all_oversold", "mtf_rsi_all_overbought"]:
            out[k] = 50 if "rsi" in k and "all" not in k else 0

    # 6) Composite MTF signal (0-100)
    score = 50.0
    if out["mtf_alignment"] == 1:
        score += 20
    elif out["mtf_alignment"] == -1:
        score -= 20
    if out["mtf_vol_surge"]:
        score += 15
    if out["mtf_bb_breakout_aligned"]:
        score += 10
    if out["mtf_rsi_all_oversold"]:
        score += 10
    if out["mtf_rsi_all_overbought"]:
        score -= 10
    if out["mtf_vol_compression"] < 0.3:
        score += 5  # squeeze building
    out["mtf_composite_score"] = float(max(0, min(100, score)))
    return out


def collect_mtf_dataset(symbols: list, days: int = 60,
                        out_path: str = "data/mtf_dataset.csv"):
    """Collect multi-timeframe features + forward returns for ML."""
    import time
    tb = ToobitClient()
    rows = []
    for sym in symbols:
        try:
            df_5m = tb.get_klines(sym, "5m", days * 288)
            df_15m = tb.get_klines(sym, "15m", days * 96)
            df_1h = tb.get_klines(sym, "1h", days * 24)
            df_4h = tb.get_klines(sym, "4h", days * 6)
        except Exception as e:
            print(f"  {sym}: ERR {e}")
            continue
        if df_4h.empty or len(df_4h) < 100:
            continue
        # For each 4h snapshot, compute MTF features
        max_idx = len(df_4h) - 4
        n = 0
        for idx in range(50, max(max_idx, 51)):
            if idx % 20 == 0:
                print(f"  {sym} idx {idx}/{max_idx}")
            # Get corresponding 15m/5m/1h sub-frames
            ts_4h = df_4h["open_time"].iloc[idx]
            # 1h: idx hours back
            n_1h = min(len(df_1h) - 1, (idx + 1) * 4)
            sub_1h = df_1h.iloc[:n_1h + 1].copy()
            # 15m
            n_15m = min(len(df_15m) - 1, (idx + 1) * 16)
            sub_15m = df_15m.iloc[:n_15m + 1].copy()
            # 5m
            n_5m = min(len(df_5m) - 1, (idx + 1) * 48)
            sub_5m = df_5m.iloc[:n_5m + 1].copy()
            if len(sub_5m) < 30 or len(sub_15m) < 20 or len(sub_1h) < 20:
                continue
            sub_4h = df_4h.iloc[:idx + 1].copy()
            # Forward returns
            fwd_4h = float(df_4h["close"].iloc[idx + 1]) if idx + 1 < len(df_4h) else float(sub_4h["close"].iloc[-1])
            fwd_12h = float(df_4h["close"].iloc[idx + 4]) if idx + 4 < len(df_4h) else float(sub_4h["close"].iloc[-1])
            cur = float(sub_4h["close"].iloc[-1])
            mtf = mtf_composite_features(sub_5m, sub_15m, sub_1h, sub_4h)
            row = {
                "symbol": sym,
                "idx": idx,
                "price": cur,
                "fwd_4h": (fwd_4h - cur) / cur * 100,
                "fwd_12h": (fwd_12h - cur) / cur * 100,
            }
            row.update(mtf)
            rows.append(row)
            n += 1
        print(f"  {sym}: {n} snapshots")
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    if not df.empty:
        print(f"Pump rate (>3% in 12h): {(df['fwd_12h'] > 3).mean()*100:.1f}%")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--out", type=str, default="data/mtf_dataset.csv")
    args = p.parse_args()
    symbols = args.symbols or [
        "EPICUSDT", "OPNUSDT", "RECALLUSDT", "TLMUSDT",
        "RESOLVUSDT", "TUTUSDT", "FIGHTUSDT"
    ]
    collect_mtf_dataset(symbols, days=args.days, out_path=args.out)
