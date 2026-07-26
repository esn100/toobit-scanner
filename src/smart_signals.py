"""
Smart signal generator using per-symbol optimized parameters.

Reads:
  - data/per_symbol_strategy.json (optimal TP/SL per symbol)
  - data/per_symbol_models_v2.json (per-symbol LR models)
  - data/learned_weights_v2.json (global weights)

Generates signals with:
  - Per-symbol TP/SL (optimized from backtest)
  - Adaptive scoring (global + per-symbol)
  - Symbol exclusion (bad performers)
  - Predictable symbol bonus
"""
from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.toobit_client import ToobitClient
from src.universe import discover_small_caps, MarketCapResolver
from src.indicators import (vwap_features, atr_features, bollinger_features,
                           relative_volume, momentum_features)
from src.technical import technical_analysis
from src.candle_quality import candle_quality_features
from src.btc_filter import BTCFilter
from src.extended_indicators import compute_all_extended
from src.anti_late import check_anti_late

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_per_symbol_strategy() -> Dict:
    path = os.path.join(DATA_DIR, "per_symbol_strategy.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_per_symbol_models() -> Dict:
    path = os.path.join(DATA_DIR, "per_symbol_models_v2.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_global_weights() -> Dict[str, float]:
    path = os.path.join(DATA_DIR, "learned_weights_v2.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
            return data.get("weights", {})
    except Exception:
        return {}


# Bad symbols (0% pump rate historically)
BAD_SYMBOLS = {"USATUSDT"}

# Predictable symbols (>=20% historical pump rate)
PREDICTABLE_SYMBOLS = {"EPICUSDT", "TLMUSDT", "OPNUSDT", "RECALLUSDT", "RESOLVUSDT",
                       "TUTUSDT", "FIGHTUSDT"}


def smart_signal_score(features: Dict, symbol: str, direction: str) -> Tuple[float, List[str]]:
    """
    Compute smart score using global + per-symbol weights.

    Returns (score_delta, reasons).
    """
    score = 0.0
    reasons = []

    # Global weights
    global_w = load_global_weights()
    for feat, w in global_w.items():
        v = features.get(feat, features.get(f"f_{feat}", 0))
        if v is None:
            continue
        try:
            v_float = float(v)
        except Exception:
            continue
        if direction == "LONG":
            contrib = v_float * w / 10.0
        else:
            contrib = -v_float * w / 10.0
        contrib = max(-5, min(5, contrib))
        score += contrib
        if abs(contrib) > 2:
            reasons.append(f"g_{feat.split('_')[-1]}({contrib:+.1f})")

    # Per-symbol model
    models = load_per_symbol_models()
    if symbol in models:
        model = models[symbol]
        sym_pump = model.get("pump_rate", 0.20)
        if sym_pump > 0.30:
            score += 10
            reasons.append(f"sym_pump={sym_pump*100:.0f}%")
        elif sym_pump > 0.20:
            score += 5
            reasons.append(f"sym_pump={sym_pump*100:.0f}%")
        # Apply per-symbol top features
        for feat, w in model.get("top_positive", []):
            v = features.get(feat, features.get(f"f_{feat}", 0))
            if v is None:
                continue
            try:
                v_float = float(v)
            except Exception:
                continue
            if direction == "LONG":
                contrib = v_float * w / 10.0
            else:
                contrib = -v_float * w / 10.0
            contrib = max(-3, min(3, contrib))
            score += contrib
            if abs(contrib) > 1.5:
                reasons.append(f"s_{feat.split('_')[-1]}({contrib:+.1f})")
        for feat, w in model.get("top_negative", []):
            v = features.get(feat, features.get(f"f_{feat}", 0))
            if v is None:
                continue
            try:
                v_float = float(v)
            except Exception:
                continue
            if direction == "LONG":
                contrib = v_float * w / 10.0
            else:
                contrib = -v_float * w / 10.0
            contrib = max(-3, min(3, contrib))
            score += contrib

    return float(max(-50, min(50, score))), reasons


def get_smart_params(symbol: str) -> Dict:
    """Get optimized TP/SL for symbol."""
    strategies = load_per_symbol_strategy()
    if symbol in strategies:
        s = strategies[symbol]
        return {
            "tp_pct": s.get("tp", 5.0),
            "sl_pct": s.get("sl", 1.5),
            "wr": s.get("wr", 0.35),
        }
    # Default
    return {"tp_pct": 5.0, "sl_pct": 1.5, "wr": 0.35}


def find_signals(min_score: float = 55.0) -> List[Dict]:
    """
    Scan universe and return valid signals with per-symbol params.
    """
    tb = ToobitClient()
    btc = BTCFilter(tb).evaluate()
    btc_1h = tb.get_klines("BTCUSDT", "1h", 100)
    resolver = MarketCapResolver(cache_dir="/home/user/toobit-scanner/data")
    tickers = tb.get_24h_tickers()
    small_caps = discover_small_caps(tickers, resolver, max_symbols=20)
    symbols = [s for s in small_caps['symbol'].tolist() if s not in BAD_SYMBOLS]
    signals = []
    for sym in symbols:
        try:
            df_4h = tb.get_klines(sym, '4h', 100)
            df_1h = tb.get_klines(sym, '1h', 100)
            df_15m = tb.get_klines(sym, '15m', 50)
            df_5m = tb.get_klines(sym, '5m', 50)
            if df_4h.empty or len(df_4h) < 60:
                continue
            ind = {}
            ind.update(vwap_features(df_4h))
            ind.update(atr_features(df_4h))
            ind.update(bollinger_features(df_4h))
            ind.update(relative_volume(df_4h))
            ind.update(momentum_features(df_4h))
            tech = technical_analysis(df_4h)
            candle = candle_quality_features(df_4h)
            ex = compute_all_extended(df_4h)
            mom_3 = ind.get('momentum_3_pct', 0)
            mom_6 = ind.get('momentum_6_pct', 0)
            atr = ind.get('atr_pct', 0)
            rvol = ind.get('rvol', 1)
            if mom_3 > 0.5 and mom_6 > 0:
                direction = 'LONG'
            elif mom_3 < -0.5 and mom_6 < 0:
                direction = 'SHORT'
            else:
                continue
            # Anti-late
            anti_feats = {
                'momentum_1_pct': ind.get('momentum_1_pct', 0),
                'momentum_3_pct': mom_3,
                'candle_strength': candle.get('candle_strength', 0.5),
                'big_wick_top': candle.get('big_wick_top', False),
            }
            al_pass, _, _ = check_anti_late(anti_feats, direction)
            if not al_pass:
                continue
            # Build features
            feats = {
                'f_atr_pct': atr, 'f_momentum_3_pct': mom_3, 'f_momentum_6_pct': mom_6,
                'f_momentum_12_pct': ind.get('momentum_12_pct', 0),
                'f_momentum_1_pct': ind.get('momentum_1_pct', 0),
                'f_momentum_acceleration': ind.get('momentum_acceleration', 0),
                'f_rvol': rvol, 'confidence': tech.get('technical_score', 50),
            }
            for k, v in ind.items():
                feats[f'ind_{k}'] = float(v) if not isinstance(v, bool) else int(v)
            for k, v in candle.items():
                feats[f'candle_{k}'] = float(v) if not isinstance(v, bool) else int(v)
            for k, v in ex.items():
                feats[f'ex_{k}'] = float(v) if not isinstance(v, bool) else int(v)
            for k, v in tech.items():
                if isinstance(v, (int, float, bool)):
                    feats[f'tech_{k}'] = v
            # Smart score
            smart, reasons = smart_signal_score(feats, sym, direction)
            base_conf = feats['confidence']
            adj_conf = base_conf + smart
            if adj_conf < min_score:
                continue
            # Per-symbol TP/SL
            params = get_smart_params(sym)
            entry = float(df_4h['close'].iloc[-1])
            tp_price = entry * (1 + params['tp_pct']/100) if direction == 'LONG' else entry * (1 - params['tp_pct']/100)
            sl_price = entry * (1 - params['sl_pct']/100) if direction == 'LONG' else entry * (1 + params['sl_pct']/100)
            signals.append({
                'symbol': sym,
                'direction': direction,
                'entry': entry,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_pct': params['tp_pct'],
                'sl_pct': params['sl_pct'],
                'base_conf': base_conf,
                'smart_score': smart,
                'adj_conf': adj_conf,
                'mom_3': mom_3,
                'mom_6': mom_6,
                'atr': atr,
                'rvol': rvol,
                'historical_wr': params['wr'],
                'reasons': reasons[:5],
            })
            time.sleep(0.2)
        except Exception as e:
            time.sleep(0.3)
    signals.sort(key=lambda x: x['adj_conf'], reverse=True)
    return signals


if __name__ == "__main__":
    print("="*100)
    print("SMART SIGNALS — Per-Symbol Optimized")
    print("="*100)
    signals = find_signals(min_score=50.0)
    if not signals:
        print("\nNo signals found.")
    else:
        for s in signals:
            print(f"\n{s['symbol']} {s['direction']}")
            print(f"   Entry: {s['entry']:.6f}")
            print(f"   TP:    {s['tp_price']:.6f} (+{s['tp_pct']}%)")
            print(f"   SL:    {s['sl_price']:.6f} (-{s['sl_pct']}%)")
            print(f"   Conf:  {s['adj_conf']:.0f} (base {s['base_conf']:.0f} + smart {s['smart_score']:+.0f})")
            print(f"   Hist WR: {s['historical_wr']*100:.0f}%")
            print(f"   Mom3: {s['mom_3']:+.1f}%, RVOL: {s['rvol']:.2f}")
            print(f"   Reasons: {', '.join(s['reasons'])}")
