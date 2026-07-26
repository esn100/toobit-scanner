"""
Adaptive ultra filter using logistic regression weights.

Combines:
  1. Global weights from all symbols (data/learned_weights.json)
  2. Per-symbol weights from per-symbol models (data/per_symbol_models.json)
  3. Smart symbol exclusion (USATUSDT never pumps)

Outputs:
  - Confidence boost/penalty based on learned feature importance
  - Per-symbol top features applied
"""
from __future__ import annotations
import os
import json
from typing import Dict, Tuple, List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GLOBAL_WEIGHTS_FILE = os.path.join(DATA_DIR, "learned_weights.json")
PER_SYMBOL_FILE = os.path.join(DATA_DIR, "per_symbol_models.json")

# Symbols that NEVER pump in 200+ snapshots
NEVER_PUMP_SYMBOLS = {"USATUSDT"}

# Symbols with high pump rate (>=25%)
PREDICTABLE_SYMBOLS = {
    "EPICUSDT", "TLMUSDT", "RESOLVUSDT", "TUTUSDT",
    "OPNUSDT", "RECALLUSDT", "FIGHTUSDT",
}


def load_global_weights() -> Dict[str, float]:
    """Load global learned weights."""
    if not os.path.exists(GLOBAL_WEIGHTS_FILE):
        return {}
    try:
        with open(GLOBAL_WEIGHTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def load_per_symbol_models() -> Dict[str, dict]:
    """Load per-symbol models."""
    if not os.path.exists(PER_SYMBOL_FILE):
        return {}
    try:
        with open(PER_SYMBOL_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def is_predictable_symbol(symbol: str) -> bool:
    """Check if symbol has historical pump rate > 20%."""
    return symbol in PREDICTABLE_SYMBOLS


def is_bad_symbol(symbol: str) -> bool:
    """Check if symbol never pumps."""
    return symbol in NEVER_PUMP_SYMBOLS


def adaptive_score(features: Dict, symbol: str = "",
                   direction: str = "LONG") -> Tuple[float, List[str]]:
    """
    Combined adaptive score (global + per-symbol).

    Returns:
      (score, reasons)
    """
    score = 0.0
    reasons = []

    # Skip bad symbols
    if is_bad_symbol(symbol):
        return -100.0, ["bad_symbol_never_pumps"]

    # === Global weights ===
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
        if abs(contrib) > 3:
            reasons.append(f"g_{feat.split('_')[-1]}({contrib:+.1f})")

    # === Per-symbol weights (stronger signal) ===
    per_sym = load_per_symbol_models()
    if symbol in per_sym:
        model = per_sym[symbol]
        sym_pump_rate = model.get("pump_rate", 0.20)
        if sym_pump_rate > 0.30:
            score += 8
            reasons.append(f"sym_high_pump({sym_pump_rate*100:.0f}%)")
        elif sym_pump_rate > 0.20:
            score += 4
            reasons.append(f"sym_good_pump({sym_pump_rate*100:.0f}%)")
        elif sym_pump_rate < 0.10:
            score -= 8
            reasons.append(f"sym_low_pump({sym_pump_rate*100:.0f}%)")
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
            if abs(contrib) > 1.5:
                reasons.append(f"s_{feat.split('_')[-1]}({contrib:+.1f})")

    return float(max(-50, min(50, score))), reasons


def get_adaptive_threshold() -> float:
    """Get current confidence threshold from self-trained params."""
    path = os.path.join(DATA_DIR, "self_trained_params.json")
    if not os.path.exists(path):
        return 60.0
    try:
        with open(path) as f:
            return float(json.load(f).get("conf_threshold", 60.0))
    except Exception:
        return 60.0


if __name__ == "__main__":
    print("="*60)
    print("ADAPTIVE FILTER STATUS")
    print("="*60)
    print(f"Global weights: {len(load_global_weights())}")
    print(f"Per-symbol models: {len(load_per_symbol_models())}")
    print(f"Never-pump symbols: {NEVER_PUMP_SYMBOLS}")
    print(f"Predictable symbols: {PREDICTABLE_SYMBOLS}")
    print(f"Threshold: {get_adaptive_threshold()}")
    print()
    print("Per-symbol top features:")
    for sym, model in load_per_symbol_models().items():
        pos = ", ".join([f"{p[0].split('_')[-1]}({p[1]:+.2f})" for p in model.get("top_positive", [])])
        print(f"  {sym:<12} pump={model.get('pump_rate', 0)*100:>5.1f}% F1={model.get('cv_f1', 0):.3f}  pos: {pos}")
