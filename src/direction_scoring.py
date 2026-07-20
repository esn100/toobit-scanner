"""
Direction-aware scoring for PumpHunter.

Given a feature dict (the keys we collect in live_collector.py), produce:
  - long_score:    0..100, how strongly this looks like a pump setup
  - short_score:   0..100, how strongly this looks like a dump setup
  - long_signals:  number of individual features voting LONG
  - short_signals: number of individual features voting SHORT
  - direction:     'LONG' | 'SHORT' | 'NEUTRAL' (best of two)
  - confidence:    max(long_score, short_score)

Rules are derived from the correlation analysis (24-cycle proxy run,
n=136). The thresholds are conservative — we want precision > recall.

For each rule we record:
  - weight: contribution to score (out of 100)
  - fired:  bool
  - reason: short text for the dashboard

When you get more labeled data, retune the thresholds in
`update_thresholds_from_data()` — that's the only function that should
change.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# ---------------------------------------------------------------------------
# Configuration — single source of truth, easy to retune later
# ---------------------------------------------------------------------------
# Each rule is (weight, predicate(features_dict) -> bool, label)
LONG_RULES: List[dict] = [
    # Volume / volatility
    {"weight": 15, "label": "atr_expanding",
     "check": lambda f: bool(f.get("f_atr_expanding", 0))},
    {"weight": 12, "label": "atr_pct>4",
     "check": lambda f: float(f.get("f_atr_pct", 0)) > 4.0},
    {"weight": 10, "label": "rvol>1.3",
     "check": lambda f: float(f.get("f_rvol", 1.0)) > 1.3},
    {"weight": 8,  "label": "5m_volume_spike",
     "check": lambda f: bool(f.get("f_m_5m_volume_spike", 0))},
    # Momentum
    {"weight": 15, "label": "momentum_6>5%",
     "check": lambda f: float(f.get("f_momentum_6_pct", 0)) > 5.0},
    {"weight": 10, "label": "momentum_3>2%",
     "check": lambda f: float(f.get("f_momentum_3_pct", 0)) > 2.0},
    {"weight": 8,  "label": "momentum_12>0",
     "check": lambda f: float(f.get("f_momentum_12_pct", 0)) > 0},
    # Structure
    {"weight": 10, "label": "vwap_above",
     "check": lambda f: float(f.get("f_vwap_distance_pct", 0)) > 0.5},
    {"weight": 8,  "label": "bos_up",
     "check": lambda f: bool(f.get("f_bos_up", 0))},
    {"weight": 8,  "label": "bb_breakout_above",
     "check": lambda f: bool(f.get("f_bb_breakout_above", 0))},
    # Ichimoku
    {"weight": 8,  "label": "ichi_above_cloud",
     "check": lambda f: bool(f.get("f_a_ichi_above_cloud", 0))},
    {"weight": 5,  "label": "ichi_thick_cloud",
     "check": lambda f: float(f.get("f_a_ichi_thickness_pct", 0)) > 3.0},
    # Fibonacci (close to support = bounce)
    {"weight": 8,  "label": "fib_dist_0.618<8%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.618", 99)) < 8.0},
    {"weight": 5,  "label": "fib_dist_0.5<8%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.500", 99)) < 8.0},
    # RSI
    {"weight": 5,  "label": "rsi_50_70",
     "check": lambda f: 50 <= float(f.get("f_rsi_value", 50)) <= 70},
    # Anti-signals (these *reduce* LONG score if present)
    {"weight": -10, "label": "NO ichi_below_cloud",
     "check": lambda f: not bool(f.get("f_a_ichi_below_cloud", 0))},
    {"weight": -8,  "label": "NO in_range",
     "check": lambda f: not bool(f.get("f_in_range", 0))},
    {"weight": -8,  "label": "NO fib_far_0.786",
     "check": lambda f: float(f.get("f_a_fib_dist_0.786", 99)) < 30.0},
    # Elliott (only if detected as bullish)
    {"weight": 5,  "label": "elliott_uptrend",
     "check": lambda f: bool(f.get("f_a_is_uptrend", 0))},
]

SHORT_RULES: List[dict] = [
    # Volume / volatility
    {"weight": 12, "label": "atr_pct>4",
     "check": lambda f: float(f.get("f_atr_pct", 0)) > 4.0},
    {"weight": 10, "label": "atr_expanding",
     "check": lambda f: bool(f.get("f_atr_expanding", 0))},
    # Fibonacci (far from fib = extended move about to reverse)
    {"weight": 15, "label": "fib_dist_0.786>50%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.786", 99)) > 50.0},
    {"weight": 12, "label": "fib_dist_0.618>50%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.618", 99)) > 50.0},
    {"weight": 10, "label": "fib_dist_0.5>50%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.500", 99)) > 50.0},
    {"weight": 8,  "label": "fib_dist_0.382>30%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.382", 99)) > 30.0},
    # Momentum negative
    {"weight": 12, "label": "momentum_6<-3%",
     "check": lambda f: float(f.get("f_momentum_6_pct", 0)) < -3.0},
    {"weight": 10, "label": "momentum_3<-2%",
     "check": lambda f: float(f.get("f_momentum_3_pct", 0)) < -2.0},
    {"weight": 8,  "label": "momentum_accel<-5",
     "check": lambda f: float(f.get("f_momentum_acceleration", 0)) < -5.0},
    # Structure
    {"weight": 8,  "label": "vwap_below",
     "check": lambda f: float(f.get("f_vwap_distance_pct", 0)) < -0.5},
    # Ichimoku
    {"weight": 10, "label": "ichi_below_cloud",
     "check": lambda f: bool(f.get("f_a_ichi_below_cloud", 0))},
    {"weight": 5,  "label": "ichi_red_cloud",
     "check": lambda f: str(f.get("f_a_ichi_cloud_color", "")) == "red"},
    # BTC independent mover (small caps that diverge from BTC often dump)
    {"weight": 5,  "label": "btc_independent",
     "check": lambda f: bool(f.get("f_btc_independent_mover", 0))},
    # Anti-signals
    {"weight": -8,  "label": "NO ichi_above_cloud",
     "check": lambda f: not bool(f.get("f_a_ichi_above_cloud", 0))},
    {"weight": -8,  "label": "NO bos_up",
     "check": lambda f: not bool(f.get("f_bos_up", 0))},
    {"weight": -6,  "label": "NO bb_breakout_above",
     "check": lambda f: not bool(f.get("f_bb_breakout_above", 0))},
    # Elliott
    {"weight": 5,  "label": "elliott_downtrend",
     "check": lambda f: (bool(f.get("f_a_is_uptrend", 0)) == False
                          and str(f.get("f_a_wave", "")) in
                          ("impulse_5_down", "correction"))},
]


@dataclass
class DirectionResult:
    symbol: str
    long_score: float
    short_score: float
    long_signals: int
    short_signals: int
    direction: str  # 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence: float
    long_fired: List[str] = field(default_factory=list)
    short_fired: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def score_features(features: Dict, rules: List[dict]) -> tuple:
    """
    Apply each rule, sum the weights of fired rules, and return:
      (score 0..100, fired_count, fired_labels)
    """
    score = 0.0
    fired = []
    for r in rules:
        try:
            if r["check"](features):
                score += r["weight"]
                fired.append(r["label"])
        except Exception:
            # Don't let a single bad feature break the whole scoring
            continue
    # Clip to [0, 100]
    score = max(0.0, min(100.0, score))
    return score, len(fired), fired


def score_long(features: Dict) -> tuple:
    return score_features(features, LONG_RULES)


def score_short(features: Dict) -> tuple:
    return score_features(features, SHORT_RULES)


def direction_score(features: Dict, symbol: str = "") -> DirectionResult:
    """
    Compute both LONG and SHORT scores, then decide direction.
    """
    long_score, long_n, long_fired = score_long(features)
    short_score, short_n, short_fired = score_short(features)
    if long_score > short_score and long_score >= 25:
        direction = "LONG"
    elif short_score > long_score and short_score >= 25:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    return DirectionResult(
        symbol=symbol,
        long_score=round(long_score, 1),
        short_score=round(short_score, 1),
        long_signals=long_n,
        short_signals=short_n,
        direction=direction,
        confidence=round(max(long_score, short_score), 1),
        long_fired=long_fired,
        short_fired=short_fired,
    )


def update_thresholds_from_data(df, target_col: str = "label",
                                current_rules: str = "LONG") -> List[dict]:
    """
    Placeholder: in the morning, after we have ~1200 labeled samples,
    we'll retune rule weights using logistic regression on (rule_fired)
    matrix -> outcome. The function will:
      1. for each rule, compute its (precision, recall, lift)
      2. if lift > 1.5 keep, if < 0.8 invert, else drop
      3. recompute weights from log-odds of each surviving rule
    Returns the new rules list.
    """
    # TODO: implement once we have labels
    raise NotImplementedError(
        "Wait for ~1200 labeled samples. Run:\n"
        "  python -m src.tune_rules --target label_LONG"
    )
