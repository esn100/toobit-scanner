"""
Direction-aware scoring for PumpHunter v2.

Key insight from data analysis (n=272, 8 cycles):
  - In small caps, pumps have NO reliable 4h leading indicator
  - 0 "early pump" samples (mom_6<2% but pump happens next cycle)
  - mom_6, mom_12, atr_expanding are LAGGING (fire AFTER pump started)
  - Real edge comes from:
      (a) FAST detection via 5m microstructure (precision ~50%)
      (b) ANTI-LATE filter: reject signals on overextended moves
      (c) Leading signals in CALM regime (small but real edge)

Rules organized as:
  1. PRIMARY (anti-late + structural): the core filter
  2. SECONDARY (leading): adds confidence in calm regime
  3. TERTIARY (5m microstructure): confirms fast move
  4. ANTI-SIGNALS: subtract weight when contradictions

The hard lesson: we previously scored +85 on BANKUSDT/TLM/ZEREBRO
even though those pumps were already over. Now score is capped when
overextended (mom_3 > 30%) and weighted toward confirmation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# ---------------------------------------------------------------------------
# Anti-late thresholds — derived from 8-cycle analysis
# ---------------------------------------------------------------------------
# mom_3 > 20% means pump is mature; expect mean-reversion
MOM_3_OVEREXTENDED_LONG = 20.0
# mom_3 < -20% means dump is mature; expect bounce
MOM_3_OVEREXTENDED_SHORT = -20.0


# ---------------------------------------------------------------------------
# LONG rules — emphasize EARLY detection and ANTI-LATE
# ---------------------------------------------------------------------------
LONG_RULES: List[dict] = [
    # === TIER 1: anti-late guard (must pass) ===
    # Pump is NOT overextended yet (still room to run)
    # Rule FIRES (penalty) only when overextended
    {"weight": -100, "label": "FAIL_overextended",
     "check": lambda f: float(f.get("f_momentum_3_pct", 0)) >= MOM_3_OVEREXTENDED_LONG},
    # === TIER 2: leading in calm regime ===
    # In calm (|mom_6| < 5%), momentum_acceleration > 0 = early pump setup
    {"weight": 20, "label": "early_calm_accel",
     "check": lambda f: (abs(float(f.get("f_momentum_6_pct", 0))) < 5
                          and float(f.get("f_momentum_acceleration", 0)) > 0)},
    # ATR expanding (volatility just started = leading, not lagging)
    {"weight": 15, "label": "atr_just_expanding",
     "check": lambda f: bool(f.get("f_atr_expanding", 0))},
    # === TIER 3: structural confirmation ===
    # Above Ichimoku cloud (trend support)
    {"weight": 12, "label": "ichi_above_cloud",
     "check": lambda f: bool(f.get("f_a_ichi_above_cloud", 0))},
    # Break of structure (real move starting)
    {"weight": 10, "label": "bos_up",
     "check": lambda f: bool(f.get("f_bos_up", 0))},
    # BB breakout
    {"weight": 8, "label": "bb_breakout_above",
     "check": lambda f: bool(f.get("f_bb_breakout_above", 0))},
    # Above VWAP
    {"weight": 8, "label": "vwap_above",
     "check": lambda f: float(f.get("f_vwap_distance_pct", 0)) > 0.5},
    # === TIER 4: 5m microstructure (fast detection) ===
    {"weight": 15, "label": "5m_volume_spike",
     "check": lambda f: bool(f.get("f_m_5m_volume_spike", 0))},
    {"weight": 10, "label": "5m_rvol>4",
     "check": lambda f: float(f.get("f_m_5m_rvol", 0)) > 4.0},
    # OBI bullish (bids > asks)
    {"weight": 8, "label": "obi_bullish",
     "check": lambda f: bool(f.get("f_m_obi_bullish", 0))},
    # === TIER 5: volume context ===
    {"weight": 8, "label": "rvol>1.3",
     "check": lambda f: float(f.get("f_rvol", 1.0)) > 1.3},
    {"weight": 6, "label": "atr_pct_elevated",
     "check": lambda f: float(f.get("f_atr_pct", 0)) > 3.0},
    # === TIER 6: Fib proximity (bounce zone) ===
    {"weight": 8, "label": "fib_dist_0.618<8%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.618", 99)) < 8.0},
    {"weight": 5, "label": "fib_dist_0.5<8%",
     "check": lambda f: float(f.get("f_a_fib_dist_0.500", 99)) < 8.0},
    # === TIER 7: momentum (lagging — but small weight) ===
    {"weight": 5, "label": "mom_3_low_positive",
     "check": lambda f: (0 < float(f.get("f_momentum_3_pct", 0)) < 10)},
    {"weight": 3, "label": "mom_12_positive",
     "check": lambda f: float(f.get("f_momentum_12_pct", 0)) > 0},
    # === TIER 8: anti-signals (deductions) ===
    {"weight": -10, "label": "NO_ichi_below",
     "check": lambda f: not bool(f.get("f_a_ichi_below_cloud", 0))},
    {"weight": -8, "label": "NO_obi_bearish",
     "check": lambda f: not bool(f.get("f_m_obi_bearish", 0))},
    {"weight": -5, "label": "rsi_50_70",
     "check": lambda f: 50 <= float(f.get("f_rsi_value", 50)) <= 70},
]


# ---------------------------------------------------------------------------
# SHORT rules
# ---------------------------------------------------------------------------
SHORT_RULES: List[dict] = [
    # === TIER 1: anti-late guard ===
    {"weight": -100, "label": "FAIL_oversold",
     "check": lambda f: float(f.get("f_momentum_3_pct", 0)) <= MOM_3_OVEREXTENDED_SHORT},
    # === TIER 2: leading signals in calm regime ===
    # ATR > 4% in calm = volatility just expanded (leading dump)
    {"weight": 18, "label": "early_calm_atr_spike",
     "check": lambda f: (abs(float(f.get("f_momentum_6_pct", 0))) < 5
                          and float(f.get("f_atr_pct", 0)) > 4)},
    # Far from fib = extended, likely to revert
    {"weight": 15, "label": "fib_far_0.618",
     "check": lambda f: float(f.get("f_a_fib_dist_0.618", 99)) > 30.0},
    {"weight": 12, "label": "fib_far_0.5",
     "check": lambda f: float(f.get("f_a_fib_dist_0.500", 99)) > 30.0},
    {"weight": 10, "label": "fib_far_0.382",
     "check": lambda f: float(f.get("f_a_fib_dist_0.382", 99)) > 20.0},
    # ATR expanding (downward)
    {"weight": 12, "label": "atr_expanding",
     "check": lambda f: bool(f.get("f_atr_expanding", 0))},
    # === TIER 3: structural ===
    {"weight": 12, "label": "ichi_below_cloud",
     "check": lambda f: bool(f.get("f_a_ichi_below_cloud", 0))},
    {"weight": 8, "label": "vwap_below",
     "check": lambda f: float(f.get("f_vwap_distance_pct", 0)) < -0.5},
    # === TIER 4: 5m microstructure (fast detection) ===
    {"weight": 12, "label": "5m_volume_spike",
     "check": lambda f: bool(f.get("f_m_5m_volume_spike", 0))},
    {"weight": 10, "label": "obi_bearish",
     "check": lambda f: bool(f.get("f_m_obi_bearish", 0))},
    {"weight": 8, "label": "cvd_negative",
     "check": lambda f: float(f.get("f_m_cvd_trend", 0)) < 0},
    # === TIER 5: momentum (small weight) ===
    {"weight": 8, "label": "mom_3_negative",
     "check": lambda f: -10 < float(f.get("f_momentum_3_pct", 0)) < 0},
    {"weight": 5, "label": "mom_accel_negative",
     "check": lambda f: float(f.get("f_momentum_acceleration", 0)) < -2},
    # === TIER 6: anti-signals ===
    {"weight": -8, "label": "NO_ichi_above",
     "check": lambda f: not bool(f.get("f_a_ichi_above_cloud", 0))},
    {"weight": -8, "label": "NO_bos_up",
     "check": lambda f: not bool(f.get("f_bos_up", 0))},
    {"weight": -5, "label": "NO_bb_breakout_above",
     "check": lambda f: not bool(f.get("f_bb_breakout_above", 0))},
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
    anti_late_blocked: bool = False  # True if FAIL_overextended/oversold fired

    def to_dict(self) -> Dict:
        return asdict(self)


def score_features(features: Dict, rules: List[dict]) -> tuple:
    """
    Apply each rule, sum the weights of fired rules, and return:
      (score 0..100, fired_count, fired_labels, anti_late_blocked)
    """
    score = 0.0
    fired = []
    anti_late_blocked = False
    for r in rules:
        try:
            if r["check"](features):
                score += r["weight"]
                fired.append(r["label"])
                if r["label"].startswith("FAIL_"):
                    anti_late_blocked = True
        except Exception:
            continue
    # Clip to [0, 100]. If anti-late blocked, score should be ~0
    if anti_late_blocked:
        score = 0.0
    else:
        score = max(0.0, min(100.0, score))
    return score, len(fired), fired, anti_late_blocked


def score_long(features: Dict) -> tuple:
    return score_features(features, LONG_RULES)


def score_short(features: Dict) -> tuple:
    return score_features(features, SHORT_RULES)


def direction_score(features: Dict, symbol: str = "") -> DirectionResult:
    """
    Compute both LONG and SHORT scores, then decide direction.
    Threshold = 25 (was 50) for more coverage.
    """
    long_score, long_n, long_fired, long_blocked = score_long(features)
    short_score, short_n, short_fired, short_blocked = score_short(features)
    if long_score > short_score and long_score >= 25 and not long_blocked:
        direction = "LONG"
    elif short_score > long_score and short_score >= 25 and not short_blocked:
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
        anti_late_blocked=long_blocked or short_blocked,
    )


def update_thresholds_from_data(df, target_col: str = "label",
                                current_rules: str = "LONG") -> List[dict]:
    """
    Placeholder: retune with labeled data using logistic regression
    on (rule_fired) matrix -> outcome.
    """
    raise NotImplementedError(
        "Wait for ~1200 labeled samples. Run:\n"
        "  python -m src.tune_rules --target label_LONG"
    )
