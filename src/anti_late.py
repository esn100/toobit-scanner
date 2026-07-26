"""
Anti-Late Filter (v2) — prevent entering pumps/dumps that already happened.

BACKTEST EVIDENCE (n=5016, pump_rate=17.7%):
  mom_1_pct by bin:
    (-5,-3]:   23.5%  ← SWEET SPOT (early pullback in uptrend)
    (-3,-1]:   24.4%  ← SWEET SPOT (mini-dip, buy the dip)
    (-1, 0]:   15.4%  ← neutral
    (0, 1]:    15.1%  ← neutral
    (1, 2]:    18.4%  ← mild up
    (2, 3]:    21.4%  ← up
    (3, 5]:    18.7%  ← up
    (5, ∞):    13.7%  ← OVEREXTENDED (rejection zone)

  mom_3_pct by bin:
    (-8,-3]:   25.6%  ← SWEET SPOT (early in move)
    (-3, 0]:   18.3%  ← neutral
    (0, 3]:    14.4%  ← getting hot
    (3, 8]:    22.4%  ← but ok with confirm
    (8, 15]:   8.1%   ← OVEREXTENDED (rejection zone)
    (15, ∞):   9.5%   ← VERY OVEREXTENDED (rejection zone)

THREE LAYERS of anti-late defense:
  1. MOMENTUM_1 overextension: reject if very recent 1-bar pump
  2. MOMENTUM_3 overextension: reject if 3-bar pump mature
  3. CANDLE QUALITY: reject if last candle is "blowoff top" (big wick, weak body)

Plus CONFIDENCE WEIGHTING: if mom_1 or mom_3 just slightly overextended
but other signals STRONG, allow entry with reduced confidence.

Anti-late REJECTS (not weights) — these moves statistically don't pump
further from here.
"""
from __future__ import annotations
from typing import Dict, Tuple
import math


# ---------------------------------------------------------------------------
# Hard thresholds (block entry)
# ---------------------------------------------------------------------------
# After these levels, the pump is statistically OVEREXTENDED.
# Source: backtest 5016 samples.
MOM_1_OVEREXTENDED_LONG = 4.0    # last 1-bar mom > 4%  → late
MOM_1_OVEREXTENDED_SHORT = -4.0  # last 1-bar mom < -4% → late (for shorts)
MOM_3_OVEREXTENDED_LONG = 10.0   # 3-bar mom > 10%  → mature
MOM_3_OVEREXTENDED_SHORT = -10.0 # 3-bar mom < -10% → mature (for shorts)

# Hard acceleration (pump speed)
# If momentum_1 > 2x momentum_3, the last bar alone is doing all the work → blowoff
ACCEL_RATIO_BLOWOFF = 1.5  # mom_1 / |mom_3| > 1.5 means last bar is parabolic

# Big wick + small body at top = distribution (smart money selling into pump)
BIG_WICK_TOP_BODY_RATIO = 0.5  # wick > 50% of total range at top = suspicious
WICK_TOP_MIN_PCT = 5.0         # only consider if top wick is meaningful


# ---------------------------------------------------------------------------
# Soft penalty (reduce confidence, but don't fully reject)
# ---------------------------------------------------------------------------
MOM_1_SOFT_LONG = 2.0          # 1-bar 2-4% → reduce confidence 10%
MOM_3_SOFT_LONG = 6.0          # 3-bar 6-10% → reduce confidence 10%
MOM_1_SOFT_SHORT = -2.0
MOM_3_SOFT_SHORT = -6.0


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def check_anti_late(features: Dict, direction: str) -> Tuple[bool, float, str]:
    """
    Returns:
      (passes: bool, confidence_penalty: float 0..1, reason: str)

    If passes=False, the entry should be REJECTED.
    If passes=True with penalty > 0, the entry is allowed but
    confidence should be reduced (multiplicative).

    Example:
      ok, pen, why = check_anti_late(feats, "LONG")
      if not ok:
          return False
      final_conf = base_confidence * (1.0 - pen)
    """
    mom_1 = float(features.get("momentum_1_pct",
                features.get("f_momentum_1_pct", 0)))
    mom_3 = float(features.get("momentum_3_pct",
                features.get("f_momentum_3_pct", 0)))
    mom_6 = float(features.get("momentum_6_pct",
                features.get("f_momentum_6_pct", 0)))
    mom_acc = float(features.get("momentum_acceleration",
                  features.get("f_momentum_acceleration", 0)))
    candle_strength = float(features.get("candle_strength",
                          features.get("f_candle_strength", 0.5)))
    big_wick_top = bool(features.get("big_wick_top",
                          features.get("f_big_wick_top", False)))
    # Best-effort wick ratio
    wick_body_ratio = float(features.get("wick_body_ratio",
                            features.get("f_wick_body_ratio", 0.0)))

    penalty = 0.0
    reasons = []

    if direction == "LONG":
        # --- HARD REJECTS (block entry) ---
        # 1. mom_1 overextended: the last bar is the pump, not early signal
        if mom_1 >= MOM_1_OVEREXTENDED_LONG:
            return False, 1.0, f"LATE_mom1({mom_1:+.2f}%>={MOM_1_OVEREXTENDED_LONG}%)"
        # 2. mom_3 overextended: pump is mature, expect mean-reversion
        if mom_3 >= MOM_3_OVEREXTENDED_LONG:
            return False, 1.0, f"LATE_mom3({mom_3:+.2f}%>={MOM_3_OVEREXTENDED_LONG}%)"
        # 3. blowoff acceleration: last bar is doing all the work
        #    mom_1 / mom_3 > 1.5 means the previous 2 bars were weak,
        #    and this bar alone is the move → likely to fade
        if mom_3 > 1.0 and mom_1 > 0 and (mom_1 / mom_3) > ACCEL_RATIO_BLOWOFF:
            return False, 1.0, f"LATE_blowoff(m1/m3={mom_1/mom_3:.2f})"
        # 4. blowoff top (distribution): big wick, weak body, at top of pump
        if big_wick_top and wick_body_ratio > BIG_WICK_TOP_BODY_RATIO and mom_3 > 3:
            return False, 1.0, f"LATE_distribution(wick={wick_body_ratio:.2f})"
        # 5. distribution candle (smart money selling) at top of move
        if candle_strength < 0.3 and mom_3 > 5:
            return False, 1.0, f"LATE_weak_candle(str={candle_strength:.2f})"

        # --- SOFT PENALTIES (reduce confidence, don't block) ---
        if mom_1 >= MOM_1_SOFT_LONG:
            penalty = max(penalty, 0.10)
            reasons.append(f"hot_m1({mom_1:+.1f}%)")
        if mom_3 >= MOM_3_SOFT_LONG:
            penalty = max(penalty, 0.10)
            reasons.append(f"hot_m3({mom_3:+.1f}%)")
        # Strong acceleration is GOOD for entry, not bad
        # (just noting in reason, no penalty)

    elif direction == "SHORT":
        if mom_1 <= MOM_1_OVEREXTENDED_SHORT:
            return False, 1.0, f"LATE_mom1({mom_1:+.2f}%<={MOM_1_OVEREXTENDED_SHORT}%)"
        if mom_3 <= MOM_3_OVEREXTENDED_SHORT:
            return False, 1.0, f"LATE_mom3({mom_3:+.2f}%<={MOM_3_OVEREXTENDED_SHORT}%)"
        if mom_3 < -1.0 and mom_1 < 0 and (abs(mom_1) / abs(mom_3)) > ACCEL_RATIO_BLOWOFF:
            return False, 1.0, f"LATE_blowoff(m1/m3={mom_1/mom_3:.2f})"
        if big_wick_top and wick_body_ratio > BIG_WICK_TOP_BODY_RATIO and mom_3 < -3:
            return False, 1.0, f"LATE_distribution(wick={wick_body_ratio:.2f})"
        if candle_strength < 0.3 and mom_3 < -5:
            return False, 1.0, f"LATE_weak_candle(str={candle_strength:.2f})"

        if mom_1 <= MOM_1_SOFT_SHORT:
            penalty = max(penalty, 0.10)
            reasons.append(f"hot_m1({mom_1:+.1f}%)")
        if mom_3 <= MOM_3_SOFT_SHORT:
            penalty = max(penalty, 0.10)
            reasons.append(f"hot_m3({mom_3:+.1f}%)")

    if reasons:
        return True, penalty, "soft: " + ",".join(reasons)
    return True, 0.0, "ok"


# ---------------------------------------------------------------------------
# Diagnostic: explain WHY a signal would be blocked
# ---------------------------------------------------------------------------
def explain_anti_late(features: Dict, direction: str) -> str:
    """
    Returns a human-readable string explaining the anti-late decision.
    """
    ok, pen, why = check_anti_late(features, direction)
    mom_1 = float(features.get("momentum_1_pct", 0))
    mom_3 = float(features.get("momentum_3_pct", 0))
    candle_strength = float(features.get("candle_strength", 0.5))
    big_wick_top = bool(features.get("big_wick_top", False))
    lines = [
        f"  Anti-late check ({direction}):",
        f"    mom_1: {mom_1:+.2f}%  | mom_3: {mom_3:+.2f}%",
        f"    candle_strength: {candle_strength:.2f} | big_wick_top: {big_wick_top}",
    ]
    if ok:
        lines.append(f"    ✅ PASS (penalty={pen*100:.0f}%, {why})")
    else:
        lines.append(f"    ⛔ BLOCK: {why}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (when run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test cases based on backtest evidence
    cases = [
        # (name, features, direction, expected_ok)
        ("early_setup_mild_up",  {"momentum_1_pct": 1.5, "momentum_3_pct": 4.0,
                                   "candle_strength": 0.7, "big_wick_top": False}, "LONG", True),
        ("sweet_spot_dip",        {"momentum_1_pct": -1.5, "momentum_3_pct": -2.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "LONG", True),
        ("overextended_mom1",     {"momentum_1_pct": 5.5, "momentum_3_pct": 7.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "LONG", False),
        ("overextended_mom3",     {"momentum_1_pct": 1.0, "momentum_3_pct": 12.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "LONG", False),
        ("blowoff_accel",         {"momentum_1_pct": 5.0, "momentum_3_pct": 3.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "LONG", False),
        ("distribution_wick",     {"momentum_1_pct": 1.0, "momentum_3_pct": 5.0,
                                   "candle_strength": 0.6, "big_wick_top": True,
                                   "wick_body_ratio": 0.7}, "LONG", False),
        ("soft_penalty",          {"momentum_1_pct": 2.5, "momentum_3_pct": 5.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "LONG", True),
        ("short_overext",         {"momentum_1_pct": -5.0, "momentum_3_pct": -7.0,
                                   "candle_strength": 0.6, "big_wick_top": False}, "SHORT", False),
    ]
    print("=" * 70)
    print("Anti-Late Filter — Self Test")
    print("=" * 70)
    passed = 0
    for name, feats, direction, expected_ok in cases:
        ok, pen, why = check_anti_late(feats, direction)
        status = "✅" if ok == expected_ok else "❌"
        if ok == expected_ok:
            passed += 1
        print(f"  {status} {name:25s} | dir={direction:5s} | "
              f"expected_ok={expected_ok} | got_ok={ok} | pen={pen:.2f} | {why}")
    print(f"\n{passed}/{len(cases)} tests passed")
