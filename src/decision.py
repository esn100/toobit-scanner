"""
Final decision layer (Layer 9 of the PumpHunter pipeline).

Combines:
  - rule-based composite score
  - ML probability
  - market regime (BTC)
  - quality flags
  - operational limits (cooldown)

Outputs one of:
  - APPROVED      : high confidence, ready to alert
  - WATCHLIST     : borderline, worth monitoring
  - REJECTED      : not actionable
"""
from __future__ import annotations
from typing import Dict, Tuple


def decide(
    composite: float,
    ml_prob: float,
    btc: dict,
    quality_ok: bool,
    cooldown_ok: bool,
    *,
    approved_threshold: float = 75.0,
    watchlist_threshold: float = 60.0,
    ml_min_prob: float = 0.45,
) -> Dict:
    """
    Return decision with reasons.

    Decision rules (in order):
      1. If BTC.RISK_OFF or !quality_ok or !cooldown_ok -> REJECTED
      2. If composite >= approved and ml_prob >= ml_min_prob -> APPROVED
      3. If composite >= watchlist_threshold -> WATCHLIST
      4. Else -> REJECTED
    """
    reasons: list = []
    state = "REJECTED"
    confidence = 0.0

    if btc.get("freeze"):
        return {
            "decision": "REJECTED",
            "reasons": ["BTC in RISK_OFF regime"],
            "confidence": 0.0,
        }
    if not quality_ok:
        return {
            "decision": "REJECTED",
            "reasons": ["data quality check failed"],
            "confidence": 0.0,
        }
    if not cooldown_ok:
        return {
            "decision": "REJECTED",
            "reasons": ["cooldown active for this symbol"],
            "confidence": 0.0,
        }
    if btc.get("state") == "BEARISH":
        reasons.append(f"BTC state BEARISH (modifier {btc.get('score_modifier')})")

    if composite >= approved_threshold and ml_prob >= ml_min_prob:
        state = "APPROVED"
        confidence = min(1.0, 0.5 * (composite / 100.0) + 0.5 * ml_prob)
        reasons.append(
            f"composite {composite:.1f} >= {approved_threshold} "
            f"and ml_prob {ml_prob:.2f} >= {ml_min_prob}"
        )
    elif composite >= watchlist_threshold:
        state = "WATCHLIST"
        confidence = min(1.0, 0.5 * (composite / 100.0) + 0.5 * ml_prob)
        reasons.append(
            f"composite {composite:.1f} between "
            f"{watchlist_threshold} and {approved_threshold}"
        )
    else:
        state = "REJECTED"
        reasons.append(
            f"composite {composite:.1f} below watchlist threshold"
        )

    return {
        "decision": state,
        "reasons": reasons,
        "confidence": round(confidence, 3),
    }
