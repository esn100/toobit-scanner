"""
Consensus filter - require multiple independent signals to agree.

Three independent "voters":
  1. Volume voter: rvol > threshold (recent volume spike)
  2. Momentum voter: short-term momentum positive and accelerating
  3. RSI voter: RSI in sweet spot (not overbought, not oversold)

If at least 2 of 3 agree, the signal passes.
"""
from __future__ import annotations
from typing import Dict, Tuple


def consensus_vote(pack: Dict) -> Tuple[bool, int, list]:
    """
    Run three voters on the pack and return consensus.
    Returns (passes, vote_count, voter_names_passed).
    """
    ind_1h = pack.get("ind_1h", {})
    tech_1h = pack.get("tech_1h", {})
    passed = []
    # Voter 1: Volume
    rvol = ind_1h.get("rvol", 1.0)
    volume_spike = ind_1h.get("volume_spike", False)
    if rvol >= 2.5 or volume_spike:
        passed.append("volume")
    # Voter 2: Momentum (positive and accelerating)
    m1 = ind_1h.get("momentum_1_pct", 0)
    m3 = ind_1h.get("momentum_3_pct", 0)
    mom_acc = ind_1h.get("momentum_acceleration", 0)
    if m1 > 0 and m3 > 0 and mom_acc > 0:
        passed.append("momentum")
    elif m1 > 1.0 or m3 > 2.0:
        # Allow strong single-bar momentum
        passed.append("momentum")
    # Voter 3: RSI sweet spot
    rsi = tech_1h.get("rsi_value", 50)
    if 30 <= rsi <= 65:
        passed.append("rsi")
    # Optional 4th voter: smart money (if available)
    sm = pack.get("smart_money", {})
    if isinstance(sm, dict) and sm.get("smart_money_score", 50) >= 60:
        passed.append("smart_money")
    # Optional 5th voter: BTC independent
    btc_corr = pack.get("btc_corr", {})
    if btc_corr.get("independent_mover", False):
        passed.append("btc_independent")
    n_passed = len(passed)
    return n_passed >= 2, n_passed, passed


def consensus_score(pack: Dict) -> float:
    """
    Score contribution from consensus (0-100).
    More voters agreeing = higher bonus.
    """
    passes, n, names = consensus_vote(pack)
    if not passes:
        return 0.0
    # Each voter is worth ~10 points
    return min(50.0, n * 10.0)
