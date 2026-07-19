"""
Microstructure Score Aggregator for PumpHunter Institutional v1.0.

Combines 8 microstructure factors:
  1. Volume Explosion (20%)
  2. Whale Activity (20%)
  3. Order Book Imbalance (15%)
  4. Liquidity Sweep (15%)
  5. Open Interest (10%)
  6. CVD (10%)
  7. Funding Rate (5%)
  8. Multi-exchange Confirmation (5%)

Output: 0-100 score, plus probability estimate and reasoning trace.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .microstructure import (
    ToobitOrderBookClient, OKXSmartMoneyClient,
    calculate_obi, calculate_spread_pct,
    detect_whales, calculate_cvd, detect_liquidity_sweep,
    multi_exchange_check,
)


# Weights from the institutional spec
FACTOR_WEIGHTS = {
    "volume_explosion": 20,
    "whale_activity": 20,
    "order_book_imbalance": 15,
    "liquidity_sweep": 15,
    "open_interest": 10,
    "cvd": 10,
    "funding_rate": 5,
    "multi_exchange": 5,
}
FACTOR_NAMES = {
    "volume_explosion": "Volume Explosion",
    "whale_activity": "Whale Activity",
    "order_book_imbalance": "Order Book Imbalance",
    "liquidity_sweep": "Liquidity Sweep",
    "open_interest": "Open Interest",
    "cvd": "CVD",
    "funding_rate": "Funding Rate",
    "multi_exchange": "Multi-exchange Confirmation",
}


def factor_volume_explosion(
    ticker: Dict, trades: List[Dict], klines: pd.DataFrame
) -> Tuple[float, bool, str]:
    """
    Factor 1: Volume explosion in last 5 min vs 1h baseline.
    Returns: (score 0-1, passed, reason)
    """
    if not ticker or klines.empty or len(klines) < 15:
        return 0.0, False, "insufficient data"
    # Recent trade volume (last 5 min)
    if not trades:
        return 0.0, False, "no trades"
    now_ms = max(t["ts"] for t in trades)
    recent_5m = [t for t in trades if t["ts"] >= now_ms - 5 * 60 * 1000]
    if not recent_5m:
        return 0.0, False, "no recent trades"
    recent_5m_vol = sum(t["qty"] for t in recent_5m)
    # Baseline: average 5-min volume over last 1h
    recent_1h = [t for t in trades if t["ts"] >= now_ms - 60 * 60 * 1000]
    if len(recent_1h) < 10:
        return 0.0, False, "insufficient 1h baseline"
    # 1h baseline = 12 buckets of 5min
    bucket_vol = sum(t["qty"] for t in recent_1h) / 12
    if bucket_vol == 0:
        return 0.0, False, "zero baseline"
    rvol = recent_5m_vol / bucket_vol
    # Trade count check
    # We don't have trade count, but if we have N trades in 5min vs 1h/12 = N
    n_5m = len(recent_5m)
    n_1h_avg = max(1, len(recent_1h) / 12)
    n_rvol = n_5m / n_1h_avg
    passed = rvol > 4.0 and n_rvol > 2.5
    if rvol > 10:
        score = 1.0
    elif rvol > 4:
        score = 0.85
    elif rvol > 2.5:
        score = 0.5
    else:
        score = 0.0
    reason = f"5m vol: {rvol:.1f}x baseline, {n_rvol:.1f}x trade count"
    return float(score), bool(passed), reason


def factor_whale_activity(trades: List[Dict]) -> Tuple[float, bool, str]:
    """Factor 2: Whale activity in last 60 sec."""
    w = detect_whales(trades, min_qty_usd=2000, min_count=3, window_sec=60)
    score = float(w["whale_score"])
    passed = w["count"] >= 3 and (w["accumulated"] or w["buy_sell_ratio"] > 2)
    if w["accumulated"]:
        reason = f"{w['count']} whales, accumulated (no price impact), buy/sell={w['buy_sell_ratio']:.1f}"
    else:
        reason = f"{w['count']} whales, buy/sell={w['buy_sell_ratio']:.1f}"
    return score, passed, reason


def factor_order_book_imbalance(depth: Dict) -> Tuple[float, bool, str]:
    """Factor 3: OBI > 1.8 (bullish) or < 0.55 (bearish)."""
    obi = calculate_obi(depth, levels=10)
    spread = calculate_spread_pct(depth)
    if spread > 0.5:
        return 0.0, False, f"spread too wide: {spread:.3f}%"
    if obi > 2.5:
        score, passed, dirn = 1.0, True, "strong bullish"
    elif obi > 1.8:
        score, passed, dirn = 0.8, True, "bullish"
    elif obi < 0.4:
        score, passed, dirn = 1.0, True, "strong bearish"
    elif obi < 0.55:
        score, passed, dirn = 0.8, True, "bearish"
    else:
        score, passed, dirn = 0.3, False, f"balanced ({obi:.2f})"
    return float(score), bool(passed), f"OBI={obi:.2f} ({dirn})"


def factor_liquidity_sweep(
    klines: pd.DataFrame, trades: List[Dict], depth: Dict
) -> Tuple[float, bool, str]:
    """Factor 4: Liquidity sweep detected."""
    s = detect_liquidity_sweep(klines, trades, depth)
    if s["sweep"]:
        score = float(s["confidence"])
        return score, True, f"sweep={s['sweep_type']}"
    return 0.0, False, "no sweep"


def factor_open_interest(oi_data: Dict) -> Tuple[float, bool, str]:
    """Factor 5: OI rising (signal confirmation)."""
    change = oi_data.get("oi_change_4h_pct", 0)
    rising = oi_data.get("oi_rising", False)
    if change > 10:
        score = 1.0
    elif change > 5:
        score = 0.8
    elif change > 1:
        score = 0.5
    elif change < -10:
        score = 0.0
    else:
        score = 0.2
    passed = bool(rising and change > 1)
    return float(score), bool(passed), f"OI change 4h: {change:+.1f}%"


def factor_cvd(cvd_data: Dict, direction: str = "long") -> Tuple[float, bool, str]:
    """Factor 6: CVD confirms direction."""
    cvd = cvd_data.get("cvd", 0)
    trend = cvd_data.get("cvd_trend", 0)
    if direction == "long":
        if trend > 0 and cvd > 0:
            score, passed = 1.0, True
        elif trend > 0:
            score, passed = 0.6, True
        else:
            score, passed = 0.0, False
        return score, passed, f"CVD={cvd:.1f} trend={trend:+.1f}"
    else:
        if trend < 0 and cvd < 0:
            score, passed = 1.0, True
        elif trend < 0:
            score, passed = 0.6, True
        else:
            score, passed = 0.0, False
        return score, passed, f"CVD={cvd:.1f} trend={trend:+.1f}"


def factor_funding_rate(funding: Dict) -> Tuple[float, bool, str]:
    """
    Factor 7: Funding extreme (short squeeze risk or long squeeze risk).
    Negative funding + OI rising = short squeeze.
    """
    rate = funding.get("funding_rate", 0)
    extreme = funding.get("extreme", False)
    if rate < -0.0005:
        # Negative funding = shorts overcrowded = squeeze risk
        score, passed = 0.9, True
        return score, passed, f"negative funding: {rate*100:.3f}%"
    elif rate > 0.0005:
        # Positive funding = longs overcrowded = dump risk
        score, passed = 0.9, True
        return score, passed, f"positive funding: {rate*100:.3f}%"
    elif abs(rate) > 0.0001:
        score, passed = 0.4, False
    else:
        score, passed = 0.0, False
    return float(score), bool(passed), f"funding neutral: {rate*100:.4f}%"


def factor_multi_exchange(mexc_data: Dict) -> Tuple[float, bool, str]:
    """Factor 8: Multi-exchange confirmation."""
    if not mexc_data.get("confirms"):
        return 0.0, False, "exchanges diverged"
    lead = mexc_data.get("lead", "synced")
    if lead == "toobit":
        return 0.7, True, "Toobit leads OKX (volume spike)"
    return 0.4, True, f"exchanges synced (lead: {lead})"


# ============================================================================
# Main Aggregator
# ============================================================================
@dataclass
class MicrostructureResult:
    symbol: str
    timestamp: float
    direction: str  # "LONG", "SHORT", or "NEUTRAL"
    confidence: str  # "High", "Medium", "Low"
    composite_score: float
    probability_long: float
    probability_short: float
    factors: Dict
    passed_factors: int
    total_factors: int
    decision: str  # "APPROVED", "WATCHLIST", "REJECTED"
    reasons: List[str]

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "confidence": self.confidence,
            "composite_score": round(self.composite_score, 1),
            "probability_long_pct": round(self.probability_long * 100, 1),
            "probability_short_pct": round(self.probability_short * 100, 1),
            "passed_factors": self.passed_factors,
            "total_factors": self.total_factors,
            "decision": self.decision,
            "factors": self.factors,
            "reasons": self.reasons,
        }


def compute_microstructure_score(
    symbol: str,
    toobit_client: ToobitOrderBookClient,
    okx_client: OKXSmartMoneyClient,
    klines_1h: pd.DataFrame,
    direction_hint: str = "long",
) -> MicrostructureResult:
    """
    Run all 8 microstructure factors and combine into a single score.
    """
    factors = {}
    reasons = []
    # 1. Volume Explosion
    ticker = toobit_client.get_ticker_24h(symbol)
    trades = toobit_client.get_recent_trades(symbol, limit=1000)
    score_vol, pass_vol, reason_vol = factor_volume_explosion(
        ticker, trades, klines_1h
    )
    factors["volume_explosion"] = {
        "score": score_vol, "passed": pass_vol, "reason": reason_vol
    }
    # 2. Whale Activity
    score_whale, pass_whale, reason_whale = factor_whale_activity(trades)
    factors["whale_activity"] = {
        "score": score_whale, "passed": pass_whale, "reason": reason_whale
    }
    # 3. Order Book Imbalance
    depth = toobit_client.get_depth(symbol, limit=20)
    score_obi, pass_obi, reason_obi = factor_order_book_imbalance(depth)
    factors["order_book_imbalance"] = {
        "score": score_obi, "passed": pass_obi, "reason": reason_obi
    }
    # 4. Liquidity Sweep
    score_sweep, pass_sweep, reason_sweep = factor_liquidity_sweep(
        klines_1h, trades, depth
    )
    factors["liquidity_sweep"] = {
        "score": score_sweep, "passed": pass_sweep, "reason": reason_sweep
    }
    # 5. Open Interest
    oi_data = okx_client.get_open_interest(symbol)
    score_oi, pass_oi, reason_oi = factor_open_interest(oi_data)
    factors["open_interest"] = {
        "score": score_oi, "passed": pass_oi, "reason": reason_oi
    }
    # 6. CVD
    cvd_data = calculate_cvd(trades)
    score_cvd, pass_cvd, reason_cvd = factor_cvd(cvd_data, direction_hint)
    factors["cvd"] = {
        "score": score_cvd, "passed": pass_cvd, "reason": reason_cvd
    }
    # 7. Funding Rate
    funding = okx_client.get_funding_rate(symbol)
    score_fund, pass_fund, reason_fund = factor_funding_rate(funding)
    factors["funding_rate"] = {
        "score": score_fund, "passed": pass_fund, "reason": reason_fund
    }
    # 8. Multi-exchange Confirmation
    okx_ticker = okx_client.get_okx_ticker(symbol)
    mexc_data = multi_exchange_check(ticker, okx_ticker)
    score_mexc, pass_mexc, reason_mexc = factor_multi_exchange(mexc_data)
    factors["multi_exchange"] = {
        "score": score_mexc, "passed": pass_mexc, "reason": reason_mexc
    }
    # Weighted sum
    total = sum(FACTOR_WEIGHTS[k] * factors[k]["score"] for k in FACTOR_WEIGHTS)
    total_weight = sum(FACTOR_WEIGHTS.values())
    composite = (total / total_weight) * 100
    # Count passed
    passed_count = sum(1 for k in FACTOR_WEIGHTS if factors[k]["passed"])
    # Determine direction
    long_signals = sum(
        factors[k]["score"] for k in ("order_book_imbalance", "cvd", "liquidity_sweep")
        if factors[k]["score"] > 0.5
    )
    short_signals = sum(
        1 for k in ("order_book_imbalance", "cvd", "liquidity_sweep")
        if factors[k]["score"] > 0.5 and k == "order_book_imbalance" and "bearish" in factors[k]["reason"]
    )
    # Simple direction: if OBI bullish + CVD positive = long
    obi_bullish = "bullish" in factors["order_book_imbalance"]["reason"]
    obi_bearish = "bearish" in factors["order_book_imbalance"]["reason"]
    cvd_positive = factors["cvd"]["score"] > 0.5
    if obi_bullish and cvd_positive:
        direction = "LONG"
    elif obi_bearish and not cvd_positive:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    # Probability (calibrated to spec)
    if direction == "LONG":
        prob_long = 0.5 + composite / 200  # 50-100%
        prob_short = 1 - prob_long
    elif direction == "SHORT":
        prob_short = 0.5 + composite / 200  # 50-100%
        prob_long = 1 - prob_short
    else:
        prob_long = 0.5
        prob_short = 0.5
    # Confidence level
    if passed_count >= 6 and composite >= 70:
        confidence = "High"
    elif passed_count >= 4 and composite >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"
    # Decision
    if passed_count >= 6 and composite >= 85:
        decision = "APPROVED"
    elif passed_count >= 4 and composite >= 60:
        decision = "WATCHLIST"
    else:
        decision = "REJECTED"
    # Reasons
    for k, f in factors.items():
        sym = "✔" if f["passed"] else "✖"
        reasons.append(f"{sym} {FACTOR_NAMES[k]}: {f['reason']}")
    return MicrostructureResult(
        symbol=symbol,
        timestamp=time.time(),
        direction=direction,
        confidence=confidence,
        composite_score=composite,
        probability_long=prob_long,
        probability_short=prob_short,
        factors=factors,
        passed_factors=passed_count,
        total_factors=len(FACTOR_WEIGHTS),
        decision=decision,
        reasons=reasons,
    )
