"""
BTC trend filter (Layer 2 of the PumpHunter pipeline).

Determines the overall market regime based on BTC's price action.
Output states: BULLISH, NEUTRAL, BEARISH, RISK_OFF

This state is then used to:
  - downweight alt signals in BEARISH
  - freeze the scanner in RISK_OFF
  - slightly relax thresholds in BULLISH
"""
from __future__ import annotations
import requests
import pandas as pd
from typing import Optional


class BTCFilter:
    """
    Reads BTCUSDT (4h) from Toobit and returns a market state.
    """
    def __init__(self, toobit_client, timeout: int = 15):
        self.toobit = toobit_client
        self.timeout = timeout
        self._cache: Optional[dict] = None
        self._cache_ts: float = 0.0

    def evaluate(self, max_age_minutes: int = 60) -> dict:
        import time
        if self._cache and (time.time() - self._cache_ts) < max_age_minutes * 60:
            return self._cache
        try:
            df = self.toobit.get_klines("BTCUSDT", "4h", 100)
        except Exception as e:
            return {
                "state": "NEUTRAL",
                "score_modifier": 1.0,
                "freeze": False,
                "reasons": [f"btc data unavailable: {e}"],
            }
        if df.empty or len(df) < 30:
            return {
                "state": "NEUTRAL",
                "score_modifier": 1.0,
                "freeze": False,
                "reasons": ["btc data insufficient"],
            }
        res = self._classify(df)
        self._cache = res
        self._cache_ts = time.time()
        return res

    def _classify(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        e20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        e50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        c = float(close.iloc[-1])

        # Multi-bar momentum (12 bars = 2 days on 4h)
        if len(close) >= 13:
            m12 = (c - float(close.iloc[-13])) / max(float(close.iloc[-13]), 1e-12) * 100.0
        else:
            m12 = 0.0
        if len(close) >= 7:
            m6 = (c - float(close.iloc[-7])) / max(float(close.iloc[-7]), 1e-12) * 100.0
        else:
            m6 = 0.0
        # Drawdown from recent high
        if len(close) >= 30:
            high30 = float(close.tail(30).max())
            dd = (c - high30) / max(high30, 1e-12) * 100.0
        else:
            dd = 0.0

        state = "NEUTRAL"
        modifier = 1.0
        freeze = False
        reasons = []

        # RISK_OFF: deep drawdown + heavy negative momentum
        if dd <= -10.0 or m12 <= -10.0:
            state = "RISK_OFF"
            modifier = 0.0
            freeze = True
            reasons.append(f"btc drawdown {dd:.1f}% / momentum {m12:.1f}%")
        elif dd <= -5.0 or m12 <= -5.0:
            state = "BEARISH"
            modifier = 0.7
            reasons.append(f"btc trending down (dd {dd:.1f}%, m12 {m12:.1f}%)")
        elif c > e20 > e50 and m6 > 0 and dd > -2.0:
            state = "BULLISH"
            modifier = 1.05
            reasons.append(f"btc trending up (m6 {m6:.1f}%, dd {dd:.1f}%)")
        else:
            state = "NEUTRAL"
            modifier = 1.0
            reasons.append("btc in neutral range")

        return {
            "state": state,
            "score_modifier": float(modifier),
            "freeze": bool(freeze),
            "btc_momentum_6_pct": float(m6),
            "btc_momentum_12_pct": float(m12),
            "btc_drawdown_pct": float(dd),
            "reasons": reasons,
        }
