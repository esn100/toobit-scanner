"""
Smart Money features from Coinglass public API.

Indicators that often precede large moves in small caps:
  - Open Interest change (rising OI + rising price = real demand)
  - Funding rate (extreme values often precede reversals)
  - Long/Short ratio (extreme ratios = crowded trade)
  - Liquidation imbalance (which side is being squeezed)
"""
from __future__ import annotations
import time
import requests
import numpy as np
from typing import Dict, Optional


class CoinglassSmart:
    BASE = "https://open-api.coinglass.com/public/v2"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pumphunter-ai/1.0"})
        if api_key:
            self.session.headers["coinglassSecret"] = api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> object:
        for attempt in range(3):
            try:
                r = self.session.get(
                    f"{self.BASE}{path}",
                    params=params or {},
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(3 + attempt * 2)
                    continue
            except requests.RequestException:
                if attempt == 2:
                    return {}
                time.sleep(1 + attempt)
        return {}

    def get_open_interest_change(self, symbol: str) -> Dict:
        """
        Open interest change over last 4h. Rising OI + price = real
        momentum; falling OI = weak rally.
        """
        params = {"symbol": symbol, "interval": "h4"}
        data = self._get("/futures/openInterest/chart", params)
        out = {"oi_change_4h_pct": 0.0, "oi_value_usd": 0.0,
               "oi_rising": False}
        if not isinstance(data, dict):
            return out
        d = data.get("data")
        if not d or not isinstance(d, list) or len(d) < 2:
            return out
        # d is list of {t, oi_usd, oi_amount}
        try:
            cur = float(d[-1].get("oi_usd", 0))
            prev = float(d[-2].get("oi_usd", 0))
            if prev > 0:
                change = (cur - prev) / prev * 100.0
                out["oi_change_4h_pct"] = change
                out["oi_value_usd"] = cur
                out["oi_rising"] = bool(change > 1.0)
        except (TypeError, ValueError, IndexError):
            pass
        return out

    def get_funding_rate(self, symbol: str) -> Dict:
        """
        Current funding rate. Extreme values (< -0.05% or > 0.05%)
        often precede reversals.
        """
        params = {"symbol": symbol}
        data = self._get("/futures/fundingRate/chart", params)
        out = {"funding_rate": 0.0, "funding_extreme": False}
        if not isinstance(data, dict):
            return out
        d = data.get("data")
        if not d or not isinstance(d, list):
            return out
        try:
            rate = float(d[-1].get("rate", 0))
            out["funding_rate"] = rate
            out["funding_extreme"] = bool(abs(rate) > 0.05)
        except (TypeError, ValueError, IndexError):
            pass
        return out

    def get_long_short_ratio(self, symbol: str) -> Dict:
        """
        Long/short account ratio. Extreme values (>2 or <0.5) indicate
        crowded trades.
        """
        params = {"symbol": symbol, "interval": "h4"}
        data = self._get("/futures/longShort/chart", params)
        out = {"long_short_ratio": 1.0, "ls_extreme": False}
        if not isinstance(data, dict):
            return out
        d = data.get("data")
        if not d or not isinstance(d, list):
            return out
        try:
            ratio = float(d[-1].get("longShortRatio", 1.0))
            out["long_short_ratio"] = ratio
            out["ls_extreme"] = bool(ratio > 2.0 or ratio < 0.5)
        except (TypeError, ValueError, IndexError):
            pass
        return out

    def get_taker_buy_sell(self, symbol: str) -> Dict:
        """
        Taker buy/sell volume ratio. >1 = aggressive buying.
        """
        params = {"symbol": symbol, "interval": "h4"}
        data = self._get("/futures/takerBuySell/chart", params)
        out = {"taker_buy_ratio": 0.5, "aggressive_buying": False}
        if not isinstance(data, dict):
            return out
        d = data.get("data")
        if not d or not isinstance(d, list):
            return out
        try:
            buy = float(d[-1].get("taker_buy_vol", 0))
            sell = float(d[-1].get("taker_sell_vol", 0))
            total = buy + sell
            if total > 0:
                ratio = buy / total
                out["taker_buy_ratio"] = ratio
                out["aggressive_buying"] = bool(ratio > 0.6)
        except (TypeError, ValueError, IndexError):
            pass
        return out

    def get_all(self, symbol: str) -> Dict:
        """Aggregate all smart money features for a symbol."""
        out = {}
        for fname, fn in (
            ("oi", self.get_open_interest_change),
            ("funding", self.get_funding_rate),
            ("ls_ratio", self.get_long_short_ratio),
            ("taker", self.get_taker_buy_sell),
        ):
            try:
                out[fname] = fn(symbol)
            except Exception:
                out[fname] = {}
        return out


def smart_money_score(smart: Dict) -> Dict:
    """
    Convert smart money features into a 0..100 score.
    """
    score = 50.0
    flags = []
    oi = smart.get("oi", {})
    if oi.get("oi_rising"):
        score += 12
        flags.append("oi_rising")
    elif oi.get("oi_change_4h_pct", 0) < -5:
        score -= 10
        flags.append("oi_falling")
    funding = smart.get("funding", {})
    fr = funding.get("funding_rate", 0)
    if fr > 0.1:
        # very high funding = overleveraged longs = reversal risk
        score -= 10
        flags.append("funding_high")
    elif fr < -0.1:
        # negative funding = shorts overcrowded = squeeze risk
        score += 8
        flags.append("funding_negative")
    ls = smart.get("ls_ratio", {})
    ratio = ls.get("long_short_ratio", 1.0)
    if ratio > 2.5:
        score -= 8  # too many longs
        flags.append("ls_crowded_long")
    elif ratio < 0.4:
        score += 8  # too many shorts -> squeeze
        flags.append("ls_crowded_short")
    taker = smart.get("taker", {})
    if taker.get("aggressive_buying"):
        score += 12
        flags.append("aggressive_buying")
    return {
        "smart_money_score": float(max(0.0, min(100.0, score))),
        "smart_money_flags": flags,
        "oi_change_4h_pct": float(oi.get("oi_change_4h_pct", 0.0)),
        "funding_rate": float(fr),
        "long_short_ratio": float(ratio),
        "taker_buy_ratio": float(taker.get("taker_buy_ratio", 0.5)),
    }
