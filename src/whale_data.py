"""
Free-tier whale / on-chain data.
- Arkham / Nansen / CryptoQuant: disabled by default (need API key).
- CoinGlass: free liquidation data for Toobit USDT-margined perps.

This module provides a single `get_whale_features(symbol)` function that
returns a normalised dict the rest of the pipeline consumes.
"""
from __future__ import annotations
import time
import requests
from typing import Dict


class CoinGlassClient:
    """
    Free CoinGlass public endpoints. No key required for the public site
    but the API needs a key - we use the public REST endpoints that work
    without auth for limited metrics like recent liquidations.
    """

    BASE = "https://open-api.coinglass.com/public/v2"

    def __init__(self, api_key: str | None = None, timeout: int = 15):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        if api_key:
            self.session.headers["coinglassSecret"] = api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None) -> dict:
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
                    time.sleep(2 + attempt * 2)
                    continue
            except requests.RequestException:
                if attempt == 2:
                    return {}
                time.sleep(1 + attempt)
        return {}

    def liquidation_summary(self, symbol: str, exchange: str = "Toobit") -> Dict:
        """
        Returns: long_liq_usd, short_liq_usd, ratio (long/(long+short)),
        and a bias score in [-1, 1] where +1 means longs are getting squeezed.
        """
        # Toobit is often not on CoinGlass for free tier; we still try.
        params = {"symbol": symbol, "exchange": exchange, "interval": "h4"}
        data = self._get("/futures/liquidation/info", params)
        result = {
            "long_liq_usd": 0.0,
            "short_liq_usd": 0.0,
            "liq_bias": 0.0,  # >0 => shorts liquidated => bullish
        }
        if not data or "data" not in data:
            return result
        d = data["data"] or {}
        try:
            result["long_liq_usd"] = float(d.get("longLiquidationUsd", 0) or 0)
            result["short_liq_usd"] = float(d.get("shortLiquidationUsd", 0) or 0)
            total = result["long_liq_usd"] + result["short_liq_usd"]
            if total > 0:
                # When shorts get liquidated more, it's bullish.
                result["liq_bias"] = (result["short_liq_usd"] - result["long_liq_usd"]) / total
        except (TypeError, ValueError):
            pass
        return result


def get_whale_features(symbol: str, coinglass: CoinGlassClient | None = None) -> Dict:
    """
    Unified whale features aggregator.
    Returns: dict with all whale-related signals (0..1 normalised where possible).
    """
    out = {
        "long_liq_usd": 0.0,
        "short_liq_usd": 0.0,
        "liq_bias": 0.0,
        "oi_change_pct": 0.0,   # placeholder
        "whale_tx_count": 0,    # placeholder
    }
    if coinglass is not None:
        try:
            liq = coinglass.liquidation_summary(symbol)
            out.update(liq)
        except Exception:
            pass
    return out
