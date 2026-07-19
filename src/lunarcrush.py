"""
LunarCrush free-tier data fetcher.
Free tier: limited but enough for social metrics per coin.
Endpoint: /v2/public/coins/:id/time-series  (or /public/coins/list)
Falls back to /v3/coins if v2 fails.
"""
from __future__ import annotations
import time
import requests
from typing import Optional, Dict


class LunarCrushClient:
    BASE_V2 = "https://lunarcrush.com/api/v2/public"
    BASE_V4 = "https://api.lunarcrush.com/v4"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                if r.status_code in (401, 403):
                    return {}
            except requests.RequestException:
                if attempt == 2:
                    return {}
                time.sleep(1 + attempt)
        return {}

    def get_coin_metrics(self, symbol: str) -> Dict:
        """
        Return a dict with social metrics for a coin symbol (e.g. 'BTC').
        If the call fails, returns zeroed metrics so the scanner keeps working.
        """
        symbol = symbol.upper()
        # v4 endpoint
        data = self._get(
            f"{self.BASE_V4}/coins",
            {"symbol": symbol, "interval": "1w", "limit": 1},
        )
        if not data:
            # v2 fallback (no key)
            data = self._get(
                f"{self.BASE_V2}/coins",
                {"symbol": symbol, "interval": "1w", "limit": 1},
            )
        if not data:
            return self._empty()

        rows = []
        if isinstance(data, dict):
            rows = data.get("data") or data.get("coins") or []
        if not rows:
            return self._empty()
        c = rows[0] if isinstance(rows, list) else rows
        try:
            return {
                "social_score": float(c.get("social_score") or 0),
                "social_volume_24h": float(c.get("social_volume_24h") or 0),
                "social_dominance": float(c.get("social_dominance") or 0),
                "sentiment": float(c.get("average_sentiment") or 0),
                "galaxy_score": float(c.get("galaxy_score") or 0),
                "alt_rank": float(c.get("alt_rank") or 9999),
                "market_cap": float(c.get("market_cap") or 0),
                "volume_24h": float(c.get("volume_24h") or 0),
                "percent_change_24h": float(c.get("percent_change_24h") or 0),
            }
        except (TypeError, ValueError):
            return self._empty()

    @staticmethod
    def _empty() -> Dict:
        return {
            "social_score": 0.0,
            "social_volume_24h": 0.0,
            "social_dominance": 0.0,
            "sentiment": 0.0,
            "galaxy_score": 0.0,
            "alt_rank": 9999.0,
            "market_cap": 0.0,
            "volume_24h": 0.0,
            "percent_change_24h": 0.0,
        }
