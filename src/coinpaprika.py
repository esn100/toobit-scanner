"""
CoinPaprika fallback client. No API key required.
Provides bulk market-cap lookup as an alternative to CoinGecko
when CoinGecko's free tier rate-limits us.
"""
from __future__ import annotations
import time
import requests
import pandas as pd
from typing import Dict, Optional


class CoinPaprikaClient:
    BASE = "https://api.coinpaprika.com/v1"

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout
        self._tickers_cache: Optional[list] = None
        self._tickers_path: Optional[str] = None

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        url = f"{self.BASE}{path}"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
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

    def get_all_tickers(self) -> list:
        """Return all tickers with USD market cap. Cached on disk if path set."""
        if self._tickers_cache is not None:
            return self._tickers_cache
        # CoinPaprika returns 5000+ coins in one shot
        data = self._get("/tickers", {"quotes": "USD", "limit": 5000})
        if isinstance(data, list) and len(data) > 100:
            self._tickers_cache = data
            if self._tickers_path:
                try:
                    import os, json
                    os.makedirs(os.path.dirname(self._tickers_path), exist_ok=True)
                    with open(self._tickers_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception:
                    pass
            return data
        return []

    def get_market_caps_for_symbols(self, symbols: list) -> Dict[str, float]:
        """Return {symbol: market_cap_usd}."""
        result: Dict[str, float] = {}
        tickers = self.get_all_tickers()
        if not tickers:
            return result
        # Build symbol -> market_cap
        # Some symbols collide (USDT, USDC, BETH); prefer the one with the
        # highest market cap (i.e. the real one).
        sym_to_mc: Dict[str, float] = {}
        for t in tickers:
            sym = (t.get("symbol") or "").upper()
            mc = (t.get("quotes", {}).get("USD", {}).get("market_cap") or 0) or 0
            if not sym or not mc:
                continue
            if sym not in sym_to_mc or mc > sym_to_mc[sym]:
                sym_to_mc[sym] = float(mc)
        for sym in symbols:
            if not sym:
                continue
            base = sym.replace("USDT", "").upper()
            if base in sym_to_mc:
                result[sym] = sym_to_mc[base]
        return result
