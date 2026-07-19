"""
Market cap filtering using CoinGecko free API.
We need this because Toobit does not expose market cap directly.
For each symbol on Toobit, we find the matching CoinGecko coin
and filter to those with market cap under $20M.
"""
from __future__ import annotations
import time
import requests
import pandas as pd
from typing import Dict, Optional


class CoinGeckoClient:
    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout
        self._coins_list_cache: Optional[list] = None
        self._mc_cache: Dict[str, float] = {}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.BASE}{path}"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 + attempt * 2)
                    continue
                r.raise_for_status()
            except requests.RequestException:
                if attempt == 2:
                    return {}
                time.sleep(1 + attempt)
        return {}

    def list_coins(self) -> list:
        """Return the global coins list (id, symbol, name)."""
        if self._coins_list_cache is not None:
            return self._coins_list_cache
        data = self._get("/coins/list", {"include_platform": "false"})
        if isinstance(data, list):
            self._coins_list_cache = data
            return data
        return []

    def symbol_to_coingecko_id(self, symbol: str) -> Optional[str]:
        """
        Map a Toobit symbol like BTCUSDT -> bitcoin.
        For each Toobit symbol we strip 'USDT' and try to find the most-traded
        coin with that ticker. To keep things fast we return the first match.
        """
        sym = symbol.replace("USDT", "").upper()
        for c in self.list_coins():
            if c.get("symbol", "").upper() == sym:
                return c.get("id")
        return None

    def get_market_caps_for_symbols(self, symbols: list) -> Dict[str, float]:
        """
        Return {symbol: market_cap_usd} for each input symbol.
        Uses /coins/markets?symbols=BTC,ETH,... with small batches.
        """
        result: Dict[str, float] = {}
        for sym in symbols:
            self._mc_cache[sym] = 0.0

        # Build symbol->id map (strip USDT)
        id_to_symbol: Dict[str, str] = {}
        all_ids: list = []
        for sym in symbols:
            cg_id = self.symbol_to_coingecko_id(sym)
            if cg_id:
                id_to_symbol[cg_id] = sym
                all_ids.append(cg_id)
        if not all_ids:
            return result

        # /coins/markets supports comma-separated ids (up to ~250)
        for i in range(0, len(all_ids), 200):
            chunk = all_ids[i:i + 200]
            params = {
                "vs_currency": "usd",
                "ids": ",".join(chunk),
                "per_page": 250,
                "page": 1,
                "sparkline": "false",
            }
            data = self._get("/coins/markets", params)
            if isinstance(data, list):
                for row in data:
                    cg_id = row.get("id")
                    mc = row.get("market_cap") or 0
                    if cg_id in id_to_symbol:
                        result[id_to_symbol[cg_id]] = float(mc)
            time.sleep(1.2)  # CoinGecko free rate limit
        return result


def filter_small_cap_symbols(
    tickers: pd.DataFrame,
    coingecko: CoinGeckoClient,
    max_market_cap_usd: float,
    min_volume_usd: float,
    max_symbols: int,
) -> pd.DataFrame:
    """Filter Toobit tickers down to small-cap names."""
    if tickers.empty:
        return tickers
    # First, rough volume filter
    t = tickers[tickers["quote_volume_24h"] >= min_volume_usd].copy()
    if t.empty:
        return t
    # Then look up market cap
    syms = t["symbol"].tolist()
    mc_map = coingecko.get_market_caps_for_symbols(syms)
    t["market_cap_usd"] = t["symbol"].map(mc_map).fillna(0.0)
    # Keep only those with market cap below threshold and > 0
    t = t[(t["market_cap_usd"] > 0) & (t["market_cap_usd"] <= max_market_cap_usd)]
    t = t.sort_values("quote_volume_24h", ascending=False).head(max_symbols)
    t = t.reset_index(drop=True)
    return t
