"""
Market cap filtering using CoinGecko.
- Uses /coins/markets for market cap lookup.
- Caches the full coins list to disk so we don't hammer CoinGecko.
- When rate-limited (429) or offline, falls back to a curated list of
  small-cap names that frequently appear on derivatives exchanges.
"""
from __future__ import annotations
import os
import time
import json
import requests
import pandas as pd
from typing import Dict, Optional, List


# Curated fallback: tickers commonly seen on Toobit and similar
# derivatives exchanges. These are *guesses* for symbols we should
# investigate when CoinGecko is unavailable. The scanner will skip
# any symbol that doesn't pass other filters.
FALLBACK_TICKERS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "PEPE", "SHIB", "WIF",
    "BONK", "FLOKI", "MEME", "TURBO", "LADYS", "AIDOGE",
    "TRX", "AVAX", "MATIC", "DOT", "LINK", "UNI", "ATOM", "LTC",
    "NEAR", "APT", "ARB", "OP", "INJ", "SUI", "SEI", "TIA",
    "FIL", "ICP", "HBAR", "VET", "ALGO", "FTM", "SAND", "MANA",
    "AXS", "GMT", "APE", "DYDX", "GMX", "BLUR", "ENA", "ETHFI",
    "JUP", "PYTH", "JTO", "STRK", "PIXEL", "PORTAL", "AEVO",
    "ONDO", "MANTA", "ALT", "TAO", "PENDLE", "ZRO", "IO", "W",
    "ENA", "BOME", "NOT", "DYM", "REZ", "IOUSDT", "ZK",
    "ZRX", "BAT", "ENJ", "CHZ", "GALA", "HOT", "IOTA",
    "KSM", "MOVR", "GLMR", "CELR", "OMG", "RVN", "ROSE",
    "MASK", "CRV", "LDO", "RPL", "SSV", "FXS", "CVX", "BAL",
    "AAVE", "MKR", "SNX", "COMP", "1INCH", "SUSHI", "YFI",
    "DODO", "PERP", "BADGER", "PICKLE", "REN", "GRT", "NMR",
    "LRC", "OGN", "TRB", "BNT", "KNC", "MLN", "BAND", "OCEAN",
    "FET", "AGIX", "RENDER", "AKT", "PHB", "ROSE", "CKB",
    "CELO", "KAVA", "OSMO", "CTK", "INJ", "BOND", "AUCTION",
    "BLZ", "ANKR", "DATA", "COS", "COTI", "CHR", "DENT", "DOCK",
    "ELEC", "KEY", "MBL", "MDT", "NULS", "ONG", "ONT", "PIVX",
    "POLY", "POWR", "REQ", "SNT", "STMX", "STORM", "SUN",
    "SYS", "TOMO", "TRU", "WAN", "WAVES", "WAXP", "WIN", "WRX",
    "XEM", "XVG", "XZC", "YGG", "ZEC", "ZEN", "ZIL",
]


class CoinGeckoClient:
    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, timeout: int = 15, cache_dir: str | None = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout
        # CoinGecko Demo API key (Pro) - set via env to enable higher rate limit
        demo_key = os.environ.get("COINGECKO_API_KEY", "")
        if demo_key:
            self.session.headers["x-cg-demo-api-key"] = demo_key
        self._coins_list_cache: Optional[list] = None
        self._coins_list_path: Optional[str] = (
            os.path.join(cache_dir, "coingecko_coins_list.json")
            if cache_dir else None
        )
        self._mc_cache: Dict[str, float] = {}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.BASE}{path}"
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    # rate limited - exponential backoff
                    time.sleep(3 + attempt * 4)
                    continue
                if r.status_code in (401, 403):
                    return {}
                r.raise_for_status()
            except requests.RequestException:
                if attempt == 3:
                    return {}
                time.sleep(2 + attempt * 2)
        return {}

    def list_coins(self) -> list:
        """Return the global coins list (id, symbol, name). Cached on disk."""
        if self._coins_list_cache is not None:
            return self._coins_list_cache
        # Try disk cache first
        if self._coins_list_path and os.path.exists(self._coins_list_path):
            try:
                with open(self._coins_list_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 1000:
                    self._coins_list_cache = data
                    return data
            except Exception:
                pass
        # Otherwise hit the API
        data = self._get("/coins/list", {"include_platform": "false"})
        if isinstance(data, list) and len(data) > 1000:
            self._coins_list_cache = data
            if self._coins_list_path:
                try:
                    os.makedirs(os.path.dirname(self._coins_list_path), exist_ok=True)
                    with open(self._coins_list_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception:
                    pass
            return data
        return []

    def symbol_to_coingecko_id(self, symbol: str) -> Optional[str]:
        """
        Map a Toobit symbol like BTCUSDT -> 'bitcoin'.
        """
        sym = symbol.replace("USDT", "").upper()
        for c in self.list_coins():
            if c.get("symbol", "").upper() == sym:
                return c.get("id")
        return None

    def get_market_caps_for_symbols(self, symbols: list) -> Dict[str, float]:
        """
        Return {symbol: market_cap_usd}.
        Uses /coins/markets in batches of 250 ids.
        """
        result: Dict[str, float] = {}
        # Build symbol->id map
        id_to_symbol: Dict[str, str] = {}
        all_ids: list = []
        for sym in symbols:
            cg_id = self.symbol_to_coingecko_id(sym)
            if cg_id:
                id_to_symbol[cg_id] = sym
                all_ids.append(cg_id)
        if not all_ids:
            return result
        # Fetch in batches
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
            # No artificial delay if we have a demo key (higher rate limit)
            if not os.environ.get("COINGECKO_API_KEY"):
                time.sleep(1.5)
        return result


def filter_small_cap_symbols(
    tickers: pd.DataFrame,
    coingecko: CoinGeckoClient,
    max_market_cap_usd: float,
    min_volume_usd: float,
    max_symbols: int,
) -> pd.DataFrame:
    """
    Filter Toobit tickers down to small-cap names.
    If market cap lookup completely fails, we fall back to symbols
    that appear in FALLBACK_TICKERS.
    """
    if tickers.empty:
        return tickers
    t = tickers[tickers["quote_volume_24h"] >= min_volume_usd].copy()
    if t.empty:
        return t

    syms = t["symbol"].tolist()
    mc_map = coingecko.get_market_caps_for_symbols(syms)
    if mc_map:
        t["market_cap_usd"] = t["symbol"].map(mc_map).fillna(0.0)
        t = t[(t["market_cap_usd"] > 0) & (t["market_cap_usd"] <= max_market_cap_usd)]
    else:
        # Fallback: keep only symbols that look like small/meme coins
        t["market_cap_usd"] = 0.0
        base_syms = t["symbol"].str.replace("USDT", "", regex=False)
        mask = base_syms.isin(FALLBACK_TICKERS)
        # If no fallback hit, just take the lowest-volume top names
        # (these are most likely to be small caps)
        if mask.sum() == 0:
            t = t.sort_values("quote_volume_24h", ascending=True)
        else:
            t = t[mask]
    t = t.sort_values("quote_volume_24h", ascending=False).head(max_symbols)
    t = t.reset_index(drop=True)
    return t
