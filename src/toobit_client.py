"""
Toobit exchange client.
Uses Toobit public REST API to:
  - list all USDT-margined perpetual symbols (format: BTC-SWAP-USDT)
  - fetch 24h ticker stats
  - fetch OHLCV klines
"""
from __future__ import annotations
import re
import time
import requests
import pandas as pd
from typing import List, Dict, Optional


class ToobitClient:
    BASE = "https://api.toobit.com"

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "toobit-scanner/1.0",
            "Content-Type": "application/json",
        })
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        url = f"{self.BASE}{path}"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        return {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def perp_to_base(sym: str) -> str:
        """BTC-SWAP-USDT -> BTC."""
        s = sym.replace("-SWAP-USDT", "").replace("USDT", "")
        return s

    @staticmethod
    def base_to_perp(sym: str) -> str:
        """BTCUSDT -> BTC-SWAP-USDT (for kline endpoint)."""
        if "-SWAP-" in sym:
            return sym
        base = sym.replace("USDT", "")
        return f"{base}-SWAP-USDT"

    # ------------------------------------------------------------ discovery
    def get_usdt_perp_symbols(self) -> List[str]:
        """
        Return all USDT-margined perpetual symbol ids, e.g. ['BTC-SWAP-USDT',
        'ETH-SWAP-USDT', ...].
        """
        info = self._get("/api/v1/exchangeInfo")
        if not isinstance(info, dict):
            return []
        contracts = info.get("contracts", []) or []
        out: List[str] = []
        for c in contracts:
            if c.get("status") != "TRADING":
                continue
            if c.get("quoteAsset") != "USDT":
                continue
            if c.get("marginToken") and c.get("marginToken") != "USDT":
                continue
            sym = c.get("symbol") or ""
            if sym and sym not in out:
                out.append(sym)
        return out

    # ------------------------------------------------------------- 24h stats
    def get_24h_tickers(self) -> pd.DataFrame:
        """Return a DataFrame of all 24h tickers on Toobit (USDT-margined perps).

        Toobit's /quote/v1/ticker/24hr returns symbols in the form XXXUSDT
        (without -SWAP-USDT suffix). It is the perpetual market feed.
        Field names are short:
          s  = symbol (e.g. BTCUSDT)
          si = symbolId
          c  = last price
          o  = open
          h  = high
          l  = low
          v  = base volume
          qv = quote volume
          pc = price change (abs)
          pcp= price change percent
        """
        data = self._get("/quote/v1/ticker/24hr")
        if isinstance(data, dict):
            data = data.get("data", data)
        rows = []
        for r in data if isinstance(data, list) else []:
            try:
                sym = r.get("s") or r.get("symbol")
                if not sym:
                    continue
                # Skip test pairs (e.g. TESTBTCTESTX8Z9USDT)
                if "TEST" in sym.upper():
                    continue
                # Skip non-USDT perps
                if not sym.endswith("USDT"):
                    continue
                rows.append({
                    "symbol": sym,
                    "base": self.perp_to_base(sym),
                    "last_price": float(r.get("c", 0) or 0),
                    "quote_volume_24h": float(r.get("qv", 0) or 0),
                    "base_volume_24h": float(r.get("v", 0) or 0),
                    "price_change_pct_24h": float(r.get("pcp", 0) or 0),
                    "high_24h": float(r.get("h", 0) or 0),
                    "low_24h": float(r.get("l", 0) or 0),
                    "open_24h": float(r.get("o", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- klines
    def get_klines(self, symbol: str, interval: str = "4h", limit: int = 300) -> pd.DataFrame:
        """Fetch OHLCV candlesticks from Toobit.

        The kline endpoint expects the perp symbol id (e.g. BTC-SWAP-USDT).
        Returns 11 fields per candle:
          0 open_time (ms)
          1 open
          2 high
          3 low
          4 close
          5 volume (base)
          6 close_time (or 0)
          7 quote_volume
          8 trades
          9 taker_buy_base
          10 taker_buy_quote
        """
        # Convert base+USDT style if user passed it
        sym = self.base_to_perp(symbol) if "USDT" in symbol and "-SWAP-" not in symbol else symbol
        params = {"symbol": sym, "interval": interval, "limit": limit}
        data = self._get("/quote/v1/klines", params=params)
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        rows = []
        for row in data:
            if not isinstance(row, list) or len(row) < 8:
                continue
            try:
                rows.append({
                    "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[7]) if len(row) > 7 else 0.0,
                })
            except (TypeError, ValueError, IndexError):
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.dropna().reset_index(drop=True)
