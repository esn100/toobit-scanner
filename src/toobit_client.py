"""
Toobit exchange client.
Uses Toobit public REST API to:
  - list all USDT-margined perpetual symbols
  - fetch 24h ticker stats
  - fetch OHLCV klines
"""
from __future__ import annotations
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

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
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

    def get_usdt_perp_symbols(self) -> List[str]:
        """List all USDT perpetual symbols on Toobit."""
        # Toobit uses /api/v1/exchangeInfo or /quote/v1/ticker/24hr
        data = self._get("/api/v1/exchangeInfo")
        symbols: List[str] = []
        try:
            for s in data.get("symbols", []):
                if (
                    s.get("status") == "TRADING"
                    and s.get("quoteAsset") == "USDT"
                    and s.get("isContract", True)  # perpetuals are contracts
                ):
                    sym = s.get("symbol", "")
                    if sym.endswith("USDT") and not sym.endswith("USDCUSDT"):
                        symbols.append(sym)
        except Exception:
            # Fallback: use 24h ticker endpoint to discover symbols
            data = self._get("/quote/v1/ticker/24hr")
            for row in data if isinstance(data, list) else []:
                sym = row.get("symbol", "")
                if sym.endswith("USDT"):
                    symbols.append(sym)
        # Deduplicate while preserving order
        seen, uniq = set(), []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def get_24h_tickers(self) -> pd.DataFrame:
        """Return a DataFrame of all 24h tickers on Toobit."""
        data = self._get("/quote/v1/ticker/24hr")
        if isinstance(data, dict):
            data = data.get("data", data)
        rows = []
        for r in data if isinstance(data, list) else []:
            try:
                rows.append({
                    "symbol": r.get("symbol"),
                    "last_price": float(r.get("lastPrice", 0) or 0),
                    "quote_volume_24h": float(r.get("quoteVolume", 0) or 0),
                    "base_volume_24h": float(r.get("volume", 0) or 0),
                    "price_change_pct_24h": float(r.get("priceChangePercent", 0) or 0),
                    "high_24h": float(r.get("highPrice", 0) or 0),
                    "low_24h": float(r.get("lowPrice", 0) or 0),
                    "open_24h": float(r.get("openPrice", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
        return pd.DataFrame(rows)

    def get_klines(self, symbol: str, interval: str = "4h", limit: int = 300) -> pd.DataFrame:
        """Fetch OHLCV candlesticks. Toobit follows Binance-style endpoints."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = self._get("/quote/v1/klines", params=params)
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]
        df = df.dropna().reset_index(drop=True)
        return df
