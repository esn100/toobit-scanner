"""
Binance public market data client (for historical data only).

Binance has 1000+ days of kline data via the public API, with
much more reliable and complete history than smaller exchanges.
This client is used ONLY for historical data; live signals are
still generated on Toobit.
"""
from __future__ import annotations
import time
import requests
import pandas as pd
from typing import List, Optional


class BinanceHistory:
    BASE = "https://api.binance.com"
    FUTURES_BASE = "https://fapi.binance.com"

    def __init__(self, timeout: int = 15, futures: bool = True):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout
        self.base = self.FUTURES_BASE if futures else self.BASE

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        for attempt in range(3):
            try:
                r = self.session.get(
                    f"{self.base}{path}",
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
                    return None
                time.sleep(1 + attempt)
        return None

    def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines. With pagination, can fetch up to 1000 bars
        per request. To get more, use start_time/end_time.
        """
        all_rows = []
        current_start = start_time
        while True:
            params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
            if current_start is not None:
                params["startTime"] = current_start
            if end_time is not None:
                params["endTime"] = end_time
            data = self._get("/fapi/v1/klines", params) if "/fapi" in self.base else \
                   self._get("/api/v3/klines", params)
            if not data or not isinstance(data, list):
                break
            if not data:
                break
            for row in data:
                all_rows.append({
                    "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": pd.to_datetime(int(row[6]), unit="ms", utc=True),
                    "quote_volume": float(row[7]),
                    "trades": int(row[8]),
                })
            if len(data) < 1000 or len(all_rows) >= limit:
                break
            # Next batch: start from last close_time + 1ms
            current_start = int(data[-1][6]) + 1
            time.sleep(0.1)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        return df.head(limit) if len(df) > limit else df

    def get_usdt_perp_symbols(self) -> List[str]:
        """Return all USDT-margined perpetual symbols on Binance."""
        data = self._get("/fapi/v1/exchangeInfo", {})
        if not isinstance(data, dict):
            return []
        out = []
        for s in data.get("symbols", []):
            if (s.get("status") == "TRADING"
                    and s.get("quoteAsset") == "USDT"
                    and s.get("contractType") == "PERPETUAL"):
                out.append(s["symbol"])
        return out

    def to_toobit_symbol(self, binance_symbol: str) -> str:
        """
        Convert BTCUSDT -> BTCUSDT (same format).
        For symbols unique to Binance, return None.
        """
        # Most USDT perps use the same symbol on both exchanges
        return binance_symbol

    def get_binance_history_for_toobit_symbol(
        self, toobit_symbol: str, interval: str = "1h", days: int = 90,
    ) -> pd.DataFrame:
        """
        Fetch historical data from Binance for a Toobit symbol.
        """
        # Most symbols are the same. For special ones we'd need a map.
        return self.get_klines(toobit_symbol, interval, days * 24 + 50)
