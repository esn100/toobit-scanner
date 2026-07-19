"""
OKX public market data client (for historical data).

OKX is accessible from Iran and has 1000+ days of kline history via
the public v5 API. This client is used ONLY for historical data
(Elliott/Fib/Ichimoku); live signals are still generated on Toobit.
"""
from __future__ import annotations
import time
import requests
import pandas as pd
from typing import List, Optional


class OKXHistory:
    BASE = "https://www.okx.com/api/v5/market"

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> object:
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
                    return None
                time.sleep(1 + attempt)
        return None

    def to_okx_inst(self, toobit_symbol: str) -> Optional[str]:
        """
        Convert a Toobit symbol like BTCUSDT to OKX SWAP instrument like
        BTC-USDT-SWAP. Returns None if cannot map.
        """
        if "USDT" not in toobit_symbol.upper():
            return None
        base = toobit_symbol.upper().replace("USDT", "")
        return f"{base}-USDT-SWAP"

    def get_klines(
        self, inst_id: str, bar: str = "1H", limit: int = 300,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV klines from OKX. With pagination, fetches up to `limit` bars."""
        all_rows = []
        current_end = end_ts
        while True:
            params = {"instId": inst_id, "bar": bar, "limit": min(limit, 300)}
            if current_end is not None:
                params["after"] = current_end  # OKX uses 'after' to get older
            data = self._get("/candles", params)
            if not data or data.get("code") != "0":
                break
            rows = data.get("data", [])
            if not rows:
                break
            for row in rows:
                try:
                    all_rows.append({
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
            if len(rows) < 300 or len(all_rows) >= limit:
                break
            # OKX returns newest first when using 'after'; use oldest ts as next 'after'
            oldest_ts = int(rows[-1][0])
            if start_ts is not None and oldest_ts <= start_ts:
                break
            current_end = oldest_ts
            time.sleep(0.1)
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        return df.head(limit) if len(df) > limit else df

    def get_history_for_toobit_symbol(
        self, toobit_symbol: str, interval: str = "1H", bars: int = 300,
    ) -> pd.DataFrame:
        """
        Fetch historical data from OKX for a Toobit symbol.
        Maps toobit_symbol -> OKX instId automatically.
        """
        inst = self.to_okx_inst(toobit_symbol)
        if inst is None:
            return pd.DataFrame()
        return self.get_klines(inst, interval, bars)

    def get_all_swap_symbols(self) -> List[str]:
        """Return all USDT-margined swap instruments."""
        data = self._get("/instruments", {"instType": "SWAP"})
        if not data or data.get("code") != "0":
            return []
        out = []
        for inst in data.get("data", []):
            if inst.get("settleCcy") == "USDT" and inst.get("state") == "live":
                out.append(inst.get("instId", ""))
        return out
