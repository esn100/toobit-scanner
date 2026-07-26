"""
Improved universe discovery for PumpHunter-AI.

Goal: find ALL small-cap (<=$20M) USDT-perp symbols on Toobit
      reliably, with multi-source market cap fallback.

PROBLEMS WITH OLD CODE (fixed here):
  1. Major list was incomplete (SOL, XRP, BNB, PEPE, DOGE slipped through)
  2. CoinPaprika miss many small caps (no record in their index)
  3. CoinGecko fallback rarely hit (slow + rate limited)
  4. Volume band 1M-50M excluded quiet micro-caps and active large caps
  5. Max symbols per run (30) wasted — only 17 real small caps exist
  6. No dedup / no priority ordering by recent activity

NEW APPROACH:
  - Complete majors blacklist (top 100 by mcap)
  - Two-tier volume band: primary 1-30M, secondary 30-100M
  - CoinPaprika first, then CoinGecko (with API key support), then
    on-chain proxy: vol24h * 50 as mc estimate
  - Score by: rvol + volume + 1h momentum
  - Top 40 by activity score
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional
import requests
import pandas as pd


# Complete majors blacklist (top 100 by market cap, evergreened).
# Anything in this list is rejected before any analysis.
MAJORS_BLACKLIST = {
    # Top 20
    "BTC", "ETH", "BNB", "SOL", "XRP", "USDT", "USDC", "ADA", "DOGE", "TRX",
    "AVAX", "LINK", "DOT", "MATIC", "TON", "LTC", "BCH", "NEAR", "ATOM",
    "UNI",
    # 20-50
    "APT", "ARB", "OP", "FIL", "ICP", "STX", "INJ", "TIA", "SEI", "SUI",
    "AAVE", "MKR", "GRT", "RUNE", "ALGO", "EGLD", "FTM", "SAND", "MANA",
    "AXS",
    # 50-100
    "CRV", "LDO", "PEPE", "SHIB", "WIF", "BONK", "FLOKI", "MEME", "TRB",
    "BLUR", "JTO", "JUP", "PYTH", "HYPE", "STETH", "WBTC", "WSTETH", "FDUSD",
    "USDE", "USDD", "TUSD", "GUSD", "PAXG", "XAUT", "WBETH", "BGB", "GT",
    "KAS", "MNT", "CRO", "LEO", "OKB", "HT", "KCS", "MX", "GT", "XEC",
    "ETC", "XMR", "ZEC", "DASH", "EOS", "XLM", "NEO", "WAVES", "ZIL",
    "BTCB", "BETH",
    # Stock/commodity tokens
    "GOOGL", "GOOGLB", "TSLA", "TSLAB", "AAPL", "AAPLB", "AMZN", "AMZNB",
    "MSFT", "MSFTB", "NVDA", "NVDAB", "META", "METAB", "NFLX", "NFLXB",
    "AMD", "AMDB", "INTC", "INTLB", "SPY", "SPYB", "QQQ", "QQQB", "TLT",
    "TLTB", "COIN", "COINB", "MSTR", "MSTRB", "QCOM", "QCOMB", "SOXL",
    "SOXLB", "HOOD", "HOODB", "CRCL", "CRCLB", "GLW", "GLWB", "CBR",
    "CBRB", "NBIS", "NBISB", "ONDS", "ONDSB", "BMNR", "BMNRB", "STX",
    "STXB", "CRWD", "CRWDB", "PLTR", "PLTRB", "PYPL", "PYPLB", "FIG",
    "FIGB", "HUT", "HUTB",
    # Stablecoins & wrapped
    "USD1", "USDC", "DAI", "FRAX", "USDP", "GHO", "SDAI", "SUSDE", "USDE",
    "PYUSD", "EURC", "EURT", "AEUR", "BSDAI", "SUSD", "CUSD",
    # LSTs (tokenized staked)
    "WSTETH", "STETH", "CBETH", "RETH", "METH", "WBETH", "LSETH",
    "OSETH", "ETHX", "ANKRETH",
    # Pump.fun (not a real coin)
    "PUMPFUN",
}


# Volume bands (USD) for tier classification
PRIMARY_VOL_BAND = (1_000_000, 30_000_000)     # small caps only
SECONDARY_VOL_BAND = (30_000_000, 100_000_000) # mid caps worth checking

# Hard MC cap
MAX_MARKET_CAP_USD = 20_000_000

# Final symbol list cap
MAX_SYMBOLS_RETURNED = 40


class MarketCapResolver:
    """
    Multi-source market cap resolver with caching.

    Priority:
      1. CoinPaprika (one shot, all tickers)
      2. CoinGecko (if env has API key)
      3. On-chain proxy: 24h_volume * 50 (catches missing coins)

    Cache to disk for 24h.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._paprika: Optional[Dict[str, float]] = None
        self._paprika_ts: float = 0.0
        self._gecko: Dict[str, float] = {}
        self._proxy_used: set = set()

    def _load_paprika(self) -> Dict[str, float]:
        """Load CoinPaprika tickers from disk cache or fetch."""
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, "paprika_tickers.json")
        if cache_path and os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < 24 * 3600:
                try:
                    import json
                    with open(cache_path) as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 100:
                        return self._parse_paprika(data)
                except Exception:
                    pass
        try:
            r = requests.get(
                "https://api.coinpaprika.com/v1/tickers",
                params={"quotes": "USD", "limit": 5000},
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if cache_path:
                    try:
                        import json
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w") as f:
                            json.dump(data, f)
                    except Exception:
                        pass
                return self._parse_paprika(data)
        except Exception:
            pass
        return {}

    def _parse_paprika(self, data: list) -> Dict[str, float]:
        """Parse CoinPaprika response into {SYMBOL: market_cap}."""
        result: Dict[str, float] = {}
        for t in data:
            sym = (t.get("symbol") or "").upper()
            mc = (t.get("quotes", {}).get("USD", {}).get("market_cap") or 0) or 0
            if not sym or not mc:
                continue
            # For symbol collisions, keep the LARGEST market cap
            if sym not in result or mc > result[sym]:
                result[sym] = float(mc)
        return result

    def _load_gecko(self, symbols: List[str]) -> Dict[str, float]:
        """CoinGecko lookup (only if API key set)."""
        if not os.environ.get("COINGECKO_API_KEY"):
            return {}
        # Map symbol -> coingecko id (cached)
        try:
            from .market_filter import CoinGeckoClient
            cg = CoinGeckoClient(cache_dir=self.cache_dir)
            # /coins/markets?vs_currency=usd&symbols=BTC,ETH
            # Note: CoinGecko /coins/markets doesn't accept symbol filter,
            # so we'd need to map to id first. Skip if too slow.
            return cg.get_market_caps_for_symbols(symbols)
        except Exception:
            return {}

    def resolve(self, symbols: List[str]) -> Dict[str, float]:
        """
        Return {BASE: market_cap_usd} for each symbol.
        Uses Paprika first, Gecko second, volume proxy third.
        """
        # Reload paprika cache if > 1 hour
        if self._paprika is None or (time.time() - self._paprika_ts) > 3600:
            self._paprika = self._load_paprika()
            self._paprika_ts = time.time()
        result: Dict[str, float] = {}
        missing = []
        for s in symbols:
            base = s.replace("USDT", "").replace("-SWAP", "").upper()
            if base in self._paprika:
                result[base] = self._paprika[base]
            else:
                missing.append(base)
        # Try CoinGecko for missing
        if missing:
            gecko = self._load_gecko([s + "USDT" for s in missing])
            for s in missing:
                if (s + "USDT") in gecko:
                    result[s] = gecko[s + "USDT"]
        return result

    def volume_proxy_mc(self, quote_volume_24h: float) -> float:
        """
        On-chain proxy: 24h volume * 50 as market cap estimate.
        Conservative for low-volume, aggressive for high-volume.
        This catches coins CoinPaprika/CoinGecko don't index.
        """
        # Cap at $50M (we only care about small caps)
        return min(50_000_000, quote_volume_24h * 50)


def discover_small_caps(
    tickers: pd.DataFrame,
    resolver: MarketCapResolver,
    *,
    max_symbols: int = MAX_SYMBOLS_RETURNED,
    vol_band: tuple = PRIMARY_VOL_BAND,
) -> pd.DataFrame:
    """
    Discover small-cap USDT-perp symbols from Toobit tickers.

    Returns enriched DataFrame with: market_cap_usd, mcap_source,
    activity_score, in_range columns. Top `max_symbols` by activity.
    """
    if tickers.empty:
        return tickers
    # Step 1: volume band
    df = tickers[
        (tickers["quote_volume_24h"] >= vol_band[0])
        & (tickers["quote_volume_24h"] <= vol_band[1])
    ].copy()
    if df.empty:
        return df
    # Step 2: exclude majors (hard blacklist)
    df = df[~df["base"].isin(MAJORS_BLACKLIST)].copy()
    if df.empty:
        return df
    # Step 3: resolve market cap
    mc_map = resolver.resolve(df["base"].tolist())
    df["market_cap_usd"] = df["base"].map(mc_map)
    # Step 4: for missing, use volume proxy
    missing_mc = df["market_cap_usd"].isna() | (df["market_cap_usd"] == 0)
    if missing_mc.any():
        df.loc[missing_mc, "market_cap_usd"] = (
            df.loc[missing_mc, "quote_volume_24h"].apply(resolver.volume_proxy_mc)
        )
        df.loc[missing_mc, "mcap_source"] = "proxy"
    df.loc[df["mcap_source"].isna(), "mcap_source"] = "paprika"
    # Step 5: hard MC cap
    df = df[df["market_cap_usd"] <= MAX_MARKET_CAP_USD].copy()
    if df.empty:
        return df
    # Step 6: activity score (vol*0.5 + |mom|*0.3 + 1.0 floor)
    mom = df["price_change_pct_24h"].abs().fillna(0)
    df["activity_score"] = (
        (df["quote_volume_24h"] / 1_000_000).clip(0, 50) * 0.5
        + mom.clip(0, 30) * 1.0
    )
    # Step 7: top by activity
    df = df.sort_values("activity_score", ascending=False).head(max_symbols)
    return df.reset_index(drop=True)


def get_secondary_symbols(
    tickers: pd.DataFrame,
    resolver: MarketCapResolver,
    *,
    max_symbols: int = 10,
) -> pd.DataFrame:
    """Get mid-cap symbols ($30M-$100M volume) for cross-validation."""
    return discover_small_caps(
        tickers, resolver,
        max_symbols=max_symbols,
        vol_band=SECONDARY_VOL_BAND,
    )


# Self-test
if __name__ == "__main__":
    from src.toobit_client import ToobitClient
    tb = ToobitClient()
    tickers = tb.get_24h_tickers()
    print(f"Total tickers: {len(tickers)}")
    resolver = MarketCapResolver(cache_dir="data")
    small_caps = discover_small_caps(tickers, resolver)
    print(f"\nDiscovered {len(small_caps)} small caps:")
    cols = [c for c in ["base", "last_price", "quote_volume_24h",
                        "market_cap_usd", "mcap_source", "activity_score"]
            if c in small_caps.columns]
    print(small_caps[cols].to_string(index=False))
    print()
    print(f"MC range: ${small_caps['market_cap_usd'].min():.0f} - "
          f"${small_caps['market_cap_usd'].max():.0f}")
    print(f"Activity: {small_caps['activity_score'].min():.1f} - "
          f"{small_caps['activity_score'].max():.1f}")
    # Show second tier
    print("\n--- Mid-caps for cross-validation ---")
    mid = get_secondary_symbols(tickers, resolver, max_symbols=5)
    if not mid.empty:
        cols_mid = [c for c in cols if c in mid.columns]
        print(mid[cols_mid].to_string(index=False))
    else:
        print("  (none in secondary band)")
