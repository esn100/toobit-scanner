"""
Enhanced multi-source sentiment for market regime detection.

Sources (all free):
  1. Fear & Greed Index (alternative.me)
  2. BTC dominance + change (CoinGecko)
  3. Stablecoin supply change 7d (DeFiLlama)
  4. CryptoPanic news sentiment (with optional auth)
  5. BTC price action (24h change, drawdown)

Output: SentimentSnapshot with aggregate 0..100 and risk_regime
  - RISK_ON: bullish bias, long signals preferred
  - NEUTRAL: both directions equally valid
  - RISK_OFF: bearish bias, short signals preferred
  - PANIC: extreme fear, only highest-conviction shorts
"""
from __future__ import annotations
import os
import json
import time
import requests
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class SentimentSnapshot:
    fear_greed: float = 50.0
    fear_greed_label: str = "Neutral"
    btc_dominance: float = 50.0
    btc_dominance_change_24h: float = 0.0
    stablecoin_supply_change_7d_pct: float = 0.0
    news_sentiment: float = 50.0
    news_volume_24h: int = 0
    bullish_news_pct: float = 50.0
    btc_change_24h_pct: float = 0.0
    btc_change_7d_pct: float = 0.0
    btc_drawdown_30d_pct: float = 0.0
    btc_volatility_24h_pct: float = 2.0
    aggregate: float = 50.0
    risk_regime: str = "NEUTRAL"     # RISK_ON / NEUTRAL / RISK_OFF / PANIC
    long_bias: float = 1.0          # multiplier for long signals
    short_bias: float = 1.0         # multiplier for short signals
    sources_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Source 1: Fear & Greed Index
# ============================================================================
class FearGreedClient:
    BASE = "https://api.alternative.me/fng/"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.timeout = timeout

    def get(self) -> tuple:
        try:
            r = self.session.get(
                f"{self.BASE}?limit=7&format=json",
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return 50.0, "Neutral", 0.0
            data = r.json()
            rows = data.get("data") or []
            if not rows:
                return 50.0, "Neutral", 0.0
            current = float(rows[0].get("value", 50))
            label = str(rows[0].get("value_classification", "Neutral"))
            # 7-day change
            if len(rows) >= 7:
                week_ago = float(rows[6].get("value", 50))
                change_7d = current - week_ago
            else:
                change_7d = 0.0
            return current, label, change_7d
        except Exception:
            return 50.0, "Neutral", 0.0


# ============================================================================
# Source 2: BTC Dominance (CoinGecko)
# ============================================================================
class BTCDominanceClient:
    BASE = "https://api.coingecko.com/api/v3/global"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout

    def get(self) -> tuple:
        try:
            r = self.session.get(self.BASE, timeout=self.timeout)
            if r.status_code != 200:
                return 50.0, 0.0
            d = r.json().get("data", {})
            dom = float(d.get("market_cap_percentage", {}).get("btc", 50.0))
            change = float(d.get("market_cap_change_percentage_24h_usd", 0.0))
            return dom, change
        except Exception:
            return 50.0, 0.0


# ============================================================================
# Source 3: Stablecoin Supply (DeFiLlama)
# ============================================================================
class StablecoinSupplyClient:
    BASE = "https://stablecoins.llama.fi"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.timeout = timeout

    def get_supply_change_7d_pct(self) -> float:
        """
        Total stablecoin supply change over last 7 days (%).
        Positive = dry powder entering the market (bullish).
        Negative = capital leaving (bearish).
        """
        try:
            r = self.session.get(
                f"{self.BASE}/stablecoincharts/all",
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return 0.0
            d = r.json()
            if not isinstance(d, list) or len(d) < 8:
                return 0.0
            recent = d[-1].get("totalCirculating", {}).get("peggedUSD", 0)
            week_ago = d[-8].get("totalCirculating", {}).get("peggedUSD", 0)
            if week_ago <= 0:
                return 0.0
            return float((recent - week_ago) / week_ago * 100.0)
        except Exception:
            return 0.0


# ============================================================================
# Source 4: CryptoPanic News
# ============================================================================
class CryptoPanicClient:
    BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: Optional[str] = None, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.auth_token = auth_token
        self.timeout = timeout

    def get_sentiment(self, symbol: str = None) -> Dict:
        try:
            params = {"kind": "news", "filter": "rising"}
            if symbol:
                params["currencies"] = symbol
            if self.auth_token:
                params["auth_token"] = self.auth_token
            r = self.session.get(self.BASE, params=params, timeout=self.timeout)
            if r.status_code != 200:
                return {"sentiment": 50.0, "volume_24h": 0, "bullish_pct": 50.0}
            d = r.json()
            results = d.get("results") or []
            if not results:
                return {"sentiment": 50.0, "volume_24h": 0, "bullish_pct": 50.0}
            bullish = 0
            bearish = 0
            volume = 0
            now = time.time()
            for post in results[:100]:
                published = post.get("published_at", "")
                try:
                    ts = pd_to_timestamp(published)
                    if now - ts > 86400:
                        continue
                except Exception:
                    pass
                volume += 1
                votes = post.get("votes", {}) or {}
                if votes.get("bullish", 0) > votes.get("bearish", 0):
                    bullish += 1
                elif votes.get("bearish", 0) > 0:
                    bearish += 1
            if volume == 0:
                return {"sentiment": 50.0, "volume_24h": 0, "bullish_pct": 50.0}
            bullish_pct = (bullish / volume) * 100.0
            sentiment = bullish_pct
            return {
                "sentiment": float(sentiment),
                "volume_24h": int(volume),
                "bullish_pct": float(bullish_pct),
            }
        except Exception:
            return {"sentiment": 50.0, "volume_24h": 0, "bullish_pct": 50.0}


def pd_to_timestamp(s: str) -> float:
    import pandas as pd
    try:
        return pd.Timestamp(s).timestamp()
    except Exception:
        return 0.0


# ============================================================================
# Source 5: BTC Price Action (via Toobit or OKX)
# ============================================================================
class BTCPriceAction:
    """Compute BTC price action metrics from recent klines."""
    def __init__(self, toobit_or_okx):
        self.provider = toobit_or_okx

    def get(self, interval: str = "1h", bars: int = 200) -> Dict:
        try:
            if hasattr(self.provider, "get_klines"):
                df = self.provider.get_klines("BTCUSDT", interval, bars)
            else:
                return {"change_24h_pct": 0.0, "change_7d_pct": 0.0,
                        "drawdown_30d_pct": 0.0, "volatility_24h_pct": 2.0}
            if df.empty or len(df) < 30:
                return {"change_24h_pct": 0.0, "change_7d_pct": 0.0,
                        "drawdown_30d_pct": 0.0, "volatility_24h_pct": 2.0}
            close = df["close"].astype(float)
            last = float(close.iloc[-1])
            # 24h change
            if len(close) >= 25:
                change_24h = float(
                    (last - float(close.iloc[-25])) / float(close.iloc[-25]) * 100
                )
            else:
                change_24h = 0.0
            # 7d change
            if len(close) >= 169:
                change_7d = float(
                    (last - float(close.iloc[-169])) / float(close.iloc[-169]) * 100
                )
            else:
                change_7d = 0.0
            # 30d drawdown
            if len(close) >= 169:
                high_30d = float(close.tail(169).max())
                dd_30d = float((last - high_30d) / high_30d * 100)
            else:
                dd_30d = 0.0
            # 24h volatility
            if len(close) >= 25:
                ret = close.pct_change().tail(24).dropna()
                vol_24h = float(ret.std() * 100)
            else:
                vol_24h = 2.0
            return {
                "change_24h_pct": change_24h,
                "change_7d_pct": change_7d,
                "drawdown_30d_pct": dd_30d,
                "volatility_24h_pct": vol_24h,
            }
        except Exception as e:
            return {"change_24h_pct": 0.0, "change_7d_pct": 0.0,
                    "drawdown_30d_pct": 0.0, "volatility_24h_pct": 2.0}


# ============================================================================
# Aggregator
# ============================================================================
def aggregate_sentiment(
    fear_greed: tuple,
    btc_dominance: tuple,
    stablecoin_change_7d: float,
    news: Dict,
    btc_action: Dict,
    cache_path: Optional[str] = None,
) -> SentimentSnapshot:
    snap = SentimentSnapshot()
    fg_value, fg_label, fg_change_7d = fear_greed
    snap.fear_greed = float(fg_value)
    snap.fear_greed_label = fg_label
    snap.sources_used.append("fear_greed")
    dom, dom_change = btc_dominance
    snap.btc_dominance = float(dom)
    snap.btc_dominance_change_24h = float(dom_change)
    snap.sources_used.append("btc_dominance")
    snap.stablecoin_supply_change_7d_pct = float(stablecoin_change_7d)
    snap.sources_used.append("stablecoin_supply")
    snap.news_sentiment = float(news.get("sentiment", 50.0))
    snap.news_volume_24h = int(news.get("volume_24h", 0))
    snap.bullish_news_pct = float(news.get("bullish_pct", 50.0))
    snap.sources_used.append("news")
    snap.btc_change_24h_pct = float(btc_action.get("change_24h_pct", 0.0))
    snap.btc_change_7d_pct = float(btc_action.get("change_7d_pct", 0.0))
    snap.btc_drawdown_30d_pct = float(btc_action.get("drawdown_30d_pct", 0.0))
    snap.btc_volatility_24h_pct = float(btc_action.get("volatility_24h_pct", 2.0))
    snap.sources_used.append("btc_action")

    # Aggregate score 0..100 (higher = more bullish)
    parts = []
    # 1) Fear & Greed (weight 0.20)
    parts.append((snap.fear_greed, 0.20))
    # 2) Stablecoin supply (-2..+2 -> 0..100) (weight 0.20)
    if snap.stablecoin_supply_change_7d_pct >= 2:
        sc_score = 100.0
    elif snap.stablecoin_supply_change_7d_pct <= -2:
        sc_score = 0.0
    else:
        sc_score = 50.0 + snap.stablecoin_supply_change_7d_pct * 25
    parts.append((sc_score, 0.20))
    # 3) News (weight 0.15)
    parts.append((snap.news_sentiment, 0.15))
    # 4) BTC dominance: low = alt season (bullish), high = risk-off
    if snap.btc_dominance <= 35:
        dom_score = 100.0
    elif snap.btc_dominance >= 65:
        dom_score = 0.0
    else:
        dom_score = 100.0 - (snap.btc_dominance - 35) * (100 / 30)
    parts.append((dom_score, 0.10))
    # 5) BTC price action 24h (weight 0.20) -/+5% -> 0..100
    change_24h = snap.btc_change_24h_pct
    if change_24h >= 5:
        pa_score = 100.0
    elif change_24h <= -5:
        pa_score = 0.0
    else:
        pa_score = 50.0 + change_24h * 10
    parts.append((pa_score, 0.20))
    # 6) BTC drawdown 30d (weight 0.15) -10% -> 0, 0% -> 100
    dd = snap.btc_drawdown_30d_pct
    if dd >= 0:
        dd_score = 100.0
    elif dd <= -20:
        dd_score = 0.0
    else:
        dd_score = 100.0 + dd * 5  # dd=-10 -> 50
    parts.append((dd_score, 0.15))
    total_w = sum(p[1] for p in parts) or 1.0
    snap.aggregate = float(sum(p[0] * p[1] for p in parts) / total_w)

    # Risk regime
    if snap.aggregate >= 65:
        snap.risk_regime = "RISK_ON"
    elif snap.aggregate <= 25:
        snap.risk_regime = "PANIC"
    elif snap.aggregate <= 40:
        snap.risk_regime = "RISK_OFF"
    else:
        snap.risk_regime = "NEUTRAL"

    # Direction bias multipliers
    if snap.risk_regime == "RISK_ON":
        snap.long_bias = 1.4
        snap.short_bias = 0.5
    elif snap.risk_regime == "NEUTRAL":
        snap.long_bias = 1.0
        snap.short_bias = 1.0
    elif snap.risk_regime == "RISK_OFF":
        snap.long_bias = 0.4
        snap.short_bias = 1.5
    elif snap.risk_regime == "PANIC":
        snap.long_bias = 0.0
        snap.short_bias = 1.8
    return snap


def build_sentiment_v2(
    toobit_or_okx,
    cryptopanic_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> SentimentSnapshot:
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "sentiment_v2.json")
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < 1800:  # 30 min cache
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    return SentimentSnapshot(**d)
                except Exception:
                    pass
    fg = FearGreedClient()
    dom = BTCDominanceClient()
    sc = StablecoinSupplyClient()
    cp = CryptoPanicClient(auth_token=cryptopanic_token)
    btc_pa = BTCPriceAction(toobit_or_okx)
    fg_v = fg.get()
    dom_v = dom.get()
    sc_v = sc.get_supply_change_7d_pct()
    news_v = cp.get_sentiment()
    btc_v = btc_pa.get()
    snap = aggregate_sentiment(fg_v, dom_v, sc_v, news_v, btc_v, cache_path)
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(asdict(snap), f, indent=2)
        except Exception:
            pass
    return snap
