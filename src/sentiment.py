"""
Multi-source sentiment analysis (Layer 12 of the pipeline).

Sources:
  - Fear & Greed Index (alternative.me, free)
  - BTC dominance (CoinGecko, free)
  - Stablecoin total supply change (defillama, free)
  - CryptoPanic news sentiment (free, with optional auth)
  - LunarCrush sentiment (already integrated)
  - Google Trends sentiment (already integrated)

Each source returns 0..100 with a confidence level. We aggregate them
into a single `market_sentiment_score` that the rule engine and ML can
consume.
"""
from __future__ import annotations
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class SentimentSnapshot:
    fear_greed: float = 50.0
    fear_greed_label: str = "Neutral"
    btc_dominance: float = 50.0
    btc_dominance_change_24h: float = 0.0
    stablecoin_supply_change_7d_pct: float = 0.0
    news_sentiment: float = 50.0     # 0..100
    news_volume_24h: int = 0
    bullish_news_pct: float = 50.0
    aggregate: float = 50.0
    risk_regime: str = "NEUTRAL"     # RISK_ON / NEUTRAL / RISK_OFF
    sources_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Fear & Greed Index
# ----------------------------------------------------------------------------
class FearGreedClient:
    BASE = "https://api.alternative.me/fng/"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.timeout = timeout

    def get(self) -> tuple:
        try:
            r = self.session.get(
                f"{self.BASE}?limit=1&format=json",
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return 50.0, "Neutral"
            data = r.json()
            d = (data.get("data") or [{}])[0]
            return float(d.get("value", 50)), str(d.get("value_classification", "Neutral"))
        except Exception:
            return 50.0, "Neutral"


# ----------------------------------------------------------------------------
# BTC Dominance
# ----------------------------------------------------------------------------
class BTCDominanceClient:
    BASE = "https://api.coingecko.com/api/v3/global"

    def __init__(self, timeout: int = 10, cache_path: str | None = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.timeout = timeout
        self.cache_path = cache_path

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


# ----------------------------------------------------------------------------
# Stablecoin Supply (DeFiLlama)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# CryptoPanic News (free, no auth for public posts)
# ----------------------------------------------------------------------------
class CryptoPanicClient:
    BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: Optional[str] = None, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "toobit-scanner/1.0"})
        self.auth_token = auth_token
        self.timeout = timeout

    def get_sentiment(self, symbol: str = None) -> Dict:
        """
        Returns:
          - sentiment: 0..100 (50 neutral, >50 bullish, <50 bearish)
          - volume_24h: number of news items in last 24h
          - bullish_pct: % of bullish-tagged news
        """
        try:
            params = {"kind": "news"}
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
            for post in results:
                # Filter to last 24h
                published = post.get("published_at", "")
                try:
                    ts = pd.Timestamp(published).timestamp()
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


# ----------------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------------
def aggregate_sentiment(
    fear_greed: tuple,
    btc_dominance: tuple,
    stablecoin_change_7d: float,
    news: Dict,
    cache_path: Optional[str] = None,
) -> SentimentSnapshot:
    """
    Combine the sources into a single SentimentSnapshot.
    The aggregate is a weighted blend that the rule engine consumes.
    """
    snap = SentimentSnapshot()
    fg_value, fg_label = fear_greed
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

    # Aggregate: each source contributes 0..100; weighted blend
    # Weighting rationale:
    #  - F&G: 30% (most studied indicator)
    #  - Stablecoin supply: 30% (capital flows)
    #  - News: 25% (real-time but noisy)
    #  - BTC dominance: 15% (inversely correlated with alt-season)
    parts = []
    parts.append((snap.fear_greed, 0.30))
    # Map stablecoin change to 0..100 (-2% to +2% maps to 0..100)
    if snap.stablecoin_supply_change_7d_pct >= 2:
        sc_score = 100.0
    elif snap.stablecoin_supply_change_7d_pct <= -2:
        sc_score = 0.0
    else:
        sc_score = 50.0 + snap.stablecoin_supply_change_7d_pct * 25
    parts.append((sc_score, 0.30))
    parts.append((snap.news_sentiment, 0.25))
    # BTC dominance: low dominance = alt season = risk-on; high = risk-off
    # Map 35%..65% to 100..0
    if snap.btc_dominance <= 35:
        dom_score = 100.0
    elif snap.btc_dominance >= 65:
        dom_score = 0.0
    else:
        dom_score = 100.0 - (snap.btc_dominance - 35) * (100 / 30)
    parts.append((dom_score, 0.15))

    total_w = sum(p[1] for p in parts) or 1.0
    snap.aggregate = float(sum(p[0] * p[1] for p in parts) / total_w)

    if snap.aggregate >= 70:
        snap.risk_regime = "RISK_ON"
    elif snap.aggregate <= 30:
        snap.risk_regime = "RISK_OFF"
    else:
        snap.risk_regime = "NEUTRAL"

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(snap.__dict__, f, indent=2)
        except Exception:
            pass
    return snap


def build_sentiment(
    symbol: Optional[str] = None,
    cryptopanic_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> SentimentSnapshot:
    """Convenience wrapper: fetch all sources and aggregate."""
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, "sentiment_snapshot.json")
        # Reuse cache if < 30 minutes old
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < 1800:
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
    fg_v = fg.get()
    dom_v = dom.get()
    sc_v = sc.get_supply_change_7d_pct()
    news_v = cp.get_sentiment(symbol=symbol)
    return aggregate_sentiment(fg_v, dom_v, sc_v, news_v, cache_path=cache_path)
