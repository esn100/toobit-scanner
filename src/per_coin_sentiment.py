"""
Per-coin social/news signal collector.

For each symbol we collect, for each cycle:
  1. Google Trends interest (7d)             — public, free, but rate-limited
  2. CryptoPanic news (posts mentioning coin) — free w/o token, better w/ token
  3. Reddit mentions (public JSON endpoint)   — free, no auth
  4. LunarCrush-like aggregate (free alt)     — CoinGecko community data
  5. CoinGecko developer/community stats     — public API

All calls fail-soft: missing data -> 0, never crash the collector.

Heavy: per symbol costs ~3-6 seconds across all sources. We sample:
  - top movers (rvol>1.3 OR atr_pct>5): full
  - everything else: trends + CoinGecko only (cheaper)
"""
from __future__ import annotations
import time
import random
import requests
from typing import Dict, List, Optional
import pandas as pd


USER_AGENT = "pumphunter-ai/1.0 (research; Iran)"


def _session(timeout: int = 8) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


# ---------------------------------------------------------------------------
# Source 1: Google Trends
# ---------------------------------------------------------------------------
class GoogleTrendsCoin:
    """PyTrends wrapper for one coin query."""

    def __init__(self):
        self._pt = None
        self._last_call = 0.0

    def _ensure(self):
        if self._pt is None:
            from pytrends.request import TrendReq
            self._pt = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.5)

    def get(self, query: str) -> dict:
        """
        Returns dict with avg/slope/peak/rising for last 7 days.
        Sleeps to avoid pytrends 429.
        """
        # Rate limit: 1 call per 3-5 seconds
        elapsed = time.time() - self._last_call
        if elapsed < 3.5:
            time.sleep(3.5 - elapsed + random.uniform(0, 0.7))
        self._ensure()
        self._last_call = time.time()
        try:
            self._pt.build_payload([query], timeframe="now 7-d", geo="")
            df = self._pt.interest_over_time()
            if df is None or df.empty or query not in df.columns:
                return {"gt_avg": 0.0, "gt_slope": 0.0, "gt_peak": 0.0,
                        "gt_rising": False, "gt_present": False}
            s = df[query].astype(float)
            n = len(s)
            avg = float(s.mean())
            peak = float(s.max())
            xs = list(range(n))
            if n >= 2:
                mean_x = (n - 1) / 2.0
                num = sum((x - mean_x) * (y - avg) for x, y in zip(xs, s))
                den = sum((x - mean_x) ** 2) or 1
                slope = float(num / den)
            else:
                slope = 0.0
            rising = bool(slope > 0 and s.iloc[-1] > avg)
            return {"gt_avg": avg, "gt_slope": slope, "gt_peak": peak,
                    "gt_rising": rising, "gt_present": True}
        except Exception:
            return {"gt_avg": 0.0, "gt_slope": 0.0, "gt_peak": 0.0,
                    "gt_rising": False, "gt_present": False}


# ---------------------------------------------------------------------------
# Source 2: CoinGecko (community + developer + social)
# ---------------------------------------------------------------------------
class CoinGeckoClient:
    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, timeout: int = 10):
        self.session = _session(timeout)
        # Cache: base symbol -> gecko id (or None if not found)
        self._id_cache: Dict[str, Optional[str]] = {}
        self._last_call = 0.0

    def _throttle(self):
        # CoinGecko free tier: ~10-30 calls/min from one IP. Be safe with
        # 3s minimum + jitter to stay well under the limit when batching.
        elapsed = time.time() - self._last_call
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed + random.uniform(0, 1.0))
        self._last_call = time.time()

    def resolve_id(self, base_symbol: str) -> Optional[str]:
        """Map TLMUSDT -> 'tlm' (CoinGecko id)."""
        if base_symbol in self._id_cache:
            return self._id_cache[base_symbol]
        self._throttle()
        for attempt in range(3):
            try:
                r = self.session.get(f"{self.BASE}/search",
                                     params={"query": base_symbol},
                                     timeout=8)
                if r.status_code == 429:
                    time.sleep(8 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    self._id_cache[base_symbol] = None
                    return None
                coins = r.json().get("coins", []) or []
                sym = base_symbol.lower()
                for c in coins:
                    if c.get("symbol", "").lower() == sym:
                        self._id_cache[base_symbol] = c["id"]
                        return c["id"]
                if coins:
                    self._id_cache[base_symbol] = coins[0]["id"]
                    return coins[0]["id"]
                self._id_cache[base_symbol] = None
                return None
            except Exception:
                if attempt == 2:
                    self._id_cache[base_symbol] = None
                    return None
                time.sleep(3)
        self._id_cache[base_symbol] = None
        return None

    def get_coin(self, base_symbol: str) -> dict:
        """
        Returns community/developer/social stats for the coin.
        """
        empty = {
            "cg_community_score": 0.0, "cg_developer_score": 0.0,
            "cg_liquidity_score": 0.0, "cg_public_interest_score": 0.0,
            "cg_twitter_followers": 0, "cg_reddit_subscribers": 0,
            "cg_telegram_users": 0, "cg_forum_users": 0,
            "cg_stars": 0, "cg_watchlist_users": 0,
            "cg_reddit_avg_48h": 0.0, "cg_reddit_avg_24h": 0.0,
            "cg_reddit_comments_24h": 0.0,
            "cg_price_change_24h_pct": 0.0,
            "cg_community_data_present": False,
        }
        cg_id = self.resolve_id(base_symbol)
        if not cg_id:
            return empty
        self._throttle()
        # Retry on 429 with longer backoff
        d = None
        for attempt in range(3):
            try:
                r = self.session.get(
                    f"{self.BASE}/coins/{cg_id}",
                    params={
                        "localization": "false",
                        "tickers": "false",
                        "market_data": "true",
                        "community_data": "true",
                        "developer_data": "true",
                        "sparkline": "false",
                    },
                    timeout=10,
                )
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)
                    continue
                if r.status_code != 200:
                    return empty
                d = r.json()
                break
            except Exception:
                if attempt == 2:
                    return empty
                time.sleep(5)
        if d is None:
            return empty
        try:
            cd = d.get("community_data") or {}
            dd_ = d.get("developer_data") or {}
            md = d.get("market_data") or {}
            rd = cd.get("reddit_average_posts_48h") or 0
            rc = cd.get("reddit_average_comments_48h") or 0
            rd_24 = (rd or 0) / 2.0
            rc_24 = (rc or 0) / 2.0
            return {
                "cg_community_score": float(cd.get("community_score") or 0),
                "cg_developer_score": float(dd_.get("developer_score") or 0),
                "cg_liquidity_score": float(cd.get("liquidity_score") or 0),
                "cg_public_interest_score": float(cd.get("public_interest_score") or 0),
                "cg_twitter_followers": int(cd.get("twitter_followers") or 0),
                "cg_reddit_subscribers": int(cd.get("reddit_subscribers") or 0),
                "cg_telegram_users": int(cd.get("telegram_channel_user_count") or 0),
                "cg_forum_users": int(cd.get("chat_users") or 0),
                "cg_stars": int(cd.get("stars") or 0),
                "cg_watchlist_users": int(d.get("watchlist_portfolio_users") or 0),
                "cg_reddit_avg_48h": float(rd),
                "cg_reddit_avg_24h": float(rd_24),
                "cg_reddit_comments_24h": float(rc_24),
                "cg_price_change_24h_pct": float(
                    (md.get("price_change_percentage_24h") or 0)
                ),
                "cg_community_data_present": True,
            }
        except Exception:
            return empty


# ---------------------------------------------------------------------------
# Source 3: Reddit (public JSON, no auth)
# ---------------------------------------------------------------------------
class RedditClient:
    """
    Use reddit.com/r/{sub}/search.json for mentions. Free, no auth.
    Only a few subs are useful: cryptomoonshots, cryptomarkets, satoshistreetbets
    """
    SUBS = ["CryptoMoonShots", "CryptoMarkets", "SatoshiStreetBets", "altcoin"]

    def __init__(self, timeout: int = 8):
        self.session = _session(timeout)
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_call = time.time()

    def mentions_24h(self, base_symbol: str) -> dict:
        """
        Count posts mentioning the coin across 4 subs in the last 24h.
        Returns: count, total_score, total_comments, top_post_url
        """
        out = {"rd_post_count_24h": 0, "rd_total_score_24h": 0,
               "rd_total_comments_24h": 0, "rd_top_post_url": ""}
        for sub in self.SUBS:
            self._throttle()
            try:
                r = self.session.get(
                    f"https://www.reddit.com/r/{sub}/search.json",
                    params={"q": base_symbol, "restrict_sr": "on",
                            "sort": "new", "t": "day", "limit": 25},
                    timeout=8,
                )
                if r.status_code != 200:
                    continue
                data = r.json().get("data", {})
                children = data.get("children", [])
                for child in children[:25]:
                    post = child.get("data", {})
                    title = (post.get("title") or "").lower()
                    selftext = (post.get("selftext") or "").lower()
                    # Require whole-word match to avoid false positives
                    if f" {base_symbol.lower()} " in f" {title} " or \
                       f" {base_symbol.lower()} " in f" {selftext} " or \
                       title.startswith(f"${base_symbol.lower()}") or \
                       base_symbol.lower() in title.split():
                        out["rd_post_count_24h"] += 1
                        out["rd_total_score_24h"] += int(post.get("score", 0) or 0)
                        out["rd_total_comments_24h"] += int(
                            post.get("num_comments", 0) or 0
                        )
                        if not out["rd_top_post_url"]:
                            url = post.get("url", "")
                            if url:
                                out["rd_top_post_url"] = str(url)
            except Exception:
                continue
        return out


# ---------------------------------------------------------------------------
# Source 4: CryptoPanic news (free without token, 200 posts/day limit)
# ---------------------------------------------------------------------------
class CryptoPanicCoinClient:
    BASE = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: Optional[str] = None, timeout: int = 8):
        self.session = _session(timeout)
        self.auth_token = auth_token
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_call = time.time()

    def news_for_coin(self, base_symbol: str) -> dict:
        """
        Returns count and bullish/bearish ratio for posts mentioning
        `base_symbol` in last 24h.
        """
        out = {"cp_post_count_24h": 0, "cp_bullish_pct": 50.0,
               "cp_bearish_pct": 50.0, "cp_sentiment": 50.0,
               "cp_top_title": ""}
        self._throttle()
        try:
            params = {"currencies": base_symbol, "filter": "hot"}
            if self.auth_token:
                params["auth_token"] = self.auth_token
            r = self.session.get(self.BASE, params=params, timeout=8)
            if r.status_code != 200:
                return out
            data = r.json()
            results = data.get("results") or []
            bullish = 0
            bearish = 0
            count = 0
            for post in results[:50]:
                votes = post.get("votes", {}) or {}
                b = int(votes.get("bullish", 0) or 0)
                br = int(votes.get("bearish", 0) or 0)
                if b > br:
                    bullish += 1
                elif br > 0:
                    bearish += 1
                count += 1
                if not out["cp_top_title"]:
                    title = (post.get("title") or "").strip()
                    if title:
                        out["cp_top_title"] = title[:200]
            out["cp_post_count_24h"] = count
            if count > 0:
                out["cp_bullish_pct"] = bullish / count * 100
                out["cp_bearish_pct"] = bearish / count * 100
                out["cp_sentiment"] = out["cp_bullish_pct"]
        except Exception:
            pass
        return out


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class CoinSocialAggregator:
    """
    Combines all per-coin social sources.
    Singleton — share across cycles.
    """

    def __init__(self, cryptopanic_token: Optional[str] = None):
        self.gt = GoogleTrendsCoin()
        self.cg = CoinGeckoClient()
        self.rd = RedditClient()
        self.cp = CryptoPanicCoinClient(auth_token=cryptopanic_token)
        # Persistent id cache across cycles
        self._id_cache = self.cg._id_cache

    def collect(self, base_symbol: str, full: bool = True,
                skip_expensive: bool = False) -> dict:
        """
        full=True: all sources (4)
        full=False: trends + coingecko only (cheaper, no Reddit/CryptoPanic)
        skip_expensive=True: CoinGecko only (no trends/reddit/panic)
        """
        out: dict = {}
        # 1. Google Trends (slow due to rate limit)
        if not skip_expensive:
            out.update(self.gt.get(base_symbol))
        else:
            out["gt_avg"] = 0.0
            out["gt_slope"] = 0.0
            out["gt_peak"] = 0.0
            out["gt_rising"] = False
            out["gt_present"] = False
        # 2. CoinGecko (still throttled but cheaper than trends)
        out.update(self.cg.get_coin(base_symbol))
        if full and not skip_expensive:
            # 3. Reddit
            out.update(self.rd.mentions_24h(base_symbol))
            # 4. CryptoPanic
            out.update(self.cp.news_for_coin(base_symbol))
        else:
            out["rd_post_count_24h"] = 0
            out["rd_total_score_24h"] = 0
            out["rd_total_comments_24h"] = 0
            out["rd_top_post_url"] = ""
            out["cp_post_count_24h"] = 0
            out["cp_bullish_pct"] = 50.0
            out["cp_bearish_pct"] = 50.0
            out["cp_sentiment"] = 50.0
            out["cp_top_title"] = ""
        return out

    def collect_many(self, base_symbols: List[str], full_mask: Dict[str, bool],
                     skip_expensive: bool = False,
                     hard_deadline_sec: float = 240.0) -> Dict[str, dict]:
        """
        Run across many symbols. full_mask tells which get full collection.
        `hard_deadline_sec`: stop and return what we have if we've spent
        more than this many seconds (CI safety net).
        Returns dict {base: features}.
        """
        results: Dict[str, dict] = {}
        start = time.time()
        for sym in base_symbols:
            if time.time() - start > hard_deadline_sec:
                # Bail out: return empty for the rest
                results[sym] = {
                    "gt_avg": 0.0, "gt_slope": 0.0, "gt_peak": 0.0,
                    "gt_rising": False, "gt_present": False,
                    "cg_community_data_present": False,
                    "rd_post_count_24h": 0, "cp_post_count_24h": 0,
                }
                continue
            try:
                results[sym] = self.collect(sym, full=full_mask.get(sym, False),
                                            skip_expensive=skip_expensive)
            except Exception:
                results[sym] = {
                    "gt_avg": 0.0, "gt_slope": 0.0, "gt_peak": 0.0,
                    "gt_rising": False, "gt_present": False,
                    "cg_community_data_present": False,
                    "rd_post_count_24h": 0, "cp_post_count_24h": 0,
                }
        return results
