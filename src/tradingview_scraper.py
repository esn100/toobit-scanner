"""
TradingView public idea/analysis scraper.
Uses TradingView's public search/ideas feed. We DO NOT need a paid plan.
We collect: count of recent 'buy' ideas, average sentiment, top authors' bias.
"""
from __future__ import annotations
import re
import time
import requests
from typing import Dict, List, Optional
from html import unescape


class TradingViewScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, timeout: int = 15):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.timeout = timeout

    def get_idea_sentiment(self, symbol: str) -> Dict:
        """
        Scrape TradingView's ideas page for a symbol and return:
          - buy_ratio  : fraction of ideas with 'long'/'buy' labels
          - idea_count : number of recent ideas found
          - avg_likes  : average likes per idea
        Fails gracefully (zeros) if the layout changes.
        """
        # TradingView uses BINANCE-style tickers for spot ideas.
        # Their ideas feed: https://www.tradingview.com/symbols/{SYMBOL}USDT/ideas/
        clean_sym = symbol.replace("USDT", "")
        urls = [
            f"https://www.tradingview.com/symbols/{clean_sym}USDT/ideas/",
            f"https://www.tradingview.com/symbols/{clean_sym}USDT.P/ideas/",  # perp
        ]
        buy = 0
        total = 0
        likes_total = 0
        for u in urls:
            try:
                r = self.session.get(u, timeout=self.timeout)
                if r.status_code != 200:
                    continue
                html = r.text
                # Heuristics: ideas with 'long' label = buy bias, 'short' = sell bias.
                # Look for JSON embedded in the page: window.__initialState or similar.
                # Fallback: simple keyword counting in HTML.
                total += len(re.findall(r'class="tv-widget-idea__title-row', html)) or \
                         len(re.findall(r'"/ideas/', html))
                # crude buy/sell tag count
                buy += html.lower().count("long") - html.lower().count("short")
                # crude likes count
                m = re.findall(r'"likesCount":(\d+)', html)
                if m:
                    likes_total += sum(int(x) for x in m)
            except requests.RequestException:
                continue
            time.sleep(1.0)
        idea_count = max(total, 1)
        return {
            "buy_ratio": max(0.0, min(1.0, (buy + idea_count) / (2 * idea_count))),
            "idea_count": int(total),
            "avg_likes": float(likes_total / idea_count) if idea_count else 0.0,
        }
