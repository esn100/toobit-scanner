"""
Google Trends data via pytrends.
Pulls a 'today 7d' interest-over-time for each coin symbol.
"""
from __future__ import annotations
import time
import random
from typing import Optional
from pytrends.request import TrendReq


class GoogleTrendsClient:
    def __init__(self, hl: str = "en-US", tz: int = 0):
        # tz=0 -> UTC, helps with deterministic results
        self.pytrends = TrendReq(hl=hl, tz=tz, retries=2, backoff_factor=0.5)

    def get_interest(self, query: str) -> dict:
        """
        Get the latest Google Trends interest for a coin.
        Returns a small dict: average, slope (rising/falling), peak.
        Falls back to zeros on failure so the rest of the pipeline keeps going.
        """
        try:
            self.pytrends.build_payload([query], timeframe="now 7-d", geo="")
            df = self.pytrends.interest_over_time()
            if df is None or df.empty or query not in df.columns:
                return {"avg": 0.0, "slope": 0.0, "peak": 0.0, "rising": False}
            series = df[query].astype(float)
            avg = float(series.mean())
            peak = float(series.max())
            # Slope = simple linear regression slope over the last 7 days
            n = len(series)
            if n < 2:
                slope = 0.0
            else:
                xs = list(range(n))
                mean_x = sum(xs) / n
                mean_y = avg
                num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series))
                den = sum((x - mean_x) ** 2 for x in xs) or 1
                slope = float(num / den)
            rising = bool(slope > 0 and series.iloc[-1] > avg)
            return {"avg": avg, "slope": slope, "peak": peak, "rising": rising}
        except Exception:
            return {"avg": 0.0, "slope": 0.0, "peak": 0.0, "rising": False}
        finally:
            # pytrends rate-limits aggressively; small jitter
            time.sleep(random.uniform(2.0, 4.0))
