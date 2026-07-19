"""
Microstructure Engine for PumpHunter Institutional v1.0.

Provides:
  - ToobitOrderBookClient: real-time depth, trades
  - OKXSmartMoneyClient: OI, funding, taker buy/sell
  - WhaleDetector: detects large orders
  - OBICalculator: Order Book Imbalance
  - LiquiditySweepDetector: stop hunts and reversals
  - CVDCalculator: Cumulative Volume Delta (from trades)
  - MicrostructureScore: combines all signals

Designed for fast scanning (every 5 seconds).
"""
from __future__ import annotations
import os
import time
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


# ============================================================================
# Toobit: Order Book + Trades
# ============================================================================
class ToobitOrderBookClient:
    BASE = "https://api.toobit.com"

    def __init__(self, timeout: int = 5):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pumphunter-inst/1.0"})
        self.timeout = timeout

    def get_depth(self, symbol: str, limit: int = 20) -> Dict:
        """Get L2 order book. Returns bids/asks as lists of [price, qty]."""
        try:
            r = self.session.get(
                f"{self.BASE}/quote/v1/depth",
                params={"symbol": symbol, "limit": min(limit, 100)},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return {"bids": [], "asks": [], "ts": 0}
            d = r.json()
            return {
                "bids": [[float(p), float(q)] for p, q in d.get("b", [])],
                "asks": [[float(p), float(q)] for p, q in d.get("a", [])],
                "ts": int(d.get("t", 0)),
            }
        except Exception:
            return {"bids": [], "asks": [], "ts": 0}

    def get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Get recent public trades. Toobit supports up to 1000."""
        try:
            r = self.session.get(
                f"{self.BASE}/quote/v1/trades",
                params={"symbol": symbol, "limit": min(limit, 1000)},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return []
            out = []
            for t in r.json()[:limit]:
                out.append({
                    "ts": int(t.get("t", 0)),
                    "price": float(t.get("p", 0)),
                    "qty": float(t.get("q", 0)),
                    "is_buyer_maker": bool(t.get("ibm", False)),
                })
            return out
        except Exception:
            return []

    def get_ticker_24h(self, symbol: str) -> Dict:
        try:
            r = self.session.get(
                f"{self.BASE}/quote/v1/ticker/24hr",
                params={"symbol": symbol},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return {}
            t = r.json()
            return {
                "open": float(t.get("o", 0)),
                "high": float(t.get("h", 0)),
                "low": float(t.get("l", 0)),
                "last": float(t.get("c", 0)),
                "bid": float(t.get("b", 0)),
                "ask": float(t.get("a", 0)),
                "volume_base": float(t.get("v", 0)),
                "volume_quote": float(t.get("qv", 0)),
                "price_change": float(t.get("pc", 0)),
                "price_change_pct": float(t.get("pcp", 0)),
                "trade_count": int(t.get("tc", 0)) if "tc" in t else 0,
            }
        except Exception:
            return {}


# ============================================================================
# OKX: Open Interest, Funding, Taker Buy/Sell
# ============================================================================
class OKXSmartMoneyClient:
    BASE = "https://www.okx.com/api/v5"

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pumphunter-inst/1.0"})
        self.timeout = timeout

    def _get(self, path: str, params: dict = None) -> object:
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
                    time.sleep(2 + attempt)
                    continue
            except requests.RequestException:
                if attempt == 2:
                    return None
                time.sleep(1 + attempt)
        return None

    def to_okx_inst(self, toobit_symbol: str) -> Optional[str]:
        if "USDT" not in toobit_symbol.upper():
            return None
        base = toobit_symbol.upper().replace("USDT", "")
        return f"{base}-USDT-SWAP"

    def get_open_interest(self, symbol: str) -> Dict:
        """Open interest change. Try futures, fall back to spot volume proxy."""
        inst = self.to_okx_inst(symbol)
        if not inst:
            return {"oi_value_usd": 0, "oi_change_4h_pct": 0, "oi_rising": False}
        # Try futures OI
        data = self._get("/rubik/stat/contracts/open-interest-volume",
                         {"ccy": symbol.replace("USDT", ""), "period": "1H"})
        if data and data.get("code") == "0" and data.get("data"):
            rows = data["data"]
            if len(rows) >= 5:
                try:
                    cur_oi = float(rows[0][1])
                    oi_4h_ago = float(rows[4][1])
                    change_pct = (cur_oi - oi_4h_ago) / oi_4h_ago * 100 if oi_4h_ago > 0 else 0
                    return {
                        "oi_value_usd": cur_oi,
                        "oi_change_4h_pct": change_pct,
                        "oi_rising": bool(change_pct > 1.0),
                        "source": "futures_oi",
                    }
                except Exception:
                    pass
        # Fallback: use spot volume change as proxy (OI not available)
        try:
            kline_data = self._get("/market/candles",
                                   {"instId": f"{symbol.replace('USDT','')}-USDT",
                                    "bar": "1H", "limit": 6})
            if kline_data and kline_data.get("code") == "0":
                candles = kline_data["data"]
                if len(candles) >= 5:
                    cur_vol = sum(float(c[5]) for c in candles[:2])  # last 2h
                    prev_vol = sum(float(c[5]) for c in candles[2:5])  # 3h before
                    if prev_vol > 0:
                        change_pct = (cur_vol - prev_vol) / prev_vol * 100
                        return {
                            "oi_value_usd": 0,
                            "oi_change_4h_pct": change_pct,
                            "oi_rising": bool(change_pct > 5.0),
                            "source": "spot_volume_proxy",
                        }
        except Exception:
            pass
        return {"oi_value_usd": 0, "oi_change_4h_pct": 0,
                "oi_rising": False, "source": "unavailable"}

    def get_funding_rate(self, symbol: str) -> Dict:
        inst = self.to_okx_inst(symbol)
        if not inst:
            return {"funding_rate": 0, "next_funding_time": 0, "extreme": False}
        # Try swap first
        data = self._get("/public/funding-rate", {"instId": inst})
        if not data or data.get("code") != "0":
            # Try spot as fallback
            return {"funding_rate": 0, "next_funding_time": 0, "extreme": False,
                    "note": "no swap data"}
        rows = data.get("data", [])
        if not rows:
            return {"funding_rate": 0, "next_funding_time": 0, "extreme": False,
                    "note": "no rows"}
        try:
            rate = float(rows[0].get("fundingRate", 0))
            return {
                "funding_rate": rate,
                "next_funding_time": int(rows[0].get("fundingTime", 0)),
                "extreme": bool(abs(rate) > 0.0005),  # > 0.05%
            }
        except Exception:
            return {"funding_rate": 0, "next_funding_time": 0, "extreme": False}

    def get_okx_ticker(self, symbol: str) -> Dict:
        """Get current ticker from OKX spot for comparison."""
        # Use spot pair like BTC-USDT for price reference
        if "USDT" not in symbol.upper():
            return {}
        base = symbol.upper().replace("USDT", "")
        inst = f"{base}-USDT"
        data = self._get("/market/ticker", {"instId": inst})
        if not data or data.get("code") != "0":
            return {}
        rows = data.get("data", [])
        if not rows:
            return {}
        try:
            r = rows[0]
            return {
                "last": float(r.get("last", 0)),
                "vol_24h": float(r.get("vol24h", 0)),
                "vol_quote_24h": float(r.get("volCcy24h", 0)),
            }
        except Exception:
            return {}

    def get_taker_flow(self, symbol: str) -> Dict:
        """
        Taker buy/sell volume. We use the contract ticker.
        Returns: taker_buy_ratio (0-1) = buy_vol / (buy_vol + sell_vol)
        """
        inst = self.to_okx_inst(symbol)
        if not inst:
            return {"taker_buy_ratio": 0.5, "aggressive_buying": False}
        # OKX taker flow needs the last 1h candle
        data = self._get("/market/taker-flow",
                         {"instId": inst, "period": "1m", "limit": 30})
        if not data or data.get("code") != "0":
            return {"taker_buy_ratio": 0.5, "aggressive_buying": False}
        rows = data.get("data", [])
        if not rows:
            return {"taker_buy_ratio": 0.5, "aggressive_buying": False}
        # rows: [ts, buy_vol, sell_vol]
        total_buy = sum(float(r[1]) for r in rows if len(r) > 1)
        total_sell = sum(float(r[2]) for r in rows if len(r) > 2)
        total = total_buy + total_sell
        if total == 0:
            return {"taker_buy_ratio": 0.5, "aggressive_buying": False}
        buy_ratio = total_buy / total
        return {
            "taker_buy_ratio": float(buy_ratio),
            "taker_buy_vol": total_buy,
            "taker_sell_vol": total_sell,
            "aggressive_buying": bool(buy_ratio > 0.6),
            "aggressive_selling": bool(buy_ratio < 0.4),
        }


# ============================================================================
# Order Book Imbalance (OBI)
# ============================================================================
def calculate_obi(depth: Dict, levels: int = 10) -> float:
    """
    Order Book Imbalance: bid_volume / ask_volume
    > 1 = more buyers, < 1 = more sellers
    """
    bids = depth.get("bids", [])[:levels]
    asks = depth.get("asks", [])[:levels]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    if ask_vol == 0:
        return 1.0
    return float(bid_vol / ask_vol)


def calculate_spread_pct(depth: Dict) -> float:
    """Bid-ask spread as percentage of mid price."""
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if not bids or not asks:
        return 100.0  # illiquid
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return 100.0
    return float((best_ask - best_bid) / mid * 100.0)


# ============================================================================
# Whale Detection
# ============================================================================
def detect_whales(trades: List[Dict], min_qty_usd: float = 5000,
                  min_count: int = 5, window_sec: int = 60) -> Dict:
    """
    Detect whale activity: min N large orders within window.
    Returns: {whale_score (0-1), count, total_buy_usd, total_sell_usd,
              buy_sell_ratio, accumulated}
    """
    if not trades:
        return {"whale_score": 0, "count": 0, "buy_sell_ratio": 1.0,
                "accumulated": False}
    now_ms = max(t["ts"] for t in trades)
    cutoff = now_ms - window_sec * 1000
    recent = [t for t in trades if t["ts"] >= cutoff]
    # Identify large trades (size > min_qty_usd)
    # We don't have quote_volume per trade, so use qty * price
    # Approximate: assume each trade's value is the trade's qty * average
    # of the surrounding trades. We use a simple heuristic.
    if not recent:
        return {"whale_score": 0, "count": 0, "buy_sell_ratio": 1.0,
                "accumulated": False}
    avg_price = sum(t["price"] for t in recent) / len(recent)
    large = []
    for t in recent:
        value_usd = t["qty"] * avg_price
        if value_usd >= min_qty_usd:
            # taker buy: is_buyer_maker=False (buyer is taker)
            is_buy = not t["is_buyer_maker"]
            large.append({"value_usd": value_usd, "is_buy": is_buy})
    if not large:
        return {"whale_score": 0, "count": 0, "buy_sell_ratio": 1.0,
                "accumulated": False}
    buy_vol = sum(t["value_usd"] for t in large if t["is_buy"])
    sell_vol = sum(t["value_usd"] for t in large if not t["is_buy"])
    total = buy_vol + sell_vol
    ratio = buy_vol / sell_vol if sell_vol > 0 else 999
    # Accumulated = price didn't move much despite whale activity
    # (high buy pressure, low price impact = absorption)
    accumulated = False
    if len(recent) >= 10:
        price_range_pct = abs(recent[-1]["price"] - recent[0]["price"]) / recent[0]["price"] * 100
        if price_range_pct < 0.3 and len(large) >= min_count:
            accumulated = True
    score = 0.0
    if len(large) >= min_count:
        score += 0.4
    if accumulated:
        score += 0.3
    if ratio > 3 or ratio < 0.33:
        score += 0.3
    elif ratio > 1.5 or ratio < 0.67:
        score += 0.15
    return {
        "whale_score": float(min(1.0, score)),
        "count": len(large),
        "buy_sell_ratio": float(ratio),
        "buy_vol_usd": float(buy_vol),
        "sell_vol_usd": float(sell_vol),
        "accumulated": accumulated,
    }


# ============================================================================
# CVD (Cumulative Volume Delta)
# ============================================================================
def calculate_cvd(trades: List[Dict]) -> Dict:
    """
    CVD = sum of (buy_vol - sell_vol) from all trades.
    Buyer is taker = buy volume (price goes up)
    Seller is taker = sell volume (price goes down)
    """
    if not trades:
        return {"cvd": 0, "cvd_trend": 0, "buy_vol": 0, "sell_vol": 0}
    # Sort by time
    sorted_t = sorted(trades, key=lambda t: t["ts"])
    buy_vol = 0
    sell_vol = 0
    for t in sorted_t:
        if not t["is_buyer_maker"]:
            buy_vol += t["qty"]
        else:
            sell_vol += t["qty"]
    cvd = buy_vol - sell_vol
    # Trend: positive if recent CVD > earlier
    half = len(sorted_t) // 2
    if half > 0:
        first_cvd = sum(
            (t["qty"] if not t["is_buyer_maker"] else -t["qty"])
            for t in sorted_t[:half]
        )
        second_cvd = sum(
            (t["qty"] if not t["is_buyer_maker"] else -t["qty"])
            for t in sorted_t[half:]
        )
        cvd_trend = second_cvd - first_cvd
    else:
        cvd_trend = 0
    return {
        "cvd": float(cvd),
        "cvd_trend": float(cvd_trend),
        "buy_vol": float(buy_vol),
        "sell_vol": float(sell_vol),
    }


# ============================================================================
# Liquidity Sweep
# ============================================================================
def detect_liquidity_sweep(
    klines: pd.DataFrame, trades: List[Dict], depth: Dict
) -> Dict:
    """
    Detect liquidity sweep:
      - Price touched recent low/high
      - Stops were triggered (volume spike + wick)
      - Price recovered quickly
      - High volume on the sweep
    """
    if klines.empty or len(klines) < 10 or not trades:
        return {"sweep": False, "sweep_type": "none", "confidence": 0}
    # Recent swing high/low (last 20 bars)
    recent = klines.tail(20)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    last = klines.iloc[-1]
    last_price = float(last["close"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    # Check if recent candles swept
    swept_high = last_high > swing_high * 1.001
    swept_low = last_low < swing_low * 0.999
    # Check for reversal
    body = abs(last_price - float(last["open"]))
    rng = last_high - last_low
    if rng <= 0:
        return {"sweep": False, "sweep_type": "none", "confidence": 0}
    close_pos = (last_price - last_low) / rng  # 1 = closed at high
    if swept_high and close_pos < 0.4:
        # Swept high, closed low = bearish reversal
        return {"sweep": True, "sweep_type": "high_sweep_bearish", "confidence": 0.7}
    if swept_low and close_pos > 0.6:
        # Swept low, closed high = bullish reversal
        return {"sweep": True, "sweep_type": "low_sweep_bullish", "confidence": 0.7}
    return {"sweep": False, "sweep_type": "none", "confidence": 0}


# ============================================================================
# Multi-exchange confirmation (using OKX as comparison)
# ============================================================================
def multi_exchange_check(
    toobit_ticker: Dict, okx_ticker: Dict
) -> Dict:
    """
    Compare price/volume on Toobit vs OKX for the same symbol.
    If Toobit leads OKX in volume spike, signal is real.
    """
    if not toobit_ticker or not okx_ticker:
        return {"confirms": False, "price_diff_pct": 0, "lead": "unknown"}
    t_price = toobit_ticker.get("last", 0)
    o_price = okx_ticker.get("last", 0)
    if t_price == 0 or o_price == 0:
        return {"confirms": False, "price_diff_pct": 0, "lead": "unknown"}
    price_diff = (t_price - o_price) / o_price * 100
    t_vol = toobit_ticker.get("volume_quote", 0)
    o_vol = okx_ticker.get("vol_quote_24h", 0)
    # Small caps often have wider spreads between exchanges
    # 5% threshold is more realistic for illiquid tokens
    confirms = abs(price_diff) < 5.0
    return {
        "confirms": bool(confirms),
        "price_diff_pct": float(price_diff),
        "toobit_volume_quote": t_vol,
        "okx_volume_quote": o_vol,
        "lead": ("toobit" if t_vol > o_vol * 1.2
                 else "okx" if o_vol > t_vol * 1.2 else "synced"),
    }
