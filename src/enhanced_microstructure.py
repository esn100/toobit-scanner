"""
Enhanced microstructure features for ML.

Collects real-time order book + trade data and creates
powerful predictive features:

  - Multi-level OBI (5, 10, 20 levels)
  - Depth imbalance (large orders vs small)
  - Trade flow toxicity (VPIN-like)
  - Volume concentration (top 1% of trades)
  - Bid-ask spread dynamics
  - Order book pressure (1%, 5%, 10% from mid)
"""
from __future__ import annotations
import os
import sys
import json
import time
from typing import Dict, List
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.microstructure import (
    ToobitOrderBookClient, OKXSmartMoneyClient,
    calculate_obi, calculate_spread_pct,
    detect_whales, calculate_cvd, detect_liquidity_sweep,
    multi_exchange_check
)
from src.toobit_client import ToobitClient
from src.universe import discover_small_caps, MarketCapResolver
from src.indicators import relative_volume, momentum_features
import time


def enhanced_orderbook_features(symbol: str, depth: Dict) -> Dict:
    """Compute enhanced order book features."""
    out = {}
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if not bids or not asks:
        return {f"ob_{k}": 0 for k in
                ["obi_5", "obi_10", "obi_20", "spread_pct",
                 "depth_1pct", "depth_5pct", "depth_10pct",
                 "bid_vol_5", "ask_vol_5", "imbalance_5",
                 "large_bid_count", "large_ask_count"]}
    # OBI at multiple levels
    for n in [5, 10, 20]:
        bid_vol = sum(q for _, q in bids[:n])
        ask_vol = sum(q for _, q in asks[:n])
        if ask_vol == 0:
            out[f"obi_{n}"] = 1.0
        else:
            out[f"obi_{n}"] = float(bid_vol / ask_vol)
    # Spread
    out["spread_pct"] = calculate_spread_pct(depth)
    # Depth at different distances from mid
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        mid = best_bid or 1
    bid_vol_1pct = sum(q for p, q in bids if p >= mid * 0.99)
    ask_vol_1pct = sum(q for p, q in asks if p <= mid * 1.01)
    bid_vol_5pct = sum(q for p, q in bids if p >= mid * 0.95)
    ask_vol_5pct = sum(q for p, q in asks if p <= mid * 1.05)
    out["depth_1pct"] = float(bid_vol_1pct + ask_vol_1pct)
    out["depth_5pct"] = float(bid_vol_5pct + ask_vol_5pct)
    out["depth_10pct"] = float(
        sum(q for p, q in bids if p >= mid * 0.9) +
        sum(q for p, q in asks if p <= mid * 1.1)
    )
    # Volume at top 5 levels
    out["bid_vol_5"] = float(sum(q for _, q in bids[:5]))
    out["ask_vol_5"] = float(sum(q for _, q in asks[:5]))
    out["imbalance_5"] = float(out["bid_vol_5"] / max(out["ask_vol_5"], 1e-9))
    # Large orders (top 10% by size)
    all_orders = [(p, q, "bid") for p, q in bids] + [(p, q, "ask") for p, q in asks]
    if all_orders:
        sizes = [q for _, q, _ in all_orders]
        threshold = np.percentile(sizes, 90) if sizes else 0
        large_bids = sum(1 for p, q, t in all_orders if t == "bid" and q >= threshold)
        large_asks = sum(1 for p, q, t in all_orders if t == "ask" and q >= threshold)
        out["large_bid_count"] = large_bids
        out["large_ask_count"] = large_asks
    else:
        out["large_bid_count"] = 0
        out["large_ask_count"] = 0
    return out


def trade_flow_features(trades: List[Dict]) -> Dict:
    """Compute trade flow toxicity / VPIN-like features."""
    out = {"flow_buy_ratio": 0.5, "flow_buy_vol": 0, "flow_sell_vol": 0,
           "flow_concentration": 0, "flow_avg_size": 0, "large_trade_count": 0}
    if not trades:
        return out
    buy_vol = sum(t["qty"] for t in trades if not t["is_buyer_maker"])
    sell_vol = sum(t["qty"] for t in trades if t["is_buyer_maker"])
    total = buy_vol + sell_vol
    if total > 0:
        out["flow_buy_ratio"] = float(buy_vol / total)
    out["flow_buy_vol"] = float(buy_vol)
    out["flow_sell_vol"] = float(sell_vol)
    # Concentration: top 1% of trades by size
    sizes = sorted([t["qty"] for t in trades], reverse=True)
    if sizes:
        top_1pct = sum(sizes[:max(1, len(sizes)//100)])
        out["flow_concentration"] = float(top_1pct / max(sum(sizes), 1e-9))
        out["flow_avg_size"] = float(np.mean(sizes))
    # Large trades (top 5% by size)
    threshold = np.percentile(sizes, 95) if sizes else 0
    out["large_trade_count"] = sum(1 for s in sizes if s >= threshold)
    return out


def collect_micro_dataset(symbols: list, out_path: str = "data/micro_dataset.csv"):
    """Collect order book + trade features for ML training."""
    tb_client = ToobitClient()
    ob_client = ToobitOrderBookClient(timeout=5)
    resolver = MarketCapResolver(cache_dir="/home/user/toobit-scanner/data")
    tickers = tb_client.get_24h_tickers()
    small_caps = discover_small_caps(tickers, resolver, max_symbols=20)
    available = set(small_caps['symbol'].tolist())
    symbols = [s for s in symbols if s in available]
    rows = []
    for sym in symbols:
        try:
            depth = ob_client.get_depth(sym, limit=20)
            trades = ob_client.get_recent_trades(sym, limit=200)
        except Exception as e:
            print(f"  {sym}: ERR {e}")
            continue
        if not depth["bids"] or not depth["asks"]:
            continue
        # Get current price
        try:
            df = tb_client.get_klines(sym, "4h", 5)
            current = float(df["close"].iloc[-1]) if not df.empty else 0
        except:
            current = 0
        # 4h forward: use 1h candles
        try:
            df_1h = tb_client.get_klines(sym, "1h", 20)
            if not df_1h.empty and len(df_1h) >= 13:
                fwd_4h = float(df_1h["close"].iloc[-1])
                fwd_12h = float(df_1h["close"].iloc[-1])
            else:
                fwd_4h = fwd_12h = current
        except:
            fwd_4h = fwd_12h = current
        # Order book features
        ob_feats = enhanced_orderbook_features(sym, depth)
        # Trade flow features
        flow_feats = trade_flow_features(trades)
        # Whale features
        whale = detect_whales(trades, min_qty_usd=2000, min_count=3, window_sec=120)
        # CVD
        cvd = calculate_cvd(trades)
        # Spread
        spread = calculate_spread_pct(depth)
        # Taker buy ratio
        if trades:
            buy_vol = sum(t["qty"] for t in trades if not t["is_buyer_maker"])
            sell_vol = sum(t["qty"] for t in trades if t["is_buyer_maker"])
            total = buy_vol + sell_vol
            taker_buy_ratio = buy_vol / total if total > 0 else 0.5
        else:
            taker_buy_ratio = 0.5
        row = {
            "symbol": sym,
            "ts": datetime.now(timezone.utc).isoformat(),
            "price": current,
            "fwd_4h": (fwd_4h - current) / current * 100 if current else 0,
            "fwd_12h": (fwd_12h - current) / current * 100 if current else 0,
        }
        row.update({f"ob_{k}": v for k, v in ob_feats.items()})
        row.update({f"flow_{k}": v for k, v in flow_feats.items()})
        row["whale_score"] = whale.get("whale_score", 0)
        row["whale_count"] = whale.get("count", 0)
        row["whale_buy_sell_ratio"] = whale.get("buy_sell_ratio", 1.0)
        row["whale_accumulated"] = int(whale.get("accumulated", False))
        row["cvd"] = cvd.get("cvd", 0)
        row["cvd_trend"] = cvd.get("cvd_trend", 0)
        row["spread_pct"] = spread
        row["taker_buy_ratio"] = taker_buy_ratio
        row["n_trades"] = len(trades)
        rows.append(row)
        time.sleep(0.5)
        print(f"  {sym}: {len(ob_feats)} OB + {len(flow_feats)} flow + whale={whale.get('count',0)}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pump"] = (df["fwd_12h"] > 3).astype(int)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    if not df.empty:
        print(f"Pump rate: {df['pump'].mean()*100:.1f}%")
    return df


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--out", type=str, default="data/micro_dataset.csv")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat collection N times (gap 5min between)")
    args = p.parse_args()
    symbols = args.symbols or [
        "EPICUSDT", "OPNUSDT", "RECALLUSDT", "TLMUSDT",
        "RESOLVUSDT", "TUTUSDT", "FIGHTUSDT", "BREVUSDT"
    ]
    print(f"Collecting microstructure for {len(symbols)} symbols, {args.repeat}x")
    all_dfs = []
    for i in range(args.repeat):
        if i > 0:
            print(f"\n--- Round {i+1}/{args.repeat} ---")
            time.sleep(300)  # 5 min between rounds
        df = collect_micro_dataset(symbols, out_path=args.out)
        if not df.empty:
            all_dfs.append(df)
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(args.out, index=False)
        print(f"\nFinal: {len(combined)} rows in {args.out}")


if __name__ == "__main__":
    main()
