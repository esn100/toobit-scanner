"""
Live feature collector for PumpHunter-AI.

Goal: collect features and 12h outcomes to build a real labeled dataset
for later optimization. NO trade decisions are made here.

Run on a schedule (e.g. every 10-30 minutes) to keep feature_history.csv
and outcome_history.csv growing.

Usage:
  python -m src.live_collector --interval 600
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .toobit_client import ToobitClient
from .data_quality import validate_ohlcv
from .indicators import (
    vwap_features,
    atr_features,
    bollinger_features,
    relative_volume,
    volume_continuity,
    momentum_features,
)
from .market_structure import structure_features
from .candle_quality import candle_quality_features
from .features import build_features
from .technical import technical_analysis
from .btc_correlation import btc_correlation_features
from .elliott_wave import detect_elliott_waves, elliott_score
from .fibonacci import compute_fib_levels, fib_score, fib_extension_target
from .ichimoku import ichimoku_features, ichimoku_score
from .microstructure import (
    ToobitOrderBookClient, OKXSmartMoneyClient,
    calculate_obi, calculate_spread_pct,
    detect_whales, calculate_cvd, detect_liquidity_sweep,
    multi_exchange_check,
)
from .microstructure_score import (
    factor_volume_explosion, factor_whale_activity,
    factor_order_book_imbalance, factor_liquidity_sweep,
    factor_open_interest, factor_cvd, factor_funding_rate,
    factor_multi_exchange, FACTOR_WEIGHTS,
)
from .direction_scoring import direction_score, score_long, score_short
from .per_coin_sentiment import CoinSocialAggregator
from .extended_indicators import compute_all_extended
from .signal_tracker import (
    open_signal, check_and_resolve, get_open_signals, get_stats,
)
from .adaptive_tp_sl import get_signal_tp_sl, format_tp_sl_for_log
from . import db as database
from .auto_trader import (
    open_ultra_signals, check_signals_smart_v2, run_auto_trader_cycle,
)

# Cache for BTC 4h klines (one fetch per cycle is enough)
_BTC_CACHE: dict = {"df": None, "ts": 0.0}
_BTC_TTL_SECONDS = 600  # refresh every 10 min


def _get_btc_df(client: ToobitClient) -> pd.DataFrame:
    """Cached BTC 4h klines (60 days) for correlation features."""
    now = time.time()
    if _BTC_CACHE["df"] is not None and (now - _BTC_CACHE["ts"]) < _BTC_TTL_SECONDS:
        return _BTC_CACHE["df"]
    try:
        df = client.get_klines("BTCUSDT", interval="4h", limit=360)
        if not df.empty:
            _BTC_CACHE["df"] = df
            _BTC_CACHE["ts"] = now
    except Exception:
        pass
    return _BTC_CACHE["df"] if _BTC_CACHE["df"] is not None else pd.DataFrame()


# Microstructure clients are heavier — cache them per cycle.
_MICRO_CACHE: dict = {
    "toobit": None, "okx": None,
}


# Social aggregator — also cached for the whole process (id cache persists).
_SOCIAL_CACHE: dict = {"agg": None}


def _get_micro_clients() -> tuple:
    if _MICRO_CACHE["toobit"] is None:
        _MICRO_CACHE["toobit"] = ToobitOrderBookClient(timeout=5)
    if _MICRO_CACHE["okx"] is None:
        _MICRO_CACHE["okx"] = OKXSmartMoneyClient(timeout=8)
    return _MICRO_CACHE["toobit"], _MICRO_CACHE["okx"]


def _micro_features(symbol: str, klines_4h: pd.DataFrame) -> dict:
    """
    Collect microstructure features for one symbol.
    Heavy: makes 5+ API calls. Caller is responsible for rate limiting.
    Returns a flat dict of features (all numeric).
    """
    out: dict = {}
    toobit_micro, okx_micro = _get_micro_clients()
    # 1. Order book imbalance (L2 depth)
    try:
        depth = toobit_micro.get_depth(symbol, limit=20)
        obi = calculate_obi(depth, levels=10)
        spread = calculate_spread_pct(depth)
        out["m_obi_10"] = obi
        out["m_spread_pct"] = spread
        out["m_obi_bullish"] = float(obi > 1.5)
        out["m_obi_bearish"] = float(obi < 0.67)
    except Exception:
        out["m_obi_10"] = 1.0
        out["m_spread_pct"] = 100.0
        out["m_obi_bullish"] = 0.0
        out["m_obi_bearish"] = 0.0
    # 2. Recent trades (whale + CVD)
    try:
        trades = toobit_micro.get_recent_trades(symbol, limit=500)
    except Exception:
        trades = []
    if trades:
        try:
            w = detect_whales(trades, min_qty_usd=2000, min_count=3, window_sec=60)
            out["m_whale_count"] = float(w["count"])
            out["m_whale_score"] = float(w["whale_score"])
            out["m_whale_accumulated"] = float(w["accumulated"])
            out["m_whale_buy_sell_ratio"] = float(
                min(w["buy_sell_ratio"], 99.0)
            )
        except Exception:
            out["m_whale_count"] = 0.0
            out["m_whale_score"] = 0.0
            out["m_whale_accumulated"] = 0.0
            out["m_whale_buy_sell_ratio"] = 1.0
        try:
            cvd = calculate_cvd(trades)
            out["m_cvd"] = cvd["cvd"]
            out["m_cvd_trend"] = cvd["cvd_trend"]
        except Exception:
            out["m_cvd"] = 0.0
            out["m_cvd_trend"] = 0.0
    else:
        out["m_whale_count"] = 0.0
        out["m_whale_score"] = 0.0
        out["m_whale_accumulated"] = 0.0
        out["m_whale_buy_sell_ratio"] = 1.0
        out["m_cvd"] = 0.0
        out["m_cvd_trend"] = 0.0
    # 3. Open interest (OKX)
    try:
        oi = okx_micro.get_open_interest(symbol)
        out["m_oi_change_4h_pct"] = float(oi.get("oi_change_4h_pct", 0))
        out["m_oi_rising"] = float(oi.get("oi_rising", False))
        out["m_oi_source"] = oi.get("source", "unavailable")
    except Exception:
        out["m_oi_change_4h_pct"] = 0.0
        out["m_oi_rising"] = 0.0
        out["m_oi_source"] = "error"
    # 4. Funding rate
    try:
        fr = okx_micro.get_funding_rate(symbol)
        out["m_funding_rate"] = float(fr.get("funding_rate", 0))
        out["m_funding_extreme"] = float(fr.get("extreme", False))
    except Exception:
        out["m_funding_rate"] = 0.0
        out["m_funding_extreme"] = 0.0
    # 5. Liquidity sweep on the 4h klines
    try:
        sweep = detect_liquidity_sweep(klines_4h, trades, depth)
        out["m_sweep"] = float(sweep.get("sweep", False))
        out["m_sweep_confidence"] = float(sweep.get("confidence", 0))
        out["m_sweep_type_bullish"] = float(
            "low_sweep_bullish" in str(sweep.get("sweep_type", ""))
        )
        out["m_sweep_type_bearish"] = float(
            "high_sweep_bearish" in str(sweep.get("sweep_type", ""))
        )
    except Exception:
        out["m_sweep"] = 0.0
        out["m_sweep_confidence"] = 0.0
        out["m_sweep_type_bullish"] = 0.0
        out["m_sweep_type_bearish"] = 0.0
    # 6. Multi-exchange confirmation
    try:
        t_ticker = toobit_micro.get_ticker_24h(symbol)
        o_ticker = okx_micro.get_okx_ticker(symbol)
        mxc = multi_exchange_check(t_ticker, o_ticker)
        out["m_mexc_confirms"] = float(mxc.get("confirms", False))
        out["m_mexc_price_diff_pct"] = float(mxc.get("price_diff_pct", 0))
        out["m_mexc_toobit_leads"] = float(mxc.get("lead") == "toobit")
    except Exception:
        out["m_mexc_confirms"] = 0.0
        out["m_mexc_price_diff_pct"] = 0.0
        out["m_mexc_toobit_leads"] = 0.0
    # 7. 5m volume explosion (uses trades)
    try:
        # Need 1h klines for baseline. We have 4h; use last bar 1h proxy:
        # take 4h bar trades distribution as proxy.
        # Simpler: use only the recent_5m_vol vs full set.
        if trades:
            now_ms = max(t["ts"] for t in trades)
            recent_5m = [t for t in trades if t["ts"] >= now_ms - 5 * 60 * 1000]
            recent_1h = [t for t in trades if t["ts"] >= now_ms - 60 * 60 * 1000]
            if recent_1h:
                bucket_vol = sum(t["qty"] for t in recent_1h) / 12
                vol_5m = sum(t["qty"] for t in recent_5m)
                rvol_5m = vol_5m / max(bucket_vol, 1e-12)
                out["m_5m_rvol"] = float(rvol_5m)
                out["m_5m_trade_count"] = float(len(recent_5m))
                out["m_5m_volume_spike"] = float(rvol_5m > 4.0)
            else:
                out["m_5m_rvol"] = 1.0
                out["m_5m_trade_count"] = 0.0
                out["m_5m_volume_spike"] = 0.0
        else:
            out["m_5m_rvol"] = 1.0
            out["m_5m_trade_count"] = 0.0
            out["m_5m_volume_spike"] = 0.0
    except Exception:
        out["m_5m_rvol"] = 1.0
        out["m_5m_trade_count"] = 0.0
        out["m_5m_volume_spike"] = 0.0
    return out


def _advanced_features(df: pd.DataFrame) -> dict:
    """Elliott Wave + Fibonacci + Ichimoku features (no API calls)."""
    out: dict = {}
    # Elliott Wave
    try:
        ew = detect_elliott_waves(df, threshold=0.05)
        out["a_wave"] = ew.get("wave", "none")
        out["a_wave_score"] = float(ew.get("score", 50.0))
        out["a_wave_elliott_score"] = float(elliott_score(ew))
        out["a_hurst"] = float(ew.get("details", {}).get("hurst", 0.5))
        out["a_wave_position"] = ew.get("details", {}).get("position", "none")
        out["a_wave3_ext"] = float(
            ew.get("details", {}).get("wave3_extension", 0.0)
        )
        out["a_wave2_retrace"] = float(
            ew.get("details", {}).get("wave2_retrace", 0.0)
        )
        out["a_is_uptrend"] = float(
            ew.get("details", {}).get("is_uptrend", False)
        )
    except Exception:
        out["a_wave"] = "error"
        out["a_wave_score"] = 50.0
        out["a_wave_elliott_score"] = 0.0
        out["a_hurst"] = 0.5
        out["a_wave_position"] = "none"
        out["a_wave3_ext"] = 0.0
        out["a_wave2_retrace"] = 0.0
        out["a_is_uptrend"] = 0.0
    # Fibonacci
    try:
        fib = compute_fib_levels(df, lookback=50)
        out["a_fib_direction"] = fib.get("direction", "none")
        out["a_fib_closest"] = fib.get("closest_level", "")
        out["a_fib_distance_pct"] = float(fib.get("distance_to_closest", 0))
        out["a_fib_score"] = float(fib_score(fib))
        out["a_fib_ext_1_618"] = float(fib_extension_target(fib, "1.618"))
        out["a_fib_ext_2_618"] = float(fib_extension_target(fib, "2.618"))
        # Distance to each level
        for lvl in ("0.382", "0.500", "0.618", "0.786"):
            level_prices = fib.get("levels", {})
            key = f"fib_{lvl}"
            if key in level_prices and fib.get("current_price", 0) > 0:
                dist = abs(level_prices[key] - fib["current_price"]) / fib["current_price"] * 100
                out[f"a_fib_dist_{lvl}"] = float(dist)
            else:
                out[f"a_fib_dist_{lvl}"] = 99.0
    except Exception:
        out["a_fib_direction"] = "none"
        out["a_fib_score"] = 50.0
        out["a_fib_ext_1_618"] = 0.0
        out["a_fib_ext_2_618"] = 0.0
        for lvl in ("0.382", "0.500", "0.618", "0.786"):
            out[f"a_fib_dist_{lvl}"] = 99.0
    # Ichimoku
    try:
        ich = ichimoku_features(df)
        out["a_ichi_price_vs_cloud"] = ich.get("price_vs_cloud", "neutral")
        out["a_ichi_cloud_color"] = ich.get("cloud_color", "neutral")
        out["a_ichi_tk_cross"] = ich.get("tk_cross", "neutral")
        out["a_ichi_thickness_pct"] = float(ich.get("cloud_thickness_pct", 0))
        out["a_ichi_score"] = float(ichimoku_score(ich))
        out["a_ichi_above_cloud"] = float(ich.get("price_vs_cloud") == "above")
        out["a_ichi_below_cloud"] = float(ich.get("price_vs_cloud") == "below")
        out["a_ichi_in_cloud"] = float(ich.get("price_vs_cloud") == "inside")
    except Exception:
        out["a_ichi_score"] = 50.0
        out["a_ichi_thickness_pct"] = 0.0
        out["a_ichi_above_cloud"] = 0.0
        out["a_ichi_below_cloud"] = 0.0
        out["a_ichi_in_cloud"] = 0.0
    return out


DATA_DIR = Path("data")
FEATURE_LOG = DATA_DIR / "feature_log.csv"
OUTCOME_LOG = DATA_DIR / "outcome_log.csv"

# Small caps: market cap < $20M is the constraint.
# We use Toobit 24h quote volume as a cheap proxy: filter to $1M-$50M
# 24h volume (skips illiquid dust and excludes majors).
MIN_24H_VOLUME_USD = 1_000_000
MAX_24H_VOLUME_USD = 50_000_000
# Don't scan more than this many symbols per cycle (rate-limit protection)
MAX_SYMBOLS_PER_CYCLE = 40


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def collect_cycle(client: ToobitClient, symbols: list[str]) -> int:
    """Run a single collection cycle. Returns number of rows appended."""
    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    failures = 0

    # Fetch BTC once per cycle (cached for 10 min)
    btc_df = _get_btc_df(client)
    btc_state = "NEUTRAL"
    btc_mom_12 = 0.0
    if not btc_df.empty and len(btc_df) >= 13:
        btc_mom_12 = float(
            (btc_df["close"].iloc[-1] - btc_df["close"].iloc[-13])
            / max(btc_df["close"].iloc[-13], 1e-12) * 100.0
        )
        if btc_mom_12 > 3.0:
            btc_state = "BULLISH"
        elif btc_mom_12 < -3.0:
            btc_state = "BEARISH"

    # Microstructure is heavy (5+ API calls per symbol). Only run on the
    # top movers (highest rvol candidates) to keep cycles short.
    micro_targets: set = set()
    for sym in symbols[:MAX_SYMBOLS_PER_CYCLE]:
        try:
            df = client.get_klines(sym, interval="4h", limit=200)
            if df.empty or len(df) < 60:
                failures += 1
                continue

            qrep = validate_ohlcv(df, min_candles=60, interval_hours=4.0)
            if not qrep.ok or qrep.cleaned is None:
                failures += 1
                continue
            df = qrep.cleaned

            ind = {}
            ind.update(vwap_features(df))
            ind.update(atr_features(df))
            ind.update(bollinger_features(df))
            ind.update(relative_volume(df))
            ind.update(volume_continuity(df))
            ind.update(momentum_features(df))
            struct = structure_features(df)
            candle = candle_quality_features(df)
            # Real technical analysis (RSI/MACD/EMA) — not hardcoded
            tech = technical_analysis(df)
            # BTC correlation (per symbol)
            btc_corr = btc_correlation_features(df, btc_df) if not btc_df.empty else {}
            # Elliott + Fibonacci + Ichimoku (no API cost, all from klines)
            adv = _advanced_features(df)
            feats = build_features(
                technical=tech,
                indicators=ind,
                structure=struct,
                candle=candle,
                mtf={"alignment_score": 50.0},  # not implemented; default
                btc={"state": btc_state,
                     "btc_momentum_12_pct": btc_mom_12},
            )
            for k, v in btc_corr.items():
                feats[k] = float(v) if isinstance(v, (int, float, bool)) else 0.0
            # Extended indicators (Stoch, ADX, MFI, etc.) — 99 new features
            try:
                ex_feats = compute_all_extended(df)
                for k, v in ex_feats.items():
                    row[f"f_{k}"] = v if not isinstance(v, bool) else int(v)
            except Exception:
                pass
            # Pick candidates for expensive microstructure fetch.
            # Heuristic: rvol >= 1.3 OR atr_pct >= 5 OR mom_6 >= 5%
            if (ind.get("rvol", 1.0) >= 1.3
                    or ind.get("atr_pct", 0.0) >= 5.0
                    or abs(ind.get("momentum_6_pct", 0.0)) >= 5.0):
                micro_targets.add(sym)
            last_close = float(df["close"].iloc[-1])
            row = {
                "ts": now.isoformat(),
                "symbol": sym,
                "close": last_close,
                "ind_rvol": ind.get("rvol", 1.0),
                "ind_atr_pct": ind.get("atr_pct", 0.0),
                "ind_vwap_distance_pct": ind.get("vwap_distance_pct", 0.0),
                "ind_bb_squeeze": int(bool(ind.get("bb_squeeze", False))),
                "ind_momentum_3_pct": ind.get("momentum_3_pct", 0.0),
                "ind_momentum_6_pct": ind.get("momentum_6_pct", 0.0),
                "btc_momentum_12_pct": btc_mom_12,
                "btc_state": btc_state,
            }
            row.update({f"f_{k}": v for k, v in feats.items()})
            row.update({f"f_{k}": v for k, v in adv.items()
                       if isinstance(v, (int, float, bool))})
            # String fields (wave, ichimoku) — keep as columns, mark f_ too
            for sk, sv in adv.items():
                if not isinstance(sv, (int, float, bool)):
                    row[f"f_{sk}"] = str(sv)
            row["quality_ok"] = int(qrep.ok)
            row["candles"] = int(qrep.stats.get("candles", 0))
            row["has_microstructure"] = int(sym in micro_targets)
            rows.append(row)
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  fail {sym}: {type(e).__name__}: {e}")
            continue
        # Light rate limit
        time.sleep(0.2)

    # ---- Pass 2: microstructure for top movers only ----
    if micro_targets:
        print(f"[{now.isoformat()}] microstructure pass on "
              f"{len(micro_targets)} symbols: {sorted(micro_targets)[:5]}...")
        for sym in micro_targets:
            try:
                # Re-fetch the 4h klines we already validated. We don't have it
                # cached in this scope, so fetch again (cheap relative to the
                # 5+ micro API calls that follow).
                df = client.get_klines(sym, interval="4h", limit=200)
                if df.empty:
                    continue
                qrep = validate_ohlcv(df, min_candles=60, interval_hours=4.0)
                if qrep.cleaned is not None:
                    df = qrep.cleaned
                micro = _micro_features(sym, df)
                # Merge into the matching row
                for r in rows:
                    if r["symbol"] == sym:
                        r.update({f"f_{k}": v for k, v in micro.items()})
                        break
            except Exception as e:
                if failures < 10:
                    print(f"  micro fail {sym}: {type(e).__name__}")
            time.sleep(0.6)  # heavier rate limit for micro API calls

    if not rows:
        print(f"[{now.isoformat()}] cycle: 0 rows (failures={failures})")
        return 0

    # ---- Pass 3: per-coin social signals (Google Trends + CoinGecko +
    #              Reddit + CryptoPanic). Heavy (~3-5s per coin). ----
    cycle_start = time.time()
    if rows:
        if _SOCIAL_CACHE["agg"] is None:
            _SOCIAL_CACHE["agg"] = CoinSocialAggregator()
        agg = _SOCIAL_CACHE["agg"]
        # Full collection only for top movers / candidates; cheap mode for the rest
        full_mask = {}
        sym_to_base = {}
        base_to_syms: Dict[str, list] = {}
        for r in rows:
            sym = r["symbol"]
            base = sym.replace("USDT", "").replace("-SWAP", "")
            sym_to_base[sym] = base
            base_to_syms.setdefault(base, []).append(sym)
            full_mask[base] = bool(
                r.get("score_long", 0) >= 50
                or r.get("score_short", 0) >= 50
                or r.get("f_rvol", 1.0) >= 1.3
                or r.get("f_atr_pct", 0.0) >= 5.0
                or abs(r.get("f_momentum_6_pct", 0.0)) >= 5.0
            )
        unique_bases = list(base_to_syms.keys())
        n_full = sum(1 for v in full_mask.values() if v)
        n_cheap = len(unique_bases) - n_full
        # If we're already past 4 minutes, only do CoinGecko (skip trends/reddit/panic)
        elapsed = time.time() - cycle_start
        skip_expensive_sources = elapsed > 240
        if skip_expensive_sources:
            print(f"[{now.isoformat()}] social pass: cycle already {elapsed:.0f}s, "
                  f"using CoinGecko only")
        print(f"[{now.isoformat()}] social pass on {len(unique_bases)} bases "
              f"({n_full} full, {n_cheap} cheap)")
        try:
            social = agg.collect_many(unique_bases, full_mask,
                                      skip_expensive=skip_expensive_sources)
        except Exception as e:
            print(f"  social pass error: {type(e).__name__}: {e}")
            social = {}
        # Fill in empty for any base that didn't return
        for b in unique_bases:
            if b not in social:
                social[b] = {
                    "gt_present": False, "cg_community_data_present": False,
                    "rd_post_count_24h": 0, "cp_post_count_24h": 0,
                }
        for r in rows:
            base = sym_to_base[r["symbol"]]
            s = social.get(base, {})
            if not s:
                s = {"gt_present": False, "cg_community_data_present": False,
                     "rd_post_count_24h": 0, "cp_post_count_24h": 0}
            for k, v in s.items():
                r[f"s_{k}"] = v
            r["s_has_social_data"] = int(
                bool(s.get("gt_present"))
                or bool(s.get("cg_community_data_present"))
                or int(s.get("rd_post_count_24h", 0)) > 0
                or int(s.get("cp_post_count_24h", 0)) > 0
            )

    # ---- Pass 4: direction scoring (LONG/SHORT) ----
    # First, backfill any rows in the existing CSV that don't have scores
    # (handles data collected before this feature was added).
    if FEATURE_LOG.exists():
        try:
            from .backfill_scores import backfill as _bf
            _bf(only_missing=True)
        except Exception as e:
            pass  # backfill is best-effort, don't fail cycle

    for r in rows:
        try:
            ds = direction_score(r, symbol=r["symbol"])
            r["score_long"] = ds.long_score
            r["score_short"] = ds.short_score
            r["direction"] = ds.direction
            r["confidence"] = ds.confidence
            r["n_long_signals"] = ds.long_signals
            r["n_short_signals"] = ds.short_signals
            r["long_fired"] = ",".join(ds.long_fired)
            r["short_fired"] = ",".join(ds.short_fired)
        except Exception as e:
            r["score_long"] = 0.0
            r["score_short"] = 0.0
            r["direction"] = "ERROR"
            r["confidence"] = 0.0
            r["n_long_signals"] = 0
            r["n_short_signals"] = 0
            r["long_fired"] = ""
            r["short_fired"] = ""

    out = pd.DataFrame(rows)
    # Make sure all expected columns exist (forward compat with new features)
    expected_min = ["ts", "symbol", "close"]
    _ensure_columns(out, expected_min)
    # Write to SQLite (primary store)
    try:
        database.init_db()
        database.insert_features(rows)
    except Exception as e:
        print(f"  SQLite insert error: {type(e).__name__}: {e}")
    # Also keep CSV as backup (for backward compat)
    if FEATURE_LOG.exists():
        try:
            old = pd.read_csv(FEATURE_LOG)
            out = pd.concat([old, out], ignore_index=True, sort=False)
        except Exception:
            pass
    out.to_csv(FEATURE_LOG, index=False)
    # Print DB size
    try:
        db_info = database.get_db_size()
        print(f"[{now.isoformat()}] cycle: {len(rows)} rows appended, "
              f"failures={failures}, total={len(out)}, "
              f"DB={db_info['size_mb']:.2f}MB "
              f"(feat={db_info['row_counts']['features']})")
    except Exception:
        print(f"[{now.isoformat()}] cycle: {len(rows)} rows appended, "
              f"failures={failures}, total={len(out)}")

    # ---- Pass 4.5: REPEATER scanner (priority: 6 known pump symbols 24/7) ----
    # Detects pre-pump patterns on symbols that have pumped before
    # (EVAA, TLM, LAB, BANK, AKE, DN). Catches ~25% of all pumps.
    n_repeater = 0
    try:
        from .repeater_scanner import run_repeater_cycle
        repeater_summary = run_repeater_cycle(verbose=False)
        n_repeater = len(repeater_summary.get("pre_pumps", [])) + \
                     len(repeater_summary.get("confirmed", []))
        if n_repeater > 0:
            print(f"  [REPEATER] Opened {n_repeater} signals "
                  f"(pre={len(repeater_summary.get('pre_pumps',[]))}, "
                  f"confirm={len(repeater_summary.get('confirmed',[]))})")
    except Exception as e:
        print(f"  [REPEATER] error: {e}")

    # ---- Pass 5: open new signals for TP/SL tracking ----
    # For each LONG/SHORT signal with high confidence, open a tracked signal
    # with adaptive TP/SL (ATR/momentum/confidence-aware).
    n_opened = 0
    n_resolved = 0
    n_tp = 0
    n_sl = 0
    tp_sl_log = []
    # ULTRA STRICT MODE: only open signals that pass ultra-strict filter
    # (14+ criteria). Expected: 0-3 signals per day.
    ultra_ids = open_ultra_signals(
        min_confidence=60.0, tp_pct=5.0, sl_pct=3.0, max_hours=8.0
    )
    n_opened = len(ultra_ids)
    if ultra_ids:
        # Build log of what was opened with their TP/SL
        from .adaptive_tp_sl import format_tp_sl_for_log
        for sid in ultra_ids:
            sig_row = None
            try:
                open_df = get_open_signals()
                matches = open_df[open_df["signal_id"] == sid]
                if not matches.empty:
                    sig_row = matches.iloc[0]
            except Exception:
                pass
            if sig_row is not None:
                tp_sl = {
                    "tp_pct": sig_row.get("tp_pct", 5),
                    "sl_pct": sig_row.get("sl_pct", 3),
                    "trailing_pct": sig_row.get("trailing_pct", 1.5),
                    "use_trailing": bool(sig_row.get("use_trailing", True)),
                }
                tp_sl_log.append(
                    f"{sig_row['symbol']:<14} {sig_row['direction']:<6} "
                    f"{format_tp_sl_for_log(tp_sl)}"
                )
    print(f"[{now.isoformat()}] signals: opened {n_opened} new (ultra-strict)")
    if tp_sl_log:
        print(f"  ultra signals with smart v2 exit:")
        for line in tp_sl_log:
            print(f"    {line}")

    # ---- Pass 6: check existing signals against current prices ----
    # Build current_prices dict from the rows we just collected
    current_prices = {r["symbol"]: float(r["close"]) for r in rows
                      if r.get("close", 0) > 0}
    if current_prices:
        try:
            # Use smart v2 logic for existing positions
            n_resolved, n_tp, n_sl, n_be = check_signals_smart_v2(current_prices)
            if n_resolved > 0:
                stats = get_stats()
                print(f"[{now.isoformat()}] resolved {n_resolved}: "
                      f"TP={n_tp}, SL={n_sl}, Breakeven={n_be}, "
                      f"win_rate={stats.get('win_rate', 0):.2f} "
                      f"({stats.get('n_total', 0)} total)")
        except Exception as e:
            if failures < 5:
                print(f"  check_signals error: {type(e).__name__}: {e}")
    return len(rows)


def label_outcomes(horizon_hours: int = 12) -> int:
    """
    For every row in feature_log.csv older than `horizon_hours`,
    fetch the price change and label it pump / dump / flat.
    Update outcome_log.csv.

    This is called by the same loop: at every cycle we re-label any
    rows whose horizon has elapsed (cheap if we cache prices).
    """
    if not FEATURE_LOG.exists():
        return 0
    feat = pd.read_csv(FEATURE_LOG)
    if feat.empty or "ts" not in feat.columns:
        return 0

    feat["ts"] = pd.to_datetime(feat["ts"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    feat = feat.dropna(subset=["ts"])
    if feat.empty:
        return 0

    # Outcomes already recorded?
    if OUTCOME_LOG.exists():
        done = pd.read_csv(OUTCOME_LOG)
        done_keys = set(zip(done["ts"].astype(str), done["symbol"]))
    else:
        done = pd.DataFrame()
        done_keys = set()

    to_label = feat[
        (now - feat["ts"] >= pd.Timedelta(hours=horizon_hours))
        & (~feat.apply(lambda r: (str(r["ts"]), r["symbol"]) in done_keys, axis=1))
    ]
    if to_label.empty:
        return 0

    client = ToobitClient()
    outcomes: list[dict] = []
    for _, r in to_label.iterrows():
        sym = r["symbol"]
        signal_close = float(r["close"])
        # We can't reliably fetch historical close from Toobit at a fixed time.
        # Use the current "now" close as a proxy for the horizon price — good
        # enough for the *next* cycle once horizon has elapsed.
        try:
            kdf = client.get_klines(sym, interval="4h", limit=2)
            if kdf.empty:
                continue
            horizon_close = float(kdf["close"].iloc[-1])
            ret = (horizon_close - signal_close) / max(signal_close, 1e-12) * 100.0
            label = "FLAT"
            if ret >= 10.0:
                label = "PUMP"
            elif ret <= -10.0:
                label = "DUMP"
            outcomes.append({
                "ts": str(r["ts"]),
                "symbol": sym,
                "horizon_hours": horizon_hours,
                "signal_close": signal_close,
                "horizon_close": horizon_close,
                "return_pct": ret,
                "label": label,
            })
        except Exception:
            continue
        time.sleep(0.2)

    if not outcomes:
        return 0
    out = pd.DataFrame(outcomes)
    # Save to SQLite
    try:
        database.init_db()
        database.insert_outcomes(outcomes)
    except Exception as e:
        pass
    # Also keep CSV as backup
    if OUTCOME_LOG.exists():
        try:
            old = pd.read_csv(OUTCOME_LOG)
            out = pd.concat([old, out], ignore_index=True, sort=False)
        except Exception:
            pass
    out.to_csv(OUTCOME_LOG, index=False)
    print(f"[{now.isoformat()}] labeled {len(outcomes)} outcomes")
    return len(outcomes)


def discover_symbols(client: ToobitClient) -> list[str]:
    """Return USDT perp symbols within our small-cap volume band."""
    df = client.get_24h_tickers()
    if df.empty:
        return []
    df = df.dropna(subset=["quote_volume_24h", "last_price"])
    df = df[(df["quote_volume_24h"] >= MIN_24H_VOLUME_USD)
            & (df["quote_volume_24h"] <= MAX_24H_VOLUME_USD)
            & (df["last_price"] > 0)]
    # Exclude obvious majors
    majors = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX",
              "LINK", "DOT", "MATIC", "TON", "LTC", "BCH", "NEAR", "ATOM",
              "UNI", "APT", "ARB", "OP", "FIL", "ICP", "STX", "INJ",
              "TIA", "SEI", "SUI", "AAVE", "MKR", "GRT", "RUNE", "ALGO",
              "EGLD", "FTM", "SAND", "MANA", "AXS", "CRV", "LDO", "PEPE",
              "SHIB", "WIF", "BONK", "FLOKI", "MEME", "TRB", "BLUR",
              "JTO", "JUP", "PYTH"}
    df = df[~df["base"].isin(majors)]
    return df.sort_values("quote_volume_24h", ascending=False)["symbol"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600,
                        help="seconds between cycles (default 600 = 10 min)")
    parser.add_argument("--once", action="store_true",
                        help="run a single cycle and exit")
    parser.add_argument("--horizon", type=int, default=12,
                        help="outcome horizon in hours (default 12)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    client = ToobitClient()

    if args.once:
        symbols = discover_symbols(client)
        print(f"discovered {len(symbols)} small-cap symbols")
        collect_cycle(client, symbols)
        label_outcomes(args.horizon)
        return

    print(f"starting live_collector loop, interval={args.interval}s")
    while True:
        try:
            symbols = discover_symbols(client)
            collect_cycle(client, symbols)
            label_outcomes(args.horizon)
        except KeyboardInterrupt:
            print("stopping")
            return
        except Exception as e:
            print(f"cycle error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
