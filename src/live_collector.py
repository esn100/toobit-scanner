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
            feats = build_features(
                technical=tech,
                indicators=ind,
                structure=struct,
                candle=candle,
                mtf={"alignment_score": 50.0},  # not implemented; default
                btc={"state": btc_state,
                     "btc_momentum_12_pct": btc_mom_12},
            )
            # Add BTC correlation features as extras
            for k, v in btc_corr.items():
                feats[k] = float(v) if isinstance(v, (int, float, bool)) else 0.0
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
            row["quality_ok"] = int(qrep.ok)
            row["candles"] = int(qrep.stats.get("candles", 0))
            rows.append(row)
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  fail {sym}: {type(e).__name__}: {e}")
            continue
        # Light rate limit
        time.sleep(0.2)

    if not rows:
        print(f"[{now.isoformat()}] cycle: 0 rows (failures={failures})")
        return 0

    out = pd.DataFrame(rows)
    # Make sure all expected columns exist (forward compat with new features)
    expected_min = ["ts", "symbol", "close"]
    _ensure_columns(out, expected_min)
    if FEATURE_LOG.exists():
        old = pd.read_csv(FEATURE_LOG)
        out = pd.concat([old, out], ignore_index=True, sort=False)
    out.to_csv(FEATURE_LOG, index=False)
    print(f"[{now.isoformat()}] cycle: {len(rows)} rows appended, "
          f"failures={failures}, total={len(out)}")
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
    if OUTCOME_LOG.exists():
        old = pd.read_csv(OUTCOME_LOG)
        out = pd.concat([old, out], ignore_index=True, sort=False)
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
