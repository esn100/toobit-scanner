"""
Data Quality Control (Layer 0 of the PumpHunter pipeline).

For every OHLCV dataframe we receive, we validate:
  - minimum number of candles
  - no NaN / zero / negative values
  - monotonic timestamp ordering
  - no excessive time gaps
  - no stale (too old) data
  - OHLC relationship integrity (low <= open,close <= high)

Returns a `QualityReport` with ok=True/False, reasons, and a cleaned df.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QualityReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned: Optional[pd.DataFrame] = None
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        lines = [f"[quality] {status}"]
        for r in self.reasons:
            lines.append(f"  ✗ {r}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        if self.stats:
            lines.append("  stats: " + ", ".join(
                f"{k}={v}" for k, v in self.stats.items()
            ))
        return "\n".join(lines)


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    min_candles: int = 60,
    max_gap_multiplier: float = 2.5,
    max_age_hours: float = 72.0,
    interval_hours: float = 4.0,
) -> QualityReport:
    """
    Run the full quality battery on an OHLCV dataframe.

    Args:
        df: dataframe with columns open_time, open, high, low, close, volume
        min_candles: minimum number of candles required
        max_gap_multiplier: a gap > multiplier * interval is a fail
        max_age_hours: data older than this is considered stale
        interval_hours: expected interval between candles (e.g. 4 for 4h)

    Returns:
        QualityReport with ok/reasons/warnings and the cleaned df
    """
    rep = QualityReport(ok=True)
    if df is None or df.empty:
        rep.ok = False
        rep.reasons.append("dataframe is empty")
        return rep

    # 1) Sufficient candles
    n = len(df)
    rep.stats["candles"] = n
    if n < min_candles:
        rep.ok = False
        rep.reasons.append(f"only {n} candles (< {min_candles})")
        # Even if we fail, return what we have for partial analysis

    # 2) Required columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.ok = False
        rep.reasons.append(f"missing columns: {missing}")
        return rep

    # 3) Nulls / NaNs
    nulls = df[required].isna().sum().sum()
    rep.stats["null_cells"] = int(nulls)
    if nulls > 0:
        # Drop rows with NaN in OHLCV
        before = len(df)
        df = df.dropna(subset=required).reset_index(drop=True)
        rep.warnings.append(f"dropped {before - len(df)} rows with NaN")

    # 4) Negative or zero prices/volumes
    neg_price = (df[required[:-1]] <= 0).any().any()
    neg_vol = (df["volume"] < 0).any()
    if neg_price:
        rep.ok = False
        rep.reasons.append("non-positive prices found")
    if neg_vol:
        rep.ok = False
        rep.reasons.append("negative volume found")

    # 5) OHLC integrity
    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    n_bad = int(bad_ohlc.sum())
    rep.stats["bad_ohlc_rows"] = n_bad
    if n_bad > 0:
        # Drop bad rows but only warn if it's a small fraction
        before = len(df)
        df = df[~bad_ohlc].reset_index(drop=True)
        if n_bad / max(1, before) > 0.05:
            rep.warnings.append(
                f"dropped {n_bad} rows with bad OHLC integrity"
            )

    # 6) Time ordering + duplicates
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        n_dup = int(df["open_time"].duplicated().sum())
        rep.stats["duplicate_timestamps"] = n_dup
        if n_dup > 0:
            df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
            rep.warnings.append(f"dropped {n_dup} duplicate timestamps")
        if not df["open_time"].is_monotonic_increasing:
            df = df.sort_values("open_time").reset_index(drop=True)
            rep.warnings.append("reordered rows by timestamp")

        # 7) Gaps
        if len(df) >= 2:
            deltas = df["open_time"].diff().dropna().dt.total_seconds() / 3600.0
            median_gap = float(deltas.median())
            rep.stats["median_gap_hours"] = round(median_gap, 2)
            max_gap = float(deltas.max())
            rep.stats["max_gap_hours"] = round(max_gap, 2)
            expected = interval_hours
            if max_gap > max_gap_multiplier * expected:
                rep.warnings.append(
                    f"large time gap detected: {max_gap:.1f}h "
                    f"(expected {expected}h, multiplier {max_gap_multiplier})"
                )
                # Mark the dataframe but don't fail outright - sometimes
                # exchanges publish patches with gaps
                if max_gap > 4 * max_gap_multiplier * expected:
                    rep.ok = False
                    rep.reasons.append(
                        f"unacceptable time gap: {max_gap:.1f}h"
                    )

        # 8) Stale data
        if not df.empty:
            last = df["open_time"].iloc[-1]
            # Ensure `last` is UTC tz-aware; `now` is tz-aware too.
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            now = pd.Timestamp.now(tz="UTC")
            age_h = (now - last).total_seconds() / 3600.0
            rep.stats["age_hours"] = round(age_h, 2)
            if age_h > max_age_hours:
                rep.ok = False
                rep.reasons.append(
                    f"data is stale ({age_h:.1f}h > {max_age_hours}h)"
                )
            elif age_h > 2 * interval_hours:
                rep.warnings.append(
                    f"data is slightly stale ({age_h:.1f}h)"
                )

    # 9) Zero volume candles
    zero_vol = int((df["volume"] == 0).sum())
    rep.stats["zero_volume_candles"] = zero_vol
    if zero_vol > 0 and zero_vol / max(1, len(df)) > 0.5:
        rep.warnings.append(
            f"{zero_vol}/{len(df)} candles have zero volume"
        )

    rep.cleaned = df
    return rep
