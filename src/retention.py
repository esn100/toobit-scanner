"""
Data retention manager.

Rules:
  - Keep last 7 days of detailed data (features, outcomes, signals)
  - For data older than 7 days, keep ONLY aggregate knowledge (no raw rows)
  - Resolve signal data is kept longer (90 days) for stats

What "knowledge" means:
  - Per-symbol daily aggregates (n_cycles, mean_score, mean_atr, etc.)
  - Per-direction aggregates (LONG vs SHORT win rate)
  - Per-confidence bucket aggregates
  - TP/SL hit rates by combo

What gets deleted:
  - All features older than 7d
  - All outcomes older than 7d
  - Resolved signals older than 7d are kept but with
    features_json stripped (no per-cycle detail)

What gets preserved (knowledge only):
  - Aggregated daily stats per symbol (1 row per symbol per day)
  - Aggregate trends (e.g., "BANKUSDT had 5 LONG signals with 60% win rate")

Runs:
  - Manually: python -m src.retention prune
  - Auto:    once per day via cron/Action
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import pandas as pd

from . import db as database


DATA_DIR = Path("data")
KNOWLEDGE_FILE = DATA_DIR / "knowledge_base.json"

# Retention windows
KEEP_DAYS = 7                # raw data
KNOWLEDGE_DAYS = 90          # aggregates go back this far


def _load_knowledge() -> Dict:
    if KNOWLEDGE_FILE.exists():
        try:
            with open(KNOWLEDGE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_update": None,
        "per_symbol": {},     # {symbol: {date: {n_cycles, mean_score_long, ...}}}
        "per_direction": {},  # {direction: {date: {n, win_rate, avg_pnl}}}
        "per_confidence": {},  # {bucket: {date: {n, win_rate}}}
        "per_tp_sl": {},       # {combo: {date: {n, win_rate}}}
        "per_hour": {},        # {hour: {date: {n, win_rate}}}
    }


def _save_knowledge(kb: Dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(KNOWLEDGE_FILE, "w") as f:
        json.dump(kb, f, indent=2, default=str)


def _extract_aggregates(df: pd.DataFrame, cutoff_date: str) -> Dict:
    """
    From a DataFrame of features older than cutoff, build per-symbol
    daily aggregates. Returns dict of {symbol: {date: {...}}}.
    """
    if df.empty:
        return {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["ts"], utc=True,
                                errors="coerce").dt.date.astype(str)
    agg = {}
    for (sym, date), group in df.groupby(["symbol", "date"]):
        if pd.isna(sym) or pd.isna(date):
            continue
        agg.setdefault(sym, {})
        agg[sym][date] = {
            "n_cycles": int(len(group)),
            "n_long": int((group["direction"] == "LONG").sum()),
            "n_short": int((group["direction"] == "SHORT").sum()),
            "mean_score_long": float(group["score_long"].mean())
                if "score_long" in group.columns else 0,
            "mean_score_short": float(group["score_short"].mean())
                if "score_short" in group.columns else 0,
            "mean_atr_pct": float(group["ind_atr_pct"].mean())
                if "ind_atr_pct" in group.columns else 0,
            "mean_mom_3": float(group["ind_momentum_3_pct"].mean())
                if "ind_momentum_3_pct" in group.columns else 0,
            "mean_rvol": float(group["ind_rvol"].mean())
                if "ind_rvol" in group.columns else 0,
            "mean_confidence": float(group["confidence"].mean())
                if "confidence" in group.columns else 0,
        }
    return agg


def _extract_outcome_aggregates(df: pd.DataFrame, cutoff_date: str) -> Dict:
    """
    From outcomes older than cutoff, build daily aggregates per symbol.
    """
    if df.empty:
        return {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["ts"], utc=True,
                                errors="coerce").dt.date.astype(str)
    agg = {}
    for (sym, date), group in df.groupby(["symbol", "date"]):
        if pd.isna(sym) or pd.isna(date):
            continue
        agg.setdefault(sym, {})
        n = len(group)
        if n == 0:
            continue
        n_pump = int((group["label"] == "PUMP").sum())
        n_dump = int((group["label"] == "DUMP").sum())
        n_flat = int((group["label"] == "FLAT").sum())
        agg[sym][date] = {
            "n_outcomes": n,
            "n_pump": n_pump,
            "n_dump": n_dump,
            "n_flat": n_flat,
            "pump_rate": n_pump / n,
            "mean_return": float(group["return_pct"].mean())
                if "return_pct" in group.columns else 0,
        }
    return agg


def prune_with_knowledge(keep_days: int = KEEP_DAYS) -> Dict:
    """
    1. Extract aggregate knowledge from data older than keep_days
    2. Delete the raw data
    3. Save aggregates to knowledge_base.json
    """
    print("=" * 70)
    print(f"RETENTION: keep {keep_days} days, extract knowledge from older")
    print("=" * 70)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_days)
    cutoff_iso = cutoff.isoformat()
    knowledge = _load_knowledge()
    knowledge["last_update"] = now.isoformat()
    # ---- Features ----
    print(f"\n[1] Features older than {cutoff.date()}:")
    all_features = database.get_features()
    all_features["ts"] = pd.to_datetime(all_features["ts"],
                                       utc=True, errors="coerce")
    old = all_features[all_features["ts"] < cutoff]
    n_old = len(old)
    if n_old > 0:
        print(f"  found {n_old} old feature rows")
        # Extract aggregates
        for sym, dates in _extract_aggregates(old, cutoff_iso).items():
            knowledge["per_symbol"].setdefault(sym, {})
            for date, stats in dates.items():
                knowledge["per_symbol"][sym][date] = stats
        # Delete
        try:
            with database.get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM features WHERE ts < ?", (cutoff_iso,)
                )
                n_del = cur.rowcount
                conn.commit()
            print(f"  deleted {n_del} rows")
        except Exception as e:
            print(f"  ERROR: {e}")
            n_del = 0
    else:
        print(f"  no old features")
        n_del = 0
    # ---- Outcomes ----
    print(f"\n[2] Outcomes older than {cutoff.date()}:")
    all_outcomes = database.get_outcomes()
    all_outcomes["ts"] = pd.to_datetime(all_outcomes["ts"],
                                       utc=True, errors="coerce")
    old = all_outcomes[all_outcomes["ts"] < cutoff]
    n_old = len(old)
    if n_old > 0:
        print(f"  found {n_old} old outcome rows")
        for sym, dates in _extract_outcome_aggregates(old, cutoff_iso).items():
            # Merge with existing
            knowledge["per_symbol"].setdefault(sym, {})
            for date, stats in dates.items():
                if date not in knowledge["per_symbol"][sym]:
                    knowledge["per_symbol"][sym][date] = {}
                # Don't overwrite features stats
                for k, v in stats.items():
                    key = f"outcome_{k}"
                    knowledge["per_symbol"][sym][date][key] = v
        try:
            with database.get_conn() as conn:
                cur = conn.execute(
                    "DELETE FROM outcomes WHERE ts < ?", (cutoff_iso,)
                )
                n_del_o = cur.rowcount
                conn.commit()
            print(f"  deleted {n_del_o} rows")
        except Exception as e:
            print(f"  ERROR: {e}")
            n_del_o = 0
    else:
        print(f"  no old outcomes")
        n_del_o = 0
    # ---- Resolved signals (keep for stats, just strip features_json) ----
    print(f"\n[3] Resolved signals older than {cutoff.date()}:")
    resolved = database.get_resolved_signals()
    if not resolved.empty and "ts_exit" in resolved.columns:
        resolved["ts_exit"] = pd.to_datetime(
            resolved["ts_exit"], utc=True, errors="coerce"
        )
        old = resolved[resolved["ts_exit"] < cutoff]
        n_old = len(old)
        if n_old > 0:
            print(f"  found {n_old} old resolved signals (keeping summary, "
                  f"stripping features_json)")
            try:
                with database.get_conn() as conn:
                    conn.execute(
                        "UPDATE resolved SET features_json = NULL "
                        "WHERE ts_exit < ?",
                        (cutoff_iso,),
                    )
                    conn.commit()
            except Exception as e:
                print(f"  ERROR: {e}")
        else:
            print(f"  no old resolved signals")
    # ---- Per-direction / per-confidence aggregates from resolved ----
    print(f"\n[4] Extracting resolved aggregates (any age):")
    if not resolved.empty:
        for direction in ["LONG", "SHORT"]:
            sub = resolved[resolved["direction"] == direction]
            if sub.empty:
                continue
            n = len(sub)
            n_tp = int((sub["status"] == "TP_HIT").sum())
            n_sl = int((sub["status"] == "SL_HIT").sum())
            n_tr = int((sub["status"] == "TRAILING_HIT").sum())
            decided = n_tp + n_sl + n_tr
            win_rate = (n_tp + n_tr) / decided if decided else 0
            avg_pnl = float(sub["exit_pct"].mean()) if n else 0
            knowledge["per_direction"][direction] = {
                "n_total": n,
                "n_tp": n_tp,
                "n_sl": n_sl,
                "n_trailing": n_tr,
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 3),
            }
        # Per confidence bucket
        for bucket in [(0, 50), (50, 60), (60, 70), (70, 80), (80, 100)]:
            sub = resolved[
                (resolved["confidence"] >= bucket[0])
                & (resolved["confidence"] < bucket[1])
            ]
            if sub.empty:
                continue
            n = len(sub)
            n_tp = int((sub["status"] == "TP_HIT").sum())
            n_sl = int((sub["status"] == "SL_HIT").sum())
            n_tr = int((sub["status"] == "TRAILING_HIT").sum())
            decided = n_tp + n_sl + n_tr
            win_rate = (n_tp + n_tr) / decided if decided else 0
            avg_pnl = float(sub["exit_pct"].mean()) if n else 0
            knowledge["per_confidence"][f"{bucket[0]}-{bucket[1]}"] = {
                "n": n,
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 3),
            }
        # Per TP/SL combo
        for tp in [3, 4, 5, 6, 8, 10]:
            for sl in [2, 3, 4, 5]:
                sub = resolved[
                    (resolved["tp_pct"] == tp) & (resolved["sl_pct"] == sl)
                ]
                if sub.empty:
                    continue
                n = len(sub)
                n_tp = int((sub["status"] == "TP_HIT").sum())
                n_sl = int((sub["status"] == "SL_HIT").sum())
                n_tr = int((sub["status"] == "TRAILING_HIT").sum())
                decided = n_tp + n_sl + n_tr
                win_rate = (n_tp + n_tr) / decided if decided else 0
                avg_pnl = float(sub["exit_pct"].mean()) if n else 0
                knowledge["per_tp_sl"][f"tp{tp}_sl{sl}"] = {
                    "n": n,
                    "win_rate": round(win_rate, 3),
                    "avg_pnl": round(avg_pnl, 3),
                }
        # Per hour of day
        resolved["ts_exit_dt"] = pd.to_datetime(
            resolved["ts_exit"], utc=True, errors="coerce"
        )
        resolved["hour"] = resolved["ts_exit_dt"].dt.hour
        for hour in range(24):
            sub = resolved[resolved["hour"] == hour]
            if sub.empty:
                continue
            n = len(sub)
            n_tp = int((sub["status"] == "TP_HIT").sum())
            n_sl = int((sub["status"] == "SL_HIT").sum())
            decided = n_tp + n_sl
            win_rate = n_tp / decided if decided else 0
            knowledge["per_hour"][f"h{hour:02d}"] = {
                "n": n,
                "win_rate": round(win_rate, 3),
            }
    # Save knowledge
    _save_knowledge(knowledge)
    print(f"\n[5] Knowledge saved to {KNOWLEDGE_FILE}")
    # VACUUM
    try:
        database.optimize()
    except Exception:
        pass
    # Summary
    print("\n" + "=" * 70)
    print("RETENTION COMPLETE")
    print("=" * 70)
    info = database.get_db_size()
    print(f"DB size after: {info['size_mb']:.2f} MB")
    print(f"  features: {info['row_counts']['features']}")
    print(f"  outcomes: {info['row_counts']['outcomes']}")
    print(f"  resolved: {info['row_counts']['resolved']}")
    print(f"  per_symbol aggregates: {len(knowledge['per_symbol'])} symbols")
    print(f"  per_confidence buckets: {len(knowledge['per_confidence'])}")
    print(f"  per_tp_sl combos: {len(knowledge['per_tp_sl'])}")
    return {
        "features_deleted": n_del,
        "outcomes_deleted": n_del_o,
        "knowledge_size_kb": KNOWLEDGE_FILE.stat().st_size / 1024,
        "db_size_mb": info["size_mb"],
    }


def show_knowledge() -> None:
    """Pretty-print the knowledge base."""
    kb = _load_knowledge()
    print("=" * 70)
    print("KNOWLEDGE BASE")
    print("=" * 70)
    print(f"last update: {kb.get('last_update')}")
    print()
    if kb.get("per_direction"):
        print("By direction:")
        for d, s in kb["per_direction"].items():
            print(f"  {d}: n={s['n_total']}, win={s['win_rate']*100:.1f}%, "
                  f"avg_pnl={s['avg_pnl']:+.2f}%")
    if kb.get("per_confidence"):
        print("\nBy confidence bucket:")
        for b, s in sorted(kb["per_confidence"].items()):
            print(f"  conf {b}: n={s['n']}, win={s['win_rate']*100:.1f}%, "
                  f"avg_pnl={s['avg_pnl']:+.2f}%")
    if kb.get("per_tp_sl"):
        print("\nBy TP/SL combo:")
        for c, s in sorted(kb["per_tp_sl"].items()):
            print(f"  {c}: n={s['n']}, win={s['win_rate']*100:.1f}%, "
                  f"avg_pnl={s['avg_pnl']:+.2f}%")
    if kb.get("per_hour"):
        print("\nBy hour of day:")
        for h, s in sorted(kb["per_hour"].items()):
            print(f"  {h}: n={s['n']}, win={s['win_rate']*100:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prune", "show"],
                        help="Action: prune old data, or show knowledge")
    parser.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    args = parser.parse_args()
    if args.action == "prune":
        prune_with_knowledge(args.keep_days)
    elif args.action == "show":
        show_knowledge()


if __name__ == "__main__":
    main()
