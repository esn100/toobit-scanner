"""
Database migration and maintenance script.

Run periodically (or via cron) to:
  1. Migrate existing CSV data to SQLite (one-time)
  2. Prune old data beyond retention period
  3. Downsample old features (keep last 24h full, older → 1h aggregates)
  4. VACUUM and ANALYZE for optimal performance
  5. Report DB size and row counts

Usage:
  python -m src.migrate_to_sqlite migrate    # one-time migration
  python -m src.migrate_to_sqlite cleanup   # prune + downsample
  python -m src.migrate_to_sqlite report    # show DB stats
  python -m src.migrate_to_sqlite all       # do everything
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import db as database


DATA_DIR = Path("data")
FEATURE_LOG = DATA_DIR / "feature_log.csv"
OUTCOME_LOG = DATA_DIR / "outcome_log.csv"
ACTIVE_LOG = DATA_DIR / "active_signals.csv"
RESOLVED_LOG = DATA_DIR / "resolved_log.csv"


def migrate_csv_to_sqlite() -> Dict:
    """Migrate existing CSV files to SQLite. Idempotent (uses INSERT OR REPLACE)."""
    print("=" * 70)
    print("MIGRATING CSV → SQLITE")
    print("=" * 70)
    database.init_db()
    counts = {}
    # Features
    if FEATURE_LOG.exists():
        print(f"  reading {FEATURE_LOG}...")
        try:
            df = pd.read_csv(FEATURE_LOG)
            print(f"  found {len(df)} rows")
            # Convert NaN to None for SQLite
            rows = df.to_dict("records")
            for r in rows:
                # Clean up NaN values
                for k, v in list(r.items()):
                    if pd.isna(v):
                        r[k] = None
            n = database.insert_features(rows)
            counts["features"] = n
            print(f"  inserted {n} features")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            counts["features"] = 0
    # Outcomes
    if OUTCOME_LOG.exists():
        print(f"  reading {OUTCOME_LOG}...")
        try:
            df = pd.read_csv(OUTCOME_LOG)
            print(f"  found {len(df)} rows")
            n = database.insert_outcomes(df.to_dict("records"))
            counts["outcomes"] = n
            print(f"  inserted {n} outcomes")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            counts["outcomes"] = 0
    # Active signals
    if ACTIVE_LOG.exists():
        print(f"  reading {ACTIVE_LOG}...")
        try:
            df = pd.read_csv(ACTIVE_LOG)
            print(f"  found {len(df)} rows")
            n = 0
            for _, r in df.iterrows():
                sig = r.to_dict()
                for k, v in list(sig.items()):
                    if pd.isna(v):
                        sig[k] = None
                try:
                    database.upsert_signal(sig)
                    n += 1
                except Exception:
                    continue
            counts["active_signals"] = n
            print(f"  inserted {n} active signals")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            counts["active_signals"] = 0
    # Resolved signals
    if RESOLVED_LOG.exists():
        print(f"  reading {RESOLVED_LOG}...")
        try:
            df = pd.read_csv(RESOLVED_LOG)
            print(f"  found {len(df)} rows")
            n = 0
            for _, r in df.iterrows():
                sig = r.to_dict()
                for k, v in list(sig.items()):
                    if pd.isna(v):
                        sig[k] = None
                try:
                    database.move_to_resolved(sig)
                    n += 1
                except Exception:
                    continue
            counts["resolved"] = n
            print(f"  inserted {n} resolved signals")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            counts["resolved"] = 0
    # Migrate signal_stats.json
    stats_file = DATA_DIR / "signal_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            database.set_stats_value("signal_stats", stats)
            print(f"  migrated signal_stats.json")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print("=" * 70)
    db_info = database.get_db_size()
    print(f"Final DB size: {db_info['size_mb']} MB")
    print(f"  features: {db_info['row_counts']['features']}")
    print(f"  outcomes: {db_info['row_counts']['outcomes']}")
    print(f"  signals: {db_info['row_counts']['signals']}")
    print(f"  resolved: {db_info['row_counts']['resolved']}")
    return counts


def cleanup_old_data(features_days: int = 7,
                     outcomes_days: int = 30,
                     resolved_days: int = 60,
                     downsample_older_than_hours: int = 24) -> Dict:
    """Prune and downsample old data."""
    print("=" * 70)
    print("CLEANUP OLD DATA")
    print("=" * 70)
    # First, downsample (preserves info but reduces row count)
    print(f"  downsampling features older than {downsample_older_than_hours}h...")
    try:
        n_saved = database.downsample_old_features(
            keep_recent_hours=downsample_older_than_hours,
            bucket_minutes=60,
        )
        print(f"  rows saved: {n_saved}")
    except Exception as e:
        print(f"  ERROR downsampling: {e}")
        n_saved = 0
    # Then prune
    print(f"  pruning features older than {features_days}d...")
    print(f"  pruning outcomes older than {outcomes_days}d...")
    print(f"  pruning resolved older than {resolved_days}d...")
    try:
        result = database.prune_old_data(
            features_days=features_days,
            outcomes_days=outcomes_days,
            resolved_days=resolved_days,
        )
        print(f"  deleted: features={result['features_deleted']}, "
              f"outcomes={result['outcomes_deleted']}, "
              f"resolved={result['resolved_deleted']}")
    except Exception as e:
        print(f"  ERROR pruning: {e}")
        result = {"features_deleted": 0, "outcomes_deleted": 0,
                  "resolved_deleted": 0}
    # Optimize
    print("  running VACUUM + ANALYZE...")
    try:
        database.optimize()
        print("  done")
    except Exception as e:
        print(f"  ERROR optimize: {e}")
    return {"downsampled_saved": n_saved, **result}


def report() -> Dict:
    """Print database status report."""
    print("=" * 70)
    print("DATABASE STATUS")
    print("=" * 70)
    info = database.get_db_size()
    print(f"DB file: {database.DB_PATH}")
    print(f"Size: {info['size_mb']:.2f} MB")
    print()
    print("Row counts:")
    for tbl, cnt in info["row_counts"].items():
        print(f"  {tbl:<15} {cnt:>6}")
    # File size comparison
    print()
    print("File sizes (CSV vs SQLite):")
    total_csv = 0
    for f in [FEATURE_LOG, OUTCOME_LOG, ACTIVE_LOG, RESOLVED_LOG]:
        if f.exists():
            sz = f.stat().st_size
            total_csv += sz
            print(f"  {f.name:<25} {sz/1024:.1f} KB")
    print(f"  {'TOTAL CSV':<25} {total_csv/1024:.1f} KB")
    print(f"  {'SQLite DB':<25} {info['size_mb']*1024:.1f} KB")
    if total_csv > 0:
        ratio = (info['size_mb'] * 1024 * 1024) / total_csv
        print(f"  SQLite is {ratio:.1f}x the size of CSV (but indexed/faster)")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["migrate", "cleanup", "report", "all"],
                        help="Action to perform")
    parser.add_argument("--features-days", type=int, default=7)
    parser.add_argument("--outcomes-days", type=int, default=30)
    parser.add_argument("--resolved-days", type=int, default=60)
    parser.add_argument("--downsample-hours", type=int, default=24)
    args = parser.parse_args()
    if args.action in ("migrate", "all"):
        migrate_csv_to_sqlite()
    if args.action in ("cleanup", "all"):
        cleanup_old_data(
            features_days=args.features_days,
            outcomes_days=args.outcomes_days,
            resolved_days=args.resolved_days,
            downsample_older_than_hours=args.downsample_hours,
        )
    if args.action in ("report", "all"):
        report()
    print("\nDone.")


if __name__ == "__main__":
    main()
