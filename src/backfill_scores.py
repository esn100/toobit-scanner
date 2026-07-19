"""
Backfill direction scores for historical rows that lack them.

The direction_score module was added in a recent collector update, so
earlier rows in feature_log.csv don't have score_long / score_short /
direction columns. Run this once after pulling data to fill them in.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd

from .direction_scoring import direction_score


DATA_DIR = Path("data")
FEATURE_LOG = DATA_DIR / "feature_log.csv"


def backfill(only_missing: bool = True) -> int:
    if not FEATURE_LOG.exists():
        print("no feature_log.csv")
        return 0
    df = pd.read_csv(FEATURE_LOG)
    n = len(df)
    if n == 0:
        print("empty")
        return 0
    needed = ["score_long", "score_short", "direction", "confidence",
              "n_long_signals", "n_short_signals", "long_fired", "short_fired"]
    if only_missing and "score_long" in df.columns:
        mask = df["score_long"].isna()
    else:
        mask = pd.Series([True] * n)
    targets = df[mask]
    print(f"backfilling {len(targets)} of {n} rows...")
    long_scores = []
    short_scores = []
    directions = []
    confidences = []
    n_longs = []
    n_shorts = []
    long_fired_list = []
    short_fired_list = []
    for _, row in targets.iterrows():
        feats = row.to_dict()
        try:
            ds = direction_score(feats, symbol=str(row.get("symbol", "")))
            long_scores.append(ds.long_score)
            short_scores.append(ds.short_score)
            directions.append(ds.direction)
            confidences.append(ds.confidence)
            n_longs.append(ds.long_signals)
            n_shorts.append(ds.short_signals)
            long_fired_list.append(",".join(ds.long_fired))
            short_fired_list.append(",".join(ds.short_fired))
        except Exception as e:
            long_scores.append(0.0)
            short_scores.append(0.0)
            directions.append("ERROR")
            confidences.append(0.0)
            n_longs.append(0)
            n_shorts.append(0)
            long_fired_list.append("")
            short_fired_list.append("")
    # Assign back
    for col, vals in [
        ("score_long", long_scores),
        ("score_short", short_scores),
        ("direction", directions),
        ("confidence", confidences),
        ("n_long_signals", n_longs),
        ("n_short_signals", n_shorts),
        ("long_fired", long_fired_list),
        ("short_fired", short_fired_list),
    ]:
        if col not in df.columns:
            df[col] = pd.NA
        # Use index to assign properly
        idx = targets.index
        df.loc[idx, col] = vals
    df.to_csv(FEATURE_LOG, index=False)
    print(f"backfilled {len(targets)} rows -> {FEATURE_LOG}")
    return len(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="re-score every row (not just missing)")
    args = parser.parse_args()
    backfill(only_missing=not args.all)


if __name__ == "__main__":
    main()
