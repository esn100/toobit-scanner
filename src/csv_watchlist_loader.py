"""
CSV Watchlist Loader - imports toobit_low_mid_cap_coins.csv into config.
Categorizes by market cap:
  - Low Cap: < $50M (highest pump potential)
  - Mid Cap: $50M - $500M (good balance)
  - High Cap: > $500M (excluded - too big to pump)
"""
import csv
import os
import sys
from typing import Dict, List, Tuple

CSV_PATH = '/home/user/uploads/toobit_low_mid_cap_coins.csv'
REPEATER_CONFIG = '/home/user/toobit-scanner/src/repeater_config.py'


def parse_csv() -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """
    Parse the CSV file.
    Returns: (low_cap, mid_cap, excluded) dicts
    """
    low_cap = {}    # < $50M
    mid_cap = {}    # $50M - $500M
    excluded = {}  # > $500M
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row['Symbol'].strip()
            cap = int(row['Market_Cap_USD'])
            cat = row['Category'].strip()
            entry = {
                'rank': int(row['Rank_by_Volume']),
                'name': row['Name'].strip(),
                'symbol': sym,
                'cap': cap,
                'category': cat,
            }
            if cap < 50_000_000:
                low_cap[sym] = entry
            elif cap < 500_000_000:
                mid_cap[sym] = entry
            else:
                excluded[sym] = entry
    return low_cap, mid_cap, excluded


def get_cap_based_config(cap_category: str) -> dict:
    """Return pump_runner config based on cap category."""
    if cap_category == 'low':
        # Aggressive: smaller cap = bigger pumps possible
        return {
            'strategy': 'pump_runner',
            'tp_pct': 30.0,
            'sl_pct': 3.0,
            'trail_pct': 7.0,
            'trail_activate_pct': 15.0,
            'max_hold_hours': 12.0,
            'reentry_on_dip': True,
            'reentry_rsi_max': 30.0,  # More aggressive entry
            'reentry_size': 0.30,
            'pre_pump_rvol_min': 0.8,
            'pre_pump_max_rvol_4h_min': 2.0,
            'pre_pump_mom_3_min': -10.0,
            'pre_pump_mom_3_max': 10.0,
            'pre_pump_flat_min_hours': 1.0,
        }
    else:  # mid
        return {
            'strategy': 'pump_runner',
            'tp_pct': 20.0,
            'sl_pct': 2.5,
            'trail_pct': 5.0,
            'trail_activate_pct': 12.0,
            'max_hold_hours': 12.0,
            'reentry_on_dip': True,
            'reentry_rsi_max': 35.0,
            'reentry_size': 0.30,
            'pre_pump_rvol_min': 0.5,
            'pre_pump_max_rvol_4h_min': 1.5,
            'pre_pump_mom_3_min': -5.0,
            'pre_pump_mom_3_max': 5.0,
            'pre_pump_flat_min_hours': 2.0,
        }


def main():
    low_cap, mid_cap, excluded = parse_csv()
    print(f'Parsed CSV:')
    print(f'  Low Cap (<$50M): {len(low_cap)} symbols')
    print(f'  Mid Cap ($50M-$500M): {len(mid_cap)} symbols')
    print(f'  Excluded (>$500M): {len(excluded)} symbols')
    print()
    print('LOW CAP:')
    for sym in sorted(low_cap.keys(), key=lambda s: low_cap[s]['cap']):
        e = low_cap[sym]
        print(f'  {sym:<14} ${e["cap"]/1e6:>7.1f}M  rank={e["rank"]:>3}  {e["name"]}')
    print()
    print('MID CAP:')
    for sym in sorted(mid_cap.keys(), key=lambda s: mid_cap[s]['cap']):
        e = mid_cap[sym]
        print(f'  {sym:<14} ${e["cap"]/1e6:>7.1f}M  rank={e["rank"]:>3}  {e["name"]}')


if __name__ == '__main__':
    main()
