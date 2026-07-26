"""
Auto-update SECONDARY_WATCHLIST from toobit_low_mid_cap_coins.csv.
Generates per-symbol config and updates repeater_config.py.
"""
import csv
import re
from datetime import datetime
from typing import Dict, List


CSV_PATH = '/home/user/uploads/toobit_low_mid_cap_coins.csv'
REPEATER_CONFIG = '/home/user/toobit-scanner/src/repeater_config.py'


def get_config_for_cap(cap: int) -> dict:
    """Return pump_runner config based on cap."""
    if cap < 50_000_000:  # low cap
        return {
            'tp_pct': 30.0,
            'sl_pct': 3.0,
            'trail_pct': 7.0,
            'trail_activate_pct': 15.0,
            'max_hold_hours': 12.0,
            'reentry_on_dip': True,
            'reentry_rsi_max': 30.0,
            'reentry_size': 0.30,
            'pre_pump_rvol_min': 0.8,
            'pre_pump_max_rvol_4h_min': 2.0,
            'pre_pump_mom_3_min': -10.0,
            'pre_pump_mom_3_max': 10.0,
            'pre_pump_flat_min_hours': 1.0,
        }
    else:  # mid cap
        return {
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


def parse_csv() -> Dict[str, dict]:
    """Parse CSV and return cap-categorized entries."""
    entries = {}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row['Symbol'].strip()
            cap = int(row['Market_Cap_USD'])
            cat = row['Category'].strip()
            entries[sym] = {
                'rank': int(row['Rank_by_Volume']),
                'name': row['Name'].strip(),
                'cap': cap,
                'category': cat,
                'config': get_config_for_cap(cap),
            }
    return entries


def main():
    entries = parse_csv()
    # Filter to low + mid cap only
    watchlist = {s: e for s, e in entries.items() if e['cap'] < 500_000_000}
    print(f'Total watchlist: {len(watchlist)} symbols')
    low = sum(1 for e in watchlist.values() if e['cap'] < 50_000_000)
    mid = sum(1 for e in watchlist.values() if 50_000_000 <= e['cap'] < 500_000_000)
    print(f'  Low Cap: {low}')
    print(f'  Mid Cap: {mid}')
    # Update config
    with open(REPEATER_CONFIG) as f:
        text = f.read()
    # Find existing SECONDARY_WATCHLIST
    match = re.search(r'SECONDARY_WATCHLIST\s*=\s*\{([^}]*)\}', text, re.DOTALL)
    if not match:
        print('Could not find SECONDARY_WATCHLIST in config!')
        return
    # Parse existing symbols
    existing = set()
    for line in match.group(1).split('\n'):
        m = re.search(r'"([A-Z0-9]+USDT)"', line)
        if m:
            existing.add(m.group(1))
    # Get new symbols (not already in primary or existing secondary)
    primary = {'EVAAUSDT', 'TLMUSDT', 'LABUSDT', 'BANKUSDT', 'AKEUSDT', 'IKAUSDT',
               'SYNUSDT', 'ACEUSDT', 'INUSDT', 'ERAUSDT', 'WODUSDT', 'XCXUSDT'}
    new_symbols = []
    for sym in sorted(watchlist.keys(), key=lambda s: watchlist[s]['cap']):
        if sym not in primary and sym not in existing:
            new_symbols.append(sym)
    print(f'\nNew symbols to add: {len(new_symbols)}')
    # Generate new SECONDARY_WATCHLIST block
    new_block = 'SECONDARY_WATCHLIST = {\n'
    # Keep existing
    for line in match.group(1).split('\n'):
        if line.strip() and ('USDT' in line or '#' in line):
            new_block += '    ' + line.strip() + '\n'
    # Add cap-based section
    new_block += '    # === CAP-BASED WATCHLIST (auto-imported from CSV) ===\n'
    new_block += '    # LOW CAP (<$50M) - highest pump potential\n'
    for sym in sorted(watchlist.keys(), key=lambda s: watchlist[s]['cap']):
        e = watchlist[sym]
        if e['cap'] < 50_000_000:
            sym_usdt = sym if sym.endswith('USDT') else sym + 'USDT'
            new_block += f'    "{sym_usdt}",  # {e["name"]} - ${e["cap"]/1e6:.1f}M\n'
    new_block += '    # MID CAP ($50M-$500M) - good balance\n'
    for sym in sorted(watchlist.keys(), key=lambda s: watchlist[s]['cap']):
        e = watchlist[sym]
        if 50_000_000 <= e['cap'] < 500_000_000:
            sym_usdt = sym if sym.endswith('USDT') else sym + 'USDT'
            new_block += f'    "{sym_usdt}",  # {e["name"]} - ${e["cap"]/1e6:.1f}M\n'
    new_block += '}\n'
    # Replace
    text = text[:match.start()] + new_block + text[match.end():]
    with open(REPEATER_CONFIG, 'w') as f:
        f.write(text)
    print(f'Updated {REPEATER_CONFIG}')
    print(f'Total symbols in SECONDARY_WATCHLIST: {len(existing) + len(new_symbols)}')


if __name__ == '__main__':
    main()
