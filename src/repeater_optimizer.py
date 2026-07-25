"""
Repeater-specific optimizer: learn from resolved repeater signals.
Updates repeater_config.py thresholds based on actual outcomes.

Currently uses 9 signals. Below MIN_SIGNALS_FOR_CALIBRATION (30) but
we can still adjust thresholds based on observed win/loss patterns.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from collections import defaultdict
import statistics
import pandas as pd

DATA_DIR = '/home/user/toobit-scanner/data'
CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
REPEATER_CONFIG = '/home/user/toobit-scanner/src/repeater_config.py'

# Load resolved signals
df = pd.read_csv(f'{DATA_DIR}/resolved_log.csv')
repeater = df[df['entry_mode'].astype(str).str.contains('PRE_PUMP|CONFIRM', na=False)].copy()

if repeater.empty:
    print('No repeater signals yet.')
    sys.exit(0)

print(f'Optimizing from {len(repeater)} repeater signals...')

# Group by symbol
by_sym = repeater.groupby('symbol')
print(f'Symbols with signals: {len(by_sym)}')

# Load current config
import importlib.util
spec = importlib.util.spec_from_file_location("repeater_config", REPEATER_CONFIG)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

# Per-symbol analysis
print(f'\n{"="*100}')
print('PER-SYMBOL PERFORMANCE')
print(f'{"="*100}\n')
print(f'{"Symbol":<12} {"n":>3} {"WR%":>5} {"avg%":>7} {"sum%":>7} {"Best":>7} {"Worst":>7}')
print('-'*60)

suggestions = {}
for sym, group in by_sym:
    n = len(group)
    wins = (group['exit_pct'] > 0).sum()
    wr = wins/n*100
    avg = group['exit_pct'].mean()
    total = group['exit_pct'].sum()
    best = group['exit_pct'].max()
    worst = group['exit_pct'].min()
    print(f'{sym:<12} {n:>3} {wr:>5.0f} {avg:>+7.2f} {total:>+7.2f} {best:>+7.2f} {worst:>+7.2f}')

    # Suggest config adjustments
    cfg = rc.REPEATERS.get(sym, {})
    if not cfg: continue
    new_cfg = dict(cfg)
    suggest = []

    # If win rate < 50%, tighten SL
    if wr < 50 and n >= 2:
        new_sl = max(cfg['sl_pct'] * 0.85, 1.5)
        new_cfg['sl_pct'] = round(new_sl, 2)
        suggest.append(f"SL: {cfg['sl_pct']}% -> {new_cfg['sl_pct']}%")
    # If avg > 0 and best > 15%, widen TP
    if best > 15 and cfg['tp_pct'] < 12:
        new_tp = min(cfg['tp_pct'] * 1.2, 12.0)
        new_cfg['tp_pct'] = round(new_tp, 2)
        suggest.append(f"TP: {cfg['tp_pct']}% -> {new_cfg['tp_pct']}%")
    # If avg < 0 and total < 0, demote to secondary
    if total < 0 and n >= 2 and wr < 50:
        suggest.append(f"DEMOTE to secondary watchlist")

    if suggest:
        suggestions[sym] = {'new_cfg': new_cfg, 'suggest': suggest, 'wr': wr, 'total': total, 'n': n}

print(f'\n{"="*100}')
print('SUGGESTED CONFIG CHANGES')
print(f'{"="*100}\n')

for sym, info in suggestions.items():
    print(f'\n{sym} (n={info["n"]}, WR={info["wr"]:.0f}%, total={info["total"]:+.2f}%):')
    for s in info['suggest']:
        print(f'  - {s}')

# Apply changes to repeater_config.py
if suggestions:
    print(f'\n{"="*100}')
    print('APPLYING CHANGES...')
    print(f'{"="*100}\n')

    # Read config file as text
    with open(REPEATER_CONFIG) as f:
        text = f.read()

    for sym, info in suggestions.items():
        if 'DEMOTE' in str(info['suggest']):
            # Move to secondary
            continue  # manual move
        new_cfg = info['new_cfg']
        old_tp = rc.REPEATERS[sym]['tp_pct']
        old_sl = rc.REPEATERS[sym]['sl_pct']
        new_tp = new_cfg['tp_pct']
        new_sl = new_cfg['sl_pct']

        # Replace tp_pct value
        import re
        # Find the sym block and replace tp_pct
        # Look for: "tp_pct": OLD_VALUE,
        pattern_tp = re.compile(
            rf'("{re.escape(sym)}":.*?"tp_pct":\s*)([\d.]+)',
            re.DOTALL
        )
        text = pattern_tp.sub(rf'\g<1>{new_tp}', text, count=1)

        pattern_sl = re.compile(
            rf'("{re.escape(sym)}":.*?"sl_pct":\s*)([\d.]+)',
            re.DOTALL
        )
        text = pattern_sl.sub(rf'\g<1>{new_sl}', text, count=1)

        print(f'  {sym}: TP {old_tp}->{new_tp}, SL {old_sl}->{new_sl}')

    with open(REPEATER_CONFIG, 'w') as f:
        f.write(text)
    print(f'\nUpdated {REPEATER_CONFIG}')

# Compute global confidence threshold
print(f'\n{"="*100}')
print('GLOBAL CONFIDENCE CALIBRATION')
print(f'{"="*100}\n')

# For each confidence range, what was WR?
bins = [(50, 65), (65, 75), (75, 85), (85, 95), (95, 101)]
print(f'{"Conf range":<14} {"n":>3} {"WR%":>5} {"avg%":>7} {"sum%":>7}')
for lo, hi in bins:
    sub = repeater[(repeater['confidence'] >= lo) & (repeater['confidence'] < hi)]
    if sub.empty: continue
    wr = (sub['exit_pct'] > 0).mean() * 100
    avg = sub['exit_pct'].mean()
    total = sub['exit_pct'].sum()
    print(f'{lo:>4}-{hi:>4}      {len(sub):>3} {wr:>5.0f} {avg:>+7.2f} {total:>+7.2f}')

# Suggest new PRE_PUMP_CONFIDENCE_MIN
# Currently 50. Should be set where avg starts being positive
best_thresh = 50
for thresh in range(50, 95, 5):
    passing = repeater[repeater['confidence'] >= thresh]
    if len(passing) < 2: continue
    avg = passing['exit_pct'].mean()
    if avg > 0:
        best_thresh = thresh

print(f'\nRecommended PRE_PUMP_CONFIDENCE_MIN: {best_thresh}')

# Save stats
stats = {
    'n_signals': len(repeater),
    'n_wins': int((repeater['exit_pct'] > 0).sum()),
    'win_rate': float((repeater['exit_pct'] > 0).mean() * 100),
    'total_pct': float(repeater['exit_pct'].sum()),
    'avg_pct': float(repeater['exit_pct'].mean()),
    'best_trade': float(repeater['exit_pct'].max()),
    'worst_trade': float(repeater['exit_pct'].min()),
    'suggested_conf_min': best_thresh,
    'per_symbol_suggestions': {k: v['suggest'] for k, v in suggestions.items()},
    'last_update': datetime.now(timezone.utc).isoformat(),
}
with open(f'{CACHE}/repeater_optimizer_state.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)
print(f'\nSaved state to {CACHE}/repeater_optimizer_state.json')
