"""
Analyze the 9 repeater signals to see how accurate the pre-pump pattern was.
For each signal, compare:
- Pre-pump confidence (predicted probability)
- Actual outcome (real P&L)
- Pattern features at entry (mom_3, rvol, flat_hours, max_rvol_4h)
"""
import json, os
import urllib.request
import pandas as pd
from datetime import datetime, timezone

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
DATA_DIR = '/home/user/toobit-scanner/data'

# Load resolved signals
df = pd.read_csv(f'{DATA_DIR}/resolved_log.csv')
print(f'Loaded {len(df)} resolved signals')

# Filter repeater signals only
repeater = df[df['entry_mode'].astype(str).str.contains('PRE_PUMP|CONFIRM', na=False)].copy()
print(f'Repeater signals: {len(repeater)}')

# Add pre-pump features from repeater_state.json
with open(f'{DATA_DIR}/repeater_state.json') as f:
    state = json.load(f)

print(f'\n{"="*100}')
print('PRE-PUMP PATTERN ANALYSIS')
print(f'{"="*100}\n')

print(f'{"Symbol":<12} {"Conf":>5} {"Mom3%":>7} {"Rvol":>6} {"Max4h":>6} {"Flat_h":>6} {"Body":>5} {"RSI":>5} {"ATR%":>5} | {"Real%":>6} {"Win":>4} {"Reason":<10}')
print('-'*100)

# Calculate pre-pump accuracy
wins = 0; losses = 0
features_list = []
for _, row in repeater.iterrows():
    sym = row['symbol']
    conf = row.get('confidence', 0)
    real_pct = float(row.get('exit_pct', 0))
    reason = row.get('exit_reason', '')
    is_win = real_pct > 0
    if is_win: wins += 1
    else: losses += 1

    # Find pre-pump features
    pending = state.get('pending_pre_pump', {}).get(sym, {})
    feats = pending.get('features', {})
    if not feats:
        # Try last_pump_time area
        last_pump = state.get('last_pump_time', {}).get(sym, '')
        feats = {'mom_3': 0, 'rvol': 0, 'max_rvol_4h': 0, 'flat_hours': 0,
                 'body': 0, 'rsi': 0, 'atr_pct': 0}

    print(f'{sym:<12} {conf:>5.0f} {feats.get("mom_3",0):>+7.2f} {feats.get("rvol",0):>6.2f} '
          f'{feats.get("max_rvol_4h",0):>6.2f} {feats.get("flat_hours",0):>6.1f} '
          f'{feats.get("body",0):>5.2f} {feats.get("rsi",0):>5.1f} {feats.get("atr_pct",0):>5.2f} | '
          f'{real_pct:>+6.2f} {"✓" if is_win else "✗":>4} {reason:<10}')
    features_list.append({
        'sym': sym, 'conf': conf, 'real_pct': real_pct, 'is_win': is_win,
        'reason': reason,
        **feats
    })

# Analyze
print(f'\n{"="*100}')
print('ACCURACY BY SIGNAL CHARACTERISTICS')
print(f'{"="*100}\n')

fdf = pd.DataFrame(features_list)
print(f'Total: {len(fdf)} signals, {wins} wins, {losses} losses ({wins/(wins+losses)*100:.1f}% win rate)')
print(f'Sum real_pct: {fdf["real_pct"].sum():+.2f}%')
print(f'Best trade: {fdf.loc[fdf["real_pct"].idxmax(), "sym"]} {fdf["real_pct"].max():+.2f}%')
print(f'Worst trade: {fdf.loc[fdf["real_pct"].idxmin(), "sym"]} {fdf["real_pct"].min():+.2f}%')

# By confidence buckets
print(f'\n--- WIN RATE BY CONFIDENCE ---')
for lo, hi in [(50, 70), (70, 85), (85, 101)]:
    sub = fdf[(fdf['conf'] >= lo) & (fdf['conf'] < hi)]
    if len(sub) > 0:
        wr = (sub['real_pct'] > 0).mean() * 100
        avg = sub['real_pct'].mean()
        print(f'  conf {lo}-{hi}: n={len(sub)}, WR={wr:.0f}%, avg_pct={avg:+.2f}%')

# By rvol
print(f'\n--- WIN RATE BY rvol (at pre-pump) ---')
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 100)]:
    sub = fdf[(fdf['rvol'] >= lo) & (fdf['rvol'] < hi)]
    if len(sub) > 0:
        wr = (sub['real_pct'] > 0).mean() * 100
        avg = sub['real_pct'].mean()
        print(f'  rvol {lo}-{hi}x: n={len(sub)}, WR={wr:.0f}%, avg_pct={avg:+.2f}%')

# By mom_3
print(f'\n--- WIN RATE BY mom_3 (at pre-pump) ---')
for lo, hi in [(-10, -3), (-3, 0), (0, 3), (3, 10), (10, 50)]:
    sub = fdf[(fdf['mom_3'] >= lo) & (fdf['mom_3'] < hi)]
    if len(sub) > 0:
        wr = (sub['real_pct'] > 0).mean() * 100
        avg = sub['real_pct'].mean()
        print(f'  mom_3 {lo} to {hi}%: n={len(sub)}, WR={wr:.0f}%, avg_pct={avg:+.2f}%')

# By flat_hours
print(f'\n--- WIN RATE BY flat_hours ---')
for lo, hi in [(0, 2), (2, 5), (5, 20), (20, 200)]:
    sub = fdf[(fdf['flat_hours'] >= lo) & (fdf['flat_hours'] < hi)]
    if len(sub) > 0:
        wr = (sub['real_pct'] > 0).mean() * 100
        avg = sub['real_pct'].mean()
        print(f'  flat {lo}-{hi}h: n={len(sub)}, WR={wr:.0f}%, avg_pct={avg:+.2f}%')

# Find best predictive thresholds
print(f'\n--- SUGGESTED FILTER THRESHOLDS ---')
# For each feature, find threshold that maximizes avg_pct
for feat in ['rvol', 'max_rvol_4h', 'flat_hours', 'mom_3', 'body', 'rsi']:
    if feat not in fdf.columns: continue
    # Test thresholds
    best_thresh = None; best_avg = -1000; best_wr = 0
    for thresh in sorted(fdf[feat].unique()):
        # Signals that PASS threshold (above for rvol/flat/body/rsi, abs() for mom_3)
        if feat == 'mom_3':
            passing = fdf[fdf['mom_3'].abs() < abs(thresh)]
        else:
            passing = fdf[fdf[feat] >= thresh]
        if len(passing) < 2: continue
        avg = passing['real_pct'].mean()
        wr = (passing['real_pct'] > 0).mean() * 100
        if avg > best_avg:
            best_avg = avg; best_thresh = thresh; best_wr = wr
    print(f'  {feat}: best_thresh={best_thresh}, avg_pct={best_avg:+.2f}%, WR={best_wr:.0f}%')

# Save features for self-train
with open(f'{CACHE}/repeater_outcomes.json', 'w') as f:
    json.dump(features_list, f, indent=2, default=str)
print(f'\nSaved to {CACHE}/repeater_outcomes.json')
