"""
Backtest the 6 advanced intel features against historical pumps.
For each pump, check what signals were active t-2h, t-1h, t-0h.
"""
import json
import sys
import os
from datetime import datetime, timezone
from collections import Counter, defaultdict

sys.path.insert(0, '/home/user/toobit-scanner')

from src.advanced_intel import get_advanced_intel
from src.repeater_scanner import _fetch_klines_1h, _compute_features

# Load pumps
CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/all_pumps.json') as f:
    pumps = json.load(f)

# For each pump, fetch klines and compute features at t-2h, t-1h, t+0
results = []
print(f'Backtesting {len(pumps)} historical pumps...')
print()

for pi, pump in enumerate(pumps, 1):
    sym = pump['sym']
    rows = _fetch_klines_1h(sym, 200)
    if len(rows) < 30:
        continue
    # Find pump index
    target_ts = pump['start_ts']
    pump_i = None
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 3600 and r['ts'] <= target_ts + 3600:
            pump_i = i
            break
    if pump_i is None:
        continue
    # Compute features at different timepoints
    pump_data = {'sym': sym, 'gain_pct': pump['gain_pct']}
    # Features at t-2h
    for h in [2, 1, 0]:
        idx = pump_i - h
        if idx < 20:
            continue
        feats = _compute_features(rows[:idx+1])
        pump_data[f't{h}h'] = feats
    results.append(pump_data)
    if pi <= 5:
        print(f'  [{pi}/{len(pumps)}] {sym}  +{pump["gain_pct"]:.0f}%')

print(f'\nLoaded {len(results)} pump features')

# Now simulate what advanced_intel would have shown
# Note: We can't actually call the real APIs in backtest (no historical data)
# So we approximate based on price/volume patterns

print('\n' + '='*80)
print('ADVANCED INTEL PREDICTIVE POWER (estimated from features)')
print('='*80)

# For each pump, estimate signal strength based on observable features
def estimate_signals(pump_data):
    """Estimate what advanced_intel would show at different timepoints."""
    estimates = {}
    for h in [2, 1, 0]:
        key = f't{h}h'
        if key not in pump_data:
            continue
        f = pump_data[key]
        if not f:
            continue
        # Approximate OBI from price action: rising price = buy pressure
        obi = 0.5 + (f.get('mom_1', 0) / 10)  # crude estimate
        obi = max(0, min(1, obi))
        # Taker buy ratio from volume vs price change
        vol = f.get('rvol', 1)
        mom = f.get('mom_3', 0)
        if vol > 1.5 and mom > 0:
            taker_buy = 0.6
        elif vol > 1.5 and mom < 0:
            taker_buy = 0.4
        else:
            taker_buy = 0.5
        # Smart money from volume surge
        if vol > 2:
            smart_score = 20
        elif vol > 1.5:
            smart_score = 10
        else:
            smart_score = 0
        estimates[key] = {
            'obi': obi,
            'taker_buy': taker_buy,
            'smart_score': smart_score,
        }
    return estimates

# Calculate win rate
correct = 0
total = 0
for pump_data in results:
    estimates = estimate_signals(pump_data)
    for key, est in estimates.items():
        # Check if signals would have triggered (score >= 40)
        score = 0
        if est['obi'] > 0.6:
            score += 20
        elif est['obi'] < 0.4:
            score -= 10
        if est['taker_buy'] > 0.6:
            score += 18
        elif est['taker_buy'] < 0.4:
            score -= 8
        score += est['smart_score']
        # We "predict" pump if score >= 30
        would_trigger = score >= 30
        # Actual: was gain > 30%? (pump)
        is_pump = pump_data['gain_pct'] >= 30
        if would_trigger == is_pump:
            correct += 1
        total += 1

print(f'\nEstimated accuracy: {correct}/{total} = {correct/total*100:.1f}%')
print(f'(rough estimate based on price/volume patterns)')

# By timepoint
print('\n--- BY TIMEPOINT ---')
for h in [2, 1, 0]:
    correct = 0
    total = 0
    for pump_data in results:
        key = f't{h}h'
        if key not in pump_data:
            continue
        estimates = estimate_signals(pump_data)
        est = estimates.get(key, {})
        if not est:
            continue
        score = 0
        if est['obi'] > 0.6:
            score += 20
        if est['taker_buy'] > 0.6:
            score += 18
        score += est['smart_score']
        would_trigger = score >= 30
        is_pump = pump_data['gain_pct'] >= 30
        if would_trigger == is_pump:
            correct += 1
        total += 1
    if total > 0:
        print(f't-{h}h: {correct}/{total} = {correct/total*100:.1f}%')

# Save analysis
with open(f'{CACHE}/advanced_intel_backtest.json', 'w') as f:
    json.dump({
        'n_pumps': len(results),
        'correct': correct,
        'total': total,
        'accuracy_pct': correct/total*100 if total else 0,
    }, f, indent=2)
print(f'\nSaved to {CACHE}/advanced_intel_backtest.json')
