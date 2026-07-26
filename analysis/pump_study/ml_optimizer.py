"""
ML-based filter optimization to reach 80%+ win rate.
Uses 2845 historical pumps across 12 repeaters (42 days each).
Tests feature combinations and finds optimal thresholds.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import itertools
import statistics

sys.path.insert(0, '/home/user/toobit-scanner')

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'

# Load all data
all_data = {}
REPEATERS = ['EVAAUSDT', 'TLMUSDT', 'LABUSDT', 'BANKUSDT', 'AKEUSDT',
             'IKAUSDT', 'SYNUSDT', 'ACEUSDT', 'INUSDT', 'ERAUSDT', 'WODUSDT', 'XCXUSDT']

print('Loading historical data...')
for sym in REPEATERS:
    cache_path = f'{CACHE}/klines_gateio_{sym}_1h.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            rows = json.load(f)
        all_data[sym] = rows
        print(f'  {sym}: {len(rows)} candles')

# Find all pumps (>= 30% in 24h for higher quality)
print('\nFinding high-quality pumps (>= 30% in 24h)...')
all_pumps = []
for sym, rows in all_data.items():
    for i in range(len(rows) - 24):
        cur_low = rows[i]['l']
        max_fwd = max((rows[i+j]['h'] for j in range(1, 25)), default=cur_low)
        pct = (max_fwd - cur_low) / cur_low * 100 if cur_low > 0 else 0
        if pct >= 30:
            peak_i = i
            for j in range(1, 25):
                if i+j < len(rows) and rows[i+j]['h'] >= max_fwd - 1e-12:
                    peak_i = i+j
            all_pumps.append({
                'sym': sym, 'start_i': i, 'peak_i': peak_i,
                'gain_pct': pct,
            })

print(f'Total high-quality pumps: {len(all_pumps)}')
sym_count = Counter(p['sym'] for p in all_pumps)
print(f'Per symbol: {dict(sym_count)}')

# Compute 20 features at t-2h, t-1h for each pump
def rsi(closes, n=14):
    if len(closes) < n+1: return 50.0
    g, l = [], []
    for i in range(1, n+1):
        ch = closes[i] - closes[i-1]
        g.append(max(ch, 0)); l.append(max(-ch, 0))
    ag = sum(g)/n; al = sum(l)/n
    for i in range(n+1, len(closes)):
        ch = closes[i] - closes[i-1]
        ag = (ag*(n-1) + max(ch,0))/n
        al = (al*(n-1) + max(-ch,0))/n
    if al == 0: return 100.0
    return 100 - 100/(1+ag/al)

def compute_features(rows, idx):
    if idx < 30: return None
    cur = rows[idx]
    closes = [r['c'] for r in rows[:idx+1]]
    vols = [r['v'] for r in rows[:idx+1]]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
    feats = {}
    # Momentum
    for k in [1, 3, 6, 12]:
        if len(rows) > k:
            feats[f'mom_{k}'] = (cur['c'] - rows[idx-k]['c']) / rows[idx-k]['c'] * 100
        else:
            feats[f'mom_{k}'] = 0
    # Volume
    feats['rvol'] = cur['v'] / avg_v if avg_v > 0 else 1
    recent = vols[-5:-1] if len(vols) > 5 else vols[:-1]
    feats['max_rvol_4h'] = max((v/avg_v for v in recent), default=1) if avg_v > 0 else 1
    # RSI
    feats['rsi'] = rsi(closes)
    # ATR
    if len(rows) > 14:
        trs = []
        for i in range(idx-13, idx+1):
            h, l, pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
            trs.append(max(h-l, abs(h-pc), abs(l-pc)))
        feats['atr_pct'] = sum(trs)/14/cur['c']*100 if cur['c'] > 0 else 0
    else:
        feats['atr_pct'] = 0
    # Body
    feats['body'] = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0
    # Flat
    flat = 0
    for j in range(idx-1, max(0, idx-168), -1):
        if abs(cur['c']-rows[j]['c'])/rows[j]['c']*100 <= 5:
            flat = idx-j
        else: break
    feats['flat_hours'] = flat
    # Volume trend
    if len(vols) >= 8:
        recent4 = sum(vols[-4:])/4
        prev4 = sum(vols[-8:-4])/4
        feats['vol_trend'] = (recent4-prev4)/prev4*100 if prev4>0 else 0
    else:
        feats['vol_trend'] = 0
    # Higher lows
    if len(rows) >= 6:
        lows = [rows[idx-i]['l'] for i in range(5, 0, -1)]
        feats['higher_lows'] = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    else:
        feats['higher_lows'] = 0
    return feats

# Compute features for all pumps
print('\nComputing features for each pump at t-1h...')
pump_features = []
for pump in all_pumps:
    sym = pump['sym']
    rows = all_data[sym]
    idx = pump['start_i'] - 1
    if idx < 30: continue
    feats = compute_features(rows, idx)
    if feats:
        pump_features.append({
            'sym': sym,
            'gain': pump['gain_pct'],
            'features': feats,
        })

print(f'Samples: {len(pump_features)}')
print(f'  Pumps (gain >= 50%): {sum(1 for p in pump_features if p["gain"] >= 50)}')
print(f'  Pumps (gain 30-50%): {sum(1 for p in pump_features if 30 <= p["gain"] < 50)}')

# Also create control samples (non-pump periods)
print('\nCreating control samples (random non-pump periods)...')
control = []
import random
random.seed(42)
for sym, rows in all_data.items():
    pump_indices = {p['start_i'] for p in all_pumps if p['sym'] == sym}
    exclude = set()
    for pi in pump_indices:
        for off in range(-24, 25): exclude.add(pi+off)
    valid = [i for i in range(50, len(rows)-1) if i not in exclude]
    random.shuffle(valid)
    for idx in valid[:30]:  # 30 per symbol
        feats = compute_features(rows, idx)
        if feats:
            control.append({'sym': sym, 'features': feats})

print(f'Control samples: {len(control)}')

# Test feature predictive power
all_samples = [{'is_pump': True, **p} for p in pump_features] + \
             [{'is_pump': False, **c} for c in control]
print(f'Total: {len(all_samples)}')

# Find best single feature
print()
print('='*80)
print('BEST SINGLE FEATURES')
print('='*80)
print(f'{"Feature":<20} {"Rule":<25} {"catch%":>8} {"FP%":>8} {"precision":>10} {"f1":>7}')
print('-'*85)

feature_names = list(pump_features[0]['features'].keys())
best_single = []

for feat in feature_names:
    pump_vals = [p['features'][feat] for p in pump_features]
    ctrl_vals = [c['features'][feat] for c in control]
    # Test multiple thresholds
    best_f1 = 0
    best_thresh = 0
    best_dir = '>'
    # Use percentiles as thresholds
    all_vals = sorted(pump_vals + ctrl_vals)
    n = len(all_vals)
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        thresh = all_vals[int(n * pct / 100)]
        # Test >
        pass_p = sum(1 for v in pump_vals if v > thresh)
        pass_c = sum(1 for v in ctrl_vals if v > thresh)
        if pass_p + pass_c > 0:
            prec = pass_p / (pass_p + pass_c)
            rec = pass_p / len(pump_vals)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_dir = '>'
        # Test <
        pass_p = sum(1 for v in pump_vals if v < thresh)
        pass_c = sum(1 for v in ctrl_vals if v < thresh)
        if pass_p + pass_c > 0:
            prec = pass_p / (pass_p + pass_c)
            rec = pass_p / len(pump_vals)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_dir = '<'
    # Calculate final stats
    if best_dir == '>':
        pass_p = sum(1 for v in pump_vals if v > best_thresh)
        pass_c = sum(1 for v in ctrl_vals if v > best_thresh)
    else:
        pass_p = sum(1 for v in pump_vals if v < best_thresh)
        pass_c = sum(1 for v in ctrl_vals if v < best_thresh)
    total = pass_p + pass_c
    prec = pass_p / total if total else 0
    rec = pass_p / len(pump_vals) if pump_vals else 0
    fp_rate = pass_c / len(ctrl_vals) if ctrl_vals else 0
    best_single.append({
        'feature': feat, 'thresh': best_thresh, 'dir': best_dir,
        'precision': prec, 'recall': rec, 'fp_rate': fp_rate, 'f1': best_f1
    })

best_single.sort(key=lambda x: -x['f1'])
for bs in best_single[:10]:
    print(f'{bs["feature"]:<20} {bs["thresh"]:>10.2f}{bs["dir"]:<14} '
          f'{bs["recall"]*100:>6.1f}%  {bs["fp_rate"]*100:>6.1f}%  '
          f'{bs["precision"]*100:>7.1f}%  {bs["f1"]:>5.2f}')

# Now test top 5 features combined (AND logic)
print()
print('='*80)
print('TOP 5 FEATURES COMBINED (AND logic)')
print('='*80)

top5 = best_single[:5]
for n in [2, 3, 4, 5]:
    best_combo_score = 0
    best_combo_rules = None
    for combo in itertools.combinations(top5, n):
        rules = [(bs['feature'], bs['thresh'], bs['dir']) for bs in combo]
        correct = 0
        total = 0
        for s in all_samples:
            passes = True
            for feat, thresh, direction in rules:
                val = s['features'][feat]
                if direction == '>' and val <= thresh: passes = False; break
                if direction == '<' and val >= thresh: passes = False; break
            if passes == s['is_pump']:
                correct += 1
            total += 1
        accuracy = correct/total if total else 0
        if accuracy > best_combo_score:
            best_combo_score = accuracy
            best_combo_rules = rules
    if best_combo_rules:
        print(f'\nBest {n}-feature combo: {best_combo_score*100:.1f}% accuracy')
        for f, t, d in best_combo_rules:
            print(f'  {f} {d} {t:.2f}')

# Save
with open(f'{CACHE}/ml_optimizer_results.json', 'w') as f:
    json.dump({
        'n_pumps': len(pump_features),
        'n_control': len(control),
        'best_single': best_single[:10],
    }, f, indent=2, default=str)
print(f'\nSaved to {CACHE}/ml_optimizer_results.json')
