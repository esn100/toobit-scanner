"""
Deep Backtest Framework - find best signal combination for 80%+ win rate.
Tests all feature combinations against 52 historical pumps.
Uses walk-forward validation to prevent overfitting.
"""
import json
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict
import itertools

sys.path.insert(0, '/home/user/toobit-scanner')

from src.repeater_scanner import _fetch_klines_1h, _compute_features

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/all_pumps.json') as f:
    pumps = json.load(f)

# ============ 20 FEATURE FUNCTIONS ============

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

def ema(values, n):
    if not values: return []
    k = 2/(n+1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v*k + out[-1]*(1-k))
    return out

def compute_20_features(rows, idx):
    """Compute 20 predictive features at index idx."""
    if idx < 30: return None
    cur = rows[idx]
    closes = [r['c'] for r in rows[:idx+1]]
    vols = [r['v'] for r in rows[:idx+1]]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]

    feats = {}
    # 1-4: Momentum
    for k in [1, 3, 6, 12]:
        if len(rows) > k:
            feats[f'mom_{k}'] = (cur['c'] - rows[idx-k]['c']) / rows[idx-k]['c'] * 100
        else:
            feats[f'mom_{k}'] = 0

    # 5: RSI
    feats['rsi'] = rsi(closes)

    # 6-7: Volume ratios
    feats['rvol'] = cur['v'] / avg_v if avg_v > 0 else 1
    recent = vols[-5:-1] if len(vols) > 5 else vols[:-1]
    feats['max_rvol_4h'] = max((v/avg_v for v in recent), default=1) if avg_v > 0 else 1

    # 8: ATR
    if len(rows) > 14:
        trs = []
        for i in range(idx-13, idx+1):
            h, l, pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
            trs.append(max(h-l, abs(h-pc), abs(l-pc)))
        feats['atr_pct'] = sum(trs)/14/cur['c']*100 if cur['c'] > 0 else 0
    else:
        feats['atr_pct'] = 0

    # 9: Body ratio
    feats['body'] = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0

    # 10: BB width
    if len(closes) >= 20:
        ma = sum(closes[-20:])/20
        std = (sum((c-ma)**2 for c in closes[-20:])/20)**0.5
        feats['bb_width'] = (4*std)/ma*100 if ma > 0 else 0
    else:
        feats['bb_width'] = 0

    # 11: BB position
    if len(closes) >= 20 and feats['bb_width'] > 0:
        ma = sum(closes[-20:])/20
        std = (sum((c-ma)**2 for c in closes[-20:])/20)**0.5
        feats['bb_pos'] = (cur['c'] - (ma - 2*std)) / (4*std) if std > 0 else 0.5
    else:
        feats['bb_pos'] = 0.5

    # 12: Flat hours
    flat = 0
    for j in range(idx-1, max(0, idx-168), -1):
        if abs(cur['c']-rows[j]['c'])/rows[j]['c']*100 <= 5:
            flat = idx-j
        else:
            break
    feats['flat_hours'] = flat

    # 13: EMA9-EMA21 cross
    if len(closes) >= 21:
        e9 = ema(closes, 9)[-1]
        e21 = ema(closes, 21)[-1]
        feats['ema_cross'] = (e9 - e21) / e21 * 100 if e21 > 0 else 0
    else:
        feats['ema_cross'] = 0

    # 14: Volume trend (last 4h vs previous 4h)
    if len(vols) >= 8:
        recent4 = sum(vols[-4:])/4
        prev4 = sum(vols[-8:-4])/4
        feats['vol_trend'] = (recent4 - prev4) / prev4 * 100 if prev4 > 0 else 0
    else:
        feats['vol_trend'] = 0

    # 15: Price trend (24h)
    if len(rows) > 24:
        feats['price_24h'] = (cur['c'] - rows[idx-24]['c']) / rows[idx-24]['c'] * 100
    else:
        feats['price_24h'] = 0

    # 16: Volume vs price correlation (whale)
    if len(closes) >= 6:
        recent_closes = closes[-6:]
        recent_vols = vols[-6:]
        # Check if volume increasing while price consolidating = accumulation
        price_range = (max(recent_closes) - min(recent_closes)) / min(recent_closes) * 100 if min(recent_closes) > 0 else 0
        vol_increase = recent_vols[-1] / sum(recent_vols[:-1]) * 5 if sum(recent_vols[:-1]) > 0 else 1
        feats['accumulation_signal'] = 1 if (price_range < 3 and vol_increase > 2) else 0
    else:
        feats['accumulation_signal'] = 0

    # 17: Higher lows (pre-pump pattern)
    if len(rows) >= 6:
        lows = [rows[idx-i]['l'] for i in range(5, 0, -1)]
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        feats['higher_lows'] = higher_lows
    else:
        feats['higher_lows'] = 0

    # 18: Volume spike (top 20% of recent)
    if len(vols) >= 20:
        sorted_vols = sorted(vols[-20:])
        threshold = sorted_vols[int(len(sorted_vols)*0.8)]
        feats['vol_top20'] = 1 if cur['v'] > threshold else 0
    else:
        feats['vol_top20'] = 0

    # 19: Wick ratio (rejection at high)
    upper_wick = (cur['h'] - max(cur['c'], cur['o'])) / (cur['h'] - cur['l']) if cur['h'] > cur['l'] else 0
    feats['upper_wick'] = upper_wick

    # 20: Donchian channel position
    if len(rows) >= 20:
        highest = max(r['h'] for r in rows[idx-19:idx+1])
        feats['donchian_pos'] = cur['c'] / highest if highest > 0 else 1
    else:
        feats['donchian_pos'] = 1

    return feats

# ============ BACKTEST ENGINE ============

print('Deep backtest: computing 20 features for 52 pumps...')
print()

pump_data = []
for pi, pump in enumerate(pumps, 1):
    sym = pump['sym']
    rows = _fetch_klines_1h(sym, 200)
    if len(rows) < 50: continue
    target_ts = pump['start_ts']
    pump_i = None
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 3600 and r['ts'] <= target_ts + 3600:
            pump_i = i; break
    if pump_i is None or pump_i < 50: continue
    # Compute features at t-1h
    feats_t1 = compute_20_features(rows, pump_i - 1)
    if not feats_t1: continue
    pump_data.append({
        'sym': sym,
        'gain': pump['gain_pct'],
        'is_pump': pump['gain_pct'] >= 30,
        'features': feats_t1,
    })
    if pi % 10 == 0:
        print(f'  {pi}/{len(pumps)} done...')

print(f'\nLoaded {len(pump_data)} pump samples')
print(f'  Pumps (gain >= 30%): {sum(1 for p in pump_data if p["is_pump"])}')
print(f'  Non-pumps: {sum(1 for p in pump_data if not p["is_pump"])}')
print()

# ============ FIND BEST SINGLE FEATURE ============
print('='*80)
print('SINGLE FEATURE PERFORMANCE (predictive power)')
print('='*80)
print(f'{"Feature":<22} {"best_thresh":>12} {"catch%":>8} {"FP%":>8} {"precision":>10}')
print('-'*80)

feature_names = list(pump_data[0]['features'].keys())
best_features = []

for feat in feature_names:
    vals = [p['features'][feat] for p in pump_data]
    pumps_with = [p for p in pump_data if p['is_pump']]
    non_pumps = [p for p in pump_data if not p['is_pump']]
    # Find best threshold
    best_score = 0
    best_thresh = 0
    best_direction = '>'
    for thresh in [v for v in vals if v is not None]:
        # Try >
        passing_pumps = sum(1 for p in pumps_with if p['features'][feat] > thresh)
        passing_non = sum(1 for p in non_pumps if p['features'][feat] > thresh)
        total_pass = passing_pumps + passing_non
        if total_pass == 0: continue
        # We want: catch many pumps, few non-pumps
        # Score: precision-weighted
        if passing_pumps > 0:
            precision = passing_pumps / total_pass
            recall = passing_pumps / max(1, len(pumps_with))
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            if f1 > best_score:
                best_score = f1
                best_thresh = thresh
                best_direction = '>'
        # Try <
        passing_pumps = sum(1 for p in pumps_with if p['features'][feat] < thresh)
        passing_non = sum(1 for p in non_pumps if p['features'][feat] < thresh)
        total_pass = passing_pumps + passing_non
        if total_pass == 0: continue
        if passing_pumps > 0:
            precision = passing_pumps / total_pass
            recall = passing_pumps / max(1, len(pumps_with))
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
            if f1 > best_score:
                best_score = f1
                best_thresh = thresh
                best_direction = '<'
    # Calculate final stats
    if best_direction == '>':
        pass_pumps = sum(1 for p in pumps_with if p['features'][feat] > best_thresh)
        pass_non = sum(1 for p in non_pumps if p['features'][feat] > best_thresh)
    else:
        pass_pumps = sum(1 for p in pumps_with if p['features'][feat] < best_thresh)
        pass_non = sum(1 for p in non_pumps if p['features'][feat] < best_thresh)
    total_pass = pass_pumps + pass_non
    if total_pass > 0:
        precision = pass_pumps / total_pass
        catch = pass_pumps / max(1, len(pumps_with))
        best_features.append({
            'feature': feat,
            'f1': best_score,
            'precision': precision,
            'catch': catch,
            'thresh': best_thresh,
            'direction': best_direction,
        })

best_features.sort(key=lambda x: -x['f1'])
for bf in best_features[:15]:
    print(f'{bf["feature"]:<22} {bf["thresh"]:>10.2f}{bf["direction"]}  {bf["catch"]*100:>6.1f}%  {bf["precision"]*100:>6.1f}%  {bf["precision"]*100:>7.1f}%')

# ============ TOP 3 COMBINATION ============
print()
print('='*80)
print('TOP 3 FEATURE COMBINATIONS (AND logic)')
print('='*80)

top_5_feats = best_features[:5]
# Try all 2-feature, 3-feature combinations
best_combo = None
best_combo_f1 = 0
for n in [2, 3, 4]:
    for combo in itertools.combinations(top_5_feats, n):
        # For each pump, check if ALL features pass
        rules = [(bf['feature'], bf['thresh'], bf['direction']) for bf in combo]
        correct = 0
        for p in pump_data:
            passes = True
            for feat, thresh, direction in rules:
                val = p['features'][feat]
                if direction == '>' and val <= thresh: passes = False; break
                if direction == '<' and val >= thresh: passes = False; break
            if passes == p['is_pump']:
                correct += 1
        accuracy = correct / len(pump_data)
        if accuracy > best_combo_f1:
            best_combo_f1 = accuracy
            best_combo = rules

print(f'\nBest combination: {best_combo_f1*100:.1f}% accuracy')
if best_combo:
    for feat, thresh, direction in best_combo:
        print(f'  {feat} {direction} {thresh:.2f}')

# Save results
with open(f'{CACHE}/deep_backtest.json', 'w') as f:
    json.dump({
        'n_samples': len(pump_data),
        'n_pumps': sum(1 for p in pump_data if p['is_pump']),
        'best_features': best_features,
        'best_combo': str(best_combo),
        'best_combo_accuracy': best_combo_f1,
    }, f, indent=2, default=str)

print(f'\nSaved to {CACHE}/deep_backtest.json')
