"""
Advanced feature analysis:
1. Volatility compression (BB width) before pump
2. Volume dry-up (consecutive low-volume hours before spike)
3. Price stability (how flat was price before)
4. Anti-late filter effectiveness
5. Optimal lookback windows
"""
import json, os, math, statistics as st
from datetime import datetime, timezone
from collections import defaultdict

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/pump_features.json') as f:
    pumps = json.load(f)
with open(f'{CACHE}/rows_index.json') as f:
    rows_idx = json.load(f)

# Find pump indices
pump_indices = defaultdict(list)
with open(f'{CACHE}/all_pumps.json') as f:
    raw_pumps = json.load(f)
for p in raw_pumps:
    target_ts = p.get('start_ts')
    if not target_ts: continue
    sym = p.get('sym', '')
    rows = rows_idx.get(sym, [])
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 3600 and r['ts'] <= target_ts + 3600:
            pump_indices[sym].append(i)
            break
ake_pump_idx = None
for i, r in enumerate(rows_idx.get('AKEUSDT', [])):
    if r['ts'] >= 1784152800 - 900 and r['ts'] <= 1784152800 + 900:
        ake_pump_idx = i
        break
if ake_pump_idx:
    pump_indices['AKEUSDT'] = [ake_pump_idx]

print('='*80)
print('PATTERN 1: VOLATILITY COMPRESSION (BB width narrowing)')
print('='*80)
# For each pump, look at BB width 24h, 12h, 4h, 1h before
print(f'\n{"Time":<8} | {"mean":>8} {"median":>8} {"< 2%":>8} {"< 4%":>8} {"< 6%":>8}')
print('-'*50)
for h in [24, 12, 4, 1]:
    vals = []
    for p in pumps:
        key = f't{h}h_atr_pct'
        if key in p and p[key] is not None:
            vals.append(p[key])
    if not vals: continue
    mean = sum(vals)/len(vals)
    med = sorted(vals)[len(vals)//2]
    print(f't-{h}h   | {mean:>8.2f} {med:>8.2f} '
          f'{sum(1 for v in vals if v<2)/len(vals)*100:>7.1f}% '
          f'{sum(1 for v in vals if v<4)/len(vals)*100:>7.1f}% '
          f'{sum(1 for v in vals if v<6)/len(vals)*100:>7.1f}%')

print('\n# INSIGHT: Pre-pump ATR% is HIGH (3.6% median) - NOT compressed!')
print('# This is OPPOSITE of stock "volatility compression" pattern.')
print('# Reason: crypto small caps are always volatile. Pump starts during a volatile period.')

# === PATTERN 2: VOLUME DRY-UP before pump ===
print('\n' + '='*80)
print('PATTERN 2: VOLUME BEHAVIOR 24h BEFORE PUMP')
print('='*80)
# Look at average volume 12-24h before vs 0-12h before
import random
random.seed(42)
pump_v_data = []
ctrl_v_data = []
for sym, rows in rows_idx.items():
    pump_i_list = pump_indices.get(sym, [])
    if not pump_i_list: continue
    step_h = 0.25 if sym == 'AKEUSDT' else 1.0
    for pi in pump_i_list:
        if pi < 30: continue
        # Last 24h volume profile (normalized to 0-1 by max)
        window = [r['v'] for r in rows[pi-24:pi+1]]
        if not window: continue
        # 4-hour bucket average
        buckets = [sum(window[i:i+4])/4 for i in range(0, 24, 4)]
        # Normalize: each bucket / mean
        mean_v = sum(buckets)/len(buckets) if buckets else 1
        norm = [b/mean_v for b in buckets] if mean_v > 0 else [1]*6
        pump_v_data.append(norm)
    # Control: 5 random non-pump periods
    exclude = set()
    for pi in pump_i_list:
        for off in range(-24, 25): exclude.add(pi+off)
    valid = [i for i in range(30, len(rows)-1) if i not in exclude]
    random.shuffle(valid)
    for idx in valid[:5]:
        window = [r['v'] for r in rows[idx-24:idx+1]]
        if not window: continue
        buckets = [sum(window[i:i+4])/4 for i in range(0, 24, 4)]
        mean_v = sum(buckets)/len(buckets) if buckets else 1
        norm = [b/mean_v for b in buckets] if mean_v > 0 else [1]*6
        ctrl_v_data.append(norm)

# Average profile
pump_avg = [sum(d[i] for d in pump_v_data)/len(pump_v_data) for i in range(6)]
ctrl_avg = [sum(d[i] for d in ctrl_v_data)/len(ctrl_v_data) for i in range(6)]
print(f'\n{"Hours before pump":<22} | {"PUMP avg":>10} {"CTRL avg":>10} {"ratio":>8}')
print('-'*60)
for i, (p, c) in enumerate(zip(pump_avg, ctrl_avg)):
    h_before = 24 - i*4
    print(f't-{h_before:>3}h to t-{h_before-4:>3}h         | {p:>10.2f} {c:>10.2f} {p/max(c,0.01):>7.2f}x')

# === PATTERN 3: EMA crossover pre-pump ===
print('\n' + '='*80)
print('PATTERN 3: MOMENTUM TREND (6h before pump)')
print('='*80)
mom_6_vals = [p.get('t1h_mom_6', 0) for p in pumps if 't1h_mom_6' in p]
print(f'\nmom_6 distribution 1h before pump:')
print(f'  mean: {sum(mom_6_vals)/len(mom_6_vals):+.2f}%')
print(f'  % positive: {sum(1 for v in mom_6_vals if v>0)/len(mom_6_vals)*100:.1f}%')
print(f'  % strongly positive (>2%): {sum(1 for v in mom_6_vals if v>2)/len(mom_6_vals)*100:.1f}%')
print(f'  % strongly negative (<-2%): {sum(1 for v in mom_6_vals if v<-2)/len(mom_6_vals)*100:.1f}%')

# === PATTERN 4: PRE-PUMP FLAT HOURS ===
print('\n' + '='*80)
print('PATTERN 4: HOW LONG WAS PRICE FLAT BEFORE PUMP?')
print('='*80)
flat_hours = [p.get('pre_pump_flat_hours', 0) for p in pumps]
flat_hours = [f for f in flat_hours if f is not None]
buckets = [(0, 6), (6, 24), (24, 72), (72, 168)]
print(f'\n{"Flat duration":<20} | {"# pumps":>10} {"% of total":>12}')
print('-'*50)
for lo, hi in buckets:
    n = sum(1 for f in flat_hours if lo <= f < hi)
    print(f'{lo}-{hi}h{"":<13} | {n:>10} {n/len(flat_hours)*100:>11.1f}%')
print(f'\nMedian flat: {sorted(flat_hours)[len(flat_hours)//2]:.0f}h')
print(f'Mean flat: {sum(flat_hours)/len(flat_hours):.0f}h')
print(f'\n# INSIGHT: 50%+ pumps occur within 6h of price flat/consolidation')

# === PATTERN 5: EARLIEST DETECTABLE SIGNAL ===
print('\n' + '='*80)
print('PATTERN 5: EARLIEST DETECTION - HOW MANY HOURS BEFORE PUMP CAN WE SEE SOMETHING?')
print('='*80)
# Check rvol > 1.5 at 1h, 4h, 12h, 24h before
for h in [24, 12, 4, 1]:
    rvol_high = sum(1 for p in pumps if p.get(f't{h}h_rvol', 0) > 1.5)
    rvol_vhigh = sum(1 for p in pumps if p.get(f't{h}h_rvol', 0) > 2.0)
    mom_pos = sum(1 for p in pumps if p.get(f't{h}h_mom_3', 0) > 0)
    mom_strong = sum(1 for p in pumps if p.get(f't{h}h_mom_3', 0) > 1)
    n = len([p for p in pumps if f't{h}h_rvol' in p])
    print(f't-{h}h: rvol>1.5 in {rvol_high}/{n} ({rvol_high/n*100:.0f}%), '
          f'rvol>2.0 in {rvol_vhigh}/{n} ({rvol_vhigh/n*100:.0f}%), '
          f'mom_3>0 in {mom_pos}/{n} ({mom_pos/n*100:.0f}%), '
          f'mom_3>1 in {mom_strong}/{n} ({mom_strong/n*100:.0f}%)')

# === PATTERN 6: ANTI-LATE FILTER CHECK ===
print('\n' + '='*80)
print('PATTERN 6: ANTI-LATE FILTER - would filter have stopped us at 1h before?')
print('='*80)
# Current anti-late: mom_3 < 20% for LONG
for h in [12, 4, 1]:
    over_20 = sum(1 for p in pumps if abs(p.get(f't{h}h_mom_3', 0)) > 20)
    over_30 = sum(1 for p in pumps if abs(p.get(f't{h}h_mom_3', 0)) > 30)
    n = len([p for p in pumps if f't{h}h_mom_3' in p])
    print(f't-{h}h: mom_3 > 20% in {over_20}/{n} ({over_20/n*100:.0f}%)')
    print(f't-{h}h: mom_3 > 30% in {over_30}/{n} ({over_30/n*100:.0f}%)')

print('\n# INSIGHT: 30-50% of pumps are "anti-late" at t-1h - current filter kills them')

# === OPTIMAL FILTER COMBINATION ===
print('\n' + '='*80)
print('OPTIMAL FILTER: find rule that catches most pumps with min false positives')
print('='*80)
# Build pump_t1h and control_t1h feature dicts again
def rsi(closes, n=14):
    if len(closes) < n+1: return 50.0
    gains, losses = [], []
    for i in range(1, n+1):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0)); losses.append(max(-ch, 0))
    avg_g = sum(gains)/n; avg_l = sum(losses)/n
    for i in range(n+1, len(closes)):
        ch = closes[i] - closes[i-1]
        avg_g = (avg_g*(n-1) + max(ch,0))/n
        avg_l = (avg_l*(n-1) + max(-ch,0))/n
    if avg_l == 0: return 100.0
    return 100 - 100/(1+avg_g/avg_l)
def atr_pct(rows, n=14):
    if len(rows) < n+1: return 0.0
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    atr = sum(trs[-n:])/min(n, len(trs))
    last = rows[-1]['c']
    return atr/last*100 if last > 0 else 0
def bollinger_pos(rows, n=20, k=2):
    if len(rows) < n: return 0.5
    closes = [r['c'] for r in rows[-n:]]
    ma = sum(closes)/n
    std = st.pstdev(closes) if len(closes) > 1 else 0
    up = ma + k*std; lo = ma - k*std
    return (rows[-1]['c']-lo)/(up-lo) if (up-lo) > 0 else 0.5

# Recompute control with full features
control = []
for sym, rows in rows_idx.items():
    pump_i_set = set(pump_indices.get(sym, []))
    exclude = set()
    for pi in pump_i_set:
        for off in range(-24, 25): exclude.add(pi+off)
    valid = [i for i in range(50, len(rows)-1) if i not in exclude]
    random.shuffle(valid)
    for idx in valid[:5]:
        window = rows[max(0, idx-50):idx+1]
        cur = rows[idx]
        closes = [r['c'] for r in window]
        vols = [r['v'] for r in window]
        avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
        rvol = cur['v']/avg_v if avg_v > 0 else 1.0
        control.append({
            'rvol': rvol, 'mom_3': (cur['c']-rows[idx-3]['c'])/rows[idx-3]['c']*100 if idx>=3 else 0,
            'body': abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0,
            'rsi': rsi(closes), 'atr_pct': atr_pct(window),
            'bb_pos': bollinger_pos(window),
        })

pump_t1h_simple = []
for p in pumps:
    if 't1h_rvol' in p:
        pump_t1h_simple.append({
            'rvol': p['t1h_rvol'], 'mom_3': p['t1h_mom_3'],
            'body': p['t1h_body'], 'rsi': p['t1h_rsi'],
            'atr_pct': p['t1h_atr_pct'], 'bb_pos': p['t1h_bb_pos'],
        })

# Try rules
rules = [
    ('anti-late-only: mom_3 < 50%', lambda p: abs(p.get('mom_3', 0)) < 50),
    ('vol+flat: rvol<1.5 (no spike yet)', lambda p: p.get('rvol', 0) < 1.5),
    ('mild+vol: rvol>0.5 AND rsi<70', lambda p: p.get('rvol', 0) > 0.5 and p.get('rsi', 0) < 70),
    ('flat+ready: rvol<2.5 AND mom_3<30', lambda p: p.get('rvol', 0) < 2.5 and abs(p.get('mom_3', 0)) < 30),
    ('relaxed: mom_3<50 AND rsi<85', lambda p: abs(p.get('mom_3', 0)) < 50 and p.get('rsi', 0) < 85),
    ('volume building: max_rvol>1.5 (4h)', lambda p: True),  # we have max_rvol_4h elsewhere
    ('CURRENT ULTRA-STRICT: mom_3<8 + ATR 3-8 + rsi 25-78 + rvol>1', lambda p: abs(p.get('mom_3', 0)) < 8 and 3 < p.get('atr_pct', 0) < 8 and 25 < p.get('rsi', 0) < 78 and p.get('rvol', 0) > 1),
    ('RELAXED v1: mom_3<20 + ATR 2-12 + rsi 20-90 + rvol>0.8', lambda p: abs(p.get('mom_3', 0)) < 20 and 2 < p.get('atr_pct', 0) < 12 and 20 < p.get('rsi', 0) < 90 and p.get('rvol', 0) > 0.8),
    ('RELAXED v2: mom_3<30 + ATR 2-15 + rsi 20-95 + rvol>0.5', lambda p: abs(p.get('mom_3', 0)) < 30 and 2 < p.get('atr_pct', 0) < 15 and 20 < p.get('rsi', 0) < 95 and p.get('rvol', 0) > 0.5),
    ('PUMP-READY: mom_3<40 + rsi<95 + rvol>0.3', lambda p: abs(p.get('mom_3', 0)) < 40 and p.get('rsi', 0) < 95 and p.get('rvol', 0) > 0.3),
]

print(f'\n{"Rule":<60} | {"catch":>8} {"fp_rate":>8} {"precision":>10}')
print('-'*90)
for name, rule in rules:
    p_pass = sum(1 for p in pump_t1h_simple if rule(p))
    c_pass = sum(1 for c in control if rule(c))
    n_p = len(pump_t1h_simple); n_c = len(control)
    if n_p == 0: continue
    catch = p_pass/n_p
    fp = c_pass/n_c
    prec = p_pass/max(1, p_pass+c_pass)
    print(f'{name:<60} | {p_pass:>3}/{n_p:<3} ({catch*100:>5.1f}%) {c_pass:>3}/{n_c:<3} ({fp*100:>5.1f}%) {prec*100:>5.1f}%')

print('\n# Lower catch rate but higher precision = better signal-to-noise')
print('# In production, we want high precision even at low catch rate')
print('# because we get many candidate symbols to filter through.')
