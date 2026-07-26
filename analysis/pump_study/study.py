"""
Statistical analysis of pre-pump features across 49 pumps.
Compare distributions: pre-pump (sample) vs random non-pump periods.
Identify the strongest predictive features.
"""
import json, os, math, statistics
from datetime import datetime, timezone
from collections import defaultdict

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/pump_features.json') as f:
    pumps = json.load(f)
with open(f'{CACHE}/rows_index.json') as f:
    rows_idx = json.load(f)

print(f'Loaded {len(pumps)} pump samples\n')

# All feature keys (excluding metadata)
META = {'sym', 'gain_pct', 'mc', 'qv_24h', 'start_time', 'tf_step_h'}
all_keys = set()
for p in pumps:
    all_keys.update(p.keys())
all_keys -= META

# === CONTROL: Random non-pump samples ===
# Pick a random non-pump snapshot from each symbol (mid-period)
print('Building control set (non-pump periods)...')
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
    std = statistics.pstdev(closes) if len(closes) > 1 else 0
    up = ma + k*std; lo = ma - k*std
    return (rows[-1]['c']-lo)/(up-lo) if (up-lo) > 0 else 0.5

def compute_features_at(rows, idx, step_h=1.0):
    if idx < 50: return None
    window = rows[max(0, idx-50):idx+1]
    cur = rows[idx]
    closes = [r['c'] for r in window]
    vols = [r['v'] for r in window]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
    rvol = cur['v']/avg_v if avg_v > 0 else 1.0
    recent_rvols = [vols[i]/avg_v for i in range(max(0,len(vols)-4), len(vols))] if avg_v > 0 else [1.0]
    max_rvol_4h = max(recent_rvols) if recent_rvols else 1.0
    if idx >= 1:
        mom_1 = (cur['c']-rows[idx-1]['c'])/rows[idx-1]['c']*100
    else: mom_1 = 0
    if idx >= 3:
        mom_3 = (cur['c']-rows[idx-3]['c'])/rows[idx-3]['c']*100
    else: mom_3 = 0
    if idx >= 6:
        mom_6 = (cur['c']-rows[idx-6]['c'])/rows[idx-6]['c']*100
    else: mom_6 = 0
    if idx >= 12:
        mom_12 = (cur['c']-rows[idx-12]['c'])/rows[idx-12]['c']*100
    else: mom_12 = 0
    body = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0
    out = {
        't1h_mom_1': mom_1, 't1h_mom_3': mom_3, 't1h_mom_6': mom_6, 't1h_mom_12': mom_12,
        't1h_rsi': rsi(closes), 't1h_atr_pct': atr_pct(window),
        't1h_rvol': rvol, 't1h_max_rvol_4h': max_rvol_4h,
        't1h_bb_pos': bollinger_pos(window), 't1h_body': body,
    }
    return out

# For each pump, mark pump_i, then for each non-pump symbol, sample 5 random points far from pumps
pump_indices = defaultdict(list)
with open(f'{CACHE}/all_pumps.json') as f:
    raw_pumps = json.load(f)
for p in raw_pumps:
    target_ts = p.get('start_ts')
    if not target_ts: continue
    sym = p.get('sym', '')
    rows = rows_idx.get(sym, [])
    if not rows: continue
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 3600 and r['ts'] <= target_ts + 3600:
            pump_indices[sym].append(i)
            break
# Also AKE
ake_pump = {'sym': 'AKEUSDT', 'start_ts': 1784152800}
for i, r in enumerate(rows_idx.get('AKEUSDT', [])):
    if r['ts'] >= 1784152800 - 900 and r['ts'] <= 1784152800 + 900:
        pump_indices['AKEUSDT'].append(i)
        break

# Build control: for each symbol, sample 5 random indices NOT within 24h of any pump
import random
random.seed(42)
control = []
for sym, rows in rows_idx.items():
    pump_i_set = set(pump_indices.get(sym, []))
    # Also exclude ±24h around each pump
    exclude = set()
    for pi in pump_i_set:
        for off in range(-24, 25):
            exclude.add(pi + off)
    valid = [i for i in range(50, len(rows)-1) if i not in exclude]
    random.shuffle(valid)
    for idx in valid[:5]:
        step_h = 0.25 if sym == 'AKEUSDT' else 1.0
        feats = compute_features_at(rows, idx, step_h)
        if feats:
            feats['sym'] = sym
            control.append(feats)

print(f'Control samples: {len(control)}')

# For comparison, take same time-points from PUMPS
# Aggregate pump samples at each timepoint
pump_t1h = []
pump_t4h = []
pump_t12h = []
pump_t24h = []
for p in pumps:
    if 't1h_rvol' in p: pump_t1h.append({k.replace('t1h_',''): v for k,v in p.items() if k.startswith('t1h_')})
    if 't4h_rvol' in p: pump_t4h.append({k.replace('t4h_',''): v for k,v in p.items() if k.startswith('t4h_')})
    if 't12h_rvol' in p: pump_t12h.append({k.replace('t12h_',''): v for k,v in p.items() if k.startswith('t12h_')})
    if 't24h_rvol' in p: pump_t24h.append({k.replace('t24h_',''): v for k,v in p.items() if k.startswith('t24h_')})

print(f'Pump samples - t1h: {len(pump_t1h)}, t4h: {len(pump_t4h)}, t12h: {len(pump_t12h)}, t24h: {len(pump_t24h)}')

# For control, take all features
control_t1h = [{k.replace('t1h_',''): v for k,v in c.items() if k.startswith('t1h_')} for c in control]
print(f'Control t1h: {len(control_t1h)}')

def stats_dict(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not vals: return None
    return {
        'n': len(vals),
        'mean': sum(vals)/len(vals),
        'median': sorted(vals)[len(vals)//2],
        'p25': sorted(vals)[len(vals)//4],
        'p75': sorted(vals)[3*len(vals)//4],
        'min': min(vals), 'max': max(vals),
    }

# ====== COMPARISON TABLE ======
print(f'\n{"="*120}')
print('FEATURE ANALYSIS: PRE-PUMP (1h before) vs CONTROL')
print(f'{"="*120}\n')
print(f'{"Feature":<18} | {"PUMP mean":>10} {"PUMP med":>10} {"PUMP p25":>10} {"PUMP p75":>10} | {"CTRL mean":>10} {"CTRL med":>10} {"CTRL p25":>10} {"CTRL p75":>10} | {"PUMP>CTRL?":>10}')
print('-'*120)

features = ['mom_1','mom_3','mom_6','mom_12','rsi','atr_pct','rvol','max_rvol_4h','bb_pos','body']
for f in features:
    pv = [d.get(f) for d in pump_t1h if f in d]
    cv = [d.get(f) for d in control_t1h if f in d]
    if not pv or not cv: continue
    ps = stats_dict(pv); cs = stats_dict(cv)
    if not ps or not cs: continue
    diff = 'YES' if ps['mean'] > cs['mean'] * 1.2 or ps['mean'] < cs['mean'] * 0.8 else 'no'
    print(f'{f:<18} | {ps["mean"]:>10.3f} {ps["median"]:>10.3f} {ps["p25"]:>10.3f} {ps["p75"]:>10.3f} | {cs["mean"]:>10.3f} {cs["median"]:>10.3f} {cs["p25"]:>10.3f} {cs["p75"]:>10.3f} | {diff:>10}')

# Pre-pump flat hours and volume accel
print(f'\n{"="*120}')
print('SPECIAL FEATURES (computed over 7-day lookback)')
print(f'{"="*120}\n')
flat_vals = [p.get('pre_pump_flat_hours', 0) for p in pumps]
flat_vals = [v for v in flat_vals if v is not None]
if flat_vals:
    print(f'pre_pump_flat_hours (PUMP):  n={len(flat_vals)}, mean={sum(flat_vals)/len(flat_vals):.1f}h, median={sorted(flat_vals)[len(flat_vals)//2]:.1f}h, max={max(flat_vals):.0f}h')

vol_accel = [p.get('vol_accel_24h', 0) for p in pumps]
if vol_accel:
    print(f'vol_accel_24h (PUMP):        n={len(vol_accel)}, mean={sum(vol_accel)/len(vol_accel):.1f}%, median={sorted(vol_accel)[len(vol_accel)//2]:.1f}%')

# Time-before-pump analysis
print(f'\n{"="*120}')
print('EVOLUTION: HOW FEATURES EVOLVE 24h, 12h, 4h, 1h BEFORE PUMP')
print(f'{"="*120}\n')
print(f'{"Feature":<14} | {"t24h mean":>11} {"t12h":>11} {"t4h":>11} {"t1h":>11} | {"trend":>20}')
print('-'*100)
trends = {}
for f in features:
    v24 = sum(d[f] for d in pump_t24h if f in d) / max(1, sum(1 for d in pump_t24h if f in d))
    v12 = sum(d[f] for d in pump_t12h if f in d) / max(1, sum(1 for d in pump_t12h if f in d))
    v4 = sum(d[f] for d in pump_t4h if f in d) / max(1, sum(1 for d in pump_t4h if f in d))
    v1 = sum(d[f] for d in pump_t1h if f in d) / max(1, sum(1 for d in pump_t1h if f in d))
    # trend
    if v24 == 0: trend = '-'
    else:
        change = (v1 - v24) / max(abs(v24), 0.01) * 100
        trend = f'{change:+.0f}% change'
    trends[f] = (v24, v12, v4, v1)
    print(f'{f:<14} | {v24:>11.3f} {v12:>11.3f} {v4:>11.3f} {v1:>11.3f} | {trend:>20}')

# === Predictive score: how often does feature X distinguish pump from control ===
print(f'\n{"="*120}')
print('PREDICTIVE POWER: % of pumps that would be caught by each rule')
print(f'{"="*120}\n')
# For each rule, find threshold that maximizes (catch_rate - false_positive_rate)
def evaluate(rule_check, name):
    pump_pass = sum(1 for p in pump_t1h if rule_check(p))
    ctrl_pass = sum(1 for c in control_t1h if rule_check(c))
    n_p = len(pump_t1h); n_c = len(control_t1h)
    if n_p == 0 or n_c == 0: return
    catch_rate = pump_pass/n_p
    fp_rate = ctrl_pass/n_c
    precision = pump_pass/max(1, pump_pass+ctrl_pass)
    print(f'  {name:<55} catch={pump_pass}/{n_p} ({catch_rate*100:>5.1f}%)  fp={ctrl_pass}/{n_c} ({fp_rate*100:>5.1f}%)  precision={precision*100:>5.1f}%')

# Try various thresholds
print('\n--- T=1h before pump ---')
evaluate(lambda p: p.get('rvol', 1) > 1.5, 'rvol > 1.5x')
evaluate(lambda p: p.get('rvol', 1) > 2.0, 'rvol > 2.0x')
evaluate(lambda p: p.get('rvol', 1) > 3.0, 'rvol > 3.0x')
evaluate(lambda p: p.get('max_rvol_4h', 1) > 2.0, 'max_rvol_4h > 2x')
evaluate(lambda p: p.get('max_rvol_4h', 1) > 3.0, 'max_rvol_4h > 3x')
evaluate(lambda p: p.get('mom_3', 0) > 0, 'mom_3 > 0')
evaluate(lambda p: p.get('mom_3', 0) > 1.0, 'mom_3 > 1%')
evaluate(lambda p: p.get('mom_3', 0) > 2.0, 'mom_3 > 2%')
evaluate(lambda p: p.get('body', 0) > 0.5, 'body > 0.5')
evaluate(lambda p: p.get('body', 0) > 0.7, 'body > 0.7')
evaluate(lambda p: 25 < p.get('rsi', 50) < 75, 'RSI 25-75')
evaluate(lambda p: p.get('rsi', 50) > 60, 'RSI > 60')

# Combined rules
print('\n--- COMBINED RULES (1h before) ---')
evaluate(lambda p: p.get('rvol', 1) > 1.5 and p.get('mom_3', 0) > 0.5, 'rvol>1.5 AND mom_3>0.5%')
evaluate(lambda p: p.get('rvol', 1) > 1.5 and p.get('body', 0) > 0.5, 'rvol>1.5 AND body>0.5')
evaluate(lambda p: p.get('rvol', 1) > 2.0 and p.get('mom_3', 0) > 1.0, 'rvol>2.0 AND mom_3>1%')
evaluate(lambda p: p.get('max_rvol_4h', 1) > 2 and p.get('mom_6', 0) > 0, 'max_rvol>2 AND mom_6>0')

# 4h before (earlier warning)
print('\n--- T=4h before pump ---')
evaluate4 = lambda rule: (
    sum(1 for p in pump_t4h if rule(p)),
    sum(1 for c in control_t1h if rule(c)),
    len(pump_t4h), len(control_t1h)
)
for name, rule in [
    ('rvol > 1.5 (4h)', lambda p: p.get('rvol', 1) > 1.5),
    ('rvol > 2.0 (4h)', lambda p: p.get('rvol', 1) > 2.0),
    ('rvol > 3.0 (4h)', lambda p: p.get('rvol', 1) > 3.0),
    ('mom_3 > 0 (4h)', lambda p: p.get('mom_3', 0) > 0),
    ('mom_3 > 1 (4h)', lambda p: p.get('mom_3', 0) > 1.0),
    ('max_rvol_4h > 2 (4h)', lambda p: p.get('max_rvol_4h', 1) > 2),
]:
    p_pass, c_pass, n_p, n_c = evaluate4(rule)
    cr = p_pass/n_p*100 if n_p else 0
    fpr = c_pass/n_c*100 if n_c else 0
    prec = p_pass/max(1,p_pass+c_pass)*100
    print(f'  {name:<40} catch={p_pass}/{n_p} ({cr:>5.1f}%)  fp={c_pass}/{n_c} ({fpr:>5.1f}%)  precision={prec:>5.1f}%')

# Save
out = {
    'pump_count': len(pumps),
    'control_count': len(control),
    'features_t1h_stats': {f: stats_dict([d.get(f) for d in pump_t1h if f in d]) for f in features},
    'features_control_t1h_stats': {f: stats_dict([d.get(f) for d in control_t1h if f in d]) for f in features},
    'trends': trends,
    'flat_hours_mean': sum(flat_vals)/len(flat_vals) if flat_vals else None,
    'vol_accel_mean': sum(vol_accel)/len(vol_accel) if vol_accel else None,
}
with open(f'{CACHE}/stats.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nStats saved to {CACHE}/stats.json')
