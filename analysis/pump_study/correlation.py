"""
Find what is COMMON across all 49 pumps:
1. Time clustering - do pumps happen together? (market-wide event)
2. BTC correlation - did BTC pump first?
3. Per-symbol pattern - do same symbols pump repeatedly?
4. Pre-pump signature - what's the COMMON sequence?
"""
import json, os
from datetime import datetime, timezone
from collections import Counter, defaultdict

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/pump_features.json') as f:
    pumps = json.load(f)
with open(f'{CACHE}/all_pumps.json') as f:
    raw_pumps = json.load(f)
with open(f'{CACHE}/rows_index.json') as f:
    rows_idx = json.load(f)

print('='*80)
print('TIME CLUSTERING: when did pumps happen?')
print('='*80)

# Group by date
dates = []
for p in raw_pumps:
    d = p['start_time'][:10]  # YYYY-MM-DD
    dates.append((d, p['sym'], p['gain_pct']))

dates.sort()
print(f'\n{"Date":<12} | {"#pumps":>7} | symbols')
print('-'*70)
date_count = Counter(d[0] for d in dates)
for d, n in sorted(date_count.items(), key=lambda x: -x[1]):
    syms = [s for date, s, _ in dates if date == d]
    print(f'{d:<12} | {n:>7} | {", ".join(syms[:8])}')

print(f'\nTotal unique dates: {len(date_count)}')
print(f'Days with multiple pumps: {sum(1 for n in date_count.values() if n>1)}')
print(f'Days with 1 pump: {sum(1 for n in date_count.values() if n==1)}')

# Check if pumps cluster in time (consecutive hours)
import statistics as st
hours = []
for p in raw_pumps:
    dt = datetime.fromisoformat(p['start_time'])
    hours.append(dt.replace(minute=0, second=0, microsecond=0))

hours.sort()
diffs_h = []
for i in range(1, len(hours)):
    diff = (hours[i] - hours[i-1]).total_seconds()/3600
    diffs_h.append(diff)

print(f'\nTime between consecutive pumps:')
print(f'  mean: {sum(diffs_h)/len(diffs_h):.1f}h')
print(f'  median: {sorted(diffs_h)[len(diffs_h)//2]:.1f}h')
print(f'  < 6h: {sum(1 for d in diffs_h if d<6)} pairs')
print(f'  6-24h: {sum(1 for d in diffs_h if 6<=d<24)} pairs')
print(f'  > 24h: {sum(1 for d in diffs_h if d>=24)} pairs')

# Symbol repetition
print(f'\n{"="*80}')
print('SYMBOL REPETITION: which symbols pumped multiple times?')
print('='*80)
sym_count = Counter(p['sym'] for p in raw_pumps)
print(f'\n{"Symbol":<14} | {"#pumps":>7} | avg gain')
print('-'*45)
for sym, n in sym_count.most_common(15):
    gains = [p['gain_pct'] for p in raw_pumps if p['sym'] == sym]
    print(f'{sym:<14} | {n:>7} | {sum(gains)/len(gains):>6.1f}%')

print(f'\nSymbols that pumped 2+ times: {sum(1 for n in sym_count.values() if n>=2)}')
print(f'Unique symbols that pumped: {len(sym_count)}')

# Check BTC at pump time
print(f'\n{"="*80}')
print('BTC CORRELATION: did BTC move before pump?')
print('='*80)

def http_get(url, timeout=15):
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except: return None

# Get BTC 1h klines for our window
print('Fetching BTC 1h klines...')
raw = http_get('https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair=BTC_USDT&interval=1h&limit=1000')
btc_rows = []
if raw:
    data = json.loads(raw)
    for r in data:
        ts = int(r[0])
        btc_rows.append({'ts': ts, 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]),
                         'c': float(r[2]), 'v': float(r[1])})
    btc_rows.sort(key=lambda x: x['ts'])
    print(f'  BTC: {len(btc_rows)} candles from {datetime.fromtimestamp(btc_rows[0]["ts"], tz=timezone.utc).isoformat()} to {datetime.fromtimestamp(btc_rows[-1]["ts"], tz=timezone.utc).isoformat()}')

def btc_at(ts, hours_before=0):
    """Get BTC return over the hours_before leading up to ts"""
    target = ts - hours_before*3600
    # Find closest candle
    closest = min(btc_rows, key=lambda r: abs(r['ts']-target))
    # And forward return
    end = min(btc_rows, key=lambda r: abs(r['ts']-ts))
    if hours_before > 0:
        return (end['c'] - closest['c'])/closest['c']*100
    else:
        return (end['c'] - btc_rows[0]['c'])/btc_rows[0]['c']*100  # just current return

# For each pump, check BTC return in 24h, 4h before
btc_pre = []
btc_24h_after = []
for p in raw_pumps:
    if not btc_rows: break
    ts = p['start_ts']
    pre_24h = btc_at(ts, 24)
    pre_4h = btc_at(ts, 4)
    fwd_4h = btc_at(ts, -4) if ts + 4*3600 < btc_rows[-1]['ts'] else 0
    fwd_24h = btc_at(ts, -24) if ts + 24*3600 < btc_rows[-1]['ts'] else 0
    btc_pre.append({'sym': p['sym'], 'gain': p['gain_pct'], 'pre_24h': pre_24h, 'pre_4h': pre_4h, 'fwd_4h': fwd_4h, 'fwd_24h': fwd_24h})

if btc_pre:
    print(f'\n{"Time":<14} | {"mean":>8} {"median":>8} {"%>0":>8} {"%>1%":>8}')
    print('-'*55)
    for label, key in [('BTC 24h before', 'pre_24h'), ('BTC 4h before', 'pre_4h'), ('BTC 4h after', 'fwd_4h'), ('BTC 24h after', 'fwd_24h')]:
        vals = [d[key] for d in btc_pre if d.get(key) is not None]
        if not vals: continue
        print(f'{label:<14} | {sum(vals)/len(vals):>7.2f}% {sorted(vals)[len(vals)//2]:>7.2f}% {sum(1 for v in vals if v>0)/len(vals)*100:>7.1f}% {sum(1 for v in vals if v>1)/len(vals)*100:>7.1f}%')

# Day of week / hour of day
print(f'\n{"="*80}')
print('TIME PATTERNS: when do pumps happen?')
print('='*80)
days = Counter()
hours_utc = Counter()
for p in raw_pumps:
    dt = datetime.fromisoformat(p['start_time'])
    days[dt.strftime('%A')] += 1
    hours_utc[dt.hour] += 1

print(f'\nDay of week:')
for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
    print(f'  {day:<10}: {days.get(day, 0):>3} pumps')

print(f'\nHour of day (UTC):')
for h in range(24):
    bar = '#' * hours_utc.get(h, 0)
    print(f'  {h:>2}:00  {hours_utc.get(h, 0):>3} {bar}')

# === COMMON PRE-PUMP SEQUENCE ===
print(f'\n{"="*80}')
print('COMMON PRE-PUMP SEQUENCE (averaged across 49 pumps)')
print('='*80)
print('\nThe "signature" that happens before every pump:')

# For each pump, show price change in last 24h, last 4h, last 1h
seq = []
for p in raw_pumps:
    sym = p['sym']
    rows = rows_idx.get(sym, [])
    if not rows: continue
    target = p['start_ts']
    pump_i = None
    for i, r in enumerate(rows):
        if r['ts'] >= target - 3600 and r['ts'] <= target + 3600:
            pump_i = i; break
    if pump_i is None or pump_i < 30: continue

    # Get prices at -24h, -12h, -4h, -1h, 0
    prices = {}
    for h in [24, 12, 4, 1, 0]:
        idx = max(0, pump_i - h)
        if idx < len(rows):
            prices[f't-{h}h'] = rows[idx]['c']
    # Volume at each point
    vols = {}
    for h in [24, 12, 4, 1, 0]:
        idx = max(0, pump_i - h)
        if idx >= 10 and idx < len(rows):
            v_window = [r['v'] for r in rows[max(0,idx-10):idx+1]]
            vols[f't-{h}h'] = sum(v_window)/len(v_window) if v_window else 0
    seq.append({'sym': sym, 'prices': prices, 'vols': vols, 'pump_i': pump_i, 'rows': rows})

# Average % change from t-24h to t-0
print(f'\n% price change relative to t-24h:')
print(f'{"Phase":<10} | {"mean":>8} {"median":>8} {"std":>8}')
print('-'*45)
for h in [24, 12, 4, 1, 0]:
    phase = f't-{h}h' if h>0 else 't-0'
    pct = []
    for s in seq:
        if f't-24h' in s['prices'] and phase in s['prices']:
            base = s['prices']['t-24h']
            if base > 0:
                pct.append((s['prices'][phase] - base)/base*100)
    if pct:
        m = sum(pct)/len(pct)
        med = sorted(pct)[len(pct)//2]
        sd = st.pstdev(pct) if len(pct)>1 else 0
        print(f'{phase:<10} | {m:>+7.2f}% {med:>+7.2f}% {sd:>7.2f}%')

# Volume profile (normalized)
print(f'\nNormalized volume (1.0 = average over 24h):')
print(f'{"Phase":<10} | {"mean":>8} {"median":>8}')
print('-'*40)
for h in [24, 12, 4, 1, 0]:
    phase = f't-{h}h' if h>0 else 't-0'
    norms = []
    for s in seq:
        if 't-24h' in s['vols'] and phase in s['vols']:
            base = s['vols']['t-24h']
            if base > 0:
                norms.append(s['vols'][phase]/base)
    if norms:
        m = sum(norms)/len(norms)
        med = sorted(norms)[len(norms)//2]
        print(f'{phase:<10} | {m:>7.2f}x {med:>7.2f}x')

# === KEY DISCOVERY: WHAT % OF PUMPS HAVE EACH FEATURE ===
print(f'\n{"="*80}')
print('THE 3-MINUTE RULES: What % of pumps satisfy X?')
print('='*80)
print('\nA pump is pre-detectable if it satisfies these in the last 4h:')

checks_4h = []
for p in pumps:
    sym = p['sym']
    if 't4h_rvol' in p:
        checks_4h.append({
            'rvol_below_2x': p.get('t4h_rvol', 99) < 2,
            'rvol_above_1': p.get('t4h_rvol', 0) > 1,
            'max_rvol_4h_above_1.5': p.get('t4h_max_rvol_4h', 0) > 1.5,
            'max_rvol_4h_above_2': p.get('t4h_max_rvol_4h', 0) > 2,
            'mom_3_positive': p.get('t4h_mom_3', -99) > 0,
            'mom_3_above_1': p.get('t4h_mom_3', -99) > 1,
            'mom_3_below_-2': p.get('t4h_mom_3', 99) < -2,
            'rsi_above_45': p.get('t4h_rsi', 0) > 45,
            'rsi_below_60': p.get('t4h_rsi', 99) < 60,
            'bb_pos_low': p.get('t4h_bb_pos', 99) < 0.4,
            'body_above_0.5': p.get('t4h_body', 0) > 0.5,
        })

n = len(checks_4h)
if n:
    print(f'\n{"Rule (4h before)":<35} | {"% of pumps":>12}')
    print('-'*50)
    for key in checks_4h[0].keys():
        c = sum(1 for d in checks_4h if d[key])
        print(f'{key:<35} | {c}/{n} ({c/n*100:>5.1f}%)')

# Best single feature for early detection
print(f'\n{"="*80}')
print('BEST EARLY-WARNING FEATURE (t-4h)')
print('='*80)
print('\nRecall@4h = how many pumps had this signal 4h before?')
print('Precision@4h = of all signals, how many were real pumps?')

# Need control set for 4h features - compute it
import urllib.request
def compute_at_idx(rows, idx, h):
    """Compute t-{h}h features at index idx (which IS the t-0 pump start)"""
    offset = h
    target = idx - offset
    if target < 30 or target >= len(rows): return None
    window = rows[max(0, target-50):target+1]
    cur = rows[target]
    closes = [r['c'] for r in window]
    vols = [r['v'] for r in window]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
    rvol = cur['v']/avg_v if avg_v > 0 else 1.0
    recent_rvols = [vols[i]/avg_v for i in range(max(0,len(vols)-4), len(vols))] if avg_v > 0 else [1.0]
    max_rvol_4h = max(recent_rvols) if recent_rvols else 1.0
    if target >= 1: mom_1 = (cur['c']-rows[target-1]['c'])/rows[target-1]['c']*100
    else: mom_1 = 0
    if target >= 3: mom_3 = (cur['c']-rows[target-3]['c'])/rows[target-3]['c']*100
    else: mom_3 = 0
    if target >= 6: mom_6 = (cur['c']-rows[target-6]['c'])/rows[target-6]['c']*100
    else: mom_6 = 0
    if target >= 12: mom_12 = (cur['c']-rows[target-12]['c'])/rows[target-12]['c']*100
    else: mom_12 = 0
    return {'rvol': rvol, 'max_rvol_4h': max_rvol_4h, 'mom_1': mom_1, 'mom_3': mom_3, 'mom_6': mom_6, 'mom_12': mom_12}

# Control at t-4h
import random
random.seed(42)
pump_indices_4h = defaultdict(list)
for p in raw_pumps:
    target_ts = p.get('start_ts')
    if not target_ts: continue
    sym = p.get('sym', '')
    rows = rows_idx.get(sym, [])
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 3600 and r['ts'] <= target_ts + 3600:
            pump_indices_4h[sym].append(i)
            break
ake_i = None
for i, r in enumerate(rows_idx.get('AKEUSDT', [])):
    if r['ts'] >= 1784152800 - 900 and r['ts'] <= 1784152800 + 900:
        ake_i = i; break
if ake_i: pump_indices_4h['AKEUSDT'] = [ake_i]

control_4h = []
for sym, rows in rows_idx.items():
    pump_i_set = set(pump_indices_4h.get(sym, []))
    exclude = set()
    for pi in pump_i_set:
        for off in range(-24, 25): exclude.add(pi+off)
    valid = [i for i in range(30, len(rows)-1) if i not in exclude]
    random.shuffle(valid)
    for idx in valid[:3]:  # 3 control per symbol
        feats = compute_at_idx(rows, idx, 4)
        if feats: control_4h.append(feats)

print(f'\nControl samples (t-4h): {len(control_4h)}')

# Best single feature at t-4h
print(f'\n{"Feature":<20} | {"Rule":<25} | {"catch%":>8} {"FP%":>8} {"prec%":>8}')
print('-'*80)
rules_4h = [
    ('rvol', lambda p: p.get('rvol', 0) > 1.5, '> 1.5x'),
    ('rvol', lambda p: p.get('rvol', 0) > 2, '> 2.0x'),
    ('rvol', lambda p: p.get('rvol', 0) > 0.8, '> 0.8x (relaxed)'),
    ('max_rvol_4h', lambda p: p.get('max_rvol_4h', 0) > 1.5, '> 1.5x'),
    ('max_rvol_4h', lambda p: p.get('max_rvol_4h', 0) > 2, '> 2.0x'),
    ('mom_3', lambda p: p.get('mom_3', -99) > 0, '> 0%'),
    ('mom_3', lambda p: p.get('mom_3', -99) > 0.5, '> 0.5%'),
    ('mom_3', lambda p: p.get('mom_3', -99) > 1, '> 1%'),
    ('mom_3', lambda p: p.get('mom_3', 99) < 0, '< 0% (pullback)'),
    ('mom_6', lambda p: p.get('mom_6', -99) > 0, '> 0%'),
    ('mom_6', lambda p: p.get('mom_6', -99) > 1, '> 1%'),
    ('mom_12', lambda p: p.get('mom_12', -99) > 0, '> 0%'),
    # Combined
    ('mom_3+rvol', lambda p: p.get('mom_3', -99) > 0 and p.get('max_rvol_4h', 0) > 1.5, 'mom>0 AND max_rvol>1.5'),
    ('mom_3+rvol', lambda p: p.get('mom_3', -99) > 0.5 and p.get('max_rvol_4h', 0) > 1.5, 'mom>0.5 AND max_rvol>1.5'),
    ('flat_then_vol', lambda p: abs(p.get('mom_3', 0)) < 2 and p.get('max_rvol_4h', 0) > 1.5, '|mom_3|<2 AND max_rvol>1.5'),
    ('flat_then_vol', lambda p: abs(p.get('mom_3', 0)) < 3 and p.get('max_rvol_4h', 0) > 1.2, '|mom_3|<3 AND max_rvol>1.2'),
    ('quiet_buildup', lambda p: p.get('rvol', 0) < 2 and p.get('max_rvol_4h', 0) > 1.5, 'rvol<2 AND max_rvol>1.5'),
    ('pullback+vol', lambda p: p.get('mom_3', 0) < -1 and p.get('max_rvol_4h', 0) > 1.5, 'mom<-1% AND max_rvol>1.5'),
]
for feat_name, rule, desc in rules_4h:
    p_pass = sum(1 for p in pumps if 't4h_'+feat_name in p and rule({'rvol': p.get('t4h_rvol', 0), 'max_rvol_4h': p.get('t4h_max_rvol_4h', 0), 'mom_3': p.get('t4h_mom_3', 0), 'mom_6': p.get('t4h_mom_6', 0), 'mom_12': p.get('t4h_mom_12', 0)}))
    c_pass = sum(1 for c in control_4h if rule(c))
    n_p = len([p for p in pumps if 't4h_'+feat_name in p])
    n_c = len(control_4h)
    if n_p == 0: continue
    catch = p_pass/n_p*100
    fp = c_pass/n_c*100 if n_c else 0
    prec = p_pass/max(1,p_pass+c_pass)*100
    print(f'{feat_name:<20} | {desc:<25} | {p_pass:>3}/{n_p:<3} ({catch:>5.1f}%) {c_pass:>3}/{n_c:<3} ({fp:>5.1f}%) {prec:>5.1f}%')
