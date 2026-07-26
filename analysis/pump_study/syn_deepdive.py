"""
SYN pump post-mortem: find ALL pumps in 30 days, analyze each.
Focus: what happened BEFORE each pump that we could detect.
"""
import json
import math
import statistics
from datetime import datetime, timezone
from collections import defaultdict

rows_raw = json.load(open('/tmp/syn_1h.json'))
rows = []
for r in sorted(rows_raw, key=lambda x: int(x[0])):
    ts = int(r[0])
    rows.append({
        'ts': ts, 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]),
        'c': float(r[2]), 'v': float(r[1]),
        't': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    })

print(f'SYN total: {len(rows)} candles, {rows[0]["t"]} to {rows[-1]["t"]}')
print(f'Price range: {min(r["l"] for r in rows):.5f} to {max(r["h"] for r in rows):.5f}')
print()

# Find ALL pumps (24h forward return >= 30%)
pumps = []
n = len(rows)
i = 0
while i < n - 24:
    cur_low = rows[i]['l']
    max_fwd = max((rows[i+j]['h'] for j in range(1, 25)), default=cur_low)
    pct = (max_fwd - cur_low)/cur_low*100 if cur_low > 0 else 0
    if pct >= 30:
        peak_i = i
        for j in range(1, 25):
            if i+j < n and rows[i+j]['h'] >= max_fwd - 1e-12:
                peak_i = i+j
        pumps.append({
            'start_i': i, 'peak_i': peak_i,
            'start_t': rows[i]['t'], 'peak_t': rows[peak_i]['t'],
            'start_c': rows[i]['c'], 'start_low': cur_low,
            'peak_h': max_fwd, 'gain_pct': pct,
        })
        i = peak_i + 6
    else:
        i += 1

print(f'Found {len(pumps)} pumps (>= 30% in 24h):')
for p in pumps:
    print(f'  {p["start_t"]}  {p["start_c"]:.5f}  ->  {p["peak_t"]}  {p["peak_h"]:.5f}  (+{p["gain_pct"]:.1f}%)')

# Analyze each pump
def rsi(closes, n=14):
    if len(closes) < n+1: return 50.0
    g, l = [], []
    for i in range(1, n+1):
        ch = closes[i] - closes[i-1]
        g.append(max(ch,0)); l.append(max(-ch,0))
    ag = sum(g)/n; al = sum(l)/n
    for i in range(n+1, len(closes)):
        ch = closes[i] - closes[i-1]
        ag = (ag*(n-1) + max(ch,0))/n
        al = (al*(n-1) + max(-ch,0))/n
    if al == 0: return 100.0
    return 100 - 100/(1+ag/al)

def atr_pct(rows, n=14):
    if len(rows) < n+1: return 0
    trs = []
    for i in range(1, len(rows)):
        h,l,pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    atr = sum(trs[-n:])/min(n,len(trs))
    return atr/rows[-1]['c']*100 if rows[-1]['c']>0 else 0

print()
print('='*100)
print('DETAILED PRE-PUMP ANALYSIS (lookback 1h, 4h, 12h, 24h before each pump)')
print('='*100)

for pi, p in enumerate(pumps, 1):
    print(f'\n{"-"*100}')
    print(f'PUMP #{pi}: {p["start_t"]} -> {p["peak_t"]}  +{p["gain_pct"]:.1f}%  ({p["start_c"]:.5f} -> {p["peak_h"]:.5f})')
    si = p['start_i']
    if si < 30: continue

    print(f'  {"lookback":<10} {"close":>10} {"rvol":>6} {"max4h":>6} {"mom_3":>7} {"mom_6":>7} {"mom_12":>7} {"rsi":>5} {"atr%":>5} {"flat_h":>7}')
    for h in [24, 12, 6, 4, 2, 1, 0]:
        idx = max(0, si - h)
        if idx < 30: continue
        window = rows[max(0, idx-50):idx+1]
        cur = rows[idx]
        closes = [r['c'] for r in window]
        vols = [r['v'] for r in window]
        avg_v = sum(vols[:-1])/max(len(vols)-1,1)
        rvol = cur['v']/avg_v if avg_v>0 else 1
        recent = vols[-5:-1] if len(vols)>5 else vols[:-1]
        max4h = max((v/avg_v for v in recent), default=1) if avg_v>0 else 1
        if idx >= 3: mom_3 = (cur['c']-rows[idx-3]['c'])/rows[idx-3]['c']*100
        else: mom_3 = 0
        if idx >= 6: mom_6 = (cur['c']-rows[idx-6]['c'])/rows[idx-6]['c']*100
        else: mom_6 = 0
        if idx >= 12: mom_12 = (cur['c']-rows[idx-12]['c'])/rows[idx-12]['c']*100
        else: mom_12 = 0
        rsi_v = rsi(closes)
        atr_v = atr_pct(window)
        # flat hours
        flat_h = 0
        for j in range(idx-1, max(0, idx-168), -1):
            if abs(cur['c']-rows[j]['c'])/rows[j]['c']*100 <= 5:
                flat_h = idx - j
            else:
                break
        print(f'  t-{h}h{" "*(3-len(str(h)))} {cur["c"]:>10.5f} {rvol:>6.2f} {max4h:>6.2f} {mom_3:>+6.2f}% {mom_6:>+6.2f}% {mom_12:>+6.2f}% {rsi_v:>5.1f} {atr_v:>5.2f} {flat_h:>5}h')

# Volume profile around all pumps (average)
print()
print('='*100)
print('AGGREGATE PRE-PUMP VOLUME PROFILE (averaged across all pumps)')
print('='*100)

all_pre_vols = []  # at each h-back
all_pre_mom3 = []
all_pre_max4h = []
for h in range(0, 25):
    rvol_vals = []
    mom3_vals = []
    max4h_vals = []
    for p in pumps:
        si = p['start_i']
        idx = max(0, si - h)
        if idx < 30: continue
        window = rows[max(0, idx-50):idx+1]
        cur = rows[idx]
        vols = [r['v'] for r in window]
        avg_v = sum(vols[:-1])/max(len(vols)-1,1)
        rvol = cur['v']/avg_v if avg_v>0 else 1
        recent = vols[-5:-1] if len(vols)>5 else vols[:-1]
        max4h = max((v/avg_v for v in recent), default=1) if avg_v>0 else 1
        if idx >= 3:
            mom_3 = (cur['c']-rows[idx-3]['c'])/rows[idx-3]['c']*100
        else: mom_3 = 0
        rvol_vals.append(rvol)
        mom3_vals.append(mom_3)
        max4h_vals.append(max4h)
    if rvol_vals:
        ar = sum(rvol_vals)/len(rvol_vals)
        am = sum(mom3_vals)/len(mom3_vals)
        am4 = sum(max4h_vals)/len(max4h_vals)
        bar = '#' * int(ar*5)
        print(f't-{h:>3}h  rvol={ar:>5.2f}x  max4h={am4:>5.2f}x  mom_3={am:>+6.2f}%  {bar}')

# What was the CONSISTENT signal?
print()
print('='*100)
print('HOW EARLY CAN WE DETECT? (% of pumps satisfying X at h-back)')
print('='*100)

for h in [24, 12, 6, 4, 2, 1]:
    counts = defaultdict(int)
    total = 0
    for p in pumps:
        si = p['start_i']
        idx = max(0, si - h)
        if idx < 30: continue
        total += 1
        window = rows[max(0, idx-50):idx+1]
        cur = rows[idx]
        vols = [r['v'] for r in window]
        avg_v = sum(vols[:-1])/max(len(vols)-1,1)
        rvol = cur['v']/avg_v if avg_v>0 else 1
        recent = vols[-5:-1] if len(vols)>5 else vols[:-1]
        max4h = max((v/avg_v for v in recent), default=1) if avg_v>0 else 1
        if idx >= 3:
            mom_3 = (cur['c']-rows[idx-3]['c'])/rows[idx-3]['c']*100
        else: mom_3 = 0
        flat_h = 0
        for j in range(idx-1, max(0, idx-168), -1):
            if abs(cur['c']-rows[j]['c'])/rows[j]['c']*100 <= 5:
                flat_h = idx - j
            else:
                break
        if rvol > 1.5: counts['rvol>1.5'] += 1
        if rvol > 2.0: counts['rvol>2.0'] += 1
        if max4h > 1.5: counts['max4h>1.5'] += 1
        if max4h > 2.0: counts['max4h>2.0'] += 1
        if abs(mom_3) < 5: counts['|mom_3|<5%'] += 1
        if abs(mom_3) < 3: counts['|mom_3|<3%'] += 1
        if flat_h > 6: counts['flat>6h'] += 1
        if flat_h > 12: counts['flat>12h'] += 1
    if total == 0: continue
    print(f'\nt-{h}h (n={total}):')
    for k, v in sorted(counts.items()):
        if v/total > 0.3:
            print(f'  {k:<20} {v}/{total} = {v/total*100:.0f}%')
