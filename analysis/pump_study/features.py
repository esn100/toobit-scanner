"""
Compute pre-pump features for all 52+ detected pumps.
For each pump, look at 1h, 4h, 12h, 24h BEFORE the pump start and compute:
- momentum, RSI, ATR, BB, volume, candle body, EMA cross
- absolute return from local low, hours of pre-pump flat
- volume trend, OBI proxy
"""
import json, os, math, statistics
from datetime import datetime, timezone

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/all_pumps.json') as f:
    all_pumps = json.load(f)
with open(f'{CACHE}/rows_index.json') as f:
    rows_idx = json.load(f)

# Add AKE manually (from earlier analysis)
raw_ake = json.load(open('/home/user/toobit-scanner/analysis/ake/klines_15m.json'))
ake_rows = []
for r in raw_ake:
    ts = int(r[0])
    ake_rows.append({'ts': ts, 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]),
                     'c': float(r[2]), 'v': float(r[1]),
                     't': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})
# 15m klines: convert to 1h-like by using them as is
# For AKE we know pump start = 2026-07-15T04:15 UTC
ake_rows.sort(key=lambda x: x['ts'])
ake_pump = {
    'sym': 'AKEUSDT', 'start_ts': 1784152800, 'start_time': '2026-07-15T04:15:00+00:00',
    'start_close': 0.000187, 'start_low': 0.0001863,
    'peak_ts': ake_rows[-1]['ts'], 'peak_time': ake_rows[-1]['t'],
    'peak_high': 0.001934, 'gain_pct': 934.0, 'mc': 5000000, 'qv_24h': 35719519,
    'tf': '15m_agg'
}
all_pumps.append(ake_pump)
rows_idx['AKEUSDT'] = ake_rows

# 15m for AKE means features computed on 15m basis
# For consistency, use 1h klines only for the rest
print(f'Total pumps: {len(all_pumps)}')

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

def ema(values, n):
    if not values: return []
    k = 2/(n+1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v*k + out[-1]*(1-k))
    return out

def bollinger_pos(rows, n=20, k=2):
    if len(rows) < n: return 0.5
    closes = [r['c'] for r in rows[-n:]]
    ma = sum(closes)/n
    std = statistics.pstdev(closes) if len(closes) > 1 else 0
    up = ma + k*std; lo = ma - k*std
    return (rows[-1]['c']-lo)/(up-lo) if (up-lo) > 0 else 0.5

def analyze_pump(pump, rows):
    """Compute features at multiple pre-pump timepoints (1h, 4h, 12h, 24h, 48h before)"""
    # Find pump start index in rows
    if 'tf' in pump and pump['tf'] == '15m_agg':
        step_hours = 0.25  # 15m
    else:
        step_hours = 1.0
    # Find nearest row
    target_ts = pump['start_ts']
    pump_i = None
    for i, r in enumerate(rows):
        if r['ts'] >= target_ts - 60*step_hours*60 and r['ts'] <= target_ts + 60*step_hours*60:
            # Closest
            if pump_i is None or abs(r['ts']-target_ts) < abs(rows[pump_i]['ts']-target_ts):
                pump_i = i
    if pump_i is None or pump_i < 50:
        return None
    result = {'sym': pump['sym'], 'gain_pct': pump['gain_pct'], 'mc': pump.get('mc'),
              'qv_24h': pump.get('qv_24h'), 'start_time': pump['start_time'],
              'tf_step_h': step_hours}

    # For each lookback window, compute features
    for h in [1, 4, 12, 24, 48, 72]:
        offset = int(h / step_hours)
        if pump_i < offset + 30: continue
        idx = pump_i - offset
        window = rows[max(0, idx-50):idx+1]
        cur = rows[idx]
        closes = [r['c'] for r in window]
        vols = [r['v'] for r in window]

        # Momentum
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

        # Volume ratio
        avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
        rvol = cur['v']/avg_v if avg_v > 0 else 1.0

        # Recent max rvol (over last 4 candles)
        recent_rvols = [vols[i]/avg_v for i in range(max(0,len(vols)-4), len(vols))] if avg_v > 0 else [1.0]
        max_rvol_4h = max(recent_rvols) if recent_rvols else 1.0

        result[f't{h}h_mom_1'] = mom_1
        result[f't{h}h_mom_3'] = mom_3
        result[f't{h}h_mom_6'] = mom_6
        result[f't{h}h_mom_12'] = mom_12
        result[f't{h}h_rsi'] = rsi(closes)
        result[f't{h}h_atr_pct'] = atr_pct(window)
        result[f't{h}h_rvol'] = rvol
        result[f't{h}h_max_rvol_4h'] = max_rvol_4h
        result[f't{h}h_bb_pos'] = bollinger_pos(window)
        body = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0
        result[f't{h}h_body'] = body
        result[f't{h}h_close'] = cur['c']

    # === SPECIAL FEATURES: pre-pump flat hours ===
    # Look back up to 168h (1 week) to find longest flat period
    flat_threshold_pct = 5.0  # within 5%
    max_lookback = int(168 / step_hours)
    if pump_i >= max_lookback:
        flat_start = pump_i
        for j in range(pump_i, max(0, pump_i - max_lookback), -1):
            pct_change = abs(rows[pump_i]['c'] - rows[j]['c']) / rows[j]['c'] * 100 if rows[j]['c'] > 0 else 0
            if pct_change <= flat_threshold_pct:
                flat_start = j
            else:
                break
        flat_hours = (pump_i - flat_start) * step_hours
    else:
        flat_hours = 0
    result['pre_pump_flat_hours'] = flat_hours

    # === Volume trend in 24h before ===
    vol_lookback = int(24 / step_hours)
    if pump_i >= vol_lookback:
        recent_vols = [r['v'] for r in rows[max(0, pump_i - vol_lookback):pump_i+1]]
        # Compare last 4h vs first 4h of this window
        first4 = sum(recent_vols[:4])/4
        last4 = sum(recent_vols[-4:])/4
        vol_accel = (last4 - first4) / first4 * 100 if first4 > 0 else 0
    else:
        vol_accel = 0
    result['vol_accel_24h'] = vol_accel

    # === Pre-pump price compression (BB width) ===
    result['bb_width_compression'] = result.get('t24h_atr_pct', 0)

    return result

print(f'Analyzing {len(all_pumps)} pumps...')
results = []
skipped = 0
for pump in all_pumps:
    rows = rows_idx.get(pump['sym'])
    if not rows:
        skipped += 1
        continue
    res = analyze_pump(pump, rows)
    if res:
        results.append(res)
print(f'Analyzed: {len(results)}, skipped: {skipped}')

# Save raw features
with open(f'{CACHE}/pump_features.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved features for {len(results)} pumps')
