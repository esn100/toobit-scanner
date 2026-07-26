"""
Check if our ultra-strict filter would have caught the AKE pump.
The pump started at 15m-tf 04:15 UTC on 2026-07-15.
We check 1, 2, 3 candles AFTER the start (i.e., 04:30, 04:45, 05:00)
which is when scanner would actually see the move.
"""
import json, math, statistics
from datetime import datetime, timezone

def load(path):
    with open(path) as f: return json.load(f)

def parse_klines(raw):
    rows = []
    for r in raw:
        ts = int(r[0])
        o, h, l, c = float(r[5]), float(r[3]), float(r[4]), float(r[2])
        v = float(r[1])
        rows.append({'ts': ts, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v,
                     'time': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})
    rows.sort(key=lambda x: x['ts'])
    return rows

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

def atr(rows, n=14):
    if len(rows) < n+1: return 0.0
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    if len(trs) < n: return sum(trs[-n:])/n
    return sum(trs)/len(trs)

def bollinger(rows, n=20, k=2):
    if len(rows) < n: return 0,0,0
    closes = [r['c'] for r in rows[-n:]]
    ma = sum(closes)/n
    std = statistics.pstdev(closes) if len(closes) > 1 else 0
    return ma, ma + k*std, ma - k*std

def ema(values, n):
    if not values: return []
    k = 2/(n+1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v*k + out[-1]*(1-k))
    return out

def check_filter(rows, i):
    """Apply our ultra-strict 14+ criteria filter at index i"""
    if i < 30: return None, "insufficient history"

    window = rows[max(0, i-50):i+1]
    cur = rows[i]
    closes = [r['c'] for r in window]

    mom_3 = (cur['c']-rows[i-3]['c'])/rows[i-3]['c']*100 if i >= 3 else 0
    mom_6 = (cur['c']-rows[i-6]['c'])/rows[i-6]['c']*100 if i >= 6 else 0
    mom_12 = (cur['c']-rows[i-12]['c'])/rows[i-12]['c']*100 if i >= 12 else 0

    atr_v = atr(window)
    atr_pct = atr_v / cur['c'] * 100 if cur['c'] > 0 else 0

    vols = [r['v'] for r in window]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
    rvol = cur['v']/avg_v if avg_v > 0 else 1.0

    ma, up, lo = bollinger(window)
    bb_width = (up-lo)/ma*100 if ma > 0 else 0
    bb_pos = (cur['c']-lo)/(up-lo) if (up-lo) > 0 else 0.5

    rsi_v = rsi(closes)
    ema9 = ema(closes, 9)[-1]
    ema21 = ema(closes, 21)[-1]
    ema_cross = 'BULL' if ema9 > ema21 else 'BEAR'

    body = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0

    # ===== Ultra-strict filter criteria (from strict_mode.py / ultra_strict.py) =====
    checks = {
        'ATR 3-8%': 3.0 <= atr_pct <= 8.0,
        'mom_3 < 8% (no overext)': abs(mom_3) < 8.0,
        'mom_6 > 1.5% (no chop)': abs(mom_6) > 1.5,
        'BB width < 12%': bb_width < 12.0,
        'BB position 0.2-0.85': 0.2 <= bb_pos <= 0.85,
        'RSI 25-78': 25 <= rsi_v <= 78,
        'rvol > 1.0': rvol > 1.0,
        'EMA cross BULL (LONG)': ema_cross == 'BULL',
        'Body ratio > 0.35': body > 0.35,
    }

    # Direction guess
    direction = 'LONG' if mom_3 > 0 else 'SHORT'

    return {
        'time': cur['time'],
        'close': cur['c'],
        'direction': direction,
        'mom_3': mom_3, 'mom_6': mom_6, 'mom_12': mom_12,
        'rsi': rsi_v,
        'atr_pct': atr_pct,
        'rvol': rvol,
        'bb_width': bb_width, 'bb_pos': bb_pos,
        'body_ratio': body,
        'ema_cross': ema_cross,
        'checks': checks,
        'passed': sum(1 for v in checks.values() if v),
        'total': len(checks),
    }, None

# Load 15m klines (covers pump window + 10 days before)
raw = load('klines_15m.json')
rows = parse_klines(raw)

# ACTUAL pump start = 2026-07-15T04:15:00 UTC (price started rising from 0.000187)
# First candle with volume spike = 2026-07-15T05:00 UTC (vol=103k, rvol=7.69x)
# Let pump_idx = the index just before the volume spike (i.e., 04:45 = last quiet candle)
pump_idx = 157  # ~04:45 UTC, just before the explosion
# Verify
print(f'Pump idx 157: {rows[157]["time"]}  c={rows[157]["c"]}  v={rows[157]["v"]}')
print(f'Pump idx 158: {rows[158]["time"]}  c={rows[158]["c"]}  v={rows[158]["v"]}  <-- first big vol')
print(f'Pump idx 159: {rows[159]["time"]}  c={rows[159]["c"]}  v={rows[159]["v"]}')
print(f'Pump start index in 5m: {pump_idx} = {rows[pump_idx]["time"]}')
print(f'Pump peak (max h after start): looking...')
peak_h = 0; peak_i = None
for j in range(pump_idx, min(pump_idx+200, len(rows))):
    if rows[j]['h'] > peak_h:
        peak_h = rows[j]['h']; peak_i = j
print(f'Peak: {rows[peak_i]["time"]} @ {peak_h:.8f}')
peak_pct = (peak_h - rows[pump_idx]['c'])/rows[pump_idx]['c']*100
print(f'Peak gain: +{peak_pct:.1f}%')

# Check filter at t+1, t+2, t+3, t+4 (5min, 10min, 15min, 20min after start)
# i.e., the FIRST 5m candle where volume spiked
print(f'\n{"="*100}')
print('FILTER CHECK AT EACH 5-MIN CANDLE AROUND PUMP START')
print(f'{"="*100}\n')
print(f'Pump start: {rows[pump_idx]["time"]} @ {rows[pump_idx]["c"]:.8f}')
print(f'Peak:       {rows[peak_i]["time"]} @ {peak_h:.8f} (+{peak_pct:.1f}%)\n')

for off in range(-2, 30):
    idx = pump_idx + off
    if idx >= len(rows) or idx < 30: continue
    result, err = check_filter(rows, idx)
    if result is None: continue
    # Compute forward return from this point to peak
    fwd = (peak_h - result['close'])/result['close']*100
    # If signal LONG, can it capture? If SHORT, no.
    pass_str = f"{result['passed']}/{result['total']}"
    chk = ' '.join(['✓' if v else '✗' for v in result['checks'].values()])
    print(f"t+{off:>3}  {result['time']}  c={result['close']:.8f}  "
          f"fwd_to_peak=+{fwd:6.1f}%  {result['direction']:>5}  "
          f"mom_3={result['mom_3']:>6.2f}%  rsi={result['rsi']:>5.1f}  "
          f"rvol={result['rvol']:>5.2f}  atr%={result['atr_pct']:>4.1f}  "
          f"pass={pass_str}  [{chk}]")

# Also check: at what point did rvol first exceed 1.5? 2? 3?
print(f'\n{"="*100}')
print('VOLUME / MOMENTUM ACCELERATION TIMELINE')
print(f'{"="*100}\n')

base_window = rows[max(0,pump_idx-100):pump_idx]
base_v = sum(r['v'] for r in base_window)/len(base_window)
print(f'Baseline avg vol (100 candles pre-pump): {base_v:,.0f}\n')

for off in range(-3, 25):
    idx = pump_idx + off
    if idx >= len(rows): break
    r = rows[idx]
    rv = r['v']/base_v if base_v > 0 else 0
    ch = (r['c']-r['o'])/r['o']*100 if r['o'] > 0 else 0
    cum_ch = (r['c']-rows[pump_idx]['c'])/rows[pump_idx]['c']*100
    marker = '  <-- PUMP START' if off == 0 else ''
    print(f"t+{off:>3}  {r['time']}  vol={r['v']:>10,.0f}  rvol={rv:>6.2f}x  "
          f"ch={ch:>+6.2f}%  cum={cum_ch:>+6.2f}%{marker}")
