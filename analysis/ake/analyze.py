"""
AKE Pump Post-Mortem
- Detects the major pump in the recent klines
- Computes features at the moment of breakout
- Compares them to our scanner's filter thresholds
"""
import json, math, statistics
from datetime import datetime, timezone

def load(path):
    with open(path) as f:
        # gate.io returns [["ts", "vol", "close", "high", "low", "open", "qvol", "is_long"], ...]
        return json.load(f)

def parse_klines(raw):
    # Sort by time asc
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
    rs = avg_g/avg_l
    return 100 - 100/(1+rs)

def atr(rows, n=14):
    if len(rows) < n+1: return 0.0
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]['h'], rows[i]['l'], rows[i-1]['c']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    if len(trs) < n: return sum(trs)/max(len(trs),1)
    return sum(trs[-n:])/n

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

def find_pump(rows, threshold_pct=300):
    """Find biggest pump event: largest 4h return >= threshold%"""
    if len(rows) < 5: return None
    best = None
    window = min(48, len(rows))  # 4h on 5m, 16h on 15m
    for i in range(len(rows)-window):
        c0 = rows[i]['c']
        max_fwd = max(rows[i+j]['h'] for j in range(1, window+1))
        pct = (max_fwd - c0)/c0*100 if c0 > 0 else 0
        if pct >= threshold_pct:
            if best is None or pct > best['pct']:
                # find the actual peak timestamp
                peak_i = i+1
                for j in range(1, window+1):
                    if rows[i+j]['h'] >= max_fwd - 1e-12:
                        peak_i = i+j; break
                best = {'start_i': i, 'peak_i': peak_i, 'pct': pct,
                        'start_ts': rows[i]['ts'], 'peak_ts': rows[peak_i]['ts'],
                        'start_c': c0, 'peak_h': max_fwd}
    return best

def compute_features_at(rows, i, lookback=20):
    """Compute scanner-relevant features at index i"""
    if i < lookback+1: return None
    window = rows[max(0,i-lookback):i+1]
    closes = [r['c'] for r in window]
    vols = [r['v'] for r in window]
    cur = rows[i]
    prev = rows[i-1]

    # Momentum
    if i >= 1:
        mom_1 = (cur['c']-prev['c'])/prev['c']*100
    else: mom_1 = 0
    if i >= 3:
        mom_3 = (cur['c']-rows[i-3]['c'])/rows[i-3]['c']*100
    else: mom_3 = 0
    if i >= 6:
        mom_6 = (cur['c']-rows[i-6]['c'])/rows[i-6]['c']*100
    else: mom_6 = 0
    if i >= 12:
        mom_12 = (cur['c']-rows[i-12]['c'])/rows[i-12]['c']*100
    else: mom_12 = 0

    # RSI
    rsi_v = rsi(closes)

    # ATR
    atr_v = atr(window)
    atr_pct = atr_v / cur['c'] * 100 if cur['c'] > 0 else 0

    # Volume ratio
    avg_v = sum(vols[:-1])/max(len(vols)-1,1) if len(vols) > 1 else vols[0]
    rvol = cur['v']/avg_v if avg_v > 0 else 1.0

    # Bollinger
    ma, up, lo = bollinger(window)
    bb_width = (up-lo)/ma*100 if ma > 0 else 0
    bb_pos = (cur['c']-lo)/(up-lo) if (up-lo) > 0 else 0.5

    # Candle body
    body = abs(cur['c']-cur['o'])/(cur['h']-cur['l']) if cur['h']>cur['l'] else 0
    body_dir = 1 if cur['c']>cur['o'] else -1

    # EMA cross
    ema9 = ema(closes, 9)[-1]
    ema21 = ema(closes, 21)[-1]
    ema_cross = 1 if ema9 > ema21 else -1

    return {
        'time': cur['time'],
        'close': cur['c'],
        'mom_1': mom_1, 'mom_3': mom_3, 'mom_6': mom_6, 'mom_12': mom_12,
        'rsi': rsi_v,
        'atr_pct': atr_pct,
        'rvol': rvol,
        'bb_width': bb_width, 'bb_pos': bb_pos,
        'body_ratio': body, 'body_dir': body_dir,
        'ema9': ema9, 'ema21': ema21, 'ema_cross': ema_cross,
    }

# Load all
for tf in ['5m', '15m', '1h']:
    raw = load(f'klines_{tf}.json')
    rows = parse_klines(raw)
    print(f'\n{"="*70}\nTIMEFRAME: {tf}  ({len(rows)} klines, {rows[0]["time"]} -> {rows[-1]["time"]})\n{"="*70}')

    # Find biggest pump
    thresh = 100 if tf != '5m' else 200
    pump = find_pump(rows, thresh)
    if not pump:
        print('  No major pump found')
        continue
    print(f'\n>>> BIGGEST PUMP DETECTED')
    print(f'  Start : {rows[pump["start_i"]]["time"]}  @ {pump["start_c"]:.8f}')
    print(f'  Peak  : {rows[pump["peak_i"]]["time"]}  @ {pump["peak_h"]:.8f}')
    print(f'  Gain  : +{pump["pct"]:.1f}%')
    hours = (pump['peak_ts']-pump['start_ts'])/3600
    print(f'  Duration: {hours:.1f} hours')

    # Compute features at pump start and 1, 3, 6, 12 candles before
    si = pump['start_i']
    print(f'\n--- PRE-PUMP FEATURE SNAPSHOTS (timeframe={tf}) ---')
    print(f'{"time":<22} {"close":>10} {"mom_3%":>7} {"rsi":>5} {"rvol":>5} {"atr%":>5} {"bb_w%":>6} {"body":>5} {"ema":>3}')
    for off in [12, 6, 3, 1, 0]:
        idx = max(0, si - off)
        feats = compute_features_at(rows, idx)
        if feats is None: continue
        ema_s = '↑' if feats['ema_cross']>0 else '↓'
        print(f"{feats['time']:<22} {feats['close']:>10.8f} {feats['mom_3']:>7.2f} {feats['rsi']:>5.1f} {feats['rvol']:>5.2f} {feats['atr_pct']:>5.1f} {feats['bb_width']:>6.1f} {feats['body_ratio']:>5.2f} {ema_s:>3}")

    # Volume profile around pump
    print(f'\n--- VOLUME PROFILE around pump (24 candles before, 12 after) ---')
    v_start = max(0, si-24)
    v_end = min(len(rows), si+12)
    base_v = sum(r['v'] for r in rows[max(0,si-100):si])/100
    print(f'Baseline vol: {base_v:,.0f}')
    for j in range(v_start, v_end):
        rel_v = rows[j]['v']/base_v if base_v > 0 else 0
        rel_t = j - si
        marker = '  <-- PUMP START' if j == si else ('  <-- PEAK' if j == pump['peak_i'] else '')
        print(f"  t-{si-j:>3}  {rows[j]['time']}  vol={rows[j]['v']:>12,.0f}  rvol={rel_v:>5.2f}x  close={rows[j]['c']:.8f}{marker}")
