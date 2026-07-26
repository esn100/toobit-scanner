"""
Step 2: For each small cap, fetch 30d 1h klines and detect major pumps.
A pump = max forward 24h return >= 100% from a local low.
"""
import json, os, time, urllib.request
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
with open(f'{CACHE}/candidates_v3.json') as f:
    small_caps = json.load(f)
print(f'Using {len(small_caps)} candidates from v3')

def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except:
        return None

def fetch_klines(pair, interval='1h', limit=1000, batches=2):
    """Fetch up to 2*1000 = 2000 hours (~83 days)"""
    cache_path = f'{CACHE}/klines_{pair}_{interval}.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    all_data = []
    for b in range(batches):
        url = f'https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={pair}&interval={interval}&limit={limit}'
        raw = http_get(url)
        if not raw: break
        try:
            data = json.loads(raw)
            if not data: break
            all_data.extend(data)
            # Set up next batch from earliest timestamp
            earliest = min(int(r[0]) for r in data) - 1
            url = f'https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={pair}&interval={interval}&limit={limit}&to={earliest}'
            # Don't continue, gate.io doesn't support 'to' well, just break
            break
        except:
            break
    if all_data:
        # Dedupe by timestamp
        seen = set()
        unique = []
        for r in all_data:
            ts = r[0]
            if ts in seen: continue
            seen.add(ts)
            unique.append(r)
        with open(cache_path, 'w') as f:
            json.dump(unique, f)
    return all_data if all_data else None

def find_pumps(klines, threshold=40):
    """Find all pumps (24h forward return >= threshold%)"""
    if not klines: return []
    klines.sort(key=lambda r: int(r[0]))
    rows = []
    for r in klines:
        ts = int(r[0])
        rows.append({'ts': ts, 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]),
                     'c': float(r[2]), 'v': float(r[1]),
                     't': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})
    pumps = []
    n = len(rows)
    window = 24  # 24 hours on 1h
    i = 0
    while i < n - window:
        # Find local low: candle whose low is lower than next 24h low
        cur_low = rows[i]['l']
        max_fwd = max((rows[i+j]['h'] for j in range(1, window+1)), default=cur_low)
        pct = (max_fwd - cur_low)/cur_low*100 if cur_low > 0 else 0
        if pct >= threshold:
            # Find exact peak
            peak_i = i
            for j in range(1, window+1):
                if i+j < n and rows[i+j]['h'] >= max_fwd - 1e-12:
                    peak_i = i+j
            pumps.append({
                'start_ts': rows[i]['ts'],
                'start_time': rows[i]['t'],
                'start_close': rows[i]['c'],
                'start_low': cur_low,
                'peak_ts': rows[peak_i]['ts'],
                'peak_time': rows[peak_i]['t'],
                'peak_high': max_fwd,
                'gain_pct': pct,
            })
            i = peak_i + 6  # skip 6 hours after peak to find next pump
        else:
            i += 1
    return pumps, rows

print(f'Processing {len(small_caps)} small caps...')
all_pumps = []
all_data = {}

def process_one(c):
    pair = c['gate_pair']
    klines = fetch_klines(pair, '1h', 720)
    if not klines: return []
    pumps, rows = find_pumps(klines, threshold=80)
    for p in pumps:
        p['sym'] = c['sym']
        p['mc'] = c['mc']
        p['qv_24h'] = c['qv_24h']
    return pumps, rows, c['sym']

# Parallel fetch
results = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(process_one, c): c for c in small_caps}
    done = 0
    for fut in as_completed(futures):
        done += 1
        c = futures[fut]
        try:
            pumps, rows, sym = fut.result()
            if pumps:
                print(f'  [{done}/{len(small_caps)}] {sym}: {len(pumps)} pumps, best +{max(p["gain_pct"] for p in pumps):.0f}%')
            results[sym] = rows
            for p in pumps:
                p['sym'] = sym
                all_pumps.append(p)
        except Exception as e:
            print(f'  [{done}/{len(small_caps)}] {c["sym"]}: ERR {e}')

# Save
print(f'\nTotal pumps detected: {len(all_pumps)}')
all_pumps.sort(key=lambda p: -p['gain_pct'])
print(f'Top 10 pumps:')
for p in all_pumps[:10]:
    print(f'  {p["sym"]:<14} +{p["gain_pct"]:>6.1f}%  {p["start_time"]}  ->  {p["peak_time"]}')

# Save the top 30+ for next analysis step
with open(f'{CACHE}/all_pumps.json', 'w') as f:
    json.dump(all_pumps, f, indent=2)
with open(f'{CACHE}/rows_index.json', 'w') as f:
    # Save only timestamps per symbol for quick lookup
    json.dump({sym: [{'ts': r['ts'], 't': r['t'], 'o': r['o'], 'h': r['h'], 'l': r['l'], 'c': r['c'], 'v': r['v']} for r in rows] for sym, rows in results.items()}, f)

print(f'\nSaved: {CACHE}/all_pumps.json + rows_index.json')
