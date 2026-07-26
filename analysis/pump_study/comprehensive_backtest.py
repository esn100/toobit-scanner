"""
Comprehensive backtest with 60+ day historical data.
Uses Gate.io as primary source (2000+ candles = 83 days of 1h data).
Finds ALL pumps, then computes features at multiple timepoints.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from collections import Counter
import statistics

sys.path.insert(0, '/home/user/toobit-scanner')

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'

def fetch_gateio_klines(symbol, days=60):
    """Fetch up to 2000 1h candles from Gate.io (~83 days)."""
    cache_path = f'{CACHE}/klines_gateio_{symbol}_1h.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    base = symbol[:-4] + '_' + symbol[-4:]
    all_rows = []
    for batch in range(3):
        try:
            url = 'https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair=' + base + '&interval=1h&limit=1000'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            if data:
                all_rows.extend(data)
                break
        except urllib.error.HTTPError as e:
            print(f'  Gate.io HTTP {e.code} for {symbol}: {e.read()[:200]}')
            break
        except Exception as e:
            print(f'  Gate.io err {symbol}: {e}')
            break
    if not all_rows: return []
    # Dedupe and sort
    seen = set()
    unique = []
    for r in all_rows:
        ts = r[0]
        if ts in seen: continue
        seen.add(ts)
        unique.append({'ts': int(ts), 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]), 'c': float(r[2]), 'v': float(r[1])})
    unique.sort(key=lambda x: x['ts'])
    with open(cache_path, 'w') as f:
        json.dump(unique, f)
    return unique

# Get all 12 repeaters
REPEATERS = ['EVAAUSDT', 'TLMUSDT', 'LABUSDT', 'BANKUSDT', 'AKEUSDT',
             'IKAUSDT', 'SYNUSDT', 'ACEUSDT', 'INUSDT', 'ERAUSDT', 'WODUSDT', 'XCXUSDT']

print('Fetching 60+ day historical data for 12 repeaters...')
import time
all_data = {}
for sym in REPEATERS:
    rows = fetch_gateio_klines(sym, days=60)
    if rows:
        all_data[sym] = rows
        print(f'  {sym}: {len(rows)} candles ({len(rows)/24:.0f} days)')
    else:
        print(f'  {sym}: FAILED')
    time.sleep(1)  # rate limit

# Find all pumps (24h forward return >= 20%)
print('\nFinding all pumps (>= 20% in 24h)...')
all_pumps = []
for sym, rows in all_data.items():
    for i in range(len(rows) - 24):
        cur_low = rows[i]['l']
        max_fwd = max((rows[i+j]['h'] for j in range(1, 25)), default=cur_low)
        pct = (max_fwd - cur_low) / cur_low * 100 if cur_low > 0 else 0
        if pct >= 20:
            peak_i = i
            for j in range(1, 25):
                if i+j < len(rows) and rows[i+j]['h'] >= max_fwd - 1e-12:
                    peak_i = i+j
            all_pumps.append({
                'sym': sym,
                'start_i': i, 'peak_i': peak_i,
                'start_ts': rows[i]['ts'],
                'start_c': rows[i]['c'],
                'peak_h': max_fwd,
                'gain_pct': pct,
            })

print(f'Total pumps: {len(all_pumps)}')
# Group by symbol
sym_count = Counter(p['sym'] for p in all_pumps)
print('Top pumpers:')
for sym, n in sym_count.most_common(5):
    print(f'  {sym}: {n} pumps')

# Save
with open(f'{CACHE}/comprehensive_pumps.json', 'w') as f:
    json.dump(all_pumps, f, indent=2, default=str)
print(f'Saved to {CACHE}/comprehensive_pumps.json')
