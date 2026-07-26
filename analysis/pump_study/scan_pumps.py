"""
Find 30+ small-cap pumps from Toobit / Gate.io and analyze pre-pump signatures.
"""
import json, os, time, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'
os.makedirs(CACHE, exist_ok=True)

# Step 1: get all Toobit USDT pairs (since scanner focuses on small caps there)
def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return None

# Try toobit first
print('Loading Toobit tickers...')
raw = http_get('https://api.toobit.com/quote/v1/ticker/24hr')
if raw:
    toobit_ticks = json.loads(raw)
    toobit_syms = set(t['s'] for t in toobit_ticks if t['s'].endswith('USDT'))
    print(f'  Toobit: {len(toobit_syms)} USDT pairs')
else:
    toobit_syms = set()
    print('  Toobit failed')

# Gate.io is more comprehensive and free
print('Loading Gate.io tickers...')
raw = http_get('https://api.gateio.ws/api/v4/spot/tickers')
if raw:
    gate_ticks = json.loads(raw)
    gate_usdt = [t for t in gate_ticks if t['currency_pair'].endswith('_USDT') and t.get('quote_volume')]
    print(f'  Gate.io USDT pairs: {len(gate_usdt)}')

# Step 2: load CoinGecko top coins to filter small caps
print('Loading CoinGecko coins list (paginated)...')
all_cgk = []
for page in range(1, 6):  # 5 pages = 500 coins
    raw = http_get(f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}')
    if raw:
        all_cgk.extend(json.loads(raw))
        time.sleep(1.0)
    else:
        break
print(f'  CoinGecko: {len(all_cgk)} coins')

# Build symbol -> market_cap map
cgk_by_symbol = {c['symbol'].upper(): c for c in all_cgk if c.get('symbol')}

# Filter small caps: market cap < $50M, on gate.io, USDT pair, decent 24h volume
small_caps = []
for t in gate_usdt:
    sym = t['currency_pair'].replace('_USDT','')
    qv = float(t.get('quote_volume', 0))
    if qv < 100_000: continue  # skip dead pairs
    if sym in ('BTC','ETH','USDT','USDC','DAI','TUSD','FDUSD','WBTC','BCH','BSV','LTC','XRP','BNB','SOL','DOGE','TRX','ADA','DOT','MATIC','AVAX','LINK','XLM','ATOM','ETC'): continue
    cgk = cgk_by_symbol.get(sym)
    mc = cgk.get('market_cap') if cgk else None
    if mc is None: continue
    if mc > 50_000_000: continue  # $50M cap
    small_caps.append({
        'sym': sym + 'USDT',
        'gate_pair': t['currency_pair'],
        'mc': mc,
        'qv_24h': qv,
        'last': float(t.get('last', 0)),
    })

# Sort by market cap ascending (smallest = most volatile)
small_caps.sort(key=lambda x: x['mc'] or 1e9)
print(f'\n  Found {len(small_caps)} small caps with MC < $50M')
print(f'  Top 5 smallest: {[c["sym"] for c in small_caps[:5]]}')
print(f'  Top 5 by vol: {sorted(small_caps, key=lambda x: -x["qv_24h"])[:5]}')

with open(f'{CACHE}/small_caps.json', 'w') as f:
    json.dump(small_caps, f, indent=2)
print(f'Saved to {CACHE}/small_caps.json')
