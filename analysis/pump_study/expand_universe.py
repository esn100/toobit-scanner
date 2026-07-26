"""
Expanded universe: get more small caps by also looking at:
- Gate.io pairs that have high volume but unknown MC (likely new/pumped)
- Toobit pairs (since scanner uses Toobit)
- Lower the threshold to 50% to get more samples
"""
import json, os, time, urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'

def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except:
        return None

# Toobit pairs (priority - scanner uses this)
print('Toobit tickers...')
raw = http_get('https://api.toobit.com/quote/v1/ticker/24hr')
toobit = json.loads(raw) if raw else []
toobit_usdt = [t for t in toobit if t.get('s','').endswith('USDT') and float(t.get('qv', 0) or 0) > 200000]
print(f'  Toobit USDT > $200k vol: {len(toobit_usdt)}')

# Gate.io pairs
print('Gate.io tickers...')
raw = http_get('https://api.gateio.ws/api/v4/spot/tickers')
gate = json.loads(raw) if raw else []
gate_usdt = [t for t in gate if t.get('currency_pair','').endswith('_USDT')]
print(f'  Gate.io USDT: {len(gate_usdt)}')

# CoinGecko list (for MC)
print('CoinGecko list (5 pages)...')
cgk_list = []
for page in range(1, 6):
    raw = http_get(f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}')
    if raw:
        cgk_list.extend(json.loads(raw))
        time.sleep(0.7)
cgk_by_sym = {c['symbol'].upper(): c for c in cgk_list if c.get('symbol')}
print(f'  CoinGecko: {len(cgk_list)}')

# Build combined universe: all Toobit pairs + Gate.io small caps
BLACKLIST = {'BTC','ETH','USDT','USDC','BCH','BSV','XRP','BNB','SOL','DOGE','TRX','ADA','DOT','MATIC','AVAX',
             'LINK','XLM','ATOM','ETC','LTC','TON','NEAR','APT','OP','ARB','IMX','INJ','SUI','SEI','TIA',
             'WIF','PEPE','SHIB','FLOKI','BONK','MEME','WLD','FIL','ICP','HBAR','VET','ALGO','FTM','SAND',
             'MANA','AXS','GALA','ENJ','CHZ','FLOW','EGLD','KSM','MINA','ROSE','CELO','KAVA','CRV','SNX',
             'COMP','MKR','AAVE','LDO','RPL','BAL','GRT','RUNE','OCEAN','FET','RNDR','PYTH','JTO','JUP',
             'BLUR','ENS','MASK','CYBER','TRB','PORTAL','PIXEL','STRK','ZRX','BAT','REP','NMR','LPT','AUCTION'}

candidates = []
seen = set()

# From Toobit (priority)
for t in toobit_usdt:
    sym = t['s']
    base = sym.replace('USDT','')
    if base in BLACKLIST: continue
    if base in seen: continue
    seen.add(base)
    cgk = cgk_by_sym.get(base)
    mc = cgk.get('market_cap') if cgk else None
    candidates.append({
        'sym': sym, 'gate_pair': f'{base}_USDT',
        'mc': mc, 'qv_24h': float(t.get('qv', 0) or 0),
        'last': float(t.get('c', 0) or 0),
        'source': 'toobit',
    })

# From Gate.io (add if not in Toobit)
for t in gate_usdt:
    base = t['currency_pair'].replace('_USDT','')
    if base in seen: continue
    if base in BLACKLIST: continue
    qv = float(t.get('quote_volume', 0) or 0)
    if qv < 500_000: continue
    cgk = cgk_by_sym.get(base)
    mc = cgk.get('market_cap') if cgk else None
    if mc and mc > 500_000_000: continue
    seen.add(base)
    candidates.append({
        'sym': base + 'USDT', 'gate_pair': t['currency_pair'],
        'mc': mc, 'qv_24h': qv, 'last': float(t.get('last', 0) or 0),
        'source': 'gate',
    })

print(f'\nTotal candidates: {len(candidates)}')
print(f'With MC known: {sum(1 for c in candidates if c["mc"])}')
print(f'With MC < $50M: {sum(1 for c in candidates if c["mc"] and c["mc"] < 50_000_000)}')

with open(f'{CACHE}/candidates_v2.json', 'w') as f:
    json.dump(candidates, f, indent=2)
print(f'Saved to cache/candidates_v2.json')
