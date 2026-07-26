"""Build candidate universe from cached cgk + toobit + gate"""
import json, os

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'

# Load cgk
with open(f'{CACHE}/cgk_full.json') as f:
    cgk_list = json.load(f)
cgk_by_sym = {c['symbol'].upper(): c for c in cgk_list if c.get('symbol')}
print(f'CGK: {len(cgk_by_sym)} unique symbols')

# Load toobit (need to refetch)
import urllib.request
def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except: return None

raw = http_get('https://api.toobit.com/quote/v1/ticker/24hr')
toobit = json.loads(raw) if raw else []
toobit_usdt = [t for t in toobit if t.get('s','').endswith('USDT') and float(t.get('qv', 0) or 0) > 200000]
print(f'Toobit USDT > $200k vol: {len(toobit_usdt)}')

raw = http_get('https://api.gateio.ws/api/v4/spot/tickers')
gate = json.loads(raw) if raw else []
gate_usdt = [t for t in gate if t.get('currency_pair','').endswith('_USDT')]
print(f'Gate.io USDT: {len(gate_usdt)}')

# BLACKLIST
BLACKLIST = {'BTC','ETH','USDT','USDC','BCH','BSV','XRP','BNB','SOL','DOGE','TRX','ADA','DOT','MATIC','AVAX',
             'LINK','XLM','ATOM','ETC','LTC','TON','NEAR','APT','OP','ARB','IMX','INJ','SUI','SEI','TIA',
             'WIF','PEPE','SHIB','FLOKI','BONK','MEME','WLD','FIL','ICP','HBAR','VET','ALGO','FTM','SAND',
             'MANA','AXS','GALA','ENJ','CHZ','FLOW','EGLD','KSM','MINA','ROSE','CELO','KAVA','CRV','SNX',
             'COMP','MKR','AAVE','LDO','RPL','BAL','GRT','RUNE','OCEAN','FET','RNDR','PYTH','JTO','JUP',
             'BLUR','ENS','MASK','CYBER','TRB','PORTAL','PIXEL','STRK','ZRX','BAT','REP','NMR','LPT','AUCTION',
             'DAI','TUSD','FDUSD','WBTC','UNI','STX','CFX','DYDX','GMX','KAS','PAXG','XAUT','USD','BGB','MNT'}

candidates = []
seen = set()

# Toobit priority
for t in toobit_usdt:
    sym = t['s']; base = sym.replace('USDT','')
    if base in BLACKLIST or base in seen: continue
    seen.add(base)
    cgk = cgk_by_sym.get(base)
    mc = cgk.get('market_cap') if cgk else None
    if mc is not None and mc > 200_000_000: continue  # skip big caps
    candidates.append({
        'sym': sym, 'gate_pair': f'{base}_USDT',
        'mc': mc, 'qv_24h': float(t.get('qv', 0) or 0),
        'last': float(t.get('c', 0) or 0), 'source': 'toobit',
    })

# Gate.io add
for t in gate_usdt:
    base = t['currency_pair'].replace('_USDT','')
    if base in BLACKLIST or base in seen: continue
    qv = float(t.get('quote_volume', 0) or 0)
    if qv < 500_000: continue
    cgk = cgk_by_sym.get(base)
    mc = cgk.get('market_cap') if cgk else None
    if mc is not None and mc > 200_000_000: continue
    seen.add(base)
    candidates.append({
        'sym': base + 'USDT', 'gate_pair': t['currency_pair'],
        'mc': mc, 'qv_24h': qv, 'last': float(t.get('last', 0) or 0), 'source': 'gate',
    })

print(f'\nTotal candidates: {len(candidates)}')
print(f'With MC < $50M: {sum(1 for c in candidates if c["mc"] and c["mc"] < 50_000_000)}')
print(f'With MC < $100M: {sum(1 for c in candidates if c["mc"] and c["mc"] < 100_000_000)}')
print(f'No MC known: {sum(1 for c in candidates if c["mc"] is None)}')

with open(f'{CACHE}/candidates_v3.json', 'w') as f:
    json.dump(candidates, f, indent=2)
print(f'Saved to {CACHE}/candidates_v3.json')
