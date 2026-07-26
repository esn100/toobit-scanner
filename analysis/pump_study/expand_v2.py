"""Fetch cgk page by page with delay"""
import json, time, urllib.request

CACHE = '/home/user/toobit-scanner/analysis/pump_study/cache'

def http_get(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except: return None

# 6 pages of cgk = 1500 coins
all_cgk = []
for page in range(1, 7):
    raw = http_get(f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}')
    if not raw:
        print(f'  page {page}: FAILED, retry...')
        time.sleep(30)
        raw = http_get(f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}')
    if raw:
        try:
            data = json.loads(raw)
            all_cgk.extend(data)
            print(f'  page {page}: {len(data)} coins')
        except:
            print(f'  page {page}: parse error')
    time.sleep(3)

with open(f'{CACHE}/cgk_full.json', 'w') as f:
    json.dump(all_cgk, f)
print(f'Total: {len(all_cgk)} coins saved to cgk_full.json')
