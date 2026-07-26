"""
Comprehensive TOP signal scan using all available tools.
Combines:
1. Repeater scanner (pre-pump pattern)
2. Breakout detector (technical)
3. Whale intel (GeckoTerminal liquidity)
4. CoinGecko sentiment
5. GitHub activity
"""
import json
import sys
from datetime import datetime, timezone
from typing import Dict

sys.path.insert(0, '/home/user/toobit-scanner')

from src.repeater_scanner import scan_repeater, REPEATERS
from src.repeater_config import REPEATERS as RC
from src.breakout_detector import detect_breakout
from src.whale_intel import get_whale_intel
from src.news_detector import detect_product_activity
import urllib.request

# Hardcoded contracts (from previous research)
CONTRACTS = {
    'SYNUSDT': '0xa4080f1778e69467e905b8d6f72f6e441f9e9484',
}

def get_current_price(symbol):
    try:
        req = urllib.request.Request(
            f'https://api.toobit.com/quote/v1/ticker/24hr?symbol={symbol}',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data:
            return {
                'price': float(data[0]['c']),
                'change_24h': float(data[0]['pcp']),
                'vol_24h': float(data[0]['qv']),
                'high_24h': float(data[0]['h']),
                'low_24h': float(data[0]['l']),
            }
    except:
        return None
    return None

print('='*80)
print('COMPREHENSIVE SCAN: All 12 repeaters + whale intel + technical')
print('='*80)
print()

results = []
for sym in REPEATERS:
    print(f'Scanning {sym}...')
    # 1. Current price
    price_data = get_current_price(sym)
    if not price_data:
        print(f'  Skipping (no price data)')
        continue

    # 2. Repeater pattern
    pattern = scan_repeater(sym)
    pattern_conf = pattern.get('pattern_confidence', 0)
    features = pattern.get('features', {})

    # 3. Breakout
    breakout = detect_breakout(sym)
    brk_score = breakout.get('score', 0)
    brk_type = breakout.get('type')

    # 4. Whale intel
    contract = CONTRACTS.get(sym)
    if contract:
        whale = get_whale_intel(sym, contract)
    else:
        whale = {'signal_strength': 0, 'sources_active': []}
    whale_score = whale.get('signal_strength', 0)

    # 5. News/product
    news = detect_product_activity(sym)
    news_score = news.get('signal_strength', 0)

    # 6. Combined score
    final_score = (
        pattern_conf * 0.35 +
        brk_score * 0.25 +
        whale_score * 0.25 +
        news_score * 0.15
    )

    results.append({
        'symbol': sym,
        'price': price_data,
        'pattern': {'confidence': pattern_conf, 'action': pattern.get('action'), 'features': features},
        'breakout': {'score': brk_score, 'type': brk_type},
        'whale': {'score': whale_score, 'sources': whale.get('sources_active', []),
                  'liquidity': whale.get('details', {}).get('geckoterminal', {}).get('total_liquidity_usd', 0)},
        'news': {'score': news_score, 'has_activity': news.get('has_product_activity', False)},
        'final_score': final_score,
    })

# Sort by final score
results.sort(key=lambda x: -x['final_score'])

print()
print('='*80)
print('RANKED SIGNALS (by combined score)')
print('='*80)
print()
print(f"{'#':<3} {'Symbol':<12} {'Price':>10} {'24h%':>7} {'Patt':>5} {'Brk':>5} {'Whale':>6} {'News':>5} {'SCORE':>6} | Verdict")
print('-'*100)

for i, r in enumerate(results, 1):
    p = r['price']
    pat = r['pattern']
    brk = r['breakout']
    whl = r['whale']
    news = r['news']
    score = r['final_score']

    # Verdict emoji
    if score >= 70:
        verdict = '🟢 STRONG BUY'
    elif score >= 55:
        verdict = '🟡 MODERATE'
    elif score >= 40:
        verdict = '⚪ WEAK'
    else:
        verdict = '🔴 SKIP'

    print(f"{i:<3} {r['symbol']:<12} {p['price']:>10.8f} {p['change_24h']:>+6.2f}% "
          f"{pat['confidence']:>4.0f}% {brk['score']:>4} {whl['score']:>5} {news['score']:>4} "
          f"{score:>5.1f}  | {verdict}")

# Top 3 detail
print()
print('='*80)
print('TOP 3 SIGNALS - DETAILED')
print('='*80)

for i, r in enumerate(results[:3], 1):
    sym = r['symbol']
    p = r['price']
    pat = r['pattern']
    cfg = RC[sym]

    print(f'\n{"#"*3} {i}. {sym} ({cfg["name"]}) - TP={cfg["tp_pct"]}% SL={cfg["sl_pct"]}%')
    print(f'  Price: {p["price"]:.8f}  24h: {p["change_24h"]:+.2f}%  Vol: ${p["vol_24h"]:,.0f}')

    # TP/SL prices
    tp_price = p['price'] * (1 + cfg['tp_pct']/100)
    sl_price = p['price'] * (1 - cfg['sl_pct']/100)
    print(f'  Entry: {p["price"]:.8f}')
    print(f'  TP:    {tp_price:.8f}  (+{cfg["tp_pct"]}%)')
    print(f'  SL:    {sl_price:.8f}  (-{cfg["sl_pct"]}%)')
    print(f'  R:R = 1:{cfg["tp_pct"]/cfg["sl_pct"]:.1f}')

    # Pattern details
    f = pat.get('features', {})
    if f:
        print(f'  Pattern: conf={pat["confidence"]:.0f}%, action={pat.get("action", "none")}')
        print(f'    rvol={f.get("rvol", 0):.2f}x  mom_3={f.get("mom_3", 0):+.2f}%  '
              f'RSI={f.get("rsi", 0):.0f}  flat={f.get("flat_hours", 0):.0f}h')

    # Breakout
    if r['breakout']['score'] > 0:
        print(f'  Breakout: score={r["breakout"]["score"]}  type={r["breakout"]["type"] or "-"}')

    # Whale
    if r['whale']['score'] > 0 or r['whale']['liquidity'] > 0:
        print(f'  Whale: score={r["whale"]["score"]}  liquidity=${r["whale"]["liquidity"]:,.0f}  '
              f'sources={r["whale"]["sources"]}')

    # News
    if r['news']['score'] > 0:
        print(f'  News: score={r["news"]["score"]}  activity={r["news"]["has_activity"]}')

    print(f'  COMBINED SCORE: {r["final_score"]:.1f}/100')

# Save to file
with open('/home/user/toobit-scanner/data/top_signals.json', 'w') as f:
    json.dump([{
        'symbol': r['symbol'],
        'price': r['price']['price'],
        'change_24h': r['price']['change_24h'],
        'vol_24h': r['price']['vol_24h'],
        'pattern_confidence': r['pattern']['confidence'],
        'breakout_score': r['breakout']['score'],
        'whale_score': r['whale']['score'],
        'news_score': r['news']['score'],
        'combined_score': r['final_score'],
    } for r in results], f, indent=2)

print()
print('Saved to /home/user/toobit-scanner/data/top_signals.json')
