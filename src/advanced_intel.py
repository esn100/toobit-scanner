"""
Advanced Intelligence Module - 6 high-impact features to boost win rate to 80%+.

Why these matter for small-cap pump detection:
1. Order Book Imbalance (OBI) - real-time buy/sell pressure
2. Long/Short Ratio - detects crowded trades that precede squeezes
3. Taker Buy/Sell Ratio - aggressive vs passive buyers
4. Smart Money Flow - large wallet movements across exchanges
5. Liquidation Heatmap - where liquidations cluster (signals volatility)
6. Cross-Exchange Arbitrage - price discrepancies = real demand

Each feature returns 0-100 score. Combined they boost pattern win rate.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math
import statistics

CACHE_DIR = '/home/user/toobit-scanner/data'
ADV_INTEL_CACHE = f'{CACHE_DIR}/advanced_intel_cache.json'


def _http_get_json(url: str, timeout: int = 10) -> Optional[any]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def _load_cache() -> dict:
    if os.path.exists(ADV_INTEL_CACHE):
        try:
            with open(ADV_INTEL_CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'obi': {},           # symbol -> {imbalance, ts}
        'ls_ratio': {},      # symbol -> {ratio, ts}
        'taker_ratio': {},   # symbol -> {buy_ratio, ts}
        'smart_money': {},   # symbol -> {flow_score, ts}
        'liquidations': {},  # symbol -> {density, ts}
        'arb_spread': {},    # symbol -> {spread_pct, ts}
        'last_scan': None,
    }


def _save_cache(cache: dict) -> None:
    with open(ADV_INTEL_CACHE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


# ============================================================
# 1. ORDER BOOK IMBALANCE (OBI) - Real-time buy/sell pressure
# ============================================================
def get_order_book_imbalance(symbol: str, exchanges: List[str] = None) -> Dict:
    """
    Get order book imbalance from multiple exchanges.
    OBI > 0.5 = more buy pressure (bullish)
    OBI < 0.5 = more sell pressure (bearish)
    Returns: {imbalance, spread, depth_ratio, sources}
    """
    if exchanges is None:
        exchanges = ['gateio', 'kucoin', 'mexc', 'htx']
    result = {
        'imbalance': 0.5,
        'spread_pct': 0,
        'bid_depth_usd': 0,
        'ask_depth_usd': 0,
        'sources': [],
    }
    imbalances = []
    spreads = []
    bid_depths = []
    ask_depths = []

    for ex in exchanges:
        try:
            if ex == 'gateio':
                base = symbol.replace('USDT', '').lower() + '_usdt'
                data = _http_get_json(f'https://api.gateio.ws/api/v4/spot/order_book?currency_pair={base}&limit=20', timeout=5)
                if data and 'bids' in data and 'asks' in data:
                    bid_vol = sum(float(b[1]) * float(b[0]) for b in data['bids'][:20])
                    ask_vol = sum(float(a[1]) * float(a[0]) for a in data['asks'][:20])
                    best_bid = float(data['bids'][0][0]) if data['bids'] else 0
                    best_ask = float(data['asks'][0][0]) if data['asks'] else 0
                    if bid_vol + ask_vol > 0:
                        imbalances.append(bid_vol / (bid_vol + ask_vol))
                    if best_bid > 0:
                        spreads.append((best_ask - best_bid) / best_bid * 100)
                    bid_depths.append(bid_vol)
                    ask_depths.append(ask_vol)
                    result['sources'].append('gateio')
            elif ex == 'kucoin':
                data = _http_get_json(f'https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={symbol}', timeout=5)
                if data and 'data' in data:
                    d = data['data']
                    bids = d.get('bids', [])
                    asks = d.get('asks', [])
                    bid_vol = sum(float(b[1]) * float(b[0]) for b in bids[:20])
                    ask_vol = sum(float(a[1]) * float(a[0]) for a in asks[:20])
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    if bid_vol + ask_vol > 0:
                        imbalances.append(bid_vol / (bid_vol + ask_vol))
                    if best_bid > 0:
                        spreads.append((best_ask - best_bid) / best_bid * 100)
                    bid_depths.append(bid_vol)
                    ask_depths.append(ask_vol)
                    result['sources'].append('kucoin')
            elif ex == 'mexc':
                data = _http_get_json(f'https://api.mexc.com/api/v3/depth?symbol={symbol}&limit=20', timeout=5)
                if data and 'bids' in data and 'asks' in data:
                    bid_vol = sum(float(b[1]) * float(b[0]) for b in data['bids'][:20])
                    ask_vol = sum(float(a[1]) * float(a[0]) for a in data['asks'][:20])
                    best_bid = float(data['bids'][0][0]) if data['bids'] else 0
                    best_ask = float(data['asks'][0][0]) if data['asks'] else 0
                    if bid_vol + ask_vol > 0:
                        imbalances.append(bid_vol / (bid_vol + ask_vol))
                    if best_bid > 0:
                        spreads.append((best_ask - best_bid) / best_bid * 100)
                    bid_depths.append(bid_vol)
                    ask_depths.append(ask_vol)
                    result['sources'].append('mexc')
            elif ex == 'htx':
                # HTX uses different symbol format
                ht_sym = symbol.lower()
                data = _http_get_json(f'https://api.huobi.pro/market/depth?symbol={ht_sym}&depth=20&type=step0', timeout=5)
                if data and 'tick' in data:
                    bids = data['tick'].get('bids', [])
                    asks = data['tick'].get('asks', [])
                    bid_vol = sum(float(b[1]) * float(b[0]) for b in bids[:20])
                    ask_vol = sum(float(a[1]) * float(a[0]) for a in asks[:20])
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    if bid_vol + ask_vol > 0:
                        imbalances.append(bid_vol / (bid_vol + ask_vol))
                    if best_bid > 0:
                        spreads.append((best_ask - best_bid) / best_bid * 100)
                    bid_depths.append(bid_vol)
                    ask_depths.append(ask_vol)
                    result['sources'].append('htx')
        except Exception:
            continue
        time.sleep(0.2)

    if imbalances:
        result['imbalance'] = sum(imbalances) / len(imbalances)
    if spreads:
        result['spread_pct'] = sum(spreads) / len(spreads)
    if bid_depths:
        result['bid_depth_usd'] = sum(bid_depths)
    if ask_depths:
        result['ask_depth_usd'] = sum(ask_depths)
    return result


# ============================================================
# 2. LONG/SHORT RATIO - Crowded trade detection
# ============================================================
def get_long_short_ratio(symbol: str) -> Dict:
    """
    Get long/short ratio from multiple exchanges.
    Extreme ratios (> 0.7 or < 0.3) often precede squeezes.
    """
    result = {
        'ratio': 0.5,  # 0.5 = balanced, 0.7 = 70% long
        'signal_type': 'neutral',  # 'squeeze_long', 'squeeze_short', 'neutral'
        'sources': [],
    }
    base = symbol.replace('USDT', '')
    # Gate.io futures
    try:
        data = _http_get_json(f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}', timeout=5)
        if data and 'funding_rate' in data:
            funding = float(data.get('funding_rate', 0) or 0)
            # Funding rate is correlated with long/short ratio
            if funding > 0.0005:  # > 0.05%
                result['signal_type'] = 'shorts_paying'  # shorts paying longs = market is long
            elif funding < -0.0005:
                result['signal_type'] = 'longs_paying'
            result['funding_rate'] = funding
            result['sources'].append('gateio_futures')
    except Exception:
        pass
    # HTX futures
    try:
        ht_sym = symbol.lower()
        data = _http_get_json(f'https://api.huobi.pro/v1/contract/exchange_turnover?symbol={ht_sym}&period=1day', timeout=5)
        if data and 'data' in data and len(data['data']) > 0:
            latest = data['data'][-1]
            # turnover fields often include buy/sell ratio
            if 'buy' in latest and 'sell' in latest:
                buy = float(latest['buy'])
                sell = float(latest['sell'])
                if buy + sell > 0:
                    result['ratio'] = buy / (buy + sell)
                    result['sources'].append('htx_futures')
    except Exception:
        pass
    # OKX ratio
    try:
        # OKX public ratio endpoint
        data = _http_get_json(
            f'https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio'
            f'?ccy={base}&period=1H',
            timeout=5
        )
        if data and 'data' in data and len(data['data']) > 0:
            ratio = float(data['data'][0].get('ratio', 0.5))
            result['ratio'] = ratio
            result['sources'].append('okx')
    except Exception:
        pass
    # Determine signal
    if result['ratio'] >= 0.65:
        result['signal_type'] = 'crowded_long'  # potential short squeeze
    elif result['ratio'] <= 0.35:
        result['signal_type'] = 'crowded_short'  # potential long squeeze
    return result


# ============================================================
# 3. TAKER BUY/SELL RATIO - Aggressive buyer detection
# ============================================================
def get_taker_buy_sell_ratio(symbol: str, hours: int = 4) -> Dict:
    """
    Get taker buy/sell ratio from recent trades.
    Taker buys = aggressive market orders (bullish)
    Taker sells = aggressive market orders (bearish)
    """
    result = {
        'buy_volume': 0,
        'sell_volume': 0,
        'buy_ratio': 0.5,
        'large_buy_count': 0,  # >$5K
        'large_sell_count': 0,
        'largest_buy_usd': 0,
        'largest_sell_usd': 0,
    }
    base = symbol.replace('USDT', '').lower()
    # Get recent trades from Gate.io
    try:
        data = _http_get_json(f'https://api.gateio.ws/api/v4/spot/trades?currency_pair={base}_usdt&limit=500', timeout=10)
        if not data or not isinstance(data, list):
            return result
        # Filter by time (last 4h)
        now = datetime.now(timezone.utc)
        cutoff_ts = int((now - timedelta(hours=hours)).timestamp())
        buy_vol = 0
        sell_vol = 0
        large_buys = 0
        large_sells = 0
        for trade in data:
            try:
                ts = int(trade.get('time', '0') if 'time' in trade else 0)
                if ts > 1e12:  # ms
                    ts = ts // 1000
                if ts < cutoff_ts:
                    continue
                amount = float(trade.get('amount', 0))
                price = float(trade.get('price', 0))
                value = amount * price
                side = trade.get('side', '')
                if side == 'buy':
                    buy_vol += value
                    if value > 5000:
                        large_buys += 1
                        if value > result['largest_buy_usd']:
                            result['largest_buy_usd'] = value
                else:
                    sell_vol += value
                    if value > 5000:
                        large_sells += 1
                        if value > result['largest_sell_usd']:
                            result['largest_sell_usd'] = value
            except Exception:
                continue
        result['buy_volume'] = buy_vol
        result['sell_volume'] = sell_vol
        if buy_vol + sell_vol > 0:
            result['buy_ratio'] = buy_vol / (buy_vol + sell_vol)
        result['large_buy_count'] = large_buys
        result['large_sell_count'] = large_sells
    except Exception:
        pass
    return result


# ============================================================
# 4. SMART MONEY FLOW - Large wallet movement proxy
# ============================================================
def get_smart_money_flow(symbol: str, hours: int = 4) -> Dict:
    """
    Detect smart money flow by analyzing:
    - Large trades (whale buys/sells)
    - Cross-exchange arbitrage
    - Liquidity changes
    """
    result = {
        'whale_buy_count': 0,
        'whale_sell_count': 0,
        'whale_net_flow_usd': 0,
        'largest_whale_trade': 0,
        'arb_signal': False,  # Cross-exchange arb detected
        'liquidity_change_pct': 0,
    }
    taker = get_taker_buy_sell_ratio(symbol, hours=hours)
    # Count whale trades
    result['whale_buy_count'] = taker.get('large_buy_count', 0)
    result['whale_sell_count'] = taker.get('large_sell_count', 0)
    result['whale_net_flow_usd'] = (
        sum([taker.get('largest_buy_usd', 0) for _ in range(result['whale_buy_count'])]) -
        sum([taker.get('largest_sell_usd', 0) for _ in range(result['whale_sell_count'])])
    )
    # Cross-exchange arbitrage
    prices = {}
    try:
        # Gate.io
        d = _http_get_json('https://api.gateio.ws/api/v4/spot/tickers?currency_pair=' + symbol.replace('USDT', '').lower() + '_usdt', timeout=5)
        if d and isinstance(d, list) and len(d) > 0:
            prices['gateio'] = float(d[0].get('last', 0))
        # MEXC
        d = _http_get_json('https://api.mexc.com/api/v3/ticker/price?symbol=' + symbol, timeout=5)
        if d and 'price' in d:
            prices['mexc'] = float(d['price'])
        # HTX
        d = _http_get_json('https://api.huobi.pro/market/detail/merged?symbol=' + symbol.lower(), timeout=5)
        if d and 'tick' in d:
            prices['htx'] = float(d['tick'].get('close', 0))
    except Exception:
        pass
    if len(prices) >= 2:
        price_vals = list(prices.values())
        if all(p > 0 for p in price_vals):
            spread = (max(price_vals) - min(price_vals)) / min(price_vals) * 100
            if spread > 1.0:  # >1% spread
                result['arb_signal'] = True
                result['arb_spread_pct'] = spread
    return result


# ============================================================
# 5. LIQUIDATION HEATMAP - Volatility prediction
# ============================================================
def get_liquidation_density(symbol: str) -> Dict:
    """
    Estimate liquidation density from OI and price.
    High OI + tight price range = cascade risk.
    """
    result = {
        'oi_usd': 0,
        'oi_change_24h_pct': 0,
        'liquidation_risk': 'low',  # 'low', 'medium', 'high', 'extreme'
        'cascade_distance_pct': 0,
    }
    # Gate.io futures
    try:
        data = _http_get_json(f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}', timeout=5)
        if data:
            mark = float(data.get('mark_price', 0) or 0)
            oi_size = float(data.get('total_size', 0) or 0)
            oi_usd = oi_size * mark
            result['oi_usd'] = oi_usd
            # 24h change
            try:
                stats = _http_get_json(f'https://api.gateio.ws/api/v4/futures/usdt/contract_stats?contract={symbol}&interval=1d', timeout=5)
                if stats:
                    result['oi_change_24h_pct'] = float(stats.get('open_interest_usd_change_pct', 0) or 0)
            except Exception:
                pass
    except Exception:
        pass
    # Estimate liquidation risk
    oi = result['oi_usd']
    if oi > 50_000_000:
        result['liquidation_risk'] = 'extreme'
    elif oi > 10_000_000:
        result['liquidation_risk'] = 'high'
    elif oi > 1_000_000:
        result['liquidation_risk'] = 'medium'
    else:
        result['liquidation_risk'] = 'low'
    return result


# ============================================================
# 6. CROSS-EXCHANGE ARBITRAGE - Real demand signal
# ============================================================
def get_cross_exchange_spread(symbol: str) -> Dict:
    """
    Compare prices across exchanges to detect real demand.
    Premium on one exchange = aggressive buying.
    """
    result = {
        'gateio': 0,
        'mexc': 0,
        'htx': 0,
        'kucoin': 0,
        'max_premium_pct': 0,  # % above min
        'signal': 'neutral',  # 'premium', 'discount', 'neutral'
    }
    base = symbol.replace('USDT', '').lower()
    prices = {}
    try:
        d = _http_get_json(f'https://api.gateio.ws/api/v4/spot/tickers?currency_pair={base}_usdt', timeout=5)
        if d and isinstance(d, list) and len(d) > 0:
            p = float(d[0].get('last', 0))
            result['gateio'] = p
            prices['gateio'] = p
    except Exception:
        pass
    try:
        d = _http_get_json(f'https://api.mexc.com/api/v3/ticker/price?symbol={symbol}', timeout=5)
        if d and 'price' in d:
            p = float(d['price'])
            result['mexc'] = p
            prices['mexc'] = p
    except Exception:
        pass
    try:
        d = _http_get_json(f'https://api.huobi.pro/market/detail/merged?symbol={base}usdt', timeout=5)
        if d and 'tick' in d:
            p = float(d['tick'].get('close', 0))
            result['htx'] = p
            prices['htx'] = p
    except Exception:
        pass
    try:
        d = _http_get_json(f'https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}', timeout=5)
        if d and 'data' in d:
            p = float(d['data'].get('price', 0))
            result['kucoin'] = p
            prices['kucoin'] = p
    except Exception:
        pass
    # Calculate premium
    valid_prices = [p for p in prices.values() if p > 0]
    if len(valid_prices) >= 2:
        min_p = min(valid_prices)
        max_p = max(valid_prices)
        premium = (max_p - min_p) / min_p * 100
        result['max_premium_pct'] = premium
        if premium > 0.5:
            result['signal'] = 'premium'
        elif premium < -0.5:
            result['signal'] = 'discount'
    return result


# ============================================================
# COMBINED SCORING
# ============================================================
def get_advanced_intel(symbol: str, verbose: bool = False) -> Dict:
    """
    Run all 6 advanced intelligence features for a symbol.
    Returns combined score + breakdown.
    """
    result = {
        'symbol': symbol,
        'has_signal': False,
        'signal_strength': 0,
        'components': {},
    }
    score = 0
    max_score = 100
    # 1. Order book imbalance (weight: 20)
    if verbose: print(f'  1. OBI...')
    obi = get_order_book_imbalance(symbol)
    result['components']['obi'] = obi
    imb = obi.get('imbalance', 0.5)
    if imb > 0.6:  # Buy pressure
        score += 20
        result['components']['obi_signal'] = 'buy_pressure'
    elif imb > 0.55:
        score += 12
        result['components']['obi_signal'] = 'mild_buy'
    elif imb < 0.4:  # Sell pressure
        score -= 10
        result['components']['obi_signal'] = 'sell_pressure'
    else:
        result['components']['obi_signal'] = 'balanced'
    # 2. Long/short ratio (weight: 15)
    if verbose: print(f'  2. L/S ratio...')
    ls = get_long_short_ratio(symbol)
    result['components']['ls'] = ls
    ratio = ls.get('ratio', 0.5)
    funding = ls.get('funding_rate', 0)
    # Extreme L/S = potential squeeze
    if 0.65 <= ratio <= 0.75:
        score += 12  # mild crowded long
    elif ratio > 0.75:
        score += 18  # very crowded long = potential short squeeze
    elif 0.25 <= ratio <= 0.35:
        score += 12
    elif ratio < 0.25:
        score += 18  # potential long squeeze
    if abs(funding) > 0.001:  # 0.1%
        score += 5
    # 3. Taker buy/sell (weight: 20)
    if verbose: print(f'  3. Taker ratio...')
    taker = get_taker_buy_sell_ratio(symbol, hours=4)
    result['components']['taker'] = taker
    buy_ratio = taker.get('buy_ratio', 0.5)
    if buy_ratio > 0.6:
        score += 18
        result['components']['taker_signal'] = 'aggressive_buying'
    elif buy_ratio > 0.55:
        score += 10
        result['components']['taker_signal'] = 'mild_buying'
    elif buy_ratio < 0.4:
        score -= 8
        result['components']['taker_signal'] = 'aggressive_selling'
    else:
        result['components']['taker_signal'] = 'balanced'
    # 4. Smart money flow (weight: 20)
    if verbose: print(f'  4. Smart money...')
    smart = get_smart_money_flow(symbol, hours=4)
    result['components']['smart_money'] = smart
    whale_net = smart.get('whale_net_flow_usd', 0)
    if whale_net > 5000:
        score += 20
        result['components']['smart_signal'] = 'whale_accumulation'
    elif whale_net > 0:
        score += 10
        result['components']['smart_signal'] = 'mild_accumulation'
    elif whale_net < -5000:
        score -= 10
        result['components']['smart_signal'] = 'whale_distribution'
    else:
        result['components']['smart_signal'] = 'neutral'
    if smart.get('arb_signal'):
        score += 5  # Cross-exchange arb = real demand
    # 5. Liquidation density (weight: 10)
    if verbose: print(f'  5. Liquidations...')
    liq = get_liquidation_density(symbol)
    result['components']['liquidations'] = liq
    oi = liq.get('oi_usd', 0)
    oi_chg = liq.get('oi_change_24h_pct', 0)
    # Rising OI = new positions = real momentum
    if oi_chg > 10:
        score += 12
        result['components']['liq_signal'] = 'rising_oi'
    elif oi_chg > 0:
        score += 6
        result['components']['liq_signal'] = 'rising_oi_mild'
    elif oi_chg < -10:
        score -= 5
        result['components']['liq_signal'] = 'falling_oi'
    # 6. Cross-exchange spread (weight: 15)
    if verbose: print(f'  6. Cross-exchange...')
    arb = get_cross_exchange_spread(symbol)
    result['components']['arb'] = arb
    premium = arb.get('max_premium_pct', 0)
    if premium > 2.0:
        score += 15  # Big premium = strong demand
        result['components']['arb_signal'] = 'strong_premium'
    elif premium > 0.5:
        score += 8
        result['components']['arb_signal'] = 'mild_premium'
    else:
        result['components']['arb_signal'] = 'aligned'
    # Final score
    result['signal_strength'] = max(0, min(100, score))
    result['has_signal'] = result['signal_strength'] >= 40
    return result


def scan_all_advanced(symbols: List[str], verbose: bool = False) -> Dict[str, Dict]:
    """Scan multiple symbols for advanced intelligence."""
    cache = _load_cache()
    now = datetime.now(timezone.utc)
    for sym in symbols:
        try:
            if verbose:
                print(f'\nScanning {sym}...')
            result = get_advanced_intel(sym, verbose=verbose)
            cache['obi'][sym] = {'value': result['components'].get('obi', {}).get('imbalance', 0.5), 'ts': now.isoformat()}
            cache['ls_ratio'][sym] = {'value': result['components'].get('ls', {}).get('ratio', 0.5), 'ts': now.isoformat()}
            cache['taker_ratio'][sym] = {'value': result['components'].get('taker', {}).get('buy_ratio', 0.5), 'ts': now.isoformat()}
            cache['smart_money'][sym] = {'value': result['components'].get('smart_money', {}).get('whale_net_flow_usd', 0), 'ts': now.isoformat()}
            cache['liquidations'][sym] = {'value': result['components'].get('liquidations', {}).get('oi_change_24h_pct', 0), 'ts': now.isoformat()}
            cache['arb_spread'][sym] = {'value': result['components'].get('arb', {}).get('max_premium_pct', 0), 'ts': now.isoformat()}
        except Exception as e:
            if verbose:
                print(f'  Error: {e}')
    cache['last_scan'] = now.isoformat()
    _save_cache(cache)
    return cache


if __name__ == '__main__':
    import sys
    from .repeater_config import REPEATERS
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        result = get_advanced_intel(sym, verbose=True)
        print('\n' + json.dumps(result, indent=2, default=str)[:2000])
    else:
        symbols = list(REPEATERS.keys())[:3]  # test first 3
        print(f'Testing advanced intel on {symbols}...')
        cache = scan_all_advanced(symbols, verbose=True)
        print()
        for sym in symbols:
            result = get_advanced_intel(sym)
            print(f'{sym}: score={result["signal_strength"]}  '
                  f'has_signal={result["has_signal"]}')
