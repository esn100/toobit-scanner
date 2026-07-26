"""
Whale Intel - comprehensive on-chain + Twitter whale tracking.

SOURCES (in priority order):
1. Whale Alert API (broad coverage, requires free API key)
2. GeckoTerminal pool data (FREE, real-time DEX info)
3. 1inch API (price/liquidity reference, requires API key for some)
4. CoinGecko community + dev metrics
5. DeBank-style holder concentration (via GeckoTerminal pools + LP analysis)
6. Twitter via syndication API (no auth, rate limited)

HOLDER CONCENTRATION DETECTION (no direct on-chain holder list):
- Total liquidity in DEX pools (low = risky)
- Volume/liquidity ratio (high = active trading)
- Pool creation age (new pools = potential rugpull)
- Top trader concentration via repeated address patterns

TWITTER WHALE ALERTS:
- @whale_alert_io (Whale Alert official)
- @lookonchain (on-chain alpha)
- @DefiIgnas, @cmsintern, @Route2FI

This module is fault-tolerant: works with any subset of sources available.
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import html
import statistics
from collections import Counter

CACHE = '/home/user/toobit-scanner/data/whale_intel_cache.json'


def _http_get_json(url: str, timeout: int = 10, headers: dict = None) -> Optional[any]:
    try:
        h = {'User-Agent': 'Mozilla/5.0'}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def _http_get_text(url: str, timeout: int = 10) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception:
        return None


# ============================================================
# 1. GECKOTERMINAL - DEX Pool & Holder Concentration (FREE)
# ============================================================
def get_geckoterminal_data(contract: str, network: str = 'bsc') -> Dict:
    """
    Get token info + all pools from GeckoTerminal.
    Returns: liquidity, volume, top pools, price.
    """
    result = {
        'total_liquidity_usd': 0,
        'total_volume_24h': 0,
        'total_volume_1h': 0,
        'pools': [],
        'price_usd': 0,
        'pool_count': 0,
        'top_pool_liquidity': 0,
    }
    # Get token info
    info = _http_get_json(f'https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{contract}', timeout=10)
    if info and 'data' in info:
        attrs = info['data'].get('attributes', {})
        result['price_usd'] = float(attrs.get('price_usd', 0) or 0)
    # Get pools
    pools = _http_get_json(f'https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{contract}/pools?page=1', timeout=10)
    if pools and 'data' in pools:
        for p in pools['data'][:10]:
            a = p.get('attributes', {})
            reserve = float(a.get('reserve_in_usd', 0) or 0)
            vol_24h = float(a.get('volume_usd', {}).get('h24', 0) or 0)
            vol_1h = float(a.get('volume_usd', {}).get('h1', 0) or 0)
            result['pools'].append({
                'name': a.get('name', ''),
                'reserve_usd': reserve,
                'vol_24h': vol_24h,
                'vol_1h': vol_1h,
                'price_change_24h': a.get('price_change_percentage', {}).get('h24', 0),
            })
            result['total_liquidity_usd'] += reserve
            result['total_volume_24h'] += vol_24h
            result['total_volume_1h'] += vol_1h
        result['pool_count'] = len(result['pools'])
        if result['pools']:
            result['top_pool_liquidity'] = max(p['reserve_usd'] for p in result['pools'])
    return result


# ============================================================
# 2. HOLDER CONCENTRATION PROXY
# ============================================================
def estimate_holder_concentration(contract: str, network: str = 'bsc') -> Dict:
    """
    Estimate holder concentration from public DEX data.
    Real holder lists require paid APIs, but we can proxy via:
    - Volume/liquidity ratio (high = many active traders = distributed)
    - Number of pools (more pools = more holders)
    - Pool age (new pools = potential rugpull)
    - LP locked status (if available)

    Returns: {concentration_score: 0-100, risk_level, metrics}
    """
    data = get_geckoterminal_data(contract, network)
    result = {
        'concentration_score': 50,  # 0=very distributed, 100=highly concentrated
        'risk_level': 'unknown',  # 'low', 'medium', 'high', 'critical'
        'liquidity_health': 'unknown',
        'metrics': data,
    }
    if data['pool_count'] == 0:
        result['risk_level'] = 'critical'
        result['liquidity_health'] = 'no_liquidity'
        return result
    liq = data['total_liquidity_usd']
    vol_24h = data['total_volume_24h']
    # Liquidity health
    if liq < 1000:
        result['liquidity_health'] = 'critical'  # rugpull territory
        result['concentration_score'] = 95
    elif liq < 10_000:
        result['liquidity_health'] = 'very_low'
        result['concentration_score'] = 80
    elif liq < 100_000:
        result['liquidity_health'] = 'low'
        result['concentration_score'] = 60
    elif liq < 1_000_000:
        result['liquidity_health'] = 'medium'
        result['concentration_score'] = 40
    else:
        result['liquidity_health'] = 'good'
        result['concentration_score'] = 20
    # Volume/liquidity ratio (turnover)
    if liq > 0 and vol_24h > 0:
        turnover = vol_24h / liq
        if turnover > 5:
            # Very high turnover = could be wash trading
            result['concentration_score'] = min(95, result['concentration_score'] + 20)
        elif turnover > 2:
            result['concentration_score'] = min(90, result['concentration_score'] + 10)
        elif turnover < 0.1:
            # Very low turnover = illiquid, concentrated holders
            result['concentration_score'] = min(95, result['concentration_score'] + 15)
    # Pool count (more pools = more distributed)
    if data['pool_count'] >= 5:
        result['concentration_score'] = max(0, result['concentration_score'] - 20)
    elif data['pool_count'] >= 3:
        result['concentration_score'] = max(0, result['concentration_score'] - 10)
    elif data['pool_count'] == 1:
        result['concentration_score'] = min(95, result['concentration_score'] + 10)
    # Risk level
    if result['concentration_score'] >= 80:
        result['risk_level'] = 'critical'
    elif result['concentration_score'] >= 60:
        result['risk_level'] = 'high'
    elif result['concentration_score'] >= 40:
        result['risk_level'] = 'medium'
    else:
        result['risk_level'] = 'low'
    return result


# ============================================================
# 3. TWITTER SYNDICATION - No auth, just public syndication API
# ============================================================
def get_twitter_syndication(handle: str, max_tweets: int = 10) -> List[Dict]:
    """
    Get recent tweets via Twitter's public syndication endpoint.
    No auth required but rate-limited (use sparingly).
    """
    handle = handle.lstrip('@')
    url = f'https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}'
    data = _http_get_text(url, timeout=10)
    if not data or len(data) < 1000:
        return []
    tweets = []
    # Parse HTML for tweet content
    # Pattern: <p class="...">tweet text</p> with timestamps
    text_pattern = re.compile(r'<p[^>]*class="[^"]*"[^>]*>([^<]+(?:<[^>]+>[^<]*)*)</p>')
    for match in text_pattern.finditer(data):
        text = re.sub(r'<[^>]+>', '', match.group(1))
        text = html.unescape(text).strip()
        if text and len(text) > 20 and not text.startswith('http'):
            tweets.append({
                'text': text[:280],
                'handle': handle,
            })
            if len(tweets) >= max_tweets:
                break
    # Try alternative parsing if first didn't work
    if not tweets:
        # Look for <a class="timeline-Tweet-text">...</a>
        text_pattern2 = re.compile(
            r'<p[^>]*class="[^"]*TweetText[^"]*"[^>]*>(.*?)</p>',
            re.DOTALL
        )
        for match in text_pattern2.finditer(data):
            text = re.sub(r'<[^>]+>', '', match.group(1))
            text = html.unescape(text).strip()
            if text and len(text) > 10:
                tweets.append({'text': text[:280], 'handle': handle})
                if len(tweets) >= max_tweets:
                    break
    return tweets


def get_whale_twitter_mentions(symbol: str, hours: int = 24) -> List[Dict]:
    """
    Search whale-related Twitter accounts for mentions of the symbol.
    Uses syndication API (no auth) - rate limited.
    """
    base = symbol.replace('USDT', '').lower()
    # Whale/informational accounts to check
    accounts = [
        'whale_alert_io',  # Whale Alert
        'lookonchain',     # On-chain alpha
        'DefiIgnas',
        'cmsintern',
        'Route2FI',
        'spotonchain',     # Another on-chain tracker
    ]
    mentions = []
    now = datetime.now(timezone.utc)
    for handle in accounts:
        try:
            tweets = get_twitter_syndication(handle, max_tweets=5)
            for t in tweets:
                text = t.get('text', '').lower()
                # Match: $SYN, SYN, synapse
                if (f'${base}' in text or f' {base} ' in text or
                    f' {base},' in text or f' {base}.' in text or
                    f' {base}!' in text or f' {base}/' in text or
                    f'{base}_usdt' in text or
                    'synapse' in text and base == 'syn'):
                    mentions.append({
                        'handle': handle,
                        'text': t.get('text', '')[:200],
                        'source': 'twitter_syndication',
                    })
        except Exception:
            continue
        time.sleep(1.0)  # rate limit
    return mentions


# ============================================================
# 4. WHALE ALERT API (when key available)
# ============================================================
def get_whale_alert_recent(symbol: str, hours: int = 24, min_value: int = 100_000) -> List[Dict]:
    """Whale Alert API for real whale transactions."""
    key = os.environ.get('WHALE_ALERT_API_KEY')
    if not key:
        # Try config file
        cfg_path = '/home/user/toobit-scanner/data/whale_config.json'
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                key = cfg.get('WHALE_ALERT_API_KEY')
            except Exception:
                pass
    if not key:
        return []
    base = symbol.replace('USDT', '').lower()
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    url = (f'https://api.whale-alert.io/v1/transactions'
           f'?api_key={key}&currency={base}&min_value={min_value}&start={cutoff}')
    data = _http_get_json(url, timeout=15)
    if not data or 'transactions' not in data:
        return []
    txs = []
    for tx in data.get('transactions', []):
        try:
            txs.append({
                'amount_usd': float(tx.get('amount_usd', 0)),
                'from': tx.get('from', {}).get('address', ''),
                'to': tx.get('to', {}).get('address', ''),
                'from_owner': tx.get('from', {}).get('owner', 'unknown'),
                'to_owner': tx.get('to', {}).get('owner', 'unknown'),
                'blockchain': tx.get('blockchain', ''),
                'symbol': tx.get('symbol', ''),
                'hash': tx.get('hash', ''),
                'timestamp': tx.get('timestamp', 0),
            })
        except Exception:
            continue
    return txs


# ============================================================
# 5. ON-CHAIN TRANSFER DETECTION (BSC RPC, fallback)
# ============================================================
def get_bsc_transfers(contract: str, from_block: int = None, max_blocks: int = 5000) -> List[Dict]:
    """Get BEP20 Transfer events from BSC RPC."""
    if from_block is None:
        # Get latest block
        from .whale_tracker import _rpc_call, BSC_RPCS
        latest = _rpc_call(BSC_RPCS, 'eth_blockNumber', [])
        if not latest:
            return []
        latest_int = int(latest, 16)
        from_block = latest_int - max_blocks
    # Note: public RPCs often limit getLogs range
    transfer_sig = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
    from .whale_tracker import _rpc_call, BSC_RPCS
    latest = _rpc_call(BSC_RPCS, 'eth_blockNumber', [])
    if not latest:
        return []
    latest_int = int(latest, 16)
    logs = _rpc_call(BSC_RPCS, 'eth_getLogs', [{
        'fromBlock': hex(from_block),
        'toBlock': hex(latest_int),
        'address': contract,
        'topics': [transfer_sig]
    }], timeout=20)
    if not logs or not isinstance(logs, list):
        return []
    transfers = []
    for log in logs:
        try:
            from_addr = '0x' + log['topics'][1][26:].lower()
            to_addr = '0x' + log['topics'][2][26:].lower()
            data = log.get('data', '0x0')
            value = int(data, 16) / (10 ** 18)
            transfers.append({
                'block': int(log.get('blockNumber', '0x0'), 16),
                'from': from_addr,
                'to': to_addr,
                'value': value,
                'tx_hash': log.get('transactionHash', ''),
            })
        except Exception:
            continue
    return transfers


# ============================================================
# 6. COMBINED WHALE INTEL
# ============================================================
def get_whale_intel(symbol: str, contract: str = None, network: str = 'bsc') -> Dict:
    """
    Combined whale intelligence from all sources.
    Returns: {strength, signal_type, has_signal, sources, details}
    """
    result = {
        'symbol': symbol,
        'has_signal': False,
        'signal_strength': 0,
        'signal_type': None,
        'sources_active': [],
        'details': {},
    }
    score = 0
    # 1. GeckoTerminal (always works)
    gt = get_geckoterminal_data(contract, network) if contract else None
    if gt:
        result['sources_active'].append('geckoterminal')
        result['details']['geckoterminal'] = gt
        # Volume surge
        if gt['total_volume_1h'] > 50_000:
            score += 25
            result['signal_type'] = 'volume_surge'
        elif gt['total_volume_1h'] > 10_000:
            score += 15
        # Liquidity check
        if gt['total_liquidity_usd'] < 1000:
            score -= 20  # too risky
            result['details']['liquidity_warning'] = 'critical_low'
    # 2. Holder concentration proxy
    hc = estimate_holder_concentration(contract, network) if contract else None
    if hc:
        result['sources_active'].append('holder_proxy')
        result['details']['holder_concentration'] = hc
        # Veto if too concentrated
        if hc['risk_level'] == 'critical':
            score -= 30
            result['details']['veto_reason'] = 'concentrated_holders'
    # 3. Whale Alert API
    wa_txs = get_whale_alert_recent(symbol, hours=24, min_value=100_000)
    if wa_txs:
        result['sources_active'].append('whale_alert')
        result['details']['whale_alert'] = {
            'count': len(wa_txs),
            'total_usd': sum(t.get('amount_usd', 0) for t in wa_txs),
        }
        score += min(40, len(wa_txs) * 10)
        if not result['signal_type']:
            result['signal_type'] = 'whale_alert'
    # 4. Twitter syndication
    tw_mentions = get_whale_twitter_mentions(symbol)
    if tw_mentions:
        result['sources_active'].append('twitter')
        result['details']['twitter_mentions'] = tw_mentions[:5]
        score += min(20, len(tw_mentions) * 5)
    # 5. On-chain transfers (BSC)
    if contract:
        try:
            transfers = get_bsc_transfers(contract, max_blocks=1000)
            if transfers:
                result['sources_active'].append('on_chain')
                # Count large transfers
                large_count = sum(1 for t in transfers if t['value'] > 1000)
                result['details']['on_chain'] = {
                    'count_1k': len(transfers),
                    'large_count': large_count,
                }
                if large_count >= 5:
                    score += 25
                    if not result['signal_type']:
                        result['signal_type'] = 'on_chain_activity'
        except Exception:
            pass
    # Final score
    result['signal_strength'] = max(0, min(100, score))
    result['has_signal'] = result['signal_strength'] >= 40
    return result


# ============================================================
# 7. SAVE / LOAD CACHE
# ============================================================
def _load_cache() -> dict:
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'whale_intel': {},
        'last_scan': None,
    }


def _save_cache(cache: dict) -> None:
    with open(CACHE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def scan_all_intel(symbols_contracts: Dict[str, str], verbose: bool = False) -> Dict[str, Dict]:
    """
    Scan all symbols for whale intel.
    symbols_contracts: {symbol: contract_address}
    """
    cache = _load_cache()
    now = datetime.now(timezone.utc)
    for sym, contract in symbols_contracts.items():
        try:
            if verbose:
                print(f'  Scanning {sym} ({contract[:10]}...)')
            result = get_whale_intel(sym, contract)
            cache['whale_intel'][sym] = {
                **result,
                'ts': now.isoformat(),
            }
            if verbose:
                print(f'    strength: {result["signal_strength"]}  sources: {result["sources_active"]}')
        except Exception as e:
            if verbose:
                print(f'    error: {e}')
        time.sleep(0.5)
    cache['last_scan'] = now.isoformat()
    _save_cache(cache)
    return cache


if __name__ == '__main__':
    import sys
    from .repeater_config import REPEATERS
    # Map symbols to contracts (hardcoded for now)
    contracts = {
        'SYNUSDT': '0xa4080f1778e69467e905b8d6f72f6e441f9e9484',
        # Add others as needed
    }
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        contract = contracts.get(sym)
        result = get_whale_intel(sym, contract)
        print(json.dumps(result, indent=2, default=str))
    else:
        print('Scanning all known repeaters...')
        cache = scan_all_intel(contracts, verbose=True)
        print()
        for sym, data in cache.get('whale_intel', {}).items():
            print(f'{sym}: strength={data.get("signal_strength")} sources={data.get("sources_active")}')
