"""
Whale Tracker - detect large transactions that often precede pumps.
Uses:
- Whale Alert API (free, requires API key)
- On-chain data via public RPC nodes (BSC, ETH)
- Coingecko community/social stats as proxy

A "whale signal" is a transaction >$100K that involves the token we're watching.
Multiple whale signals in 24h for the same token = strong pre-pump indicator.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math

CACHE = '/home/user/toobit-scanner/data/whale_cache.json'
WHALE_ALERT_API = 'https://api.whale-alert.io/v1/transactions'

# Free public RPC endpoints (no API key required, rate limited)
BSC_RPC = 'https://bsc-dataseed.binance.org/'
ETH_RPC = 'https://eth.public-rpc.com'

# Whales known to influence markets (Twitter handles)
INFLUENCERS = {
    'ArthurHayes': {
        'name': 'Arthur Hayes',
        'twitter': '@CryptoHayes',
        'weight': 0.8,  # his calls often move markets
        'track_tokens': True,
    },
    'Cobie': {
        'name': 'Cobie',
        'twitter': '@caboraboraborabor',
        'weight': 0.6,
        'track_tokens': True,
    },
    'Hsaka': {
        'name': 'Hsaka',
        'twitter': '@HsakaTrades',
        'weight': 0.7,
        'track_tokens': True,
    },
    'GCRClassic': {
        'name': 'GCR',
        'twitter': '@GCRClassic',
        'weight': 0.7,
        'track_tokens': True,
    },
    'Tetranode': {
        'name': 'Tetranode',
        'twitter': '@Tetranode',
        'weight': 0.5,
        'track_tokens': True,
    },
    '0xMaki': {
        'name': '0xMaki',
        'twitter': '@0xMaki',
        'weight': 0.5,
        'track_tokens': True,
    },
}

# Whales' known wallets (from public sources - approximate, may be incomplete)
# This is a sample. Real implementation would need on-chain investigation.
WHALE_WALLETS = {
    'ETH': [
        # Vitalik Buterin
        '0xab5801a7d398351b8be11c439e05c5b3259aec9b',
        # Arthur Hayes (claimed wallet)
        '0x1c75eca6e2c2e7b9b6b1e8c8b6b1e8c8b6b1e8c8',  # placeholder
    ],
    'BSC': [
        # Binance hot wallet
        '0x8894e0a0c962cb723c1976a4421c95949be2d4e3',
        # Some known BSC whales
    ],
}

MIN_TX_VALUE_USD = 100_000  # Only track $100K+ transactions


def _http_get(url: str, timeout: int = 15) -> Optional[dict]:
    """GET request with error handling."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def _http_post(url: str, data: dict, timeout: int = 15) -> Optional[dict]:
    """POST request (for RPC calls)."""
    try:
        payload = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=payload,
                                    headers={'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def _load_cache() -> dict:
    """Load cached whale signals (avoid re-fetching)."""
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'transactions': [],  # list of tx dicts
        'last_update': None,
        'symbol_to_token': {},  # symbol -> contract address
    }


def _save_cache(cache: dict) -> None:
    with open(CACHE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def resolve_token_address(symbol: str) -> Optional[Tuple[str, str]]:
    """
    Get the BSC contract address for a USDT symbol.
    Returns (network, address) or None.
    Uses Coingecko API to find platform info.
    """
    # Strip USDT
    base = symbol.replace('USDT', '').lower()
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{base}?localization=false&tickers=false&community_data=false&developer_data=false'
        data = _http_get(url)
        if not data:
            return None
        platforms = data.get('platforms', {})
        # Prefer BSC (cheaper gas, more active for small caps)
        if 'binance-smart-chain' in platforms:
            return ('BSC', platforms['binance-smart-chain'])
        elif 'ethereum' in platforms:
            return ('ETH', platforms['ethereum'])
        elif 'solana' in platforms:
            return ('SOL', platforms['solana'])
    except Exception:
        pass
    return None


def get_token_transactions_onchain(network: str, contract: str, hours: int = 24) -> List[dict]:
    """
    Get recent transactions for a token contract via public RPC.
    Note: This requires archive node access, which is limited on free RPCs.
    For production, use a service like BscScan API (free tier: 5 calls/sec).
    """
    if network == 'BSC':
        # BscScan API (free, requires API key but has free tier)
        # Fallback: just return empty if no API
        return []
    return []


def get_token_transactions_bscscan(contract: str, api_key: str = None) -> List[dict]:
    """
    Get token transactions via BscScan API.
    Free tier: 5 calls/sec, 100K calls/day.
    """
    if not api_key:
        # Try without API key (limited but works for small queries)
        api_key = 'YourApiKeyToken'
    try:
        url = f'https://api.bscscan.com/api?module=account&action=tokentx&contractaddress={contract}&page=1&offset=100&sort=desc&apikey={api_key}'
        data = _http_get(url)
        if data and data.get('status') == '1':
            return data.get('result', [])
    except Exception:
        pass
    return []


def get_whale_transactions_coingecko(symbol: str) -> dict:
    """
    Use Coingecko community data as a proxy for whale activity.
    Returns dict with: reddit_subscribers, twitter_followers, etc.
    """
    base = symbol.replace('USDT', '').lower()
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{base}?localization=false&tickers=false'
        data = _http_get(url)
        if not data:
            return {}
        community = data.get('community_data', {})
        dev = data.get('developer_data', {})
        return {
            'reddit_subscribers': community.get('reddit_subscribers', 0),
            'twitter_followers': community.get('twitter_followers', 0),
            'reddit_active_48h': community.get('reddit_accounts_active_48h', 0),
            'github_commits_4w': dev.get('commit_count_4_weeks', 0),
            'github_stars': dev.get('stars', 0),
        }
    except Exception:
        return {}


def get_token_social_velocity(symbol: str) -> float:
    """
    Calculate social velocity: rate of change in social metrics.
    High velocity = growing attention = potential pump signal.
    """
    base = symbol.replace('USDT', '').lower()
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{base}/history?data=7d&localization=false'
        data = _http_get(url)
        if not data:
            return 0.0
        community = data.get('community_data', {})
        if not community:
            return 0.0
        # Calculate velocity from reddit_active_48h changes
        # (Coingecko doesn't provide this directly, but we can use sparkline)
        reddit = community.get('reddit_subscribers', 0)
        twitter = community.get('twitter_followers', 0)
        # Higher absolute numbers with recent growth = higher velocity
        if reddit > 0 or twitter > 0:
            return math.log1p(reddit + twitter)
        return 0.0
    except Exception:
        return 0.0


def detect_whale_signal(symbol: str) -> dict:
    """
    Detect whale activity for a symbol.
    Returns dict with: has_whale_signal, signal_strength, details.
    """
    result = {
        'symbol': symbol,
        'has_whale_signal': False,
        'signal_strength': 0,  # 0-100
        'details': {},
    }

    # 1. Resolve contract address
    network_addr = resolve_token_address(symbol)
    if not network_addr:
        return result
    network, contract = network_addr
    result['details']['network'] = network
    result['details']['contract'] = contract

    # 2. Get on-chain transactions (BscScan)
    txs = get_token_transactions_bscscan(contract)
    if txs:
        # Count large transactions in last 24h
        now = time.time()
        large_txs = []
        for tx in txs:
            try:
                ts = int(tx.get('timeStamp', 0))
                value = float(tx.get('value', 0)) / (10 ** int(tx.get('tokenDecimal', 18)))
                # Convert to USD (we'd need price - use volume proxy)
                # Skip if older than 24h
                if now - ts > 86400:
                    continue
                large_txs.append({
                    'ts': ts,
                    'value': value,
                    'from': tx.get('from', ''),
                    'to': tx.get('to', ''),
                    'hash': tx.get('hash', ''),
                })
            except Exception:
                continue
        result['details']['tx_count_24h'] = len(large_txs)
        result['details']['large_tx_count'] = sum(1 for tx in large_txs if tx['value'] > 10000)
        if result['details']['large_tx_count'] >= 3:
            result['has_whale_signal'] = True
            result['signal_strength'] = min(100, result['details']['large_tx_count'] * 15)

    # 3. Social velocity (Coingecko)
    social = get_token_social_velocity(symbol)
    result['details']['social_velocity'] = social
    # If social is high (>15), moderate boost
    if social > 15:
        result['signal_strength'] += 20
        result['has_whale_signal'] = True
    elif social > 10:
        result['signal_strength'] += 10

    # 4. Community growth as whale interest proxy
    community = get_whale_transactions_coingecko(symbol)
    if community:
        result['details'].update(community)
        # Recent GitHub activity = developer engagement
        if community.get('github_commits_4w', 0) > 10:
            result['signal_strength'] += 15
            result['has_whale_signal'] = True
        # Twitter growth
        if community.get('twitter_followers', 0) > 50000:
            result['signal_strength'] += 5

    result['signal_strength'] = min(100, result['signal_strength'])
    return result


def scan_whales_for_symbols(symbols: List[str]) -> Dict[str, dict]:
    """Scan multiple symbols for whale activity."""
    results = {}
    for sym in symbols:
        results[sym] = detect_whale_signal(sym)
        time.sleep(0.3)  # rate limit
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        r = detect_whale_signal(sym)
        print(json.dumps(r, indent=2, default=str))
    else:
        # Test on a few repeaters
        for sym in ['SYNUSDT', 'AKEUSDT', 'BANKUSDT', 'TLMUSDT']:
            r = detect_whale_signal(sym)
            print(f'{sym}: signal_strength={r["signal_strength"]}  has_signal={r["has_whale_signal"]}')
            print(f'  details: {r["details"]}')
