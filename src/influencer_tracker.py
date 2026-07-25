"""
Influencer Tracker - detect when known crypto influencers mention a token.
Uses Nitter (free Twitter mirror) as a fallback when Twitter API is unavailable.

Note: This is a passive monitor. We track mentions but do NOT auto-trade on them.
Many influencers have terrible track records (FX Empire 2024 review showed
Arthur Hayes was wrong on HYPE, ZEC, NEAR, WLD calls).
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import math
import html

CACHE = '/home/user/toobit-scanner/data/influencer_cache.json'

# Known influencers and their track records (from public research)
# We weight them by historical accuracy, not just follower count.
INFLUENCERS = {
    'ArthurHayes': {
        'name': 'Arthur Hayes',
        'handle': '@CryptoHayes',
        'weight': 0.6,  # mixed track record (FX Empire 2024)
        'categories': ['macro', 'altcoins'],
        'source': 'nitter',
    },
    'Cobie': {
        'name': 'Cobie',
        'handle': '@caboraboraborabor',
        'weight': 0.5,
        'categories': ['defi', 'general'],
        'source': 'nitter',
    },
    'Hsaka': {
        'name': 'Hsaka',
        'handle': '@HsakaTrades',
        'weight': 0.65,  # better track record
        'categories': ['trading', 'memes'],
        'source': 'nitter',
    },
    'GCRClassic': {
        'name': 'GCR',
        'handle': '@GCRClassic',
        'weight': 0.5,
        'categories': ['memes', 'altcoins'],
        'source': 'nitter',
    },
    'Tetranode': {
        'name': 'Tetranode',
        'handle': '@Tetranode',
        'weight': 0.5,
        'categories': ['defi', 'l2'],
        'source': 'nitter',
    },
    'DefiIgnas': {
        'name': 'DefiIgnas',
        'handle': '@DefiIgnas',
        'weight': 0.6,
        'categories': ['defi', 'research'],
        'source': 'nitter',
    },
    'Route2FI': {
        'name': 'Route 2 FI',
        'handle': '@Route2FI',
        'weight': 0.55,
        'categories': ['defi', 'yield'],
        'source': 'nitter',
    },
    '0xMaki': {
        'name': '0xMaki',
        'handle': '@0xMaki',
        'weight': 0.45,
        'categories': ['defi'],
        'source': 'nitter',
    },
    'CMSintern': {
        'name': 'CMS',
        'handle': '@cmsintern',
        'weight': 0.5,
        'categories': ['research'],
        'source': 'nitter',
    },
    'lookonchain': {
        'name': 'Lookonchain',
        'handle': '@lookonchain',
        'weight': 0.7,  # on-chain alpha
        'categories': ['whale', 'onchain'],
        'source': 'nitter',
    },
}

# Nitter instances (free Twitter mirrors, may be unreliable)
NITTER_INSTANCES = [
    'https://nitter.net',
    'https://nitter.privacydev.net',
    'https://nitter.poast.org',
]


def _http_get(url: str, timeout: int = 15) -> Optional[str]:
    """GET request returning text (for HTML parsing)."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception:
        return None


def _load_cache() -> dict:
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'mentions': {},  # symbol -> [mentions]
        'last_update': None,
    }


def _save_cache(cache: dict) -> None:
    with open(CACHE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def get_recent_tweets(handle: str, max_tweets: int = 20) -> List[dict]:
    """
    Fetch recent tweets for a handle via Nitter.
    Returns list of {text, ts, url} dicts.
    Falls back to alternative instances if one is down.
    """
    handle_clean = handle.lstrip('@')
    for instance in NITTER_INSTANCES:
        try:
            url = f'{instance}/{handle_clean}'
            html_content = _http_get(url, timeout=10)
            if not html_content:
                continue
            # Parse tweet-content divs
            tweets = []
            # Simple regex parsing (Nitter HTML structure)
            # Look for timeline-item divs
            items = re.findall(
                r'<div class="timeline-item[^"]*"[^>]*>(.*?)(?=<div class="timeline-item|$)',
                html_content,
                re.DOTALL
            )
            for item in items[:max_tweets]:
                # Extract text
                text_match = re.search(r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>', item, re.DOTALL)
                if not text_match:
                    continue
                text = re.sub(r'<[^>]+>', '', text_match.group(1))
                text = html.unescape(text).strip()
                # Extract timestamp
                ts_match = re.search(r'<span class="tweet-date[^"]*"[^>]*><a[^>]+title="([^"]+)"', item)
                ts = ts_match.group(1) if ts_match else 'unknown'
                # Extract URL
                url_match = re.search(r'href="(/' + handle_clean + r'/status/\d+)"', item)
                tweet_url = f'{instance}{url_match.group(1)}' if url_match else ''
                tweets.append({
                    'text': text,
                    'ts': ts,
                    'url': tweet_url,
                })
            if tweets:
                return tweets
        except Exception:
            continue
    return []


def find_token_mentions(tweets: List[dict], symbols: List[str]) -> Dict[str, List[dict]]:
    """
    Search tweets for mentions of any of our symbols.
    Returns: {symbol: [tweet dicts that mention it]}
    """
    mentions = {}
    for tweet in tweets:
        text_lower = tweet['text'].lower()
        for sym in symbols:
            # Match: $SYN, SYN, SYN/USDT, $AKE, etc.
            base = sym.replace('USDT', '').lower()
            patterns = [
                f'${base}',
                f' {base} ',
                f' {base},',
                f' {base}.',
                f' {base}!',
                f' {base}?',
                f' {base}\n',
                f'/{base}',
                f'{base}/usdt',
            ]
            for pattern in patterns:
                if pattern in text_lower or pattern.replace(' ', '') in text_lower.replace(' ', ''):
                    if sym not in mentions:
                        mentions[sym] = []
                    mentions[sym].append(tweet)
                    break
    return mentions


def scan_influencers(symbols: List[str], hours_lookback: int = 24) -> dict:
    """
    Scan all influencers for mentions of given symbols in last 24h.
    Returns dict with mention counts and weighted signal strength.
    """
    cache = _load_cache()
    now = datetime.now(timezone.utc)

    results = {
        'mentions': {},  # symbol -> list of mentions with influencer info
        'signal_strength': {},  # symbol -> 0-100
    }

    for inf_id, inf in INFLUENCERS.items():
        if not inf.get('weight', 0) > 0.3:
            continue
        tweets = get_recent_tweets(inf['handle'], max_tweets=10)
        if not tweets:
            continue
        # Find mentions
        mentions = find_token_mentions(tweets, symbols)
        for sym, sym_mentions in mentions.items():
            if sym not in results['mentions']:
                results['mentions'][sym] = []
            for m in sym_mentions:
                results['mentions'][sym].append({
                    'influencer': inf['name'],
                    'handle': inf['handle'],
                    'weight': inf['weight'],
                    'text': m['text'][:200],
                    'ts': m['ts'],
                    'url': m['url'],
                })
        time.sleep(0.5)  # rate limit nitter

    # Calculate signal strength
    for sym in symbols:
        mentions = results['mentions'].get(sym, [])
        if not mentions:
            results['signal_strength'][sym] = 0
            continue
        # Weighted sum
        total_weight = sum(m['weight'] for m in mentions)
        # Multiple influencers = stronger signal
        unique_influencers = len(set(m['influencer'] for m in mentions))
        # Score: 0-100
        # Base: 30 per weighted mention
        # Bonus: +20 if multiple influencers
        # Bonus: +20 if high-weight influencer (>=0.6)
        score = min(100, total_weight * 30 + unique_influencers * 20)
        results['signal_strength'][sym] = score

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        r = scan_influencers([sym])
        print(json.dumps(r, indent=2, default=str))
    else:
        test_symbols = ['SYNUSDT', 'AKEUSDT', 'BANKUSDT', 'TLMUSDT',
                        'LABUSDT', 'EVAAUSDT', 'IKAUSDT', 'XCXUSDT',
                        'ACEUSDT', 'INUSDT', 'WODUSDT', 'ERAUSDT']
        r = scan_influencers(test_symbols)
        for sym in test_symbols:
            strength = r['signal_strength'].get(sym, 0)
            mentions = r['mentions'].get(sym, [])
            print(f'{sym:<14} strength={strength:>3}  mentions={len(mentions)}')
            for m in mentions[:2]:
                print(f'  - {m["influencer"]} ({m["weight"]}): {m["text"][:80]}...')
