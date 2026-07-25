"""
News/Product Activity Detector - distinguishes real development from hype.

Real product activity signals:
- GitHub commits in last 4 weeks (developer engagement)
- API documentation updates
- Whitepaper revisions
- Roadmap updates
- Smart contract deployments

Hype-only signals (we IGNORE these):
- Twitter announcement without code
- Listing announcements (often 'sell the news')
- Influencer mentions alone
- Rumors

This module uses:
- GitHub commits API (free, no auth)
- CryptoPanic API (free, with auth optional)
- CoinGecko developer stats
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import math

CACHE = '/home/user/toobit-scanner/data/news_cache.json'

# Known project GitHub repos for our repeaters
# This would be expanded with proper research per symbol
KNOWN_REPOS = {
    'SYNUSDT': 'https://github.com/synapseprotocol',  # Synapse
    'AKEUSDT': 'https://github.com/akedo',  # placeholder
    'EVAAUSDT': 'https://github.com/evaa-protocol',  # placeholder
    'TLMUSDT': None,  # Alien Worlds - no public repo
    'BANKUSDT': None,  # unknown
    'LABUSDT': None,  # unknown
    'IKAUSDT': None,
    'ACEUSDT': 'https://github.com/fusionist',  # Fusionist
    'INUSDT': None,
    'WODUSDT': 'https://github.com/dypius',  # World of Dypians
    'XCXUSDT': None,
    'ERAUSDT': None,
}


def _http_get(url: str, timeout: int = 15) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception:
        return None


def _http_get_json(url: str, timeout: int = 15) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def get_github_activity(repo_url: str) -> dict:
    """
    Get GitHub repo activity metrics.
    repo_url: full URL like https://github.com/owner/repo
    """
    if not repo_url:
        return {}
    # Extract owner/repo
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+)/?$', repo_url)
    if not m:
        return {}
    owner, repo = m.group(1), m.group(2)

    result = {
        'owner': owner,
        'repo': repo,
        'stars': 0,
        'forks': 0,
        'commits_4w': 0,
        'last_commit': None,
        'open_issues': 0,
        'open_pulls': 0,
    }

    # Get repo info
    repo_data = _http_get_json(f'https://api.github.com/repos/{owner}/{repo}')
    if repo_data:
        result['stars'] = repo_data.get('stargazers_count', 0)
        result['forks'] = repo_data.get('forks_count', 0)
        result['open_issues'] = repo_data.get('open_issues_count', 0)
        result['last_commit'] = repo_data.get('pushed_at')

    # Get commits in last 4 weeks
    try:
        since = (datetime.now(timezone.utc) - timedelta(weeks=4)).isoformat()
        commits_data = _http_get_json(
            f'https://api.github.com/repos/{owner}/{repo}/commits?since={since}&per_page=100'
        )
        if commits_data and isinstance(commits_data, list):
            result['commits_4w'] = len(commits_data)
    except Exception:
        pass

    return result


def get_cryptopanic_news(symbol: str, hours: int = 24) -> List[dict]:
    """
    Get news from CryptoPanic (free API, optional auth).
    """
    base = symbol.replace('USDT', '').lower()
    # Without auth, public API has rate limits
    url = f'https://cryptopanic.com/api/v1/posts/?currencies={base}&kind=news'
    data = _http_get_json(url)
    if not data or 'results' not in data:
        return []
    news = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    for item in data.get('results', [])[:20]:
        try:
            pub = item.get('published_at', '')
            if pub:
                pub_dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                if pub_dt < cutoff:
                    continue
            news.append({
                'title': item.get('title', ''),
                'source': item.get('source', {}).get('title', 'unknown'),
                'published': pub,
                'url': item.get('url', ''),
                'votes': item.get('votes', {}),
            })
        except Exception:
            continue
    return news


def classify_news_quality(news_list: List[dict]) -> dict:
    """
    Classify news as product/hype/neutral.
    Real product news: API, integration, mainnet, audit, partnership with real product.
    Hype: listing, prediction, "to the moon", giveaway.
    """
    real_keywords = [
        'api', 'integration', 'mainnet', 'testnet', 'audit', 'partnership',
        'sdk', 'documentation', 'release', 'launch', 'v2', 'v3', 'update',
        'github', 'code', 'audit', 'security', 'smart contract',
    ]
    hype_keywords = [
        'listing', 'to the moon', 'pump', 'giveaway', 'airdrop',
        'prediction', 'price target', 'moonshot', '1000x', 'lambo',
        'bullish', 'fomo', 'hodl', 'wen',
    ]
    real_count = 0
    hype_count = 0
    for n in news_list:
        title = n.get('title', '').lower()
        for kw in real_keywords:
            if kw in title:
                real_count += 1
                break
        for kw in hype_keywords:
            if kw in title:
                hype_count += 1
                break
    return {
        'real_news_count': real_count,
        'hype_news_count': hype_count,
        'ratio': real_count / max(hype_count, 1),
    }


def detect_product_activity(symbol: str) -> dict:
    """
    Detect real product development activity for a symbol.
    """
    result = {
        'symbol': symbol,
        'has_product_activity': False,
        'signal_strength': 0,  # 0-100
        'details': {},
    }

    # 1. GitHub activity
    repo_url = KNOWN_REPOS.get(symbol)
    if repo_url:
        gh = get_github_activity(repo_url)
        if gh:
            result['details'].update(gh)
            # Score based on activity
            if gh.get('commits_4w', 0) > 20:
                result['signal_strength'] += 30
                result['has_product_activity'] = True
            elif gh.get('commits_4w', 0) > 5:
                result['signal_strength'] += 20
                result['has_product_activity'] = True
            elif gh.get('commits_4w', 0) > 0:
                result['signal_strength'] += 5
            # Stars
            if gh.get('stars', 0) > 1000:
                result['signal_strength'] += 10
            # Recent push
            if gh.get('last_commit'):
                try:
                    last = datetime.fromisoformat(gh['last_commit'].replace('Z', '+00:00'))
                    days_ago = (datetime.now(timezone.utc) - last).days
                    if days_ago < 7:
                        result['signal_strength'] += 15
                    elif days_ago < 30:
                        result['signal_strength'] += 5
                except Exception:
                    pass

    # 2. CoinGecko developer data
    base = symbol.replace('USDT', '').lower()
    cgk = _http_get_json(f'https://api.coingecko.com/api/v3/coins/{base}?localization=false&tickers=false')
    if cgk:
        dev = cgk.get('developer_data', {})
        if dev:
            commits_4w = dev.get('commit_count_4_weeks', 0)
            result['details']['cgk_commits_4w'] = commits_4w
            if commits_4w > 30:
                result['signal_strength'] += 20
                result['has_product_activity'] = True
            elif commits_4w > 10:
                result['signal_strength'] += 10
            stars = dev.get('stars', 0)
            if stars > 500:
                result['signal_strength'] += 5
            # PRs merged recently
            prs = dev.get('pull_request_count_4_weeks', 0)
            if prs > 5:
                result['signal_strength'] += 10

    # 3. News classification (real vs hype)
    news = get_cryptopanic_news(symbol, hours=72)
    if news:
        quality = classify_news_quality(news)
        result['details'].update(quality)
        result['details']['news_count_72h'] = len(news)
        if quality['real_news_count'] > quality['hype_news_count']:
            result['signal_strength'] += 15
            result['has_product_activity'] = True
        elif quality['hype_news_count'] > 5:
            # Lots of hype = warning
            result['signal_strength'] -= 10

    result['signal_strength'] = min(100, max(0, result['signal_strength']))
    return result


def scan_news_activity(symbols: List[str]) -> Dict[str, dict]:
    """Scan multiple symbols for product activity."""
    results = {}
    for sym in symbols:
        results[sym] = detect_product_activity(sym)
        time.sleep(0.3)  # rate limit
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        r = detect_product_activity(sym)
        print(json.dumps(r, indent=2, default=str))
    else:
        test_symbols = ['SYNUSDT', 'AKEUSDT', 'BANKUSDT', 'TLMUSDT',
                        'LABUSDT', 'EVAAUSDT', 'IKAUSDT', 'XCXUSDT',
                        'ACEUSDT', 'INUSDT', 'WODUSDT', 'ERAUSDT']
        r = scan_news_activity(test_symbols)
        for sym in test_symbols:
            res = r[sym]
            marker = '🟢' if res['has_product_activity'] else ('🟡' if res['signal_strength'] > 20 else '⚪')
            print(f'{marker} {sym:<14} strength={res["signal_strength"]:>3}  '
                  f'details={list(res["details"].keys())[:5]}')
