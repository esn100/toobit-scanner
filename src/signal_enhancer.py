"""
Signal Enhancer - combines 4 intelligence sources to improve signal quality.

Integrates:
1. Whale tracker (on-chain large transactions)
2. Influencer tracker (Twitter/X mentions)
3. Breakout detector (technical patterns)
4. News/product activity (GitHub, real development)

Each source returns a 0-100 score. Combined "intelligence score" can boost
or veto a pre-pump signal from the repeater scanner.

Usage:
    from src.signal_enhancer import enhance_signal
    result = enhance_signal('SYNUSDT', pre_pump_pattern={'confidence': 75})
    if result['final_score'] > 70:
        # STRONG signal - take trade
    elif result['veto']:
        # VETOED - skip trade despite technical pattern
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional
import statistics

from .whale_tracker import detect_whale_signal
from .influencer_tracker import scan_influencers
from .breakout_detector import detect_breakout
from .news_detector import detect_product_activity


CACHE_DIR = '/home/user/toobit-scanner/data'
SIGNAL_ENHANCER_CACHE = f'{CACHE_DIR}/enhancer_cache.json'


def _load_cache() -> dict:
    if os.path.exists(SIGNAL_ENHANCER_CACHE):
        try:
            with open(SIGNAL_ENHANCER_CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'whale': {},  # symbol -> {strength, ts}
        'influencer': {},  # symbol -> {strength, mentions, ts}
        'breakout': {},  # symbol -> {score, type, ts}
        'news': {},  # symbol -> {strength, ts}
        'last_full_scan': None,
    }


def _save_cache(cache: dict) -> None:
    with open(SIGNAL_ENHANCER_CACHE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


def get_whale_score(symbol: str, cache: dict, max_age_min: int = 60) -> int:
    """Get whale activity score, using cache if recent."""
    entry = cache['whale'].get(symbol, {})
    if not entry:
        return 0
    try:
        ts = datetime.fromisoformat(entry['ts'].replace('Z', '+00:00'))
        age_min = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60
        if age_min > max_age_min:
            return 0  # stale
        return entry.get('strength', 0)
    except Exception:
        return 0


def get_influencer_score(symbol: str, cache: dict, max_age_min: int = 60) -> tuple:
    """Get influencer mention score, returns (strength, mention_count)."""
    entry = cache['influencer'].get(symbol, {})
    if not entry:
        return 0, 0
    try:
        ts = datetime.fromisoformat(entry['ts'].replace('Z', '+00:00'))
        age_min = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60
        if age_min > max_age_min:
            return 0, 0
        return entry.get('strength', 0), entry.get('mention_count', 0)
    except Exception:
        return 0, 0


def get_breakout_score(symbol: str, cache: dict, max_age_min: int = 30) -> tuple:
    """Get breakout score, returns (score, type)."""
    entry = cache['breakout'].get(symbol, {})
    if not entry:
        return 0, None
    try:
        ts = datetime.fromisoformat(entry['ts'].replace('Z', '+00:00'))
        age_min = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60
        if age_min > max_age_min:
            return 0, None
        return entry.get('score', 0), entry.get('type')
    except Exception:
        return 0, None


def get_news_score(symbol: str, cache: dict, max_age_min: int = 120) -> int:
    """Get news/product activity score."""
    entry = cache['news'].get(symbol, {})
    if not entry:
        return 0
    try:
        ts = datetime.fromisoformat(entry['ts'].replace('Z', '+00:00'))
        age_min = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60
        if age_min > max_age_min:
            return 0
        return entry.get('strength', 0)
    except Exception:
        return 0


def scan_all(symbols: list, verbose: bool = False) -> dict:
    """
    Run all 4 intelligence scanners and update cache.
    Returns updated cache.
    """
    cache = _load_cache()
    now = datetime.now(timezone.utc)

    if verbose:
        print(f'[ENHANCER] Scanning {len(symbols)} symbols across 4 intelligence sources...')

    # 1. Whale tracker
    if verbose:
        print(f'  [1/4] Whale tracker...')
    for sym in symbols:
        try:
            r = detect_whale_signal(sym)
            cache['whale'][sym] = {
                'strength': r.get('signal_strength', 0),
                'has_signal': r.get('has_whale_signal', False),
                'details': r.get('details', {}),
                'ts': now.isoformat(),
            }
        except Exception as e:
            if verbose:
                print(f'    whale err {sym}: {e}')

    # 2. Influencer tracker
    if verbose:
        print(f'  [2/4] Influencer tracker...')
    try:
        inf_result = scan_influencers(symbols, hours_lookback=24)
        for sym in symbols:
            mentions = inf_result['mentions'].get(sym, [])
            cache['influencer'][sym] = {
                'strength': inf_result['signal_strength'].get(sym, 0),
                'mention_count': len(mentions),
                'mentions': mentions[:5],  # top 5
                'ts': now.isoformat(),
            }
    except Exception as e:
        if verbose:
            print(f'    influencer err: {e}')

    # 3. Breakout detector
    if verbose:
        print(f'  [3/4] Breakout detector...')
    for sym in symbols:
        try:
            r = detect_breakout(sym)
            cache['breakout'][sym] = {
                'score': r.get('score', 0),
                'type': r.get('type'),
                'has_breakout': r.get('has_breakout', False),
                'details': r.get('details', {}),
                'ts': now.isoformat(),
            }
        except Exception as e:
            if verbose:
                print(f'    breakout err {sym}: {e}')

    # 4. News detector
    if verbose:
        print(f'  [4/4] News/product activity...')
    for sym in symbols:
        try:
            r = detect_product_activity(sym)
            cache['news'][sym] = {
                'strength': r.get('signal_strength', 0),
                'has_activity': r.get('has_product_activity', False),
                'details': r.get('details', {}),
                'ts': now.isoformat(),
            }
        except Exception as e:
            if verbose:
                print(f'    news err {sym}: {e}')

    cache['last_full_scan'] = now.isoformat()
    _save_cache(cache)
    return cache


def enhance_signal(symbol: str, pre_pump_confidence: float = 0, pre_pump_features: dict = None) -> dict:
    """
    Combine 4 intelligence sources with pre-pump pattern to give final score.
    Returns: {
        'final_score': 0-100,
        'veto': bool,
        'boost': int,
        'components': {...},
        'action': 'TAKE' / 'SKIP' / 'WATCH'
    }
    """
    cache = _load_cache()

    # Get all 4 scores
    whale_str = get_whale_score(symbol, cache)
    inf_str, inf_mentions = get_influencer_score(symbol, cache)
    breakout_str, breakout_type = get_breakout_score(symbol, cache)
    news_str = get_news_score(symbol, cache)

    components = {
        'whale': whale_str,
        'influencer': inf_str,
        'influencer_mentions': inf_mentions,
        'breakout': breakout_str,
        'breakout_type': breakout_type,
        'news': news_str,
        'pre_pump': pre_pump_confidence,
    }

    # Combined intelligence score (weighted)
    intel_score = (
        whale_str * 0.30 +         # whales are strong signal
        inf_str * 0.15 +           # influencers are noisy
        breakout_str * 0.35 +      # technical breakout is strong
        news_str * 0.20            # real development is strong
    )

    # Final score: 60% pattern + 40% intelligence
    final_score = pre_pump_confidence * 0.6 + intel_score * 0.4

    # Veto logic: ignore if fundamentals are bad
    veto = False
    veto_reason = None

    # Veto if too much hype-only news (no real product)
    news_data = cache.get('news', {}).get(symbol, {})
    if news_data.get('details', {}).get('hype_news_count', 0) > 5:
        if news_data.get('details', {}).get('real_news_count', 0) < 2:
            veto = True
            veto_reason = 'high_hype_low_substance'

    # Action recommendation
    if veto:
        action = 'SKIP'
    elif final_score >= 70:
        action = 'TAKE'  # strong signal
    elif final_score >= 50:
        action = 'WATCH'  # moderate
    else:
        action = 'SKIP'  # weak

    return {
        'final_score': round(final_score, 1),
        'intel_score': round(intel_score, 1),
        'veto': veto,
        'veto_reason': veto_reason,
        'boost': round(intel_score - pre_pump_confidence, 1),
        'action': action,
        'components': components,
    }


if __name__ == '__main__':
    import sys
    from .repeater_config import REPEATERS
    symbols = list(REPEATERS.keys())
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        result = enhance_signal(sym, pre_pump_confidence=70)
        print(json.dumps(result, indent=2, default=str))
    else:
        cache = scan_all(symbols, verbose=True)
        print()
        for sym in symbols:
            result = enhance_signal(sym, pre_pump_confidence=70)
            print(f'{sym:<14}  final={result["final_score"]:>5}  '
                  f'whale={result["components"]["whale"]:>3}  '
                  f'breakout={result["components"]["breakout"]:>3}  '
                  f'news={result["components"]["news"]:>3}  '
                  f'action={result["action"]}')
