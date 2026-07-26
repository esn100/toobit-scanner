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

from .whale_tracker import detect_whale_signal_v2 as detect_whale_signal
from .whale_intel import get_whale_intel
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

    # 5. ADVANCED INTEL (6 features)
    if verbose:
        print(f'  [5/5] Advanced intel (6 features)...')
    from .advanced_intel import get_advanced_intel
    for sym in symbols:
        try:
            r = get_advanced_intel(sym, verbose=False)
            # Cache each component
            obi_data = r.get('components', {}).get('obi', {})
            cache.setdefault('obi', {})
            cache['obi'][sym] = {
                'value': obi_data.get('imbalance', 0.5),
                'signal': r.get('components', {}).get('obi_signal', 'balanced'),
                'ts': now.isoformat(),
            }
            ls_data = r.get('components', {}).get('ls', {})
            cache.setdefault('ls_ratio', {})
            cache['ls_ratio'][sym] = {
                'value': ls_data.get('ratio', 0.5),
                'signal': ls_data.get('signal_type', 'neutral'),
                'ts': now.isoformat(),
            }
            taker_data = r.get('components', {}).get('taker', {})
            cache.setdefault('taker_ratio', {})
            cache['taker_ratio'][sym] = {
                'value': taker_data.get('buy_ratio', 0.5),
                'signal': r.get('components', {}).get('taker_signal', 'balanced'),
                'ts': now.isoformat(),
            }
            sm_data = r.get('components', {}).get('smart_money', {})
            cache.setdefault('smart_money', {})
            cache['smart_money'][sym] = {
                'value': sm_data.get('whale_net_flow_usd', 0),
                'signal': r.get('components', {}).get('smart_signal', 'neutral'),
                'ts': now.isoformat(),
            }
            liq_data = r.get('components', {}).get('liquidations', {})
            cache.setdefault('liquidations', {})
            cache['liquidations'][sym] = {
                'value': liq_data.get('oi_change_24h_pct', 0),
                'signal': r.get('components', {}).get('liq_signal', 'neutral'),
                'ts': now.isoformat(),
            }
            arb_data = r.get('components', {}).get('arb', {})
            cache.setdefault('arb_spread', {})
            cache['arb_spread'][sym] = {
                'value': arb_data.get('max_premium_pct', 0),
                'signal': r.get('components', {}).get('arb_signal', 'aligned'),
                'ts': now.isoformat(),
            }
        except Exception as e:
            if verbose:
                print(f'    advanced err {sym}: {e}')

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

    # Get all 4 base scores
    whale_str = get_whale_score(symbol, cache)
    inf_str, inf_mentions = get_influencer_score(symbol, cache)
    breakout_str, breakout_type = get_breakout_score(symbol, cache)
    news_str = get_news_score(symbol, cache)

    # Get 6 ADVANCED scores
    obi = cache.get('obi', {}).get(symbol, {}).get('value', 0.5)
    ls_ratio = cache.get('ls_ratio', {}).get(symbol, {}).get('value', 0.5)
    taker_buy = cache.get('taker_ratio', {}).get(symbol, {}).get('value', 0.5)
    smart_money = cache.get('smart_money', {}).get(symbol, {}).get('value', 0)
    liq_oi_chg = cache.get('liquidations', {}).get(symbol, {}).get('value', 0)
    arb_premium = cache.get('arb_spread', {}).get(symbol, {}).get('value', 0)

    # Calculate advanced intel score (0-100)
    adv_score = 0
    if obi > 0.65:
        adv_score += 20
    elif obi > 0.55:
        adv_score += 12
    elif obi < 0.4:
        adv_score -= 10
    if ls_ratio > 0.75 or ls_ratio < 0.25:
        adv_score += 18
    elif ls_ratio > 0.65 or ls_ratio < 0.35:
        adv_score += 10
    if taker_buy > 0.65:
        adv_score += 20
    elif taker_buy > 0.55:
        adv_score += 12
    elif taker_buy < 0.4:
        adv_score -= 10
    if smart_money > 10000:
        adv_score += 20
    elif smart_money > 0:
        adv_score += 10
    elif smart_money < -10000:
        adv_score -= 12
    if liq_oi_chg > 15:
        adv_score += 12
    elif liq_oi_chg > 5:
        adv_score += 6
    elif liq_oi_chg < -10:
        adv_score -= 5
    if arb_premium > 2:
        adv_score += 15
    elif arb_premium > 0.5:
        adv_score += 8
    adv_score = max(0, min(100, adv_score))

    components = {
        'whale': whale_str,
        'influencer': inf_str,
        'influencer_mentions': inf_mentions,
        'breakout': breakout_str,
        'breakout_type': breakout_type,
        'news': news_str,
        'pre_pump': pre_pump_confidence,
        'obi': round(obi, 2),
        'ls_ratio': round(ls_ratio, 2),
        'taker_buy': round(taker_buy, 2),
        'smart_money': round(smart_money, 0),
        'liq_oi_chg': round(liq_oi_chg, 1),
        'arb_premium': round(arb_premium, 2),
        'advanced_score': adv_score,
    }

    # Combined intelligence score (weighted)
    base_intel = (
        whale_str * 0.20 +
        inf_str * 0.10 +
        breakout_str * 0.25 +
        news_str * 0.15
    )
    intel_score = base_intel * 0.7 + adv_score * 0.3

    # Final score: 50% pattern + 50% intelligence (more weight on intel)
    final_score = pre_pump_confidence * 0.5 + intel_score * 0.5

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
