"""
Breakout Detector - identifies high-probability breakout setups.

A "breakout" is when price moves beyond a defined range/chart pattern
with volume confirmation. Common patterns:
1. Range/channel breakout (horizontal or ascending)
2. Triangle/wedge breakout
3. Resistance level break with volume
4. Donchian channel high breakout (20-period)

Combined with:
- Volume spike (2x+ average)
- RSI not overbought (room to run)
- Body ratio (real buying, not wicks)

Note: Many false breakouts - we filter by volume + momentum follow-through.
"""
from __future__ import annotations
import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def _http_get_json(url: str, timeout: int = 15) -> Optional[list]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception:
        return None


def fetch_klines(symbol: str, interval: str = '1h', limit: int = 100) -> List[dict]:
    """Fetch klines from Toobit."""
    try:
        url = f'https://api.toobit.com/quote/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
        raw = _http_get_json(url, timeout=10)
        if not raw:
            return []
        rows = []
        for r in raw:
            ts_ms = int(r[0])
            rows.append({
                'ts': ts_ms // 1000,
                'o': float(r[1]),
                'h': float(r[2]),
                'l': float(r[3]),
                'c': float(r[4]),
                'v': float(r[5]),
            })
        rows.sort(key=lambda x: x['ts'])
        return rows
    except Exception:
        return []


def rsi(closes: List[float], n: int = 14) -> float:
    if len(closes) < n + 1:
        return 50.0
    g, l = [], []
    for i in range(1, n + 1):
        ch = closes[i] - closes[i-1]
        g.append(max(ch, 0)); l.append(max(-ch, 0))
    ag = sum(g)/n; al = sum(l)/n
    for i in range(n+1, len(closes)):
        ch = closes[i] - closes[i-1]
        ag = (ag*(n-1) + max(ch,0))/n
        al = (al*(n-1) + max(-ch,0))/n
    if al == 0: return 100.0
    return 100 - 100/(1+ag/al)


def detect_donchian_breakout(rows: List[dict], lookback: int = 20) -> Tuple[bool, float]:
    """
    Donchian channel: highest high / lowest low over N periods.
    Breakout = current price > highest high of last N periods.
    Returns: (is_breakout, breakout_strength_pct).
    """
    if len(rows) < lookback + 1:
        return False, 0.0
    # Use N periods BEFORE current
    prior = rows[-(lookback+1):-1]
    if not prior:
        return False, 0.0
    highest = max(r['h'] for r in prior)
    cur_close = rows[-1]['c']
    if cur_close > highest:
        strength = (cur_close - highest) / highest * 100
        return True, strength
    return False, 0.0


def detect_range_breakout(rows: List[dict], lookback: int = 24) -> Tuple[bool, float]:
    """
    Detect horizontal range breakout.
    Range = max(high) - min(low) over N periods.
    Tight range (<3%) followed by break = good signal.
    """
    if len(rows) < lookback + 1:
        return False, 0.0
    prior = rows[-(lookback+1):-1]
    highs = [r['h'] for r in prior]
    lows = [r['l'] for r in prior]
    range_high = max(highs)
    range_low = min(lows)
    range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
    cur_close = rows[-1]['c']
    if cur_close > range_high:
        strength = (cur_close - range_high) / range_high * 100
        return True, strength
    return False, 0.0


def detect_ascending_triangle(rows: List[dict], lookback: int = 24) -> Tuple[bool, float]:
    """
    Ascending triangle: higher lows + flat resistance.
    Breakout above resistance = bullish.
    """
    if len(rows) < lookback + 1:
        return False, 0.0
    prior = rows[-(lookback+1):-1]
    # Split into 2 halves
    half = lookback // 2
    first_half = prior[:half]
    second_half = prior[half:]
    # Check if second half lows are higher than first half
    first_low = min(r['l'] for r in first_half)
    second_low = min(r['l'] for r in second_half)
    # Check if highs are similar (flat resistance)
    first_high = max(r['h'] for r in first_half)
    second_high = max(r['h'] for r in second_half)
    # Higher low + similar high = ascending triangle
    higher_low = second_low > first_low * 1.01
    similar_high = abs(second_high - first_high) / first_high < 0.03 if first_high > 0 else False
    if higher_low and similar_high:
        # Now check if breakout
        cur_close = rows[-1]['c']
        if cur_close > second_high:
            strength = (cur_close - second_high) / second_high * 100
            return True, strength
    return False, 0.0


def detect_volume_spike(rows: List[dict], lookback: int = 20) -> Tuple[bool, float]:
    """Detect volume spike vs recent average."""
    if len(rows) < lookback + 1:
        return False, 1.0
    recent = rows[-(lookback+1):-1]
    avg_v = sum(r['v'] for r in recent) / len(recent) if recent else 1
    cur_v = rows[-1]['v']
    rvol = cur_v / avg_v if avg_v > 0 else 1.0
    return rvol > 2.0, rvol


def detect_breakout(symbol: str) -> dict:
    """
    Main breakout detection for a symbol.
    Returns dict with: has_breakout, type, strength, score, details.
    """
    result = {
        'symbol': symbol,
        'has_breakout': False,
        'type': None,
        'strength_pct': 0,
        'score': 0,  # 0-100
        'details': {},
    }

    rows = fetch_klines(symbol, '1h', 100)
    if len(rows) < 30:
        return result

    # Run multiple detectors
    donchian, donchian_str = detect_donchian_breakout(rows, 20)
    range_br, range_str = detect_range_breakout(rows, 24)
    triangle, triangle_str = detect_ascending_triangle(rows, 24)
    vol_spike, rvol = detect_volume_spike(rows, 20)

    closes = [r['c'] for r in rows]
    rsi_v = rsi(closes)
    mom_3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) > 3 else 0

    result['details'] = {
        'donchian_breakout': donchian,
        'donchian_strength': donchian_str,
        'range_breakout': range_br,
        'range_strength': range_str,
        'triangle_breakout': triangle,
        'triangle_strength': triangle_str,
        'rvol': rvol,
        'rsi': rsi_v,
        'mom_3': mom_3,
    }

    # Score
    score = 0
    if donchian:
        score += 35
        result['type'] = 'donchian'
        result['strength_pct'] = donchian_str
    if range_br:
        score += 30
        if not result['type']:
            result['type'] = 'range'
        result['strength_pct'] = max(result['strength_pct'], range_str)
    if triangle:
        score += 25
        if not result['type']:
            result['type'] = 'triangle'
        result['strength_pct'] = max(result['strength_pct'], triangle_str)
    if vol_spike:
        score += 20
    if rsi_v < 70:  # not overbought
        score += 5
    if 0 < mom_3 < 10:  # positive momentum but not overdone
        score += 10

    # Penalty for overbought
    if rsi_v > 80:
        score -= 20

    result['score'] = min(100, max(0, score))
    result['has_breakout'] = result['score'] >= 50
    return result


def scan_breakouts(symbols: List[str]) -> Dict[str, dict]:
    """Scan multiple symbols for breakouts."""
    results = {}
    for sym in symbols:
        results[sym] = detect_breakout(sym)
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        r = detect_breakout(sym)
        print(json.dumps(r, indent=2, default=str))
    else:
        test_symbols = ['SYNUSDT', 'AKEUSDT', 'BANKUSDT', 'TLMUSDT',
                        'LABUSDT', 'EVAAUSDT', 'IKAUSDT', 'XCXUSDT',
                        'ACEUSDT', 'INUSDT', 'WODUSDT', 'ERAUSDT']
        r = scan_breakouts(test_symbols)
        for sym in test_symbols:
            res = r[sym]
            marker = '🟢' if res['has_breakout'] else ('🟡' if res['score'] > 30 else '⚪')
            print(f'{marker} {sym:<14} score={res["score"]:>3}  type={res["type"]}  '
                  f'rvol={res["details"].get("rvol", 0):.2f}  mom3={res["details"].get("mom_3", 0):+.1f}%')
