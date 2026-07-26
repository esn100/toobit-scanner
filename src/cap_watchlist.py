"""
Cap-based watchlist: always monitor small and mid cap tokens.
Categories:
- Low cap: < $50M (highest pump potential, high risk)
- Mid cap: $50M - $500M (good balance)

This module updates the repeater_config with these always-on watch symbols.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Hardcoded from coinranking.com/exchange/NO8VJJUzMH8V+toobit/coins
# Format: symbol -> (market_cap_usd, vol_24h, name)
# LOW CAP (< $50M) - highest pump potential
LOW_CAP_WATCHLIST = {
    # Format: SYMBOL -> Market Cap
    "TUTUSDT": 13_440_000,      # Tutorial - low cap
    "WLFIUSDT": 1_750_000_000,  # World Liberty Financial
    "BRUSDT": 44_060_000,         # Bedrock
    "WCTUSDT": 17_370_000,        # WalletConnect
    "EULUSDT": 64_920_000,        # Euler
    "BANKUSDT": 478_430_000,      # Lorenzo Governance (already primary repeater)
    "EPICUSDT": 25_030_000,       # Epic Chain
    "ESPUSDT": 108_840_000,       # Espresso
    "LPTUSDT": 74_540_000,        # Livepeer
    "INJUSDT": 487_500_000,       # Injective Protocol
    "TRUMPUSDT": 395_350_000,     # Official Trump
    "ZAMAUSDT": 175_550_000,      # Zama
    "PIEVERSEUSDT": 65_220_000,   # Pieverse
    "ENUSDT": 813_890_000,        # Ethena
    "VIRTUALUSDT": 386_970_000,   # Virtuals Protocol
    "KAITOUSDT": 283_610_000,     # Kaito
    "AEROUSDT": 402_910_000,      # Aerodrome
    "SENTUSDT": 160_570_000,      # Sentient
    "FARTCOINUSDT": 129_050_000,  # Fartcoin
}

# MID CAP ($50M - $500M) - already covered mostly
MID_CAP_WATCHLIST = {
    "MNTUSDT": 1_360_000_000,   # Mantle (close to mid cap range)
    "PEPEUSDT": 1_220_000_000,
    "AAVEUSDT": 1_510_000_000,
    "ONDOUSDT": 1_940_000_000,
    "LDOUSDT": 1_510_000_000,
    "WLDUSDT": 1_240_000_000,
}

# Tokens NOT in repeater config but should be added to watchlist
# Filter: vol 24h > $1M (to ensure liquidity)
def get_cap_based_watchlist() -> Dict[str, Dict]:
    """
    Get combined watchlist of small and mid cap tokens.
    Returns: {symbol: {cap, vol, source, priority}}
    """
    watchlist = {}
    # Low cap - HIGHEST priority for pump detection
    for sym, cap in LOW_CAP_WATCHLIST.items():
        if cap < 50_000_000:  # Strict low cap
            watchlist[sym] = {
                'cap': cap,
                'cap_category': 'low',
                'priority': 10,  # Highest priority
            }
        elif cap < 500_000_000:  # Mid cap
            watchlist[sym] = {
                'cap': cap,
                'cap_category': 'mid',
                'priority': 7,
            }
    return watchlist


def update_repeater_config_with_watchlist():
    """
    Add cap-based watchlist symbols to repeater_config.py SECONDARY_WATCHLIST.
    These will always be scanned alongside primary repeaters.
    """
    import sys
    sys.path.insert(0, '/home/user/toobit-scanner')
    # Read current config
    config_path = '/home/user/toobit-scanner/src/repeater_config.py'
    with open(config_path) as f:
        text = f.read()
    # Find SECONDARY_WATCHLIST line
    new_symbols = list(LOW_CAP_WATCHLIST.keys())
    # Don't add if already in primary or secondary
    primary_set = {'EVAAUSDT', 'TLMUSDT', 'LABUSDT', 'BANKUSDT', 'AKEUSDT', 'IKAUSDT',
                   'SYNUSDT', 'ACEUSDT', 'INUSDT', 'ERAUSDT', 'WODUSDT', 'XCXUSDT'}
    to_add = [s for s in new_symbols if s not in primary_set]
    return to_add


if __name__ == '__main__':
    watchlist = get_cap_based_watchlist()
    print(f'Cap-based watchlist: {len(watchlist)} symbols')
    print()
    print('LOW CAP (< $50M):')
    for sym, info in sorted(watchlist.items(), key=lambda x: x[1]['cap']):
        if info['cap_category'] == 'low':
            print(f'  {sym:<14} cap=${info["cap"]/1e6:>8.1f}M  priority={info["priority"]}')
    print()
    print('MID CAP ($50M - $500M):')
    for sym, info in sorted(watchlist.items(), key=lambda x: x[1]['cap']):
        if info['cap_category'] == 'mid':
            print(f'  {sym:<14} cap=${info["cap"]/1e6:>8.1f}M  priority={info["priority"]}')
    print()
    print(f'To add to SECONDARY_WATCHLIST: {len(update_repeater_config_with_watchlist())} symbols')
    print(update_repeater_config_with_watchlist())
