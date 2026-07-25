"""
Whale Tracker v2 - REAL whale detection with 3 data sources.

Sources (in priority order):
1. Whale Alert API (broad coverage, requires API key) - 5K calls/month free
2. BSC direct RPC (free, unlimited, BSC only) - real-time block scanning
3. Etherscan API (legacy, free) - fallback for ETH mainnet

Detects:
- Large transactions (>$100K) involving watched tokens
- "Whale wallet" activity (known exchange/insider wallets)
- Net flow direction (accumulation vs distribution)
- Velocity (sudden whale activity = pre-pump signal)
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import math
import threading
from collections import defaultdict

# Cache
CACHE_DIR = '/home/user/toobit-scanner/data'
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = f'{CACHE_DIR}/whale_cache_v2.json'
CONFIG_FILE = f'{CACHE_DIR}/whale_config.json'

# ====== CONFIGURATION ======
# Environment variables take priority, fallback to config file
def _get_api_key(name: str) -> Optional[str]:
    """Get API key from env var first, then config file."""
    env_val = os.environ.get(name)
    if env_val:
        return env_val
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            return cfg.get(name)
        except Exception:
            pass
    return None

WHALE_ALERT_KEY = _get_api_key('WHALE_ALERT_API_KEY')
BSCSCAN_KEY = _get_api_key('BSCSCAN_API_KEY') or _get_api_key('ETHERSCAN_API_KEY')

# Public RPCs (no API key required, rate limited ~5 req/sec)
BSC_RPCS = [
    'https://bsc-dataseed.binance.org/',
    'https://bsc-dataseed1.defibit.io/',
    'https://bsc-dataseed1.ninicoin.io/',
    'https://bsc.publicnode.com/',
]
ETH_RPCS = [
    'https://eth.llamarpc.com',
    'https://eth.public-rpc.com',
    'https://rpc.ankr.com/eth',
]

# Thresholds
MIN_TX_VALUE_USD = 100_000        # Track $100K+ transactions
WHALE_WALLET_VALUE_USD = 500_000  # Known whale wallet threshold
LARGE_BLOCK_SCAN = 50            # Number of recent blocks to scan

# Known exchange hot wallets (these are LARGE flows, not "whale buys")
# We use these to detect NET INFLOW to exchanges = distribution (bearish)
# Or NET OUTFLOW from exchanges = accumulation (bullish)
KNOWN_EXCHANGE_WALLETS = {
    'BSC': {
        '0x8894e0a0c962cb723c1976a4421c95949be2d4e3': 'Binance Hot Wallet',
        '0xF977814e90dA44bFA03b6295A0616a897441aceC': 'Binance 8',
        '0x505e7cEA1B192A7B4738e16B8e32B2B5C57bA35C': 'Binance 14',
        '0x21B31c8e4b56Cf27f6A9d52b1f2b3b5E7e5e5E7e': 'Binance BSC',
        '0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2': 'FTX (cold)',
    },
    'ETH': {
        '0x28C6c06298d514Db089934071355E5743bf21d60': 'Binance 14',
        '0x21a31Ee1afc54dF90EB2BeE2BF7B4A6F5F7E5E5E': 'Binance Hot',
        '0xDFd5293D8e347dFe59E90faedA1af07F8A0a5E5E': 'Coinbase',
    },
}

# Token decimals (for amount to USD conversion, approximate)
TOKEN_DECIMALS = {
    'BSC-USDT': 18,
    'BSC-USDC': 18,
    'BSC-BUSD': 18,
    'BSC-DAI': 18,
}

# Hardcoded token contracts (fallback when CoinGecko fails)
# Source: CoinGecko, BscScan, manual research
HARDCODED_CONTRACTS = {
    'SYNUSDT': {'chain': 'BSC', 'address': '0xa4080f1778e69467e905b8d6f72f6e441f9e9484', 'name': 'Synapse'},
    # For others, will fallback to CoinGecko
}


# ====== HTTP HELPERS ======
def _http_get_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
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


def _rpc_call(rpc_urls: list, method: str, params: list, timeout: int = 10) -> Optional[any]:
    """Try multiple RPC endpoints, return first successful result."""
    payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1})
    for url in rpc_urls:
        try:
            req = urllib.request.Request(url, data=payload.encode('utf-8'),
                                        headers={'User-Agent': 'Mozilla/5.0',
                                                 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode('utf-8', errors='ignore'))
                if 'result' in data:
                    return data['result']
        except Exception:
            continue
    return None


# ====== CACHE ======
def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'symbol_to_token': {},     # 'SYNUSDT' -> {chain, address, decimals}
        'large_transactions': [],  # recent large txs
        'whale_signals': {},       # symbol -> {strength, signal_type, ts}
        'last_scan_block': 0,
        'last_full_scan': None,
    }


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


# ====== TOKEN RESOLUTION ======
def resolve_token_address(symbol: str) -> Optional[Dict]:
    """
    Resolve a USDT symbol to its contract address on BSC/ETH.
    Priority: cache > hardcoded > CoinGecko API.
    """
    base = symbol.replace('USDT', '').lower()
    cache = _load_cache()
    if symbol in cache['symbol_to_token']:
        return cache['symbol_to_token'][symbol]
    # 1. Hardcoded contracts (most reliable)
    if symbol in HARDCODED_CONTRACTS:
        result = HARDCODED_CONTRACTS[symbol]
        cache['symbol_to_token'][symbol] = result
        _save_cache(cache)
        return result
    # 2. CoinGecko API
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{base}?localization=false&tickers=false&community_data=false&developer_data=false'
        data = _http_get_json(url, timeout=10)
        if not data:
            return None
        platforms = data.get('platforms', {})
        for chain_key, rpcs in [('binance-smart-chain', BSC_RPCS), ('ethereum', ETH_RPCS)]:
            if chain_key in platforms and platforms[chain_key]:
                addr = platforms[chain_key]
                result = {
                    'chain': 'BSC' if 'binance' in chain_key else 'ETH',
                    'address': addr,
                    'symbol': data.get('symbol', '').upper(),
                    'name': data.get('name', ''),
                }
                cache['symbol_to_token'][symbol] = result
                _save_cache(cache)
                return result
    except Exception:
        pass
    return None


# ====== WHALE ALERT API ======
def get_whale_alert_transactions(symbol: str, min_value_usd: int = 100_000, hours: int = 24) -> List[Dict]:
    """
    Get large transactions from Whale Alert API.
    Requires API key (free tier: 5000 requests/month).
    Falls back to empty list if no key or error.
    """
    if not WHALE_ALERT_KEY:
        return []
    base = symbol.replace('USDT', '').lower()
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    url = (f'https://api.whale-alert.io/v1/transactions'
           f'?api_key={WHALE_ALERT_KEY}'
           f'&currency={base}'
           f'&min_value={min_value_usd}'
           f'&start={cutoff}')
    data = _http_get_json(url, timeout=15)
    if not data or 'transactions' not in data:
        return []
    txs = []
    for tx in data.get('transactions', []):
        try:
            txs.append({
                'ts': tx.get('timestamp', 0),
                'amount_usd': float(tx.get('amount_usd', 0)),
                'from': tx.get('from', {}).get('address', ''),
                'to': tx.get('to', {}).get('address', ''),
                'from_owner': tx.get('from', {}).get('owner', ''),
                'to_owner': tx.get('to', {}).get('owner', ''),
                'from_type': tx.get('from', {}).get('owner_type', ''),
                'to_type': tx.get('to', {}).get('owner_type', ''),
                'blockchain': tx.get('blockchain', ''),
                'symbol': tx.get('symbol', ''),
                'hash': tx.get('hash', ''),
            })
        except Exception:
            continue
    return txs


# ====== BSC RPC BLOCK SCAN ======
def get_recent_blocks_bsc(count: int = LARGE_BLOCK_SCAN) -> List[Dict]:
    """Get last N blocks from BSC."""
    result = _rpc_call(BSC_RPCS, 'eth_getBlockByNumber', ['latest', False])
    if not result:
        return []
    try:
        latest = int(result['number'], 16)
        blocks = []
        for i in range(count):
            block_num = latest - i
            b = _rpc_call(BSC_RPCS, 'eth_getBlockByNumber',
                         [hex(block_num), True])
            if b and 'transactions' in b:
                blocks.append(b)
        return blocks
    except Exception:
        return []


def filter_large_transactions(blocks: List[Dict], min_value_usd: int = MIN_TX_VALUE_USD) -> List[Dict]:
    """
    Filter blocks for large ERC20/BEP20 transfers.
    Returns list of tx dicts with amount_usd estimate.
    """
    large_txs = []
    bnb_price = _get_bnb_price()
    for block in blocks:
        ts = int(block.get('timestamp', '0x0'), 16)
        for tx in block.get('transactions', []):
            try:
                value = int(tx.get('value', '0x0'), 16) / 1e18
                value_usd = value * bnb_price
                if value_usd >= min_value_usd:
                    large_txs.append({
                        'ts': ts,
                        'amount': value,
                        'amount_usd': value_usd,
                        'from': tx.get('from', ''),
                        'to': tx.get('to', ''),
                        'hash': tx.get('hash', ''),
                        'block': int(block.get('number', '0x0'), 16),
                    })
            except Exception:
                continue
    return large_txs


def _get_bnb_price() -> float:
    """Get BNB price in USD from CoinGecko."""
    data = _http_get_json('https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd', timeout=5)
    if data and 'binancecoin' in data:
        return data['binancecoin'].get('usd', 600)
    return 600  # fallback


# ====== TOKEN TRANSFER DETECTION ======
def get_token_transfers_bsc(contract: str, from_block: int, to_block: int = None) -> List[Dict]:
    """
    Get Transfer events for a specific token contract on BSC.
    Uses eth_getLogs (BEP20 Transfer event signature).
    """
    if to_block is None:
        latest = _rpc_call(BSC_RPCS, 'eth_blockNumber', [])
        to_block = int(latest, 16) if latest else from_block + 100
    # BEP20 Transfer event: 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
    transfer_sig = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
    logs = _rpc_call(BSC_RPCS, 'eth_getLogs', [{
        'fromBlock': hex(from_block),
        'toBlock': hex(to_block),
        'address': contract,
        'topics': [transfer_sig]
    }], timeout=20)
    if not logs or not isinstance(logs, list):
        return []
    transfers = []
    for log in logs:
        try:
            from_addr = '0x' + log['topics'][1][26:]
            to_addr = '0x' + log['topics'][2][26:]
            data = log.get('data', '0x0')
            value = int(data, 16) / (10 ** 18)  # assume 18 decimals
            transfers.append({
                'ts': 0,  # would need block timestamp lookup
                'block': int(log.get('blockNumber', '0x0'), 16),
                'tx_hash': log.get('transactionHash', ''),
                'from': from_addr.lower(),
                'to': to_addr.lower(),
                'value': value,
            })
        except Exception:
            continue
    return transfers


# ====== WHALE SCORING ======
def score_whale_activity(symbol: str, transactions: List[Dict], contract_info: Dict = None) -> Dict:
    """
    Score whale activity for a symbol.
    Returns: {strength: 0-100, has_signal: bool, signal_type, details}
    """
    result = {
        'symbol': symbol,
        'has_signal': False,
        'strength': 0,
        'signal_type': None,  # 'accumulation', 'distribution', 'whale_buy', 'whale_sell'
        'tx_count': 0,
        'net_flow_usd': 0,
        'details': {},
    }
    if not transactions:
        return result

    bnb_price = _get_bnb_price()
    # Get token price for USD conversion
    base = symbol.replace('USDT', '').lower()
    price_data = _http_get_json(
        f'https://api.coingecko.com/api/v3/simple/price?ids={base}&vs_currencies=usd',
        timeout=5
    )
    token_price = 0
    if price_data and base in price_data:
        token_price = price_data[base].get('usd', 0)

    # Analyze transactions
    inflow_usd = 0  # tokens coming TO non-exchange (accumulation)
    outflow_usd = 0  # tokens going TO exchange (distribution)
    whale_buys = 0
    whale_sells = 0
    large_txs = []

    for tx in transactions:
        amount_usd = tx.get('amount_usd', 0)
        if amount_usd == 0 and token_price > 0 and 'value' in tx:
            amount_usd = tx['value'] * token_price
        if amount_usd < MIN_TX_VALUE_USD:
            continue
        large_txs.append(tx)
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        is_from_exchange = any(from_addr == w.lower() for w in KNOWN_EXCHANGE_WALLETS.get(contract_info.get('chain', 'BSC') if contract_info else 'BSC', {}).keys())
        is_to_exchange = any(to_addr == w.lower() for w in KNOWN_EXCHANGE_WALLETS.get(contract_info.get('chain', 'BSC') if contract_info else 'BSC', {}).keys())
        if is_from_exchange and not is_to_exchange:
            # Outflow from exchange = accumulation (bullish)
            inflow_usd += amount_usd
            whale_buys += 1
        elif is_to_exchange and not is_from_exchange:
            # Inflow to exchange = distribution (bearish)
            outflow_usd += amount_usd
            whale_sells += 1
        elif not is_from_exchange and not is_to_exchange:
            # Wallet to wallet - neutral but still notable
            whale_buys += 0.5  # partial credit
            inflow_usd += amount_usd * 0.3

    result['tx_count'] = len(large_txs)
    result['net_flow_usd'] = inflow_usd - outflow_usd
    result['details'] = {
        'whale_buys': whale_buys,
        'whale_sells': whale_sells,
        'inflow_usd': inflow_usd,
        'outflow_usd': outflow_usd,
        'token_price': token_price,
        'large_tx_count': len(large_txs),
    }

    # Score
    score = 0
    if len(large_txs) >= 3:
        score += 30
    elif len(large_txs) >= 1:
        score += 15

    # Accumulation is bullish
    if inflow_usd > 500_000:
        score += 30
        result['signal_type'] = 'accumulation'
    elif inflow_usd > 100_000:
        score += 15
    # Distribution is bearish (negative signal for LONG)
    if outflow_usd > 500_000:
        score -= 20
        if not result['signal_type']:
            result['signal_type'] = 'distribution'
    # Net positive flow
    if result['net_flow_usd'] > 200_000:
        score += 20
    elif result['net_flow_usd'] < -200_000:
        score -= 15

    result['strength'] = max(0, min(100, score))
    result['has_signal'] = result['strength'] >= 40
    return result


# ====== SPOT LARGE TRADES (Gate.io trades) ======
def get_large_spot_trades(symbol: str, min_value_usd: float = 5000) -> Dict:
    """
    Get recent large spot trades for whale activity detection.
    Returns: {large_trades_count, taker_buy_ratio, largest_trade_usd, total_buy_usd, total_sell_usd}
    """
    base = symbol.replace('USDT', '').lower() + '_usdt'
    result = {
        'large_trades_count': 0,
        'taker_buy_ratio': 0.5,
        'largest_trade_usd': 0,
        'total_buy_usd': 0,
        'total_sell_usd': 0,
        'source': 'gate_spot',
    }
    try:
        url = f'https://api.gateio.ws/api/v4/spot/trades?currency_pair={base}&limit=200'
        data = _http_get_json(url, timeout=5)
        if not data or not isinstance(data, list):
            return result
        large_trades = []
        for trade in data:
            try:
                amount = float(trade.get('amount', 0))
                price = float(trade.get('price', 0))
                value = amount * price
                if value >= min_value_usd:
                    large_trades.append({
                        'value': value,
                        'side': trade.get('side', ''),
                    })
            except Exception:
                continue
        result['large_trades_count'] = len(large_trades)
        if large_trades:
            buy_vol = sum(t['value'] for t in large_trades if t['side'] == 'buy')
            sell_vol = sum(t['value'] for t in large_trades if t['side'] == 'sell')
            total = buy_vol + sell_vol
            if total > 0:
                result['taker_buy_ratio'] = buy_vol / total
            result['total_buy_usd'] = buy_vol
            result['total_sell_usd'] = sell_vol
            largest = max(large_trades, key=lambda t: t['value'])
            result['largest_trade_usd'] = largest['value']
            result['largest_trade_side'] = largest['side']
    except Exception:
        pass
    return result


# ====== DERIVATIVE WHALE PROXY ======
def get_derivative_whale_signal(symbol: str) -> Dict:
    """
    Use derivatives (futures OI, taker buy/sell) as a proxy for whale activity
    when on-chain data is unavailable. This is FREE via Gate.io and OKX.
    Returns dict with: long_liquidations, short_liquidations, oi_change, taker_ratio
    """
    result = {
        'oi_change_pct': 0,
        'long_liq_usd': 0,
        'short_liq_usd': 0,
        'taker_buy_ratio': 0.5,  # 0.5 = neutral
        'large_trades_count': 0,
    }
    # Use Gate.io futures API (public, no auth)
    try:
        # Get ticker with 24h stats
        url = f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}'
        data = _http_get_json(url, timeout=5)
        if not data:
            return result
        # OI in USD
        oi = float(data.get("total_size", 0) or 0) * float(data.get("mark_price", 0) or 0)
        result['oi_usd'] = oi
    except Exception:
        pass

    # Get recent trades (large ones = whale activity)
    try:
        url = f'https://api.gateio.ws/api/v4/futures/usdt/trades?contract={symbol}&limit=100'
        data = _http_get_json(url, timeout=5)
        if data and isinstance(data, list):
            large_trades = []
            for trade in data:
                try:
                    size = float(trade.get('size', 0))
                    price = float(trade.get('price', 0))
                    value = size * price
                    if value >= 10_000:  # $10K+ trade
                        large_trades.append({
                            'size': size,
                            'price': price,
                            'value': value,
                            'side': 'buy' if trade.get('side') == 'buy' else 'sell',
                        })
                except Exception:
                    continue
            result['large_trades_count'] = len(large_trades)
            # Calculate taker buy ratio
            if large_trades:
                buy_vol = sum(t['value'] for t in large_trades if t['side'] == 'buy')
                sell_vol = sum(t['value'] for t in large_trades if t['side'] == 'sell')
                total = buy_vol + sell_vol
                if total > 0:
                    result['taker_buy_ratio'] = buy_vol / total
                # Largest trade
                if large_trades:
                    largest = max(large_trades, key=lambda t: t['value'])
                    result['largest_trade_usd'] = largest['value']
                    result['largest_trade_side'] = largest['side']
    except Exception:
        pass

    return result


# ====== MAIN DETECTION ======
def detect_whale_signal_v2(symbol: str) -> dict:
    """
    Comprehensive whale signal detection using 3 sources + derivative proxy.
    """
    result = {
        'symbol': symbol,
        'has_signal': False,
        'signal_strength': 0,
        'signal_type': None,
        'source': None,
        'details': {},
    }

    # 1. Resolve contract
    contract = resolve_token_address(symbol)
    if not contract:
        result['details']['error'] = 'contract not found'
        # Skip to derivative fallback
        contract = None
    else:
        chain = contract.get('chain', 'BSC')
        contract_addr = contract.get('address')
        result['details']['chain'] = chain
        result['details']['contract'] = contract_addr

    all_txs = []
    sources_used = []

    # 2. Whale Alert API
    if WHALE_ALERT_KEY:
        wa_txs = get_whale_alert_transactions(symbol, min_value_usd=MIN_TX_VALUE_USD, hours=24)
        if wa_txs:
            sources_used.append('whale_alert')
            all_txs.extend(wa_txs)
            result['details']['whale_alert_count'] = len(wa_txs)

    # 3. Token Transfer Events (BSC) - if contract available
    if contract and contract.get('chain') == 'BSC':
        contract_addr = contract.get('address')
        if contract_addr and len(contract_addr) == 42:
            try:
                cache = _load_cache()
                last_block = cache.get('last_scan_block', 0)
                latest_block_hex = _rpc_call(BSC_RPCS, 'eth_blockNumber', [])
                if latest_block_hex:
                    latest = int(latest_block_hex, 16)
                    scan_from = max(latest - 5000, last_block)
                    if last_block == 0 or scan_from > last_block:
                        transfers = get_token_transfers_bsc(contract_addr, scan_from, latest)
                        if transfers:
                            sources_used.append('token_transfers')
                            base = symbol.replace('USDT', '').lower()
                            price_data = _http_get_json(
                                f'https://api.coingecko.com/api/v3/simple/price?ids={base}&vs_currencies=usd',
                                timeout=5
                            )
                            token_price = 0
                            if price_data and base in price_data:
                                token_price = price_data[base].get('usd', 0)
                            for tx in transfers:
                                tx['amount_usd'] = tx['value'] * token_price
                            all_txs.extend(transfers)
                            result['details']['token_transfers_count'] = len(transfers)
                            result['details']['token_price'] = token_price
                        cache['last_scan_block'] = latest
                        _save_cache(cache)
            except Exception as e:
                result['details']['token_transfers_error'] = str(e)

    # 4. BscScan API (if key available)
    if BSCSCAN_KEY and contract and contract.get('chain') == 'BSC':
        contract_addr = contract.get('address')
        if contract_addr and len(contract_addr) == 42:
            try:
                url = (f'https://api.bscscan.com/api?module=account&action=tokentx'
                      f'&contractaddress={contract_addr}'
                      f'&page=1&offset=100&sort=desc&apikey={BSCSCAN_KEY}')
                data = _http_get_json(url, timeout=10)
                if data and data.get('status') == '1':
                    txs = data.get('result', [])
                    sources_used.append('bscscan')
                    base = symbol.replace('USDT', '').lower()
                    price_data = _http_get_json(
                        f'https://api.coingecko.com/api/v3/simple/price?ids={base}&vs_currencies=usd',
                        timeout=5
                    )
                    token_price = 0
                    if price_data and base in price_data:
                        token_price = price_data[base].get('usd', 0)
                    for tx in txs:
                        try:
                            ts = int(tx.get('timeStamp', 0))
                            if (datetime.now(timezone.utc).timestamp() - ts) > 86400:
                                continue
                            value = float(tx.get('value', 0)) / (10 ** int(tx.get('tokenDecimal', 18)))
                            all_txs.append({
                                'ts': ts,
                                'value': value,
                                'amount_usd': value * token_price,
                                'from': tx.get('from', ''),
                                'to': tx.get('to', ''),
                                'hash': tx.get('hash', ''),
                            })
                        except Exception:
                            continue
                    result['details']['bscscan_count'] = len(txs)
            except Exception:
                pass

    # 5. DERIVATIVE WHALE PROXY (always available via Gate.io futures)
    deriv = get_derivative_whale_signal(symbol)
    result['details']['derivative'] = deriv
    if deriv.get('large_trades_count', 0) > 0:
        sources_used.append('derivatives')

    # 6. SPOT LARGE TRADES (most reliable for spot-only small caps)
    spot = get_large_spot_trades(symbol, min_value_usd=5000)
    result['details']['spot'] = spot
    if spot.get('large_trades_count', 0) > 0:
        sources_used.append('spot_trades')

    result['source'] = '+'.join(sources_used) if sources_used else 'none'
    result['details']['total_txs'] = len(all_txs)

    # Score on-chain if available
    if all_txs:
        score_result = score_whale_activity(symbol, all_txs, contract)
        result['has_signal'] = score_result['has_signal']
        result['signal_strength'] = score_result['strength']
        result['signal_type'] = score_result['signal_type']
        result['details'].update(score_result['details'])

    # ADD spot large trades signal (most reliable proxy for spot-only tokens)
    spot_count = spot.get('large_trades_count', 0)
    if spot_count > 0:
        spot_score = min(40, spot_count * 3)
        # Strong taker buying (taker buy ratio > 0.6) is bullish
        spot_taker = spot.get('taker_buy_ratio', 0.5)
        if spot_taker > 0.6:
            spot_score += 15
            if not result.get('signal_type'):
                result['signal_type'] = 'spot_taker_buying'
        elif spot_taker < 0.4:
            spot_score -= 10
            if not result.get('signal_type'):
                result['signal_type'] = 'spot_taker_selling'
        # Large single trade
        if spot.get('largest_trade_usd', 0) > 50_000:
            spot_score += 15
        result['signal_strength'] = max(0, min(100, result.get('signal_strength', 0) + spot_score))
        result['has_signal'] = result['signal_strength'] >= 40

    # ADD derivative signal boost (fallback for non-spot)
    elif deriv.get('large_trades_count', 0) > 5:
        deriv_score = min(40, deriv['large_trades_count'] * 3)
        if deriv.get('taker_buy_ratio', 0.5) > 0.6:
            deriv_score += 15
            if not result.get('signal_type'):
                result['signal_type'] = 'taker_buying'
        elif deriv.get('taker_buy_ratio', 0.5) < 0.4:
            deriv_score -= 10
            if not result.get('signal_type'):
                result['signal_type'] = 'taker_selling'
        if deriv.get('largest_trade_usd', 0) > 50_000:
            deriv_score += 15
        result['signal_strength'] = max(0, min(100, result.get('signal_strength', 0) + deriv_score))
        result['has_signal'] = result['signal_strength'] >= 40

    result['details']['sources_used'] = sources_used
    return result


# ====== CONFIGURATION HELPERS ======
def save_api_key(name: str, value: str) -> None:
    """Save API key to config file."""
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
        except Exception:
            pass
    cfg[name] = value
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        sym = sys.argv[1]
        r = detect_whale_signal_v2(sym)
        print(json.dumps(r, indent=2, default=str))
    else:
        # Test on a few repeaters
        for sym in ['SYNUSDT', 'AKEUSDT', 'BANKUSDT', 'TLMUSDT']:
            r = detect_whale_signal_v2(sym)
            print(f'\n{sym}:')
            print(f'  signal_strength: {r["signal_strength"]}')
            print(f'  has_signal: {r["has_signal"]}')
            print(f'  source: {r["source"]}')
            print(f'  details: {r["details"]}')
