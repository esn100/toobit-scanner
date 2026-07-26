"""
SYN +145% pump: full hour-by-hour analysis to find the EARLIEST signal.
Window: 2026-06-26T21:00 to 2026-06-30T13:00 UTC
"""
import json
import statistics
from datetime import datetime, timezone

rows_raw = json.load(open('/tmp/syn_1h.json'))
rows = []
for r in sorted(rows_raw, key=lambda x: int(x[0])):
    ts = int(r[0])
    rows.append({
        'ts': ts, 'o': float(r[5]), 'h': float(r[3]), 'l': float(r[4]),
        'c': float(r[2]), 'v': float(r[1]),
        't': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    })

# Find the 145% pump
# Start: 2026-06-27T21:00 = ts ~1780016400
# Peak:   2026-06-30T13:00 = ts ~1780256400
# Find indices
start_ts = int(datetime(2026,6,27,21,0, tzinfo=timezone.utc).timestamp())
end_ts = int(datetime(2026,6,30,13,0, tzinfo=timezone.utc).timestamp())

si = None; ei = None
for i, r in enumerate(rows):
    if r['ts'] == start_ts: si = i
    if r['ts'] == end_ts: ei = i
print(f'Start index: {si}, End index: {ei}')
print(f'Start: {rows[si]["c"]:.5f}, End (peak): {rows[ei]["h"]:.5f}')

# Go back 48h before start
lookback = 48
start_i = si - lookback
print(f'\nLooking at {lookback}h before pump start:')
print(f'  Window: {rows[start_i]["t"]} to {rows[si]["t"]}')

# Print hour-by-hour
def rsi(closes, n=14):
    if len(closes) < n+1: return 50.0
    g, l = [], []
    for i in range(1, n+1):
        ch = closes[i] - closes[i-1]
        g.append(max(ch,0)); l.append(max(-ch,0))
    ag = sum(g)/n; al = sum(l)/n
    for i in range(n+1, len(closes)):
        ch = closes[i] - closes[i-1]
        ag = (ag*(n-1) + max(ch,0))/n
        al = (al*(n-1) + max(-ch,0))/n
    if al == 0: return 100.0
    return 100 - 100/(1+ag/al)

def atr_pct(window):
    if len(window) < 15: return 0
    trs = []
    for i in range(1, len(window)):
        h,l,pc = window[i]['h'], window[i]['l'], window[i-1]['c']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(trs[-14:])/14/window[-1]['c']*100 if window[-1]['c']>0 else 0

# Show key signals
print(f'\n{"time":<22} {"close":>9} {"vs_start":>9} {"rvol":>5} {"max4h":>5} {"mom3%":>7} {"rsi":>5} {"atr%":>5} {"flat_h":>6}')
print('-'*100)
for i in range(start_i, ei+1):
    if i < 30: continue
    window = rows[max(0,i-50):i+1]
    cur = rows[i]
    closes = [r['c'] for r in window]
    vols = [r['v'] for r in window]
    avg_v = sum(vols[:-1])/max(len(vols)-1,1)
    rvol = cur['v']/avg_v if avg_v>0 else 1
    recent = vols[-5:-1] if len(vols)>5 else vols[:-1]
    max4h = max((v/avg_v for v in recent), default=1) if avg_v>0 else 1
    if i >= 3: mom_3 = (cur['c']-rows[i-3]['c'])/rows[i-3]['c']*100
    else: mom_3 = 0
    flat_h = 0
    for j in range(i-1, max(0, i-168), -1):
        if abs(cur['c']-rows[j]['c'])/rows[j]['c']*100 <= 5:
            flat_h = i-j
        else: break
    rsi_v = rsi(closes)
    atr_v = atr_pct(window)
    pct_change = (cur['c']-rows[si]['c'])/rows[si]['c']*100
    marker = ' <-- PEAK' if i == ei else ' <-- START' if i == si else ''
    print(f'{cur["t"]:<22} {cur["c"]:>9.5f} {pct_change:>+8.1f}% {rvol:>5.2f} {max4h:>5.2f} {mom_3:>+6.2f}% {rsi_v:>5.1f} {atr_v:>5.2f} {flat_h:>4}h{marker}')

# What if we had entered at different times?
print(f'\n{"="*100}')
print('BACKTEST: Enter LONG at each hour before peak, hold 6h, TP=+10%, SL=-3%')
print(f'{"="*100}\n')
peak_price = rows[ei]['h']

# Simulate entry at each t-back
results = []
for entry_offset in [48, 24, 12, 6, 4, 2, 1, 0]:
    entry_i = ei - entry_offset
    if entry_i < 30: continue
    entry_price = rows[entry_i]['c']
    # Hold 6h, check if TP or SL hit
    tp_price = entry_price * 1.10
    sl_price = entry_price * 0.97
    # Walk forward
    result = 'TIMEOUT'
    pnl = 0
    for j in range(1, 7):
        if entry_i+j >= len(rows): break
        high = rows[entry_i+j]['h']
        low = rows[entry_i+j]['l']
        # Conservative: SL first, then TP
        if low <= sl_price:
            result = 'SL'; pnl = -3.0; break
        if high >= tp_price:
            result = 'TP'; pnl = 10.0; break
    if result == 'TIMEOUT':
        # Final price
        end_idx = min(entry_i+6, len(rows)-1)
        pnl = (rows[end_idx]['c']-entry_price)/entry_price*100
    results.append((entry_offset, entry_price, result, pnl, rows[entry_i]['t']))

print(f'{"Entry":<10} {"Time":<22} {"Price":>9} {"Result":<8} {"P&L%":>7}')
for off, ep, res, pnl, t in results:
    print(f't-{off}h{"":<5} {t:<22} {ep:>9.5f} {res:<8} {pnl:>+6.2f}%')

# What about 30% TP (more realistic for SYN)?
print(f'\n\nWhat if we used TP=+30%, SL=-3%? (SYN pumps 30-40% typically)')
print(f'{"Entry":<10} {"Result":<8} {"P&L%":>7}')
for off in [48, 24, 12, 6, 4, 2, 1, 0]:
    entry_i = ei - off
    if entry_i < 30: continue
    entry_price = rows[entry_i]['c']
    tp_price = entry_price * 1.30
    sl_price = entry_price * 0.97
    result = 'TIMEOUT'
    pnl = 0
    for j in range(1, 25):  # 24h window
        if entry_i+j >= len(rows): break
        high = rows[entry_i+j]['h']
        low = rows[entry_i+j]['l']
        if low <= sl_price:
            result = 'SL'; pnl = -3.0; break
        if high >= tp_price:
            result = 'TP'; pnl = 30.0; break
    if result == 'TIMEOUT':
        end_idx = min(entry_i+24, len(rows)-1)
        pnl = (rows[end_idx]['c']-entry_price)/entry_price*100
    print(f't-{off}h{"":<5} {result:<8} {pnl:>+6.2f}%')
