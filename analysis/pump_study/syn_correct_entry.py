"""
CORRECTED backtest: Entry at 28 Jun 11:00 (peak of wave 1, before wave 2+3)
Simulate pump_runner strategy on SYN's full +145% pump.
"""
import json
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

def simulate_strategy(entry_i, rows, tp_pct, sl_pct, trail_pct, trail_activate_pct, max_hold):
    """Simulate new strategy on actual price series."""
    entry_price = rows[entry_i]['c']
    position_size = 0.30
    reentry_size = 0.30
    highest = 0
    current_sl = entry_price * (1 - sl_pct/100)
    tp_price = entry_price * (1 + tp_pct/100)
    total_pnl_pct = 0
    log = []
    reentry_done = False
    reentry_pnl = 0

    n = len(rows)
    for j in range(entry_i, min(entry_i + max_hold, n)):
        cur = rows[j]
        cur_pct_high = (cur['h'] - entry_price)/entry_price*100
        cur_pct_low = (cur['l'] - entry_price)/entry_price*100
        cur_pct_close = (cur['c'] - entry_price)/entry_price*100
        if cur_pct_high > highest:
            highest = cur_pct_high
        # Check exits (within the hour)
        if cur['h'] >= tp_price:
            pnl = tp_pct * position_size
            total_pnl_pct += pnl
            log.append(f"  t+{j-entry_i:>2}h: TP_HIT @ {tp_price:.5f}  P&L=+{pnl:.2f}%")
            return total_pnl_pct + reentry_pnl, log
        if cur['l'] <= current_sl:
            pnl = (current_sl - entry_price)/entry_price*100 * position_size
            total_pnl_pct += pnl
            log.append(f"  t+{j-entry_i:>2}h: SL_HIT @ {current_sl:.5f}  P&L={pnl:+.2f}%")
            return total_pnl_pct + reentry_pnl, log
        # Smart exit v2 (pump_runner mode)
        if cur_pct_close >= 2.0 and current_sl < entry_price:
            current_sl = entry_price
            log.append(f"  t+{j-entry_i:>2}h: BREAKEVEN @ {entry_price:.5f} (cur_pct={cur_pct_close:.1f}%)")
        if cur_pct_close >= 8.0:
            target_sl = entry_price * 1.05
            if current_sl < target_sl:
                current_sl = target_sl
                log.append(f"  t+{j-entry_i:>2}h: LOCK_5% @ {target_sl:.5f}  (cur_pct={cur_pct_close:.1f}%)")
        if cur_pct_close >= 15.0:
            target_sl = entry_price * 1.10
            if current_sl < target_sl:
                current_sl = target_sl
                log.append(f"  t+{j-entry_i:>2}h: LOCK_10% @ {target_sl:.5f}  (cur_pct={cur_pct_close:.1f}%)")
        if cur_pct_close >= trail_activate_pct and highest > trail_activate_pct:
            trail_price = cur['c'] * (1 - trail_pct/100)
            if trail_price > current_sl:
                current_sl = trail_price
                log.append(f"  t+{j-entry_i:>2}h: TRAIL @ {trail_price:.5f}  (highest={highest:.1f}%)")
        # Re-entry trigger: -10% pullback from highest while still in profit
        if not reentry_done and highest > 10.0 and cur_pct_close < highest - 8.0 and cur_pct_close > 3.0:
            log.append(f"  t+{j-entry_i:>2}h: REENTRY @ {cur['c']:.5f} (pullback from {highest:.1f}% to {cur_pct_close:.1f}%)")
            reentry_done = True
            # Simulate reentry as separate position
            reentry_entry = cur['c']
            reentry_high = cur['c']
            reentry_sl = reentry_entry * 0.97
            reentry_tp = reentry_entry * 1.20  # 20% TP for reentry
            for k in range(j+1, min(j + max_hold, n)):
                rk = rows[k]
                rk_pct_high = (rk['h'] - reentry_entry)/reentry_entry*100
                if rk_pct_high > reentry_high:
                    reentry_high = rk_pct_high
                # TP
                if rk['h'] >= reentry_tp:
                    rpnl = 20.0 * reentry_size
                    reentry_pnl += rpnl
                    log.append(f"    REENTRY t+{k-entry_i:>2}h: TP_HIT @ {reentry_tp:.5f}  P&L=+{rpnl:.2f}%")
                    break
                # SL
                if rk['l'] <= reentry_sl:
                    rpnl = (reentry_sl - reentry_entry)/reentry_entry*100 * reentry_size
                    reentry_pnl += rpnl
                    log.append(f"    REENTRY t+{k-entry_i:>2}h: SL_HIT @ {reentry_sl:.5f}  P&L={rpnl:+.2f}%")
                    break
                # Trail 5% after +10%
                rk_pct = (rk['c'] - reentry_entry)/reentry_entry*100
                if rk_pct > 10.0 and reentry_high > 10.0:
                    trail = rk['c'] * 0.95
                    if trail > reentry_sl:
                        reentry_sl = trail
            else:
                # Timeout
                end_k = min(j + max_hold - 1, n - 1)
                rpnl = (rows[end_k]['c'] - reentry_entry)/reentry_entry*100 * reentry_size
                reentry_pnl += rpnl
                log.append(f"    REENTRY timeout @ {rows[end_k]['c']:.5f}  P&L={rpnl:+.2f}%")
    # Timeout
    end_idx = min(entry_i + max_hold - 1, n - 1)
    final_close = rows[end_idx]['c']
    pnl = (final_close - entry_price)/entry_price*100 * position_size
    total_pnl_pct += pnl
    log.append(f"  t+{end_idx-entry_i:>2}h: TIMEOUT @ {final_close:.5f}  P&L={pnl:+.2f}%")
    return total_pnl_pct + reentry_pnl, log

# Entry: 29 Jun 00:00 UTC (low point, before main wave 2+3)
entry_ts = int(datetime(2026,6,29,0,0, tzinfo=timezone.utc).timestamp())
entry_i = None
for i, r in enumerate(rows):
    if r['ts'] == entry_ts:
        entry_i = i; break

entry_price = rows[entry_i]['c']
print(f'CORRECTED: Pump Runner Strategy on SYN +145% pump')
print(f'Entry: 2026-06-28 11:00 @ {entry_price:.5f}')
print(f'Peak:  2026-06-30 13:00 @ 0.72000 (max forward 64h)')
print()

total_pnl, log = simulate_strategy(
    entry_i, rows,
    tp_pct=30, sl_pct=3.0, trail_pct=8.0,
    trail_activate_pct=20.0, max_hold=64
)
print('Strategy execution (entry at wave-1 peak, holding through wave 2+3):')
for line in log:
    print(line)
print(f'\nTotal P&L (weighted by size): {total_pnl:+.2f}%')
print(f'Full-position equivalent: {total_pnl/0.30:+.1f}%')

# Compare to old strategy
print(f'\n{"="*70}')
print('COMPARISON: Old (TP=8%, hold=6h) vs New pump_runner')
print(f'{"="*70}')

# Old: simple TP/SL with 6h hold
old_highest = 0
old_pnl_pct = 0
old_entry = entry_price
old_sl = old_entry * 0.97
old_tp = old_entry * 1.08
old_log = []
for j in range(entry_i, min(entry_i+6, len(rows))):
    cur = rows[j]
    if cur['h'] >= old_tp:
        old_pnl_pct = 8.0 * 0.30
        old_log.append(f"  t+{j-entry_i}h: TP_HIT @ {old_tp:.5f}  P&L=+{old_pnl_pct:.2f}%")
        break
    if cur['l'] <= old_sl:
        old_pnl_pct = -3.0 * 0.30
        old_log.append(f"  t+{j-entry_i}h: SL_HIT @ {old_sl:.5f}  P&L={old_pnl_pct:+.2f}%")
        break
else:
    end_idx = min(entry_i+5, len(rows)-1)
    pnl = (rows[end_idx]['c'] - old_entry)/old_entry*100 * 0.30
    old_pnl_pct = pnl
    old_log.append(f"  t+{end_idx-entry_i}h: TIMEOUT @ {rows[end_idx]['c']:.5f}  P&L={pnl:+.2f}%")

print(f'\nOld (TP=8%, hold=6h):')
for line in old_log:
    print(line)
print(f'Old P&L: {old_pnl_pct:+.2f}%')

print(f'\nNEW pump_runner P&L: {total_pnl:+.2f}%')
print(f'IMPROVEMENT: {total_pnl - old_pnl_pct:+.2f}% ({(total_pnl/max(old_pnl_pct,0.01)):.1f}x better)')
