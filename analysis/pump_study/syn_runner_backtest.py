"""
Simulate the new pump_runner strategy on SYN's +145% pump.
Verify it would have captured significantly more than the old +8% TP.
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

# Entry: 2026-06-27 11:00 (where scanner would have triggered)
# peak: 2026-06-30 13:00

def simulate_pump_runner(entry_i, exit_i, entry_price, tp_pct, sl_pct, trail_pct, trail_activate_pct, max_hold):
    """Simulate new strategy on the actual price series."""
    position_size = 0.30  # initial 30%
    entry_value = entry_price * position_size
    total_pnl = 0
    reentry_done = False
    reentry_size = 0.30
    highest = 0
    current_sl = entry_price * (1 - sl_pct/100)
    tp_price = entry_price * (1 + tp_pct/100)

    log = []
    for j in range(entry_i, min(entry_i + max_hold, exit_i + 1, len(rows))):
        cur = rows[j]
        # Update highest
        if cur['h'] > 0:
            cur_pct = (cur['h'] - entry_price)/entry_price*100
            if cur_pct > highest:
                highest = cur_pct
        # Check exit conditions
        # TP hit?
        if cur['h'] >= tp_price:
            pnl = (tp_price - entry_price)/entry_price*100 * position_size
            total_pnl += pnl
            log.append(f"  t+{j-entry_i}h: TP_HIT at {tp_price:.5f}  P&L={pnl:+.2f}%")
            break
        # SL hit?
        if cur['l'] <= current_sl:
            pnl = (current_sl - entry_price)/entry_price*100 * position_size
            total_pnl += pnl
            log.append(f"  t+{j-entry_i}h: SL_HIT at {current_sl:.5f}  P&L={pnl:+.2f}%")
            break
        # Apply smart_exit_v2 logic
        # Stage 1: breakeven @ +2%
        cur_close = cur['c']
        cur_pct_close = (cur_close - entry_price)/entry_price*100
        if cur_pct_close >= 2.0 and current_sl < entry_price:
            current_sl = entry_price
            log.append(f"  t+{j-entry_i}h: BREAKEVEN @ {entry_price:.5f}")
        # Stage 2: lock 5% at +8%
        if cur_pct_close >= 8.0:
            target_sl = entry_price * 1.05
            if current_sl < target_sl:
                current_sl = target_sl
                log.append(f"  t+{j-entry_i}h: LOCK_5% @ {target_sl:.5f}  (price={cur_close:.5f})")
        # Stage 3: lock 10% at +15%
        if cur_pct_close >= 15.0:
            target_sl = entry_price * 1.10
            if current_sl < target_sl:
                current_sl = target_sl
                log.append(f"  t+{j-entry_i}h: LOCK_10% @ {target_sl:.5f}  (price={cur_close:.5f})")
        # Stage 4: trail 8% below highest after +20%
        if cur_pct_close >= 20.0 and highest > 20.0:
            trail_price = cur_close * (1 - trail_pct/100)
            if trail_price > current_sl:
                current_sl = trail_price
                log.append(f"  t+{j-entry_i}h: TRAIL @ {trail_price:.5f}  (price={cur_close:.5f}, highest={highest:.1f}%)")
        # Re-entry check: RSI-like (we use price drop from highest)
        if not reentry_done:
            # Simple: if price dipped 5% from highest AND still in profit
            if highest > 8.0 and cur_pct_close < highest * 0.85 and cur_pct_close > 0:
                # Re-entry triggered
                reentry_value = cur_close * reentry_size
                log.append(f"  t+{j-entry_i}h: REENTRY @ {cur_close:.5f} (30% size)")
                reentry_done = True
                # Add a separate tracking
                reentry_entry = cur_close
                reentry_highest = cur_close
                # Continue with reentry
                reentry_tp = cur_close * 1.20  # TP for reentry
                reentry_sl = cur_close * 0.97
                # Simulate reentry from this point
                for k in range(j+1, min(j + max_hold, len(rows))):
                    rk = rows[k]
                    if rk['h'] > reentry_highest:
                        reentry_highest = rk['h']
                    reentry_pct = (rk['h'] - reentry_entry)/reentry_entry*100
                    # TP
                    if rk['h'] >= reentry_tp:
                        rpnl = 20 * reentry_size
                        total_pnl += rpnl
                        log.append(f"    REENTRY t+{k-entry_i}h: TP @ {reentry_tp:.5f}  P&L=+{rpnl:.2f}%")
                        break
                    # SL
                    if rk['l'] <= reentry_sl:
                        rpnl = (reentry_sl - reentry_entry)/reentry_entry*100 * reentry_size
                        total_pnl += rpnl
                        log.append(f"    REENTRY t+{k-entry_i}h: SL @ {reentry_sl:.5f}  P&L={rpnl:+.2f}%")
                        break
                    # Trail
                    if reentry_pct > 8:
                        trail = rk['c'] * 0.95
                        if trail > reentry_sl:
                            reentry_sl = trail
                # If reentry didn't close, close at max_hold
                else:
                    end_idx = min(j + max_hold - 1, len(rows) - 1)
                    rpnl = (rows[end_idx]['c'] - reentry_entry)/reentry_entry*100 * reentry_size
                    total_pnl += rpnl
                    log.append(f"    REENTRY end @ {rows[end_idx]['c']:.5f}  P&L={rpnl:+.2f}%")
    # If no exit triggered, close at end
    else:
        end_idx = min(entry_i + max_hold - 1, len(rows) - 1)
        cur_close = rows[end_idx]['c']
        pnl = (cur_close - entry_price)/entry_price*100 * position_size
        total_pnl += pnl
        log.append(f"  t+{end_idx-entry_i}h: TIMEOUT @ {cur_close:.5f}  P&L={pnl:+.2f}%")
    return total_pnl, log

# Find the entry point: 2026-06-27 11:00 UTC
entry_ts = int(datetime(2026,6,27,11,0, tzinfo=timezone.utc).timestamp())
exit_ts = int(datetime(2026,6,30,13,0, tzinfo=timezone.utc).timestamp())

entry_i = None; exit_i = None
for i, r in enumerate(rows):
    if r['ts'] == entry_ts: entry_i = i
    if r['ts'] == exit_ts: exit_i = i

entry_price = rows[entry_i]['c']
print(f'Pump Runner Strategy on SYN +145% pump')
print(f'Entry: 2026-06-27 11:00 @ {entry_price:.5f}')
print(f'Peak: 2026-06-30 13:00 @ {rows[exit_i]["h"]:.5f}')
print()

# Run simulation
total_pnl, log = simulate_pump_runner(
    entry_i, exit_i, entry_price,
    tp_pct=30, sl_pct=3.0, trail_pct=8.0,
    trail_activate_pct=20.0, max_hold=24
)
print('Strategy execution:')
for line in log:
    print(line)
print(f'\nTotal P&L (weighted by size): {total_pnl:+.2f}%')
print(f'Equivalent full-size return: {total_pnl/0.30:+.1f}%')

# Compare to old strategy
print(f'\n{"="*70}')
print('COMPARISON: Old strategy (TP=8%, hold=6h) vs new pump_runner')
print(f'{"="*70}')

# Old: TP=8%, SL=3%, hold 6h
old_pnl = 0
entry_v_old = entry_price * 0.30
old_highest = 0
old_sl = entry_price * 0.97
old_tp = entry_price * 1.08
old_log = []
for j in range(entry_i, min(entry_i+6, len(rows))):
    cur = rows[j]
    if cur['h'] >= old_tp:
        pnl = 8.0 * 0.30
        old_pnl = pnl
        old_log.append(f"  t+{j-entry_i}h: TP_HIT @ {old_tp:.5f}  P&L=+2.40%")
        break
    if cur['l'] <= old_sl:
        pnl = -3.0 * 0.30
        old_pnl = pnl
        old_log.append(f"  t+{j-entry_i}h: SL_HIT @ {old_sl:.5f}  P&L=-0.90%")
        break
print(f'\nOld strategy (TP=8%, hold=6h):')
for line in old_log:
    print(line)
print(f'Old P&L: {old_pnl:+.2f}%')

print(f'\nIMPROVEMENT: {total_pnl - old_pnl:+.2f}% ({(total_pnl/old_pnl if old_pnl != 0 else 0):.1f}x)')
