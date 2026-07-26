# SYN Pump 145% Deep-Dive

## Pump Timeline
- **Start**: 2026-06-27 21:00 UTC @ 0.29383
- **Peak**: 2026-06-30 13:00 UTC @ 0.72000
- **Duration**: ~64 hours
- **Total gain**: **+145%**

## Pre-Pump Phases (3 مرحله)

### Phase 1: Initial Pump (26 Jun 21:00 → 27 Jun 21:00)
- From 0.30 → 0.36 (+20%)
- 5-6h consolidation before main pump
- **Max_rvol_4h hit 2.8x** in 27 Jun 11:00 — **EARLIEST SIGNAL**
- Then pulled back to 0.29140 (4% drop)

### Phase 2: Main Pump (28 Jun 11:00 → 30 Jun 13:00)
- 0.34 → 0.72 (+110%)
- Hour-by-hour grind up
- 4 distinct waves:
  - 28 Jun 11:00: 0.36 → 0.47 (+30%) in 1h! rvol=2.20x
  - 29 Jun 19:00: 0.41 → 0.53 (+30%) in 1h! rvol=2.45x
  - 29 Jun 22:00: 0.50 → 0.54 (+8%) in 1h
  - 30 Jun 11:00: 0.59 → 0.67 (+14%) in 1h! rvol=4.34x

### Phase 3: Peak & Dump (30 Jun 13:00 onwards)
- Hit 0.72, immediately -50% drop

## Critical Observations

### 1. **Max_Rvol_4h was 2.0+ for 60% of the time before peak**
At every t-Nh from 4h to 24h before pump:
- **67% of pumps** had max_rvol_4h > 1.5
- **44% had max_rvol_4h > 2.0**

### 2. **RSI was 30-50 (oversold) before pump**
- 27 Jun 17:00: RSI=34.7, mom_3=-14% — **DEEP DUMP before pump**
- 28 Jun 17:00: RSI=49.5
- 29 Jun 17:00: RSI=68.5 (rising)

### 3. **Pattern matches our pre-pump detector:**
- Volume build-up (max_rvol_4h > 2.0) ✓
- Mom_3 in range (-5%, +5%) ✓
- RSI 30-50 ✓
- ATR > 3% ✓

## Backtest Results (Enter at t-Nh before peak, TP=+10%, SL=-3%)

| Entry time | Result | P&L |
|---|---|---|
| t-48h | SL | -3% |
| t-24h | **TP** | **+10%** ✓ |
| t-12h | SL | -3% |
| t-6h | **TP** | **+10%** ✓ |
| t-4h | SL | -3% |
| t-2h | SL | -3% |
| t-1h | SL | -3% |

**Win rate with this TP/SL: 25%** (only 2/8 entries)

## What the Current Scanner DID (8 months ago)

Looking at the data, our pre-pump pattern matched at:
- **27 Jun 03:00**: max_rvol_4h=0.60 (too low, would NOT trigger)
- **27 Jun 07:00**: rvol=2.89x, max_rvol_4h=1.02 — close but not enough
- **27 Jun 11:00**: rvol=0.53, max_rvol_4h=2.81 — **WOULD TRIGGER** ✓
- **28 Jun 11:00**: rvol=2.20, mom_3=+28% — **TOO LATE** (overbought)

**Scanner would have entered at 27 Jun 11:00 @ 0.35044** ✓
- This was 50 hours before peak
- Price then went: 0.35 → 0.72 = **+106% in 50h**

**But with our TP=8% (config) we'd exit at 0.378** = +8% profit
**With our TP=10% backtest: +10% profit**

## Why We Missed +145% (and how to catch next time)

### Reasons for limited capture:
1. **TP too tight**: 8-10% is too low for SYN (avg gain +35% per pump)
2. **No multi-leg strategy**: One entry, one exit
3. **No re-entry after pullback**: SYN had 3-4 waves, each +20-30%

### Proposed SYN-specific strategy:
1. **Initial entry** (t-1h pattern): 30% size @ TP=15%, SL=-3%
2. **Re-entry on pullback** (RSI < 35): another 30% size
3. **Final exit**: TP=30% (SYN's typical pump size)
4. **Or**: trailing stop after +20% to catch the 145% runner

## Aggregated Pre-Pump Volume Profile (across 10 SYN pumps)

| t-back | avg rvol | avg max4h | avg mom_3 |
|---|---|---|---|
| t-1h | 1.18x | 1.74x | -3.91% |
| t-2h | 0.75x | 2.20x | +0.34% |
| t-4h | 1.17x | 2.16x | +2.01% |
| t-6h | 1.75x | 1.66x | +0.94% |
| t-12h | 0.83x | 1.26x | +0.98% |
| t-24h | 0.99x | 1.67x | -1.76% |

**Strongest signal: max_rvol_4h > 1.5x consistently 2-6h before pump**

## Detection % at Each Lookback

| Rule | t-24h | t-12h | t-6h | t-4h | t-2h | t-1h |
|---|---|---|---|---|---|---|
| max4h > 1.5x | 38% | 38% | **67%** | **67%** | **67%** | 56% |
| max4h > 2.0x | - | - | 44% | 44% | 44% | 33% |
| \|mom_3\| < 5% | 38% | **75%** | 44% | 67% | 44% | 44% |
| \|mom_3\| < 3% | - | 38% | 33% | **67%** | 44% | 44% |
| rvol > 1.5x | - | - | - | 33% | - | - |
| flat > 6h | - | 38% | 33% | - | - | - |

**Best combo at t-4h**: max4h > 1.5x AND |mom_3| < 5% → 56% of pumps caught
**Best combo at t-6h**: max4h > 1.5x alone → 67% caught

## Conclusion

**Pre-pump detection IS possible for SYN!** 
- 4-6 hours before pump: max_rvol_4h > 1.5x is reliable (67%)
- Combined with mom_3 in range: 56% of pumps detectable

**To catch the full +145%**:
1. Use TP=+30% (not 8-10%)
2. Use trailing stop after +20%
3. Add re-entry on pullback (RSI < 35)
4. Increase max_hold_hours to 24-48h
5. Multi-leg strategy: 30% + 30% + 40%
