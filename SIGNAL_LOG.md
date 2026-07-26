# Signal Log — 2026-07-22

## Open Signal: OPNUSDT LONG

| Field | Value |
|---|---|
| **Time** | 2026-07-22 23:06:55 UTC |
| **Symbol** | OPNUSDT |
| **Direction** | LONG |
| **Signal ID** | OPNUSDT_LONG_1784761615 |
| **Entry Price** | 0.070400 |
| **TP Target** | 0.073920 (+5.0%) |
| **SL** | 0.069344 (-1.5%) |
| **Trailing** | 0.5% |
| **Max Hold** | 8h |
| **Confidence** | 58 (base 54 + adaptive +4) |

### Why this signal:

✅ **Historical pump rate: 25.7%** (RECALLUSDT-style)
- 35 pumps out of 136 snapshots (3 months data)
- Per-symbol F1: 0.274, Acc: 59.5%

✅ **Features (from data):**
- Mom3: +6.0% (early in move, not overextended)
- Mom6: +3.8% (medium trend up)
- ATR: 4.0% (sweet spot)
- Adaptive boost: +4.6 (predictable symbol)

✅ **Strategy:**
- Smart v2 exit: breakeven @+1%, lock @+2%, trail 0.5%
- 8h max hold
- Max loss: -1.5% (after breakeven lock)
- Max gain: +5% (TP) or higher (trail)

### Exit Plan:

| Price Level | Action |
|---|---|
| 0.070400 (entry) | Open |
| 0.071104 (+1%) | SL → 0.070400 (breakeven) |
| 0.072208 (+2%) | SL → 0.071104 (lock +1%) |
| 0.073200 (+4%) | Trail active, SL = 0.073200 × 0.995 |
| 0.073920 (+5%) | **TP HIT** |
| 0.069344 (-1.5%) | **SL HIT** (loss) |
| 8h elapsed | **Max hold exit** |

### Risk/Reward:
- Risk: 1.5% loss
- Reward: 5% gain (or more with trail)
- R:R = 1:3.3

### Status: 🟡 OPEN (age 0.0h)

Updated: 2026-07-22 23:07:00 UTC
