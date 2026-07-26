# 📊 Strategy Comparison — PumpHunter-AI

**Date:** 2026-07-21
**Data:** 17 small-cap symbols, 60 days, 880 snapshots
**Tested:** 10 strategies × 5 TP/SL configs = 50 combinations

---

## 🏆 Best Strategy: S3_MOMENTUM (Momentum Only)

**Criteria:** mom_3 > 2% (LONG) or < -2% (SHORT) + rvol > 1.3
- **Win rate: 79.2%** ✅ (close to 80% target)
- **Profit factor: 94.9** 🤯
- **Total P&L: +196.8%** (60 days)
- **TP/SL:** 2.0-3.0% / 2.0% / trail 0.5%
- **Signals:** 48 over 60 days = 0.8 per day

---

## 📈 Top 5 by Win Rate

| # | Strategy | WR% | Signals | P&L | PF | sig/day |
|---|---|---|---|---|---|---|
| 1 | S3_MOMENTUM | 79.2% | 48 | +196.8% | 94.9 | 0.047 |
| 2 | S10_MULTI | 76.1% | 46 | +117.4% | 40.1 | 0.045 |
| 3 | S7_PULLBACK | 71.4% | 63 | +87.1% | 21.0 | 0.062 |
| 4 | S2_LOOSE | 71.0% | 558 | +829.7% | 22.2 | 0.547 |
| 5 | S5_ANTI_LATE | 67.6% | 757 | +1045.8% | 21.8 | 0.742 |

---

## 💰 Top 5 by Total P&L (volume matters)

| Strategy | P&L | WR% | Signals |
|---|---|---|---|
| S5_ANTI_LATE | +1045.8% | 67.6% | 757 |
| S2_LOOSE | +829.7% | 71.0% | 558 |
| S3_MOMENTUM | +196.8% | 79.2% | 48 |
| S10_MULTI | +117.4% | 76.1% | 46 |
| S1_ULTRA_STRICT | +108.7% | 61.1% | 36 |

---

## 🎯 Current Strategy (S1_ULTRA_STRICT) Analysis

| TP | SL | Trail | WR% | P&L | PF |
|---|---|---|---|---|---|
| 2.0 | 1.5 | 0.5 | 61.1% | +108.7% | 28.9 |
| 2.5 | 2.0 | 0.5 | 61.1% | +108.7% | 28.9 |
| 3.0 | 2.0 | 0.5 | 61.1% | +108.7% | 28.9 |
| 5.0 | 3.0 | 1.5 | 61.1% | +80.3% | 5.5 |
| 7.0 | 3.0 | 1.5 | 61.1% | +80.3% | 5.5 |

**Note:** Current ultra-strict catches fewer signals (36) but with 61% win rate.

---

## 🔬 Strategy Definitions

| ID | Name | Filter |
|---|---|---|
| S1 | ULTRA_STRICT | conf≥60, ATR 3-8, mom3<8, mom6>1.5, anti-late |
| S2 | LOOSE | conf≥40, ATR 2-12 |
| S3 | MOMENTUM | mom_3>±2%, rvol>1.3 |
| S4 | VOLUME | rvol_4h>1.5, rvol_15m>2.0 |
| S5 | ANTI_LATE | only anti-late filter (mom1<4, mom3<10) |
| S6 | BREAKOUT | bb_breakout + rvol>1.5 + mom6>0 |
| S7 | PULLBACK | mom3 in [-3,+1], rvol>1, atr>3 (buy dip) |
| S8 | TREND_FOLLOW | mom6>±2, rvol>1.2, NOT in range |
| S9 | CONF_HIGH | conf≥70 (very high) |
| S10 | MULTI | rvol>1.3 AND mom6>1 AND atr in 3-8 |

---

## 🎯 Recommendations

### For WIN RATE priority (target 80%):
**→ S3_MOMENTUM** with TP=2.5%, SL=2.0%, trail=0.5%
- 79.2% win rate, only 0.8 signals/day
- Best for "quality > quantity" approach

### For P&L priority (maximize total profit):
**→ S5_ANTI_LATE** with TP=2.5%, SL=2.0%, trail=0.5%
- 67.6% win rate, 12.6 signals/day
- Higher volume = more total profit despite lower WR

### Balanced (compromise):
**→ S10_MULTI** with TP=2.5%, SL=2.0%, trail=0.5%
- 76.1% win rate, 0.77 signals/day
- Multi-factor confirmation reduces false signals

---

## 📊 Data Status (as of 2026-07-21 08:00 UTC)

- **Feature log:** 200 rows, 6 cycles, 37 unique symbols
- **Outcomes:** 167 labeled (2 PUMP, 13 DUMP, 152 FLAT)
- **Active signals:** 1 (ALICEUSDT LONG, opened 2026-07-20 13:24)
- **Closed signals:** 2 (ACEUSDT +5.83%, ALICEUSDT -3.88%)
- **Real win rate:** 50% (n=2, statistically not meaningful)

---

## ⚠️ Caveats

1. **Backtest bias:** Historical data doesn't account for slippage, fees, or liquidity issues
2. **Sample size:** 880 snapshots is decent but real-world variance could be higher
3. **No realistic spread modeling:** Real entry/exit prices would have ~0.1-0.2% spread
4. **Survivorship bias:** Symbols that delisted wouldn't appear in 60-day data
5. **Regime dependency:** 60 days may not cover bear market conditions

---

## 🔄 Live Data Collection

- **Status:** Active (background process)
- **Cycle:** Every 10 minutes
- **Log:** /tmp/pumphunter_collector.log
- **Next cycle:** ~10 min after each completion

