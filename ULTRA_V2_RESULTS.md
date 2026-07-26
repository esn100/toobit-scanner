# Ultra-Strict V2 — Extended Indicators Integration

**Date:** 2026-07-21
**Test:** 17 small-cap symbols, 60 days, 6h snapshot interval
**TP/SL:** 2.5% / 2.0% / trail 0.5%

---

## 🏆 Results

| Version | N | Wins | Win Rate | Total P&L | Avg | PF |
|---|---|---|---|---|---|---|
| V1 (no extended) | 10 | 10 | **100%** | +49.2% | +4.92% | ∞ |
| **V2 (with extended)** | 9 | 9 | **100%** | +47.0% | **+5.23%** | ∞ |

### Key Insights:
- **V2 filtering is more selective**: 9 vs 10 signals (rejected 1 with ex penalty)
- **V2 picks are higher quality**: avg P&L +5.23% vs +4.92%
- **90% overlap** between V1 and V2 — extended adds marginal value
- **Both achieve 100% win rate** (but n=10 is small)

---

## 🎯 Extended Indicators Integrated (8)

| Indicator | Weight | Direction |
|---|---|---|
| **ex_ulcer_score** | ±12 | Downside volatility |
| **ex_cmf** | ±12 | Accumulation/Distribution |
| **ex_ulcer** | ±10 | Ulcer value |
| **ex_uo** | ±8 | Ultimate Oscillator |
| **ex_obv_slope_norm** | ±8 | Volume momentum |
| **ex_stoch_k** | ±8 | Stochastic %K |
| **ex_williams_r** | ±5 | Williams %R |
| **ex_tsi_score** | ±5 | TSI |
| **ex_roc_score** | ±3 | ROC |

Total max: +71 boost / -52 penalty

---

## 📁 Files Modified/Created

- `src/extended_indicators.py` — 99 new features
- `src/ultra_strict_v2.py` — new filter with extended
- `src/live_collector.py` — collects ex_* features every cycle
- `src/backtest_ultra_v2.py` — comparison test
- `src/backtest_extended.py` — indicator importance analysis

---

## ⚠️ Caveats

1. **Sample size: 9-10 trades** — not statistically significant
2. **Both versions had 100% WR** — could be luck
3. **No slippage/spread** modeled
4. **Backtest bias**: only tested on 17 symbols that have full history
5. **Extended indicators** will accumulate in DB over next 7 days
   for real-time validation

---

## 🔄 Next Steps

1. **Live data accumulation**: 7 days × 144 cycles/day = 1000+ cycles
   with all 99 extended features
2. **Real calibration**: Once 30+ outcomes are labeled, run
   logistic regression on extended features
3. **A/B test**: Run V1 and V2 in parallel for 1 week
4. **Drop indicators** that show no edge in real data
