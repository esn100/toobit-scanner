# Extended Indicator Analysis — PumpHunter-AI

**Date:** 2026-07-21
**Methodology:** 933 snapshots, 17 small-cap symbols, 60 days
**Label:** +3% in 12h = PUMP (28.1% positive rate)
**Total new indicators tested:** 99

---

## 🏆 Top 10 (KEEP — actually predictive)

| # | Indicator | Type | Pearson | WR Top 25% | WR Bot 25% | Diff |
|---|---|---|---|---|---|---|
| 1 | ex_ulcer_score | Volatility | -0.116 | 17.1% | 35.9% | **+18.8%** |
| 2 | ex_cmf | Volume | -0.122 | 17.5% | 32.5% | **+15.0%** |
| 3 | ex_ulcer | Volatility | +0.067 | 35.9% | 17.1% | +18.8% |
| 4 | ex_uo (Ultimate Osc) | Momentum | -0.110 | 18.8% | 31.6% | -12.8% |
| 5 | ex_obv_norm | Volume | -0.045 | 21.8% | 34.2% | -12.4% |
| 6 | ex_stoch_k | Momentum | -0.090 | 20.5% | 32.1% | -11.5% |
| 7 | ex_williams_r | Momentum | -0.090 | 20.5% | 32.1% | -11.5% |
| 8 | ex_tsi | Momentum | -0.101 | 25.2% | 37.2% | -12.0% |
| 9 | ex_tsi_score | Momentum | -0.101 | 25.2% | 37.2% | -12.0% |
| 10 | ex_roc_score | Momentum | -0.082 | 24.4% | 32.5% | -8.1% |

### Interpretation:
- **Ulcer Index** (downside volatility): High = no pump, Low = pump likely
  - This means: when coin has been volatile on the downside recently, less likely to pump
  - Useful for AVOIDING bad entries
- **CMF** (Chaikin Money Flow): Negative = distribution, Positive = accumulation
  - Distribution phase → less likely to pump
- **OBV normalized**: Volume momentum - useful confirmation
- **Stochastic / Williams %R / TSI / ROC**: All classic oversold indicators
  - When VERY oversold → bounce/pump more likely
  - When OVERBOUGHT → reversal

---

## 🗑️ Bottom 10 (DROP — no signal)

| Indicator | Reason |
|---|---|
| ex_trix_positive | Chi-p = 1.0 (no signal) |
| ex_ao_saucer | Chi-p = 1.0 (too rare) |
| ex_psar_* (3) | Chi-p = 1.0 (no direction edge) |
| ex_adx_strong | Chi-p = 1.0 (no signal in small caps) |
| ex_mfi_oversold | Binary thresholds don't work |
| ex_cci_oversold | Binary thresholds don't work |
| ex_uo_oversold | Binary thresholds don't work |

### Key Finding:
**Binary "oversold/overbought" thresholds DON'T work for small caps.**
Better to use the continuous values directly (e.g., `ex_stoch_k`, `ex_cci` as numbers).

---

## 📊 Category Performance

| Category | Top score | Best indicator | Useful? |
|---|---|---|---|
| Volatility | 0.458 (ulcer_score) | ex_ulcer | ✅ YES |
| Volume | 0.452 (cmf) | ex_cmf | ✅ YES |
| Momentum | 0.431 (uo) | ex_uo | ✅ YES |
| Trend | 0.39 (tsi in mom category) | ex_tsi | ⚠️ Limited |
| Reversal | All chi-p=1 | n/a | ❌ NO |

---

## 🎯 Recommendations

### ✅ KEEP (high value):
- **Ulcer Index** (`ex_ulcer`, `ex_ulcer_score`) — best predictor
- **CMF** (`ex_cmf`, `ex_cmf_positive`) — volume proxy
- **OBV normalized** (`ex_obv_norm`, `ex_obv_slope_norm`) — momentum
- **Stochastic** (`ex_stoch_k`, `ex_stoch_d`, `ex_stoch_crossover`)
- **Williams %R** (`ex_williams_r`)
- **TSI** (`ex_tsi`, `ex_tsi_score`)
- **Ultimate Oscillator** (`ex_uo`)
- **ROC** (`ex_roc_score`)

### 🗑️ DROP (no value):
- All Parabolic SAR features (chi-p = 1.0)
- All binary "oversold/overbought" flags
- Mass Index (was 27.4 but no edge)
- ADX strong/weak flags (no edge)

### ⚠️ TEST MORE:
- SuperTrend (need more data)
- Aroon (need trend confirmation)
- Vortex, KST, TRIX (close to top 30)

---

## 🔧 Implementation Status

| Component | Status |
|---|---|
| Module `extended_indicators.py` | ✅ 99 features |
| Live collector integration | ✅ Active |
| Feature log accumulation | 🔄 In progress |
| Real-time backtest | ⏳ Pending 30+ samples |
| Production filter | ⏳ After validation |

---

## 📈 Next Steps

1. **Wait for 7 days** of live data with extended indicators
2. **Run real calibration** on actual outcomes
3. **Integrate top 8-10** into the ultra_strict filter
4. **Test new combined strategy** with extended features
