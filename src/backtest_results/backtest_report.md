# PumpHunter Backtest Report

Generated: 2026-07-19T07:53:21.326773+00:00

**Samples**: 5016
**Features**: 28
**Positive rate**: 0.177

**Random Forest CV F1**: 0.283 ± 0.082
**Logistic Regression CV F1**: 0.335 ± 0.034

## 1. Pearson correlation with success
| Feature | r |
|---|---:|
| in_range | -0.1540 |
| atr_pct | +0.0932 |
| candle_strength | -0.0616 |
| vwap_distance_pct | -0.0563 |
| ema_alignment | -0.0558 |
| rsi_value | -0.0530 |
| price_above_vwap | -0.0505 |
| momentum_6_pct | -0.0498 |
| macd_divergence | +0.0449 |
| momentum_12_pct | -0.0417 |
| big_wick_top | +0.0395 |
| mtf_alignment | -0.0338 |
| momentum_3_pct | -0.0332 |
| higher_highs | +0.0329 |
| momentum_1_pct | -0.0281 |

## 2. Chi-Square significance
| Feature | chi2 | p | significant |
|---|---:|---:|:---:|
| atr_pct | 190.38 | 0.0000 | ✓ |
| rvol | 20.03 | 0.0000 | ✓ |
| vwap_distance_pct | 20.03 | 0.0000 | ✓ |
| momentum_6_pct | 19.38 | 0.0000 | ✓ |
| ema_alignment | 17.08 | 0.0000 | ✓ |
| candle_strength | 15.09 | 0.0001 | ✓ |
| macd_hist | 11.84 | 0.0006 | ✓ |
| momentum_3_pct | 10.84 | 0.0010 | ✓ |
| mtf_alignment | 8.11 | 0.0044 | ✓ |
| rsi_value | 7.29 | 0.0069 | ✓ |
| momentum_12_pct | 6.90 | 0.0086 | ✓ |
| momentum_1_pct | 4.76 | 0.0291 | ✓ |
| momentum_acceleration | 3.02 | 0.0821 | — |
| macd_divergence | 1.17 | 0.2788 | — |
| rsi_divergence | 0.28 | 0.5978 | — |

## 3. Mutual Information
| Feature | MI |
|---|---:|
| atr_pct | 0.0332 |
| macd_hist | 0.0194 |
| in_range | 0.0183 |
| momentum_12_pct | 0.0134 |
| momentum_1_pct | 0.0101 |
| mtf_alignment | 0.0092 |
| momentum_6_pct | 0.0088 |
| higher_lows | 0.0083 |
| volume_spike | 0.0067 |
| vwap_distance_pct | 0.0051 |
| momentum_3_pct | 0.0047 |
| price_above_vwap | 0.0046 |
| big_wick_top | 0.0034 |
| candle_strength | 0.0030 |
| rsi_value | 0.0017 |

## 4. Logistic Regression coefficients
| Feature | coef |
|---|---:|
| in_range | -0.5145 |
| atr_pct | +0.1719 |
| higher_highs | +0.1539 |
| bb_squeeze | +0.1477 |
| bb_breakout_above | +0.1320 |
| candle_strength | -0.1257 |
| vwap_distance_pct | +0.1215 |
| atr_expanding | -0.1212 |
| momentum_6_pct | -0.1096 |
| rsi_value | -0.0957 |
| momentum_1_pct | -0.0916 |
| power_streak | +0.0887 |
| macd_divergence | +0.0881 |
| momentum_acceleration | +0.0833 |
| ema_alignment | -0.0811 |

## 5. Random Forest feature importance
| Feature | importance |
|---|---:|
| atr_pct | 0.1895 |
| macd_hist | 0.0977 |
| momentum_6_pct | 0.0688 |
| momentum_acceleration | 0.0661 |
| momentum_12_pct | 0.0628 |
| vwap_distance_pct | 0.0597 |
| rsi_value | 0.0595 |
| momentum_3_pct | 0.0563 |
| mtf_alignment | 0.0555 |
| in_range | 0.0555 |
| rvol | 0.0554 |
| momentum_1_pct | 0.0554 |
| candle_strength | 0.0498 |
| macd_divergence | 0.0086 |
| higher_highs | 0.0085 |

## 6. Sub-score bucket correlation
| Sub-score | Pearson r |
|---|---:|
| technical | -0.0261 |
| momentum | -0.0414 |
| volume | +0.0168 |
| vwap | -0.0577 |
| atr_bb | +0.0900 |
| structure | -0.0615 |
| candle | +0.0113 |
| mtf | -0.0338 |
| pattern | +0.0201 |

## 7. Grid-search optimal weights
**Top-10% precision**: 0.205 (baseline 0.183)
**Score separation**: -0.93

| Sub-score | Weight | Normalised |
|---|---:|---:|
| technical | 8 | 9.9% |
| momentum | 8 | 9.9% |
| volume | 12 | 14.8% |
| vwap | 4 | 4.9% |
| atr_bb | 10 | 12.3% |
| structure | 15 | 18.5% |
| candle | 12 | 14.8% |
| mtf | 4 | 4.9% |
| pattern | 8 | 9.9% |
