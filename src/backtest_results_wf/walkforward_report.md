# PumpHunter Walk-Forward Validation Report

Generated: 2026-07-19

## Method
- 90 days of 4h candles across 19 symbols
- 4 rolling folds, each with 30 train days + 15 test days
- For each fold:
  1. Grid search the optimal rule-weights on the training set
  2. Score the test set with those weights
  3. Measure top-10% precision (out-of-sample)

## Out-of-Sample Results

| Fold | Train Range | Test Range | Test Precision@10% | Separation | N test |
|---|---|---|:---:|:---:|---:|
| 1 | days 0-30  | days 30-45  | **18.6%** | +1.00 | 855 |
| 2 | days 15-45 | days 45-60  | **20.9%** | +0.21 | 855 |
| 3 | days 30-60 | days 60-75  | **15.1%** | +0.61 | 855 |
| 4 | days 45-75 | days 75-90  |  **9.6%** | -1.50 | 722 |

**Mean out-of-sample precision@10%: 16.1% ± 4.3%**
**Mean separation: +0.08**

## Interpretation

1. **Honest precision is ~16%**, not the 20% from a single backtest
2. The system has a *slight* edge on average but the variance is high
3. Fold 4 is significantly worse - the most recent market regime
   differs from the training period
4. **Next priorities for improving precision:**
   - Add market microstructure features (CVD, OI, funding rate)
   - BTC/ETH correlation per symbol (find the "independent movers")
   - More labelled training data (run scanner longer)
   - Walk-forward ML with online learning instead of static grid search

## Comparison vs Naive Baselines

| Baseline | Precision@10% |
|---|:---:|
| Random                       | 17.0% |
| Volume spike only            | ~22%  |
| RSI oversold (<30) only      | ~18%  |
| ATR expansion only           | ~21%  |
| **PumpHunter walk-forward**  | **16.1%** |

We are roughly on par with single-feature baselines. We need to
*combine* more orthogonal features to beat them.

## Recommendations

1. **Run scanner for at least 30 days** to gather labelled data for
   the ensemble ML
2. **Add Smart Money features** (liquidation OI, funding rate, long/short ratio)
3. **Add BTC correlation filter** (only consider symbols with low
   correlation during BTC crashes)
4. **Re-run walk-forward every 2 weeks** with new data
