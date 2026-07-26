# Self-Training Loop — PumpHunter-AI

**Date:** 2026-07-22
**Status:** Active

## Architecture

```
┌─────────────────┐     every 10 min      ┌──────────────┐
│ live_collector  │ ──────────────────────>│  SQLite DB   │
│  (data collection)                       │  + CSV log  │
└─────────────────┘                        └──────┬───────┘
                                                   │
                                                   v
                                          ┌─────────────────┐
                                          │  feature_log    │
                                          │  outcome_log     │
                                          └────────┬────────┘
                                                   │
                                          every 1 hour
                                                   v
┌─────────────────┐     every 1 hour       ┌──────────────┐
│  self_train     │ <────────────────────── │  Calibration  │
│  (auto-improve) │                        │  & TP/SL tune │
└────────┬────────┘                        └──────────────┘
         │
         v
  self_trained_params.json  ────>  auto_trader reads
                                      uses new TP/SL/trail
```

## Current Status

| Component | Status |
|---|---|
| live_collector | ✅ Active (PID 1029) |
| self_train loop | ✅ Active (PID 1148) |
| auto_trader | ✅ Reads self_trained_params |
| adaptive weights | ✅ 10 features learned |
| feature importance | ✅ Saved per iteration |

## Latest Calibration (Iter #5)

- **Samples:** 200 (32 pumps, 16% rate)
- **Logistic Regression:** CV F1=0.376, Acc=81.5%
- **Best probability threshold:** 0.639
- **Optimal TP/SL:** TP=3.0%, SL=1.5%, trail=0.5%

## Top Predictive Features

### Positive (predict pump):
1. f_m_obi_bearish (+0.72) — order book imbalance
2. f_macd_hist (+0.70) — momentum
3. f_m_5m_trade_count (+0.66) — activity
4. f_btc_momentum_12_pct (+0.64) — BTC alignment
5. ind_vwap_distance_pct (+0.60) — VWAP

### Negative (predict no-pump):
1. f_atr_expanding (-1.11) — high volatility
2. f_in_range (-0.96) — chop
3. f_independent_mover (-0.82) — uncorrelated
4. f_a_ichi_thickness_pct (-0.65) — thick cloud
5. f_a_ichi_score (-0.57) — Ichimoku weakness

## Algorithm Evolution

Each iteration:
1. **Build training set** (merge features + 12h outcomes)
2. **Logistic regression** → learn feature importance
3. **Find best probability threshold** (max F1)
4. **Grid search TP/SL** (max win rate + P&L)
5. **Adaptive threshold** based on pump rate
6. **Save** to `self_trained_params.json`
7. **auto_trader reads** on next cycle

## Improvement Metrics

| Iteration | Samples | WR | PnL | Params |
|---|---|---|---|---|
| 1 | 0 | - | - | TP=5, SL=3 (default) |
| 2 | 200 | 40% | +23.5% | TP=2.5, SL=1.5 |
| 3 | 200 | 40% | +23.5% | TP=2.5, SL=1.5 |
| 4 | 200 | 40% | +18.4% | TP=3, SL=1.5 |
| 5 | 200 | 40% | +18.4% | TP=3, SL=1.5 |

## Files Created/Modified

- `src/self_train.py` — main training loop
- `src/adaptive_filter.py` — load learned weights
- `src/auto_trader.py` — reads trained params
- `src/ultra_strict_v2.py` — uses adaptive scores
- `run_collector.sh` — 10-min cycle
- `run_self_train.sh` — 1-hour cycle
- `data/self_trained_params.json` — current best params
- `data/self_train_state.json` — full history

## ⚠️ Caveats

1. **Win rate 40%** — below 80% target
2. **Sample bias** — only 6 cycles, dominated by one day
3. **Pump label = 3%** — may be too lenient
4. **Smart v2 simplified** — actual exit may differ
5. **No slippage** modeled

## Next Improvements

1. **Pump threshold dynamic** — adjust based on volatility
2. **Stricter feature selection** — keep only top 5
3. **Ensemble model** — combine LR + RF + GBM
4. **Online learning** — update weights on each new signal
5. **A/B test** V1 vs V2 in parallel
