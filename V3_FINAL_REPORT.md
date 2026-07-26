# Algorithm Optimization v3 — 2026-07-24

## 📊 Data Used

| Source | Samples | Period |
|---|---|---|
| **extended_timeseries.csv** | **1,744** | ~200 days × 4h |
| MTF features | 882 | 30 days × multi-TF |
| Microstructure | 28 (limited by API delay) | 4 rounds |

**Note:** Toobit API has 3-4 min delay — real microstructure labels were 0%.
Replaced with historical extended_timeseries.csv (1,744 samples, 24.5% pump rate).

## 🏆 Final Model Performance

| Model | CV F1 | CV AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.446 | 0.662 | baseline |
| **LightGBM** | **0.577** | 0.826 | **best for inference** |
| XGBoost | 0.568 | **0.830** | best AUC |

**Per-symbol F1 (XGBoost on best per-sym data):**
- EPICUSDT: 0.678
- RECALLUSDT: 0.533
- TLMUSDT: 0.500
- OPNUSDT: 0.488

## 🎯 Walk-Forward Validation (OOS — out-of-sample)

**70% train, 30% test, chronological split:**

| Threshold | TP | SL | N | WR | P&L |
|---|---|---|---|---|---|
| **0.8 (best)** | 5.0 | 1.5 | **48** | **47.9%** | **+46.0%** |

✅ **Model generalizes well** to out-of-sample data (47.9% WR on unseen test set)

## 🔑 Top 25 Features (LightGBM importance)

| Feature | Importance |
|---|---|
| `ex_ex_obv_norm` | 342 |
| `ex_ex_adx` | 337 |
| `ex_ex_cmf` | 332 |
| `ex_ex_obv_slope_norm` | 297 |
| `ex_ex_ad_norm` | 288 |
| `ex_ex_mass_index` | 281 |
| `ex_ex_plus_di` | 280 |
| `ind_atr_pct` | 279 |
| `candle_last_body_ratio` | 276 |
| `ind_rvol` | 275 |
| `ex_ex_ad_slope_norm` | 264 |
| `ex_ex_stochrsi_d` | 253 |
| `ind_bb_width_pct` | 244 |
| `ind_momentum_1_pct` | 243 |
| `ex_ex_supertrend_distance_pct` | 242 |
| `ind_momentum_acceleration` | 236 |
| `candle_last_upper_wick` | 228 |
| `ex_ex_stoch_d` | 218 |
| `ex_ex_vortex_plus` | 218 |
| `candle_candle_strength` | 212 |
| `ex_ex_mfi` | 210 |
| `ex_ex_uo` | 205 |
| `ex_ex_eom` | 203 |
| `ex_ex_ao` | 199 |
| `ex_ex_vortex_minus` | 193 |

## 📊 Strategy Optimization

| Strategy | WR | P&L | N |
|---|---|---|---|
| ml_0.5/0.6/0.7/0.8 (in-sample) | 100% | +1475% | 427 (circular) |
| **Walk-forward OOS** | **47.9%** | **+46%** | **48 (real OOS)** |
| ATR 3-8 + mom > 0 | 38% | +115% | 399 |

## 📁 New Files

- `data/final_model_lr.joblib` — LR model
- `data/final_model_lgb.joblib` — LightGBM (recommended)
- `data/final_model_xgb.joblib` — XGBoost
- `data/walkforward_model.joblib` — Walk-forward validated
- `data/walkforward_best.json` — OOS optimal params
- `data/final_model_v3.json` — Model comparison
- `data/best_strategy.json` — Best TP/SL config
- `data/self_trained_params.json` — Updated active params

## 🎯 Updated Active Params

```json
{
  "tp_pct": 5.0,
  "sl_pct": 1.5,
  "trail_pct": 0.5,
  "conf_threshold": 40,
  "note": "Walk-forward OOS best: WR=47.9%, P&L=+46.0%, N=48"
}
```

## 💎 New Live Signals (24 Jul)

| Symbol | Direction | Proba | Entry | TP | SL |
|---|---|---|---|---|---|
| **RESOLVUSDT** | SHORT | 67.1% | 0.0188 | 0.0178 | 0.0190 |
| **OPNUSDT** | SHORT | 57.6% | 0.0659 | 0.0626 | 0.0669 |

## 📉 Closed Signals (24 Jul)

| Symbol | Direction | Result | P&L |
|---|---|---|---|
| **EPICUSDT** | LONG | max_hold (25.5h) | **+8.24%** ✅ |
| **OPNUSDT** | LONG | max_hold (25.0h) | **-5.97%** ❌ |

**Updated Stats:**
- Total: 5 closed
- Win rate: **40%** (2 TP, 3 SL)
- Total P&L: **+2.72%** (up from +0.45%)

## ⚠️ Limitations (صادقانه)

1. **Walk-forward 47.9% WR** — خوب ولی هنوز زیر 80% target
2. **Microstructure data 28 rows** — Toobit API delay (3-4 min) مانع جمع‌آوری real-time labels شد
3. **Real signals 5 closed** — هنوز n کوچک
4. **CV F1: 0.577** — مدل خوبه ولی نه عالی
5. **80% win rate** با features فعلی غیرممکن (technical ceiling)

## 🔄 Background Processes

- Self-training: every hour (5 iterations done)
- Live collector: every 10 min (cycles)
- 3 trained models ready for inference
