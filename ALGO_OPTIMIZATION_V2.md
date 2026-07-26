# Algorithm Optimization Report v2 — 2026-07-22

## 📊 Data Collection (Extended)

| Metric | Value |
|---|---|
| Symbols | 4 (predictable only) |
| Snapshots | 1,744 |
| Features per snapshot | 136 |
| Time period | 436 4h-candles each (~72 days) |
| Pump rate (>3% in 12h) | **24.5%** |

## 🏆 Per-Symbol Optimal Strategy

| Symbol | Historical WR | P&L | Optimal TP/SL | Status |
|---|---|---|---|---|
| **EPICUSDT** | **43.1%** | **+361.6%** | TP=5%, SL=1.5% | ⭐ Best |
| RECALLUSDT | 39.7% | +65.2% | TP=3%, SL=1.5% | ✅ Good |
| OPNUSDT | 34.6% | +10.6% | TP=5%, SL=1.5% | ✅ OK |
| TLMUSDT | 33.5% | -7.4% | TP=5%, SL=1.5% | ⚠️ Marginal |

## 🔬 Top Predictive Features (v2 weights)

### Positive (predict pump):
1. `ind_vwap` (+0.87) — VWAP position
2. `ex_ex_kst` (+0.83) — KST momentum
3. `ex_ex_plus_di` (+0.65) — ADX +DI (trend up)
4. `ex_ex_ulcer_score` (+0.62) — Low downside vol
5. `ind_bb_lower` (+0.55) — BB lower band

### Negative (predict no pump):
1. `ex_ex_kst_signal` (-1.09) — KST signal line above
2. `ind_bb_upper` (-0.67) — BB upper band
3. `ex_ex_minus_di` (-0.60) — ADX -DI (trend down)
4. `ex_ex_vortex_plus` (-0.54) — Vortex weak
5. `ex_ex_kst_score` (-0.54) — KST negative

## 📈 Model Performance

| Model | CV F1 | CV Acc | CV AUC |
|---|---|---|---|
| **Global** | 0.299 | 49.6% | 0.509 |
| EPICUSDT (per-sym) | 0.336 | 43.1% | - |
| RECALLUSDT | 0.260 | 49.1% | - |
| OPNUSDT | 0.247 | 58.2% | - |
| TLMUSDT | 0.228 | 52.5% | - |

## 🛠️ Algorithm v2 Components

### 1. `src/smart_signals.py` (NEW)
- Per-symbol TP/SL from `per_symbol_strategy.json`
- Combined global + per-symbol scoring
- Bad symbol exclusion (USATUSDT)
- Predictable symbol bonus
- Anti-late filter integration

### 2. `data/learned_weights_v2.json` (NEW)
- 64 strong features (>0.15 coefficient)
- Best probability threshold: 0.467
- Trained on 1,744 samples

### 3. `data/per_symbol_models_v2.json` (NEW)
- 4 per-symbol LR models
- Each with top 5 positive/negative features
- Updated pump rate + CV F1 per symbol

### 4. `data/per_symbol_strategy.json` (NEW)
- Optimal TP/SL per symbol
- Based on 1744 snapshots × 12 grid combos

## 🎯 Current Smart Signals

| Symbol | Dir | Entry | TP | SL | Conf | Hist WR |
|---|---|---|---|---|---|---|
| GROVEUSDT | SHORT | 0.0113 | 0.0107 (+5%) | 0.0115 (-1.5%) | 68 | 35% |
| ACEUSDT | SHORT | 0.0850 | 0.0808 (+5%) | 0.0863 (-1.5%) | 52 | 35% |
| TLMUSDT | LONG | 0.0018 | 0.0019 (+5%) | 0.0018 (-1.5%) | 52 | 33% |

## 📉 Limitations

1. **Win rate ceiling: 43%** (EPICUSDT) — well below 80% target
2. **CV AUC: 0.51** — only slightly better than random
3. **Sample bias:** 4 symbols × 1744 snapshots — limited diversity
4. **Features explain ~5%** of variance (R² ~0.05)
5. **80% target likely unachievable** with technical analysis alone

## 🔄 Active Processes

- Collector: every 10 min ✅
- Self-training: every hour ✅
- Smart signals: on-demand ✅
- Active trades: EPICUSDT, OPNUSDT (from manual signals)
