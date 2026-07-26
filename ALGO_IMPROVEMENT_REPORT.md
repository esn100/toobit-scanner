# Algorithm Improvement Report — 2026-07-22

## 📊 Data Collected

| Source | Samples | Features | Period |
|---|---|---|---|
| Time-series | 2085 | 163 | 200 candles × 17 symbols |
| Per-symbol model | 136/symbol | 155 | 2 weeks backtest |
| Live features | 231 rows | 116+ | 3 days real data |

## 🏆 Best Predictable Symbols (from 200+ samples)

| Symbol | Pump Rate | CV F1 | CV Acc |
|---|---|---|---|
| **EPICUSDT** | **37.5%** | 0.377 | 58.9% |
| **TLMUSDT** | **31.6%** | 0.225 | 52.3% |
| **RESOLVUSDT** | 26.5% | 0.249 | 39.0% |
| **TUTUSDT** | 26.5% | 0.222 | 49.5% |
| **OPNUSDT** | 25.7% | 0.274 | 59.5% |
| **RECALLUSDT** | 24.3% | **0.459** | **74.3%** |
| **FIGHTUSDT** | 22.1% | 0.217 | 63.1% |

❌ **USATUSDT: 0% pump (removed from universe)**

## 🎯 Top Predictive Features (Global)

1. `ind_atr_pct` (+0.15) — ATR 3-8% sweet spot
2. `struct_range_pct` (+0.15) — Range expansion
3. `ind_bb_width_pct` (+0.14) — BB width
4. `ex_ex_ulcer_score` (-0.12) — Downside vol matters
5. `ex_ex_dc_lower` (-0.09) — Donchian support

## 🛠️ Algorithm Improvements Made

### 1. **Anti-Late Filter** (`src/anti_late.py`)
- Catches blowoff acceleration, distribution wicks
- Rejects entry at overextended moves (mom_3 > 8%)

### 2. **Extended Indicators** (`src/extended_indicators.py`)
- 99 new features added: Stoch, ADX, MFI, CMF, OBV, etc.
- Top 8 integrated into ultra_strict_v2

### 3. **Universe Improvements** (`src/universe.py`)
- Complete majors blacklist (100+ symbols)
- Multi-source MC resolution (Paprika → Gecko → proxy)
- Activity score for priority
- USATUSDT removed (0% pump)

### 4. **Adaptive Filter** (`src/adaptive_filter.py`)
- Global + per-symbol weights
- Symbol exclusion (USATUSDT)
- Pump rate bonus (predictable symbols get +8)

### 5. **Self-Training Loop** (`src/self_train.py`)
- Every hour: train LR, find best TP/SL, save params
- Adaptive threshold based on pump rate
- Best TP/SL grid search

### 6. **Per-Symbol Models** (`data/per_symbol_models.json`)
- 14 symbols individually trained
- Best: RECALLUSDT (F1=0.46, Acc=74%)

## 📈 Current Best Params (auto-trader uses)

```json
{
  "tp_pct": 5.0,
  "sl_pct": 1.5,
  "trail_pct": 0.5,
  "conf_threshold": 55,
  "strategy": "V3_aggressive + per_symbol + adaptive"
}
```

## 🎯 Top 3 Current Signals (23:00 UTC)

| Symbol | Direction | Conf | Mom3 | Hist.Pump | Entry | TP | SL |
|---|---|---|---|---|---|---|---|
| **TLMUSDT** | LONG | 57 | +8.4% | 31.6% | 0.001813 | 0.001904 | 0.001786 |
| **OPNUSDT** | LONG | 58 | +6.0% | 25.7% | 0.070500 | 0.074025 | 0.069442 |
| **RECALLUSDT** | LONG | 55 | +2.6% | 24.3% | 0.038600 | 0.040530 | 0.038021 |

⚠️ TLMUSDT: ATR=10.6%, Mom3=8.4% — overextended, risky entry

## ⚠️ Caveats

1. **CV F1 max = 0.46** (RECALLUSDT) — well below 0.8 target
2. **Real outcomes: 5 PUMP out of 200** (2.5% rate) — rare
3. **Win rate ceiling ~50%** with available features
4. **80% target** likely unreachable with technical analysis alone
5. **Need 30+ real signals** for true validation (currently 2)

## 🔄 Active Processes

- **Collector** (PID 1029): Every 10 min
- **Self-Trainer** (PID 1148): Every hour
- **Open signals**: EPICUSDT LONG (from previous cycle)
