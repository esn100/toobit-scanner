# گزارش نهایی: تحلیل ۴۹ Pump بزرگ small-cap (مطالعه آماری)

## دیتاست
- **Universe**: ۳۳۰ کاندید small cap از Toobit + Gate.io + CoinGecko
- **Threshold**: pump = 24h forward return ≥ 40% از local low
- **Time window**: ۳۰ روز گذشته (1h klines)
- **Pump count**: ۵۲ عدد (با AKE = ۵۳) → آنالیز شد: **۴۹ pump**
- **Control set**: ۱۳۱۵ نمونه non-pump تصادفی

## یافته‌های کلیدی

### ۱. Pump profile
| Metric | Value |
|---|---|
| میانگین سود | ~+۹۵٪ (24h) |
| بزرگ‌ترین | EVAAUSDT +358%, IKAUSDT +235%, OXTUSDT +401% |
| میانه flat قبل | ۱ ساعت (price range < 5%) |
| ۶۷٪ pumpها | بعد از ≤ ۶ ساعت flat |

### ۲. Pre-pump feature distribution (1h قبل)

| Feature | PUMP mean | CTRL mean | تفاوت |
|---|---|---|---|
| mom_3 | **-۱.۵۵٪** | -0.01٪ | پامپ‌ها اغلب **consolidation** یا **dip** دارن نه breakout |
| rvol | ۱.۵x | ۱.۲x | تفاوت کم - pump خیلی quiet شروع میشه |
| max_rvol_4h | ۲.۷x | ۲.۰x | **بهترین سیگنال volume** - اوج موج 4h |
| ATR% | **۶.۳٪** | ۱.۷٪ | small caps inherently volatile |
| RSI | ۴۴.۷ | ۴۸.۴ | تفاوت ناچیز - نه OB, نه OS |
| body_ratio | ۰.۴۵ | ۰.۴۸ | تفاوت ناچیز |

### ۳. INSIGHT های غافلگیرکننده

**🔍 یافته ۱: pump با MOMENTUM منفی شروع میشه**
- ۵۷٪ pumpها در t-1h، `mom_3 < 0` (دارن consolidation یا pullback میدن)
- یعنی فیلتر "momentum_3 < 8%" ما داره خیلی چیزها رو میگیره ولی دیر!

**🔍 یافته ۲: volume building pattern (نه spike فوری)**
- ۴h قبل: rvol avg 1.27x
- 8h قبل: rvol avg 0.92x
- یعنی **تدریجی** بالا میره، نه انفجاری
- ۲۹٪ pumpها max_rvol_4h > 2x در t-4h (early warning!)

**🔍 یافته ۳: Anti-late filter داره pump واقعی رو می‌کشه!**
- ۴۹ pump آنالیز شد
- در t-1h فقط **۲٪ (۱ pump)** mom_3 > 20% داشت
- ولی در **t+1h** (یعنی بعد از شروع پامپ) خیلی‌ها overextend میشن
- یعنی anti-late filter ما (mom_3 < 20%) **روی t-1h خوبه** ولی **نباید روی t+1h اعمال بشه**

**🔍 یافته ۴: pre-pump flat pattern**
- **۶۷٪ pumpها بعد از ≤ ۶h flat** شروع میشن
- یعنی اگه scanner ما **pre-pump flat detector** داشت، می‌تونست **قبل از شروع** سیگنال بده

**🔍 یافته ۵: 6h momentum مثبت = build-up**
- ۴۳٪ pumpها در t-1h، `mom_3 > 0` داشتن
- ۲۹٪ pumpها `mom_3 > 1%` در t-1h
- یعنی "momentum +1% الی +5%" یه **pre-pump setup** هست

### ۴. Predictive power (precision)

| Rule | Catch | False Positive | Precision |
|---|---|---|---|
| **CURRENT ULTRA-STRICT** (mom_3<8, ATR 3-8, RSI 25-78, rvol>1) | ۶٪ | ۳٪ | **۷.۳٪** |
| **RELAXED v1** (mom_3<20, ATR 2-12, RSI 20-90, rvol>0.8) | ۲۵٪ | ۱۰٪ | **۸.۱٪** |
| **RELAXED v2** (mom_3<30, ATR 2-15, RSI 20-95, rvol>0.5) | ۴۳٪ | ۱۵٪ | **۹.۷٪** |
| **PUMP-READY** (mom_3<40, RSI<95, rvol>0.3) | ۸۰٪ | ۷۲٪ | ۴.۰٪ |

### ۵. پیشنهاد: NEW FILTER

```python
# Pre-pump detection (4h+ قبل)
PRE_PUMP_FILTER = {
    # Strong pre-pump signals
    'min_flat_hours': 3,           # ≥ 3h flat
    'max_dd_before_pump': -8.0,    # max DD in last 24h
    'vol_buildup_4h': 0.5,         # rvol_avg(last_4h) / rvol_avg(prev_4h) > 0.5
    'mom_6_trend': (-5, 8),        # 6h momentum between -5% and +8%
    
    # Confirmation (1h قبل)
    'mom_3_max': 15,               # mom_3 < 15% (relaxed from 8%)
    'rsi_range': (20, 90),         # wider than (25, 78)
    'atr_min': 1.5,                # lower than 3% (capture early)
    'rvol_min': 0.5,               # lower than 1.0 (don't need spike yet)
    'max_rvol_4h_min': 1.5,        # at some point in last 4h had volume
}
```

### ۶. Strategy جدید: EARLY WARNING MODE

1. **t-4h**: scan برای `flat > 3h` + `vol buildup starting`
2. **t-1h**: confirm با `mom_3 in (-5, +15)` + `max_rvol_4h > 1.5`
3. **t+0 (start)**: ENTER long، نه بعد از pump
4. **t+1h onwards**: anti-late filter فعال بشه (max profit protection)

### ۷. Limitations
- ۴۹ pump نمونه کوچیکه (آماری p<0.05 trust)
- Control set متعادل نیست (1315 vs 49)
- داده 30 روز محدود - pumpهای بزرگ‌تر miss شدن
- Toobit data delay (3-4 min) vs Gate.io real-time
- Market regime یکسان نبوده (BTC bull/bear متفاوت)

## فایل‌ها

- `cache/all_pumps.json` - لیست ۵۲ pump detected
- `cache/pump_features.json` - features computed for 49 pumps
- `cache/stats.json` - aggregate statistics
- `cache/rows_index.json` - klines data per symbol
- `study.py` - main analysis script
- `advanced.py` - pattern analysis
- `find_pumps.py` - pump detection
