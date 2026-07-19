# PumpHunter-AI 🤖

اسکنر حرفه‌ای بازار Toobit که **روزی ۲ بار** به‌صورت خودکار اجرا می‌شود، کوین‌های زیر ۲۰ میلیون دلار را با یک pipeline ۱۰ لایه‌ای تحلیل می‌کند و در صورت معتبر بودن سیگنال از طریق **تلگرام** هشدار می‌دهد.

> **درباره نام**: این پروژه ابتدا با نام `toobit-scanner` شروع شد و سپس به یک موتور تصمیم‌گیری چندلایه به نام **PumpHunter-AI** ارتقا یافت.

---

## ✨ ویژگی‌ها

### داده‌ها
- 📡 **Toobit USDT Perpetual** — REST API
- 💰 **CoinPaprika / CoinGecko** — فیلتر زیر ۲۰M دلار
- 🌐 **LunarCrush + Google Trends + TradingView** — سوشال
- 🐋 **CoinGlass** — لیکوئیدیشن (hooks برای Arkham/Nansen/CryptoQuant)
- ₿ **BTC regime** — فیلتر بازار

### تحلیل تکنیکال (چندتایم‌فریم: 1h + 4h)
- **اندیکاتورهای کلاسیک**: RSI، MACD، EMA (20/50/100/200)
- **اندیکاتورهای پیشرفته**: VWAP، ATR، Bollinger Bands + Squeeze، Relative Volume، Momentum (1/3/6/12)، Multi-timeframe alignment
- **ساختار بازار**: Higher Highs/Lows، BOS (Break of Structure)، Range
- **کیفیت کندل**: Body ratio، Upper/Lower wick، Close position، Power streak
- **الگوها**: Engulfing، Hammer، Double top/bottom، Higher lows

### یادگیری ماشین
- 🧠 **Logistic Regression** → `P(success)` بین 0 و 1
- 📊 **Chi-Square Test** → significance هر feature
- ⚖️ **Soft weight adaptation** → وزن‌ها به‌تدریج اصلاح می‌شوند
- 💾 **Persistence** → مدل و ضرایب روی دیسک ذخیره می‌شوند
- 🔬 **Train/test split** → ارزیابی روی داده unseen

### تصمیم‌گیری نهایی
- سه خروجی: **APPROVED** / **WATCHLIST** / **REJECTED**
- **APPROVED** = composite ≥ 75 **و** ML prob ≥ 0.45
- **WATCHLIST** = composite بین 60-75
- **REJECTED** = composite < 60 یا فیلتر منفی critical

### عملیات
- 🛡 **Duplicate guard / cooldown** — هر نماد حداکثر هر ۶ ساعت
- 🧹 **Data Quality Control** — تشخیص NaN، gap، OHLC نامعتبر، staleness
- 📬 **Telegram digest + per-coin alert**
- 🖥 **HTML dashboard** با تم تیره
- ⏰ **GitHub Actions** — هر ۱۲ ساعت (۰۰:۰۰ و ۱۲:۰۰ UTC)

---

## 🏗 ساختار پروژه

```
toobit-scanner/
├── config.yaml                  # کانفیگ اصلی
├── requirements.txt             # وابستگی‌ها
├── .github/workflows/scan.yml   # GitHub Actions schedule
└── src/
    ├── scanner.py               # pipeline اصلی
    ├── toobit_client.py         # Toobit REST
    ├── market_filter.py         # CoinGecko filter
    ├── coinpaprika.py           # CoinPaprika filter
    ├── lunarcrush.py            # سوشال
    ├── google_trends.py         # ترندها
    ├── tradingview_scraper.py   # TV
    ├── whale_data.py            # CoinGlass
    ├── data_quality.py          # لایه ۰: کنترل کیفیت
    ├── btc_filter.py            # لایه ۲: فیلتر BTC
    ├── cooldown.py              # لایه ۱: duplicate guard
    ├── technical.py             # RSI/MACD/EMA/الگوها
    ├── indicators.py            # VWAP/ATR/BB/RVol/Momentum/MTF
    ├── market_structure.py      # HH/HL/BOS/Range
    ├── candle_quality.py        # Body/Wick/Power streak
    ├── features.py              # Feature engineering
    ├── scoring.py               # لایه ۵: rule-based score
    ├── ml_engine.py             # لایه ۶/۷/۹: ML
    ├── outcome.py               # لایه ۱۰: post-signal eval
    ├── decision.py              # لایه ۹: APPROVED/REJECTED/WATCHLIST
    ├── telegram_notifier.py     # هشدار تلگرام
    ├── dashboard.py             # داشبورد HTML
    └── smoke_test.py            # ۱۲ تست end-to-end
```

---

## 🧠 Pipeline (۱۰ لایه)

1. **جمع‌آوری داده** → Toobit + CoinPaprika + LunarCrush + TV + Trends + CoinGlass
2. **کنترل کیفیت** → تعداد کندل، NaN، gap، OHLC integrity، staleness
3. **پیش‌فیلتر** → نقدشوندگی، حذف TEST pairs، فیلتر مارکت‌کپ < 20M
4. **Cooldown** → جلوگیری از تکرار سیگنال (۶ ساعت پیش‌فرض)
5. **BTC regime filter** → BULLISH / NEUTRAL / BEARISH / RISK_OFF
6. **اندیکاتورها** → RSI, MACD, EMA, VWAP, ATR, Bollinger, RVol, Momentum, MTF
7. **Feature engineering** → 28 feature مهندسی‌شده
8. **Rule-based scoring** → ترکیب وزنی 9 زیرسیستم + جریمه‌ها
9. **ML probability** → Logistic Regression با train/test split
10. **Final decision** → APPROVED / WATCHLIST / REJECTED + alerts

---

## 🚀 راه‌اندازی

### ۱) ربات تلگرام
1. به [@BotFather](https://t.me/BotFather) → `/newbot`
2. **Bot Token** و **chat_id** را کپی کن

### ۲) اتصال به GitHub
```bash
git init && git add . && git commit -m "init"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

### ۳) Secrets
**Settings → Secrets → Actions**:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### ۴) اجرای دستی (تست)
از تب Actions، workflow را با `workflow_dispatch` اجرا کن.

### ۵) مشاهده داشبورد
Artifact `toobit-scanner-output` را دانلود و `dashboard.html` را در مرورگر باز کن.

---

## 🤖 یادگیری ماشین — چطور کار می‌کند؟

1. هر سیگنال (حتی REJECTED) به `data/signal_history.csv` اضافه می‌شود همراه با 28 feature + composite score + ML prob + entry_price
2. در اجرای بعدی، **outcome** ارزیابی می‌شود: 12 ساعت بعد چند درصد بالا رفته؟ peak/drawdown چقدر بوده؟ یک outcome_score 0-100 + label (1=success, 0=fail) ثبت می‌شود
3. وقتی ≥ 30 سیگنال labeled داشته باشیم، **Logistic Regression** آموزش می‌بیند
4. **Chi-Square** significance هر feature محاسبه می‌شود
5. **Soft weight adaptation**: وزن‌های rule-based با feature importances از logistic regression ترکیب می‌شوند (blend 30%)
6. سیستم هرچه بیشتر اجرا شود، هوشمندتر می‌شود.

---

## 🧪 تست
```bash
cd src
python smoke_test.py
```
خروجی مورد انتظار:
```
✅ data_quality ok
✅ all indicator modules returned dicts
✅ structure features
✅ candle_quality
✅ btc_filter
✅ features: 28 fields
✅ scoring
✅ cooldown
✅ decision: APPROVED / WATCHLIST / REJECTED
✅ ml_engine: train ok
✅ outcome
✅ dashboard
```

---

## ⚙️ شخصی‌سازی

### آستانه هشدار
`config.yaml`:
```yaml
alerting:
  notify_threshold: 75.0
  max_alerts_per_run: 5
```

### وزن‌های rule-based
```yaml
rule_weights:
  technical: 12
  momentum: 12
  volume: 18
  vwap: 8
  atr_bb: 6
  structure: 10
  candle: 8
  mtf: 8
  pattern: 8
```

### Cooldown
```yaml
cooldown:
  hours: 6.0
```

### زمان‌بندی
`.github/workflows/scan.yml`:
```yaml
- cron: "0 0,12 * * *"  # هر 12 ساعت
```

---

## 🛡 محدودیت‌ها

- **CoinPaprika/CoinGecko free** rate-limit → در GitHub Actions گاهی rate-limited می‌شویم، به همین خاطر fallback اضافه شده
- **LunarCrush/Google Trends** بدون API key → ممکن است 0 برگردانند
- **CoinGlass** برای Toobit معمولاً داده liquidation ندارد
- **ML training** حداقل 30 labeled signal نیاز دارد (حدود 2 هفته با 2 اجرا در روز)

---

## 📜 لایسنس
MIT
