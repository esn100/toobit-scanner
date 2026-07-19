# Toobit Scanner 🤖

اسکنر حرفه‌ای بازار Toobit که **روزی ۲ بار** به‌صورت خودکار اجرا می‌شود، کوین‌های زیر ۲۰ میلیون دلار را تحلیل می‌کند و در صورت بالا بودن امتیاز (پیش‌فرض > ۸۵) از طریق **تلگرام** هشدار می‌دهد.

> به‌علاوه، یک **داشبورد HTML** به‌صورت خودکار ساخته می‌شود که در هر اجرا قابل مشاهده است.

---

## ✨ ویژگی‌ها

| قابلیت | جزئیات |
|---|---|
| 📡 منبع داده | Toobit USDT Perpetual (REST API) |
| 💰 فیلتر مارکت‌کپ | CoinGecko (رایگان، فقط کوین‌های زیر ۲۰M دلار) |
| 📈 تحلیل تکنیکال | RSI + واگرایی، MACD + واگرایی، EMA (20/50/100/200)، الگوهای شمعی و کلاسیک |
| 🌐 سوشال | LunarCrush (Galaxy Score، Sentiment)، Google Trends (PyTrends)، TradingView (Scraping) |
| 🐋 نهنگ‌ها | CoinGlass (Liquidations) — قابل ارتقا به Arkham / Nansen / CryptoQuant |
| 🎯 امتیازدهی | ترکیب وزنی ۴ بعد (تکنیکال، الگو، سوشال، نهنگ) |
| 🤖 یادگیری ماشین | تنظیم خودکار وزن‌ها با `RandomForest` بر اساس نتایج گذشته |
| ⏰ زمان‌بندی | GitHub Actions، هر ۱۲ ساعت (۰۰:۰۰ و ۱۲:۰۰ UTC) |
| 📬 هشدار | تلگرام (digest + per-coin) — فقط امتیاز > ۸۵ |
| 🖥 داشبورد | `dashboard.html` خودکار ساخته می‌شود |

---

## 🏗 ساختار پروژه

```
toobit-scanner/
├── config.yaml                  # کانفیگ اصلی
├── requirements.txt             # وابستگی‌ها
├── .github/workflows/scan.yml   # GitHub Actions schedule
└── src/
    ├── scanner.py               # pipeline اصلی
    ├── toobit_client.py         # دریافت داده از Toobit
    ├── market_filter.py         # فیلتر مارکت‌کپ با CoinGecko
    ├── technical.py             # RSI/MACD/EMA/واگرایی/الگوها
    ├── lunarcrush.py            # داده‌های سوشال
    ├── google_trends.py         # روند جستجو
    ├── tradingview_scraper.py   # تحلیل‌های TV
    ├── whale_data.py            # CoinGlass + جای خالی Arkham/Nansen
    ├── ml_weights.py            # امتیازدهی + ML تنظیم وزن‌ها
    ├── telegram_notifier.py     # ارسال هشدار تلگرام
    ├── dashboard.py             # ساخت داشبورد HTML
    ├── data/                    # JSON خروجی و تاریخچه
    ├── reports/                 # گزارش‌های JSON هر اجرا
    └── models/                  # مدل ML ذخیره‌شده
```

---

## 🚀 راه‌اندازی سریع

### ۱) ساخت ربات تلگرام
1. به [@BotFather](https://t.me/BotFather) پیام بده و `/newbot` بزن.
2. **Bot Token** را کپی کن.
3. به ربات پیام بده، سپس این URL را در مرورگر باز کن تا **chat_id** را پیدا کنی:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   مقدار `chat.id` از پاسخ JSON همان چت تو است (اگر گروه است، chat_id گروه را بگیر).

### ۲) اتصال به GitHub
```bash
cd toobit-scanner
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

### ۳) تنظیم Secrets
در ریپو: **Settings → Secrets and variables → Actions → New repository secret**

| نام | مقدار |
|---|---|
| `TELEGRAM_BOT_TOKEN` | توکن ربات از BotFather |
| `TELEGRAM_CHAT_ID` | شناسه چت/کانال |

### ۴) فعال‌سازی Actions
از تب **Actions** در ریپو، workflow را enable کن. اولین اجرا به‌صورت دستی از `workflow_dispatch` هم ممکن است.

### ۵) مشاهده داشبورد
بعد از هر اجرا، از artifact `toobit-scanner-output` فایل `dashboard.html` را دانلود و در مرورگر باز کن.

---

## ⚙️ شخصی‌سازی

### تغییر آستانه هشدار
`config.yaml`:
```yaml
alerting:
  notify_threshold: 85.0   # فقط بالای ۸۵ هشدار بده
  max_alerts_per_run: 5    # حداکثر ۵ هشدار در هر اجرا
```

### تغییر وزن‌های اولیه (قبل از اینکه ML کافی داده داشته باشد)
```yaml
weights:
  technical: 40
  pattern: 20
  social: 25
  whale: 15
```

### افزودن API Key برای LunarCrush
اگر کلید LunarCrush داری، در `src/lunarcrush.py` این خط را فعال کن:
```python
def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
    self.api_key = api_key  # مقدار را اینجا قرار بده
```

### فعال‌سازی Arkham / Nansen / CryptoQuant
در `src/whale_data.py` کلاس‌های مربوط به هر سرویس را اضافه کن و در `config.yaml` آنها را `enabled: true` کن.

### تغییر زمان‌بندی
در `.github/workflows/scan.yml` کرون را عوض کن، مثلاً هر ۶ ساعت:
```yaml
- cron: "0 */6 * * *"
```

---

## 🤖 یادگیری ماشین چطور کار می‌کند؟

1. هر اجرا، **برای هر کوین** یک ردیف در `data/signal_history.csv` ذخیره می‌شود (شامل ویژگی‌ها + وزن‌های استفاده‌شده + امتیاز نهایی).
2. در اجرای بعدی، ردیف‌های قبلی **label** می‌گیرند: آیا قیمت ۱۲ ساعت بعد بالا رفته؟ (۱ = بله، ۰ = نه).
3. وقتی حداقل ۲۰ ردیف برچسب‌خورده داشته باشیم، یک **RandomForestClassifier** آموزش می‌بیند.
4. feature importances مدل، وزن‌های جدید را پیشنهاد می‌دهد (ترکیب ۷۰٪ ML + ۳۰٪ prior برای پایداری).
5. هر ۵ اجرا یک‌بار مدل بازآموزی می‌شود.

> یعنی اسکنر هر چه بیشتر اجرا شود، وزن‌هایش هوشمندتر می‌شود.

---

## 🧪 اجرای محلی (تست)

```bash
pip install -r requirements.txt
cd src
export TELEGRAM_BOT_TOKEN=xxx
export TELEGRAM_CHAT_ID=yyy
python scanner.py
python dashboard.py
# باز کردن dashboard.html در مرورگر
```

---

## 🛡 محدودیت‌ها

- **CoinGecko Free** ~ ۱۰-۳۰ درخواست در دقیقه؛ به همین خاطر `max_symbols_per_run` روی ۵۰ محدود شده.
- **Google Trends** از pytrends استفاده می‌کند؛ در GitHub Actions گاهی timeout می‌دهد ولی بازیابی خودکار دارد.
- **CoinGlass** برای Toobit ممکن است داده‌ی لیکوئید ندهد؛ در این صورت `liq_bias=0` خواهد بود.
- **TradingView Scraping** وابسته به ساختار HTML سایت است؛ در صورت تغییر، الگوریتم heuristic بازسازی می‌شود.

---

## 📜 لایسنس
MIT — هر طور که خواستی استفاده کن.
