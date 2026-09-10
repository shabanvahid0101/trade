# گزارش تحویل پروژه Trader Bot

تاریخ تهیه: 2026-09-10  
مسیر پروژه روی سیستم فعلی: `D:\code\projects\trade\trader_bot`

> این گزارش عمداً هیچ کلید API، توکن، رمز عبور، مقدار Secret یا اطلاعات حساس عملیاتی را شامل نمی‌شود. فایل‌های محرمانه مثل `.env` باید جداگانه و فقط از مسیر امن منتقل شوند.

## 1. هدف پروژه

هدف این پروژه ساخت یک ربات تحقیق، هشدار و Paper Trading برای بازار کریپتو، مخصوصاً `BTC/USDT` است. ایده اولیه پیش‌بینی کوتاه‌مدت قیمت با مدل‌های یادگیری ماشین بود، اما بعد از چند دور تست مشخص شد که تمرکز صرف روی پیش‌بینی قیمت، edge قابل اتکا ایجاد نکرده است.

مسیر پروژه به‌تدریج به سمت «آزمایشگاه کشف edge معاملاتی» تغییر کرده است؛ یعنی هر استراتژی یا مدل باید با داده تاریخی، هزینه معامله، اسپرد، اسلیپیج، مدیریت ریسک، گزارش عملکرد و مقایسه با buy-and-hold سنجیده شود.

اهداف عملی فعلی:

- تولید و نگهداری دیتاست‌های کندلی و فاندامنتال
- آموزش مدل‌های چندافقی برای `1h`، `15m` و `5m`
- تولید سیگنال `LONG`، `SHORT` یا `HOLD`
- اجرای Paper Trading با سرمایه مجازی
- تحلیل عملکرد و توقف خودکار در صورت ضعف عملکرد
- ساخت بانک فیچر تکنیکال و فاندامنتال برای تست مدل‌های بهتر
- جلوگیری از اجرای واقعی یا نوتیفیکیشن‌های مزاحم تا وقتی مدل edge قابل دفاع ندارد

## 2. معماری کلی بات

پروژه چند ماژول اصلی دارد:

### 2.1 دریافت و آماده‌سازی داده

فایل‌های مرتبط:

- `crypto_predictor.py`
- `dataset_maintenance.py`
- `data_quality.py`
- `fundamental_data.py`

وظایف:

- دریافت کندل‌های بازار با `ccxt`
- خواندن و به‌روزرسانی فایل‌های CSV داخل `dataset/`
- پاک‌سازی داده و کنترل gap زمانی
- اتصال داده‌های فاندامنتال به کندل‌ها با روش backward as-of merge
- ساخت فیچرهای تکنیکال، فاندامنتال و زمانی

دیتاست‌های مهم:

- `dataset/5m_btc_history.csv`
- `dataset/15m_btc_history_5000.csv`
- `dataset/1h-btc_history.csv`
- `dataset/1h-btc_fundamentals.csv`
- `dataset/15m_btc_fundamentals.csv`

### 2.2 مدل و آموزش

فایل اصلی:

- `crypto_predictor.py`

مدل فعلی بر پایه LSTM ساخته شده و دو حالت هدف دارد:

- regression: پیش‌بینی بازده آینده
- classification: طبقه‌بندی حرکت آینده به `SHORT`، `HOLD` یا `LONG`

در نسخه‌های اخیر بیشتر از classification استفاده شده، چون برای تصمیم معاملاتی مستقیم‌تر است.

مدل‌ها و artifactها داخل پوشه `models/` ذخیره می‌شوند. هر artifact شامل scaler، ستون‌های فیچر، تنظیمات horizon، threshold، متریک‌ها و بازه train/validation/test است.

### 2.3 بانک فیچر تکنیکال

فایل جدید:

- `technical_feature_bank.py`

Feature setهای مهم:

- `core`
- `advanced`
- `advanced-fundamental`
- `ta-wide`
- `ta-wide-fundamental`

Feature set جدید `ta-wide` روی `advanced` ساخته شده و تعداد زیادی فیچر تکنیکال اضافه می‌کند، از جمله:

- EMA / SMA / HMA trend structure
- Aroon
- Ichimoku
- Donchian Channel
- Keltner Channel
- Bollinger/Keltner squeeze
- Supertrend
- Chandelier Exit
- PPO / StochRSI / TSI / Fisher
- CMF / Chaikin Oscillator / VPT / Force Index
- الگوهای کندلی مثل doji، hammer، engulfing، morning/evening star
- gap و liquidity sweep

برای جلوگیری از overfit، استفاده از `ta-wide` باید همراه feature selection و walk-forward test باشد.

### 2.4 تولید سیگنال و استراتژی

فایل‌های مرتبط:

- `alert_bot.py`
- `strategy_rules.py`
- `signal_explainer.py`

خروجی مدل‌های چند horizon با هم ترکیب می‌شود. سپس بسته به تنظیمات، سیگنال نهایی از یکی از این حالت‌ها می‌آید:

- `model`: فقط رأی مدل‌ها
- `range`: منطق mean reversion در بازار رنج
- `hybrid`: ترکیب مدل در بازار رونددار و منطق range در بازار رنج

سیگنال‌ها:

- `LONG`
- `SHORT`
- `HOLD`

برای هر سیگنال، توضیح تصمیم نیز ساخته می‌شود تا مشخص شود رأی horizonها، confidence، regime و استراتژی چه بوده است.

### 2.5 Paper Trading و مدیریت ریسک

فایل‌های مرتبط:

- `paper_trader.py`
- `risk_manager.py`
- `trading_gate.py`

Paper Trading سرمایه مجازی را در فایل state ذخیره می‌کند و می‌تواند پوزیشن مجازی باز/بسته کند.

فایل‌های state مهم:

- `paper_state.json`
- `paper_state_15m_staging.json`
- `paper_state_5m_staging.json`

قابلیت‌های مدیریت ریسک:

- position sizing
- کاهش سایز بعد از drawdown
- توقف بعد از loss streak
- حداقل equity
- recovery mode با سایز کوچک‌تر
- stop loss
- take profit
- max holding candles
- محاسبه fee، spread و slippage

`trading_gate.py` قبل از اجرای paper trading وضعیت عملکرد را بررسی می‌کند و در صورت عبور از محدودیت‌ها trading را pause می‌کند.

### 2.6 گزارش‌ها و مانیتورینگ

فایل‌های مرتبط:

- `performance_report.py`
- `signal_report.py`
- `portfolio_report.py`
- `health_check.py`
- `edge_report.py`

گزارش‌ها وضعیت سرمایه، drawdown، win rate، تعداد معاملات، بهترین/بدترین معامله، عملکرد روزانه، عملکرد سیگنال‌ها و edge نسبت به buy-and-hold را بررسی می‌کنند.

`edge_report.py` برای تصمیم‌گیری سخت‌گیرانه‌تر اضافه شده است: هر سیستم Paper Trading باید نسبت به buy-and-hold، profit factor، expectancy، drawdown و حداقل تعداد معامله بررسی شود.

## 3. استراتژی معاملاتی فعلی

استراتژی فعلی هنوز در مرحله تحقیق و paper trading است و برای معامله واقعی آماده نیست.

هسته استراتژی:

1. دریافت کندل‌های جدید
2. ساخت فیچرهای تکنیکال/فاندامنتال
3. پیش‌بینی چندافقی با مدل‌های LSTM
4. ترکیب رأی horizonها
5. اعمال فیلتر confidence و min-agree
6. تشخیص تقریبی regime بازار
7. اجرای منطق `model`، `range` یا `hybrid`
8. کنترل risk قبل از ورود
9. ثبت معامله مجازی در state
10. ارزیابی عملکرد با گزارش‌ها و kill switch

در تست‌های قبلی، مدل‌های `1h`، `15m` و `5m` سوددهی قابل اتکا نشان ندادند. به همین دلیل اجرای خودکار `1h` و `15m` محدود/دستی شده و `5m staging` هم با kill switch کنترل می‌شود.

## 4. زبان و کتابخانه‌ها

زبان اصلی:

- Python

کتابخانه‌های اصلی:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `ccxt`
- `python-dotenv`
- `requests`
- `joblib`

فایل dependency:

- `requirements.txt`

نسخه فعلی پروژه به‌صورت عمدی از `TA-Lib` یا `pandas-ta` وابسته نشده تا اجرای GitHub Actions ساده‌تر و پایدارتر بماند. فیچرهای تکنیکال جدید با `pandas` و `numpy` پیاده‌سازی شده‌اند.

## 5. روش راه‌اندازی روی لپ‌تاپ جدید

### 5.1 دریافت پروژه

اگر پروژه روی GitHub موجود است:

```powershell
git clone https://github.com/shabanvahid0101/trade.git
cd trade
```

اگر قبلاً clone شده:

```powershell
git pull origin main
```

### 5.2 ساخت محیط Python

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 فایل‌های محرمانه

فایل `.env` در گزارش نوشته نشده و نباید داخل Git commit شود. اگر برای دریافت داده یا اجرای برخی بخش‌ها لازم است، باید جداگانه و امن منتقل شود.

فعلاً ارسال تلگرام به‌صورت پیش‌فرض خاموش است. برای روشن کردن دوباره باید صراحتاً kill switch تغییر کند.

## 6. روش‌های اجرا

### 6.1 آموزش مدل 5 دقیقه‌ای با feature set عادی

```powershell
python crypto_predictor.py train --data dataset/5m_btc_history.csv --sequence-length 96 --horizon 1 --epochs 40 --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

### 6.2 آموزش مدل 5 دقیقه‌ای با بانک تکنیکال گسترده

```powershell
python crypto_predictor.py train --data dataset/5m_btc_history.csv --symbol BTC/USDT --timeframe 5m --model-dir models/5m_ta_wide_local_test --sequence-length 96 --horizon 1 --epochs 12 --batch-size 32 --training-verbose 2 --max-train-rows 5000 --feature-set ta-wide --feature-selection correlation --max-selected-features 24 --min-selected-features 10 --target-mode classification --threshold 0.0002 --min-confidence 0.50 --fee-rate 0.001 --spread-bps 2 --slippage-bps 2
```

### 6.3 پیش‌بینی چندافقی

```powershell
python crypto_predictor.py predict-multi --data dataset/5m_btc_history.csv --symbol BTC/USDT --timeframe 5m --model-dir models/5m_staging --horizons 1,3,6 --min-agree 2 --min-confidence 0.52
```

### 6.4 اجرای یک مرحله Paper Trading

```powershell
python paper_trader.py --mode single --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --model-dir models/5m_staging --horizons 1,3,6 --initial-capital 100 --min-agree 2 --min-confidence 0.52 --strategy hybrid --threshold 0.0004 --fee-rate 0.001 --spread-bps 2 --slippage-bps 2 --state-file paper_state_5m_staging.json
```

### 6.5 بک‌تست

```powershell
python paper_trader.py --mode backtest --data dataset/5m_btc_history.csv --timeframe 5m --horizons 1,3,6 --initial-capital 100 --days 7 --fee-rate 0.001 --spread-bps 2 --slippage-bps 2
```

### 6.6 گزارش عملکرد

```powershell
python performance_report.py --symbol BTC/USDT --data dataset/5m_btc_history.csv --state-file paper_state_5m_staging.json --initial-capital 100
```

### 6.7 گزارش edge

```powershell
python edge_report.py --initial-capital 100 --min-closed-trades 50 --min-return-pct 0 --min-alpha-pct 0 --min-profit-factor 1.10 --max-drawdown-pct 2
```

## 7. روش تست

تست‌های سبک و مهم:

```powershell
python -m compileall crypto_predictor.py technical_feature_bank.py telegram_utils.py performance_report.py health_check.py signal_report.py trading_gate.py edge_report.py
```

Smoke test مدل‌ها:

```powershell
python smoke_test_models.py --symbol BTC/USDT --timeframe 5m --horizons 1,3,6 --model-dir models/5m_staging
```

تست آماده‌سازی feature set جدید:

```powershell
python -c "from crypto_predictor import load_price_csv, prepare_datasets, FEATURE_SETS; df=load_price_csv('dataset/5m_btc_history.csv'); split=prepare_datasets(df, sequence_length=96, horizon=1, timeframe='5m', max_rows=5000, feature_columns=FEATURE_SETS['ta-wide'], target_mode='classification', target_threshold=0.0002, feature_selection='correlation', max_selected_features=24, min_selected_features=10); print(split.X_train.shape, split.X_val.shape, split.X_test.shape, len(split.feature_columns))"
```

## 8. وضعیت فعلی پروژه

آخرین وضعیت مهم شناخته‌شده:

- Telegram notifications به‌صورت پیش‌فرض خاموش شده‌اند.
- kill switch سراسری در `telegram_utils.py` اضافه شده است.
- `TELEGRAM_DISABLED` به‌صورت پیش‌فرض فعال است؛ یعنی ارسال تلگرام انجام نمی‌شود مگر اینکه صراحتاً غیرفعال شود.
- بانک فیچر تکنیکال `ta-wide` اضافه شده و به `crypto_predictor.py` وصل شده است.
- آموزش آزمایشی local با `ta-wide` انجام شده، اما مدل هنوز edge سودده نشان نداده است.
- دیتاست 5m روی لوکال بزرگ است و قبلاً حدود 63 هزار کندل از `2025-12-25` تا `2026-08-03` داشت.
- یک مشکل NaN در کندل‌های zero-range پیدا و اصلاح شد: برای wickها مقدار صفر و برای close position مقدار خنثی `0.5` استفاده می‌شود.
- اجرای زمان‌بندی‌شده تلگرام/گزارش‌ها اگر هنوز در workflowها `--telegram` داشته باشند، به خاطر kill switch نباید پیام ارسال کنند.

نتایج آخرین تست local `ta-wide` با 5000 کندل آخر:

- تعداد فیچر کاندید: 139
- فیچرهای انتخاب‌شده: 24
- دقت کلاس‌بندی حدودی: 42٪
- actionable accuracy حدودی: 31٪
- بک‌تست داخلی: منفی، حدود `-0.34%`
- نتیجه: هنوز قابل قبول برای معامله واقعی نیست.

## 9. مشکلات شناخته‌شده

### 9.1 نبود edge پایدار

مدل‌های فعلی تا این مرحله سوددهی قابل دفاع نشان نداده‌اند. مشکل اصلی این است که پیش‌بینی جهت حرکت کوتاه‌مدت BTC بسیار نویزی است و بعد از fee/spread/slippage، edge از بین می‌رود.

### 9.2 تعداد معامله کم در بعضی تست‌ها

در برخی train/testها مدل بسیار محافظه‌کار شده و معاملات کمی باز کرده است. این از نظر ریسک خوب است، ولی برای ارزیابی آماری کافی نیست.

### 9.3 overfit احتمالی

با اضافه شدن `ta-wide` تعداد فیچرها زیاد شده است. اگر feature selection، walk-forward و تست خارج از نمونه جدی نباشد، ریسک overfit بالاست.

### 9.4 وابستگی به کیفیت دیتاست

وجود gap، کندل zero-volume یا کندل zero-range می‌تواند train را خراب کند یا باعث حذف ناخواسته ردیف‌ها شود. بخش data quality باید قبل از هر train جدی اجرا شود.

### 9.5 GitHub Actions و فایل‌های local

بعضی فایل‌های local test مثل مدل‌های آزمایشی ممکن است untracked باشند و نباید بی‌هدف commit شوند. قبل از هر commit باید `git status` و `git diff --stat` بررسی شود.

### 9.6 وضعیت Telegram

تلگرام فعلاً عمداً خاموش است. اگر دوباره فعال شود، باید با دقت بررسی شود که workflowها پیام‌های زیاد یا تکراری نفرستند.

## 10. کارهای باقی‌مانده پیشنهادی

اولویت‌های پیشنهادی:

1. پاک‌سازی و audit کامل دیتاست‌های 5m، 15m و 1h
2. train چندافقی با `ta-wide` برای horizonهای `1,3,6`
3. اجرای walk-forward واقعی روی `5m` و `15m`
4. مقایسه `advanced`، `advanced-fundamental`، `ta-wide` و `ta-wide-fundamental`
5. ساخت benchmark جدی در `edge_report.py`
6. اضافه کردن معیارهای سخت‌گیرانه برای قبول مدل:
   - profit factor بالای 1.10
   - drawdown کنترل‌شده
   - بازده بهتر از buy-and-hold
   - حداقل تعداد معاملات بسته‌شده
7. تست چند symbol به‌جای فقط BTC
8. اضافه کردن multi-timeframe confirmation:
   - ورودی 5m
   - تأیید 15m یا 1h
9. بهبود استراتژی خروج:
   - stop/take پویا با ATR
   - trailing stop
   - time-based exit
   - early loss cut
10. ساخت یک dashboard ساده برای دیدن وضعیت مدل‌ها، paper state و گزارش‌ها
11. نگه داشتن Telegram خاموش تا وقتی مدل واقعاً از gate عبور کند
12. قبل از هر اجرای واقعی، حداقل چند هفته paper trading بدون دخالت دستی

## 11. پیشنهاد مسیر ادامه برای لپ‌تاپ جدید

بعد از انتقال پروژه، بهتر است در Codex جدید این ترتیب دنبال شود:

1. اجرای `git status -sb`
2. بررسی اینکه `.env` وجود دارد یا نه، بدون چاپ محتوای آن
3. نصب dependencyها
4. اجرای compileall
5. اجرای data quality روی دیتاست‌ها
6. اجرای smoke test مدل‌های فعلی
7. اجرای `edge_report.py`
8. اگر همه چیز سالم بود، train چندافقی `ta-wide` روی 5m
9. سپس walk-forward و تصمیم‌گیری بر اساس عدد، نه حس

## 12. جمع‌بندی

پروژه در حال حاضر بیشتر یک آزمایشگاه جدی برای کشف استراتژی و edge است تا یک ربات آماده معامله واقعی. زیرساخت‌های خوبی اضافه شده‌اند: دیتاست، فیچرهای تکنیکال و فاندامنتال، Paper Trading، مدیریت ریسک، گزارش عملکرد، gate توقف و GitHub Actions. اما مدل هنوز به سطح سوددهی قابل اعتماد نرسیده است.

برای ادامه، باید با داده بیشتر، تست walk-forward، feature selection سخت‌گیرانه، چند horizon، چند تایم‌فریم و چند symbol دنبال edge واقعی گشت. تا قبل از عبور از این gateها، معامله واقعی توصیه نمی‌شود.
