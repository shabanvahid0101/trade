# Crypto Price Predictor / پیش‌بینی قیمت ارز دیجیتال

## فارسی

این پروژه یک ابزار تحقیق، هشدار و Paper Trading برای پیش‌بینی کوتاه‌مدت قیمت ارزهای دیجیتال است. تمرکز فعلی پروژه روی `BTC/USDT` در تایم‌فریم `1h` است و از مدل‌های چند افقی، فیچرهای تکنیکال، داده‌های فیوچرز، فیلتر بازار رنج، کنترل ریسک و پیام تلگرام استفاده می‌کند.

> هشدار مهم: هیچ مدل هوش مصنوعی نمی‌تواند قیمت کریپتو را بدون خطا پیش‌بینی کند. این پروژه برای تحقیق، تست، هشدار و Paper Trading ساخته شده است. قبل از معامله واقعی باید نتایج را روی داده خارج از آموزش، با کارمزد، اسلیپیج و مدیریت ریسک بررسی کنی.

### امکانات اصلی

- دریافت و به‌روزرسانی کندل‌های جدید بازار
- آموزش مدل روی داده‌های پیوسته و بدون گپ
- پیش‌بینی چند افقی با مدل‌های `1,3,6` کندل آینده
- استفاده از فیچرهای تکنیکال و داده‌های فاندامنتال فیوچرز مثل Funding Rate، Open Interest، ازدحام لانگ/شورت، فشار خرید/فروش Taker، فشار اهرمی، ریسک Short/Long Squeeze و واگرایی Open Interest با قیمت
- انتخاب فیچرهای مهم با روش همبستگی روی پنجره آموزش
- سیگنال‌های `LONG`، `SHORT` و `HOLD`
- پشتیبانی از Paper Trading با سرمایه مجازی 100 دلار
- استراتژی Hybrid برای استفاده از مدل در بازار رونددار و Mean Reversion در بازار رنج
- کنترل ریسک برای کاهش حجم یا جلوگیری از ورود بعد از افت سرمایه یا ضررهای پشت سر هم
- تنظیم هوشمند حجم معامله بر اساس کیفیت سیگنال
- شبیه‌سازی واقع‌بینانه‌تر اجرا با Spread و Slippage
- ارسال پیام تلگرام حتی برای `HOLD`
- پیام‌های فارسی تلگرام برای Alert، Paper Trading، Health Check، Retrain و گزارش عملکرد
- توضیح دلیل سیگنال در پیام تلگرام و ذخیره دلیل تصمیم در Paper Trading
- اجرای خودکار با GitHub Actions
- بازآموزی شبانه مدل در ساعت `02:00 UTC`
- بازآموزی امن با مدل staging، quality gate و گزارش تلگرام
- گزارش عملکرد روزانه و Health Check دوره‌ای
- گزارش عملکرد سیگنال‌ها بر اساس دلیل تصمیم مدل

### نصب و راه‌اندازی

از ریشه پروژه:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

نسخه‌های TensorFlow و Keras در `requirements.txt` ثابت شده‌اند، چون فایل‌های مدل `.keras` نسبت به تغییر نسخه Keras حساس هستند.

### فایل `.env`

برای اجرای محلی تلگرام، این مقدارها را داخل فایل `.env` بگذار:

```text
TELEGRAM_TOKEN=telegram_bot_token
TELEGRAM_CHAT_ID=telegram_chat_id
```

اگر از API صرافی استفاده می‌کنی:

```text
Access_ID=coinex_api_key
Secret_Key=coinex_secret
```

برای GitHub Actions باید همین مقدارهای تلگرام را در مسیر زیر به عنوان Repository Secret وارد کنی:

```text
Settings -> Secrets and variables -> Actions -> Repository secrets
```

نام Secretها باید دقیقا این‌ها باشد:

```text
TELEGRAM_TOKEN
TELEGRAM_CHAT_ID
```

### آموزش مدل 5 دقیقه‌ای

```powershell
python crypto_predictor.py train --data dataset/5m_btc_history.csv --sequence-length 96 --horizon 1 --epochs 40 --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

برای دریافت کندل‌های جدید قبل از آموزش:

```powershell
python crypto_predictor.py train --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

### آموزش مدل 1 ساعته با داده فاندامنتال فیوچرز

اول داده فاندامنتال را دریافت کن:

```powershell
python fundamental_data.py --symbol BTC/USDT --timeframe 1h --market-data dataset/1h-btc_history.csv --output dataset/1h-btc_fundamentals.csv
```

بعد مدل‌های چند افقی را آموزش بده:

```powershell
python crypto_predictor.py train-multi --update --update-fundamentals --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --sequence-length 96 --epochs 40 --max-train-rows 5000 --feature-set advanced-fundamental --feature-selection correlation --max-selected-features 32 --min-selected-features 12 --target-mode classification --threshold 0.0015 --min-confidence 0.50
```

داده فاندامنتال با روش backward as-of merge به کندل‌ها وصل می‌شود؛ یعنی هر کندل فقط داده‌هایی را می‌بیند که در همان زمان یا قبل از آن منتشر شده‌اند. فایل فاندامنتال با کندل‌های بازار هم‌تراز می‌شود و فیچر `fundamental_source_age_hours` به مدل می‌گوید داده واقعی فاندامنتال چند ساعت قدیمی است. فیچرهای مشتق‌شده فیوچرز برای گرفتن نیروهای کوتاه‌مدت اصلی بازار بیت‌کوین ساخته شده‌اند: رشد اهرم، شلوغ شدن سمت لانگ یا شورت، جریان سفارش‌های تهاجمی، افراط Funding، ریسک Squeeze و حالت‌هایی که Open Interest زیاد می‌شود ولی قیمت آن را تایید نمی‌کند.

### تست سالم بودن مدل‌ها

```powershell
python smoke_test_models.py --symbol BTC/USDT --timeframe 1h --horizons 1,3,6 --sanitize
```

گزینه `--sanitize` کلیدهای ناسازگار Keras را از آرشیو مدل پاک می‌کند و بعد مدل‌ها را Load و تست می‌کند.

### پیش‌بینی یک‌باره

```powershell
python crypto_predictor.py predict --data dataset/5m_btc_history.csv
```

با دریافت دیتای تازه و ارسال تلگرام:

```powershell
python crypto_predictor.py predict --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --telegram
```

### هشدار تلگرام

اجرای یک مرحله هشدار با مدل‌های 1 ساعته:

```powershell
python alert_bot.py --update --update-fundamentals --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --min-agree 1 --min-confidence 0.45 --strategy hybrid
```

اجرای دائمی روی سرور پایتون:

```powershell
python alert_bot.py --mode loop --update --update-fundamentals --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --min-agree 1 --min-confidence 0.45 --strategy hybrid --sleep-seconds 300
```

Workflow فایل `.github/workflows/telegram-alert.yml` هر 15 دقیقه اجرا می‌شود و حتی پیام `HOLD` هم ارسال می‌کند.

پیام تلگرام بخش `Why` هم دارد که نشان می‌دهد رأی هر افق مدل چه بوده، بازار رنج/رونددار تشخیص داده شده یا نه، confidence چقدر است و چرا سیگنال نهایی `LONG`، `SHORT` یا `HOLD` شده است.

### Paper Trading

بک‌تست با سرمایه مجازی 100 دلار:

```powershell
python paper_trader.py --mode backtest --data dataset/1h-btc_history.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100 --days 7 --fee-rate 0.001 --spread-bps 2 --slippage-bps 2
```

اجرای یک مرحله Paper Trading زنده:

```powershell
python paper_trader.py --mode single --update --update-fundamentals --telegram --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100 --min-agree 1 --min-confidence 0.45 --strategy hybrid --risk-enabled --risk-max-drawdown-pct 5 --risk-max-loss-streak 3 --dynamic-position-sizing --dynamic-min-position-size-pct 0.25 --dynamic-max-position-size-pct 1.0 --spread-bps 2 --slippage-bps 2
```

Workflow فایل `.github/workflows/paper-trading.yml` هر ساعت اجرا می‌شود، دیتای بازار و فاندامنتال را آپدیت می‌کند، استراتژی Hybrid و کنترل ریسک را اعمال می‌کند، پیام تلگرام می‌فرستد و وضعیت حساب مجازی را در `paper_state.json` ذخیره می‌کند.

Dynamic Position Sizing بعد از Risk Manager اجرا می‌شود. اگر کنترل ریسک اجازه ورود بدهد، کیفیت سیگنال با confidence، تعداد رأی‌های موافق، قوی‌ترین افق مدل و رژیم بازار سنجیده می‌شود و حجم معامله بین حداقل و حداکثر تعیین‌شده تنظیم می‌شود.

برای نزدیک‌تر شدن به اجرای واقعی، `--spread-bps` و `--slippage-bps` قیمت ورود و خروج را علیه معامله تنظیم می‌کنند. مثلا با `--spread-bps 2 --slippage-bps 2` هر ورود/خروج حدود 3 bps بدتر از قیمت mid شبیه‌سازی می‌شود.

Paper Trading دلیل آخرین سیگنال را در `last_signal_reason` و `last_signal_reasons` ذخیره می‌کند. اگر معامله‌ای باز یا بسته شود، خلاصه دلیل تصمیم در همان رکورد معامله هم ذخیره می‌شود.

### گزارش عملکرد سیگنال‌ها

```powershell
python signal_report.py --symbol BTC/USDT --state-file paper_state.json --telegram
```

این گزارش از تاریخچه Paper Trading می‌خواند و نتیجه را بر اساس سیگنال، جهت معامله، رژیم بازار، استراتژی و دلیل بسته شدن معامله گروه‌بندی می‌کند. Workflow فایل `.github/workflows/signal-report.yml` هر روز ساعت `03:30 UTC` آن را به تلگرام می‌فرستد.

### گزارش عملکرد

```powershell
python performance_report.py --telegram --symbol BTC/USDT --data dataset/1h-btc_history.csv --state-file paper_state.json --initial-capital 100
```

Workflow فایل `.github/workflows/performance-report.yml` هر روز ساعت `03:00 UTC` گزارش سرمایه، سود و زیان، Win Rate، Drawdown، بهترین/بدترین معامله و پوزیشن فعلی را ارسال می‌کند.

### Health Check

```powershell
python health_check.py --telegram --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --paper-state paper_state.json --alert-state alert_state.json
```

Workflow فایل `.github/workflows/health-check.yml` هر 6 ساعت سلامت پروژه را بررسی می‌کند: تازگی دیتای بازار، تازگی دیتای فاندامنتال، وجود مدل‌ها، گپ کندل‌ها، وضعیت Alert، وضعیت Paper Trading و سرمایه مجازی.

### بهینه‌سازی استراتژی

جستجوی تنظیمات بهتر برای Paper Trading:

```powershell
python strategy_optimizer.py --data dataset/1h-btc_history.csv --days 14 --initial-capital 100 --top 10 --spread-bps 2 --slippage-bps 2
```

Walk-forward validation:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --train-days 14 --test-days 3 --step-days 3 --walkforward-days 45 --initial-capital 100 --spread-bps 2 --slippage-bps 2
```

تست استراتژی Hybrid و Range:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --train-days 7 --test-days 2 --step-days 2 --walkforward-days 16 --initial-capital 100 --horizon-sets "1,3,6;1,6;3,6;6" --confidence-grid "0.45,0.50" --min-agree-grid "1,2" --strategy-grid "model,hybrid,range" --range-lower-grid "0.15,0.20,0.25" --range-upper-grid "0.75,0.80,0.85" --range-atr-max-grid "0.006,0.008,0.012" --range-trend-max-grid "0.002,0.003,0.005"
```

استراتژی `hybrid` در بازار رونددار از مدل استفاده می‌کند و وقتی بازار رنج تشخیص داده شود، نزدیک حمایت/مقاومت از منطق Mean Reversion کمک می‌گیرد.

### بازآموزی خودکار مدل

Workflow فایل `.github/workflows/model-retrain.yml` هر شب ساعت `02:00 UTC` مدل‌های 1 ساعته را دوباره آموزش می‌دهد. مراحل آن:

1. دریافت کندل‌ها و داده فاندامنتال جدید
2. آموزش مدل‌های افق `1,3,6` داخل `models/staging`
3. Smoke test مدل‌های staging
4. اجرای Walk-forward validation روی staging
5. بررسی کیفیت با Quality Gate
6. اگر قبول شد، staging به مدل production در `models/` منتقل می‌شود
7. در هر حالت، گزارش تلگرام ارسال می‌شود و `model_registry.json` وضعیت آخرین retrain را نگه می‌دارد

### اجرای رایگان روی GitHub Actions

این پروژه برای اجرای زمان‌بندی‌شده روی GitHub Actions آماده شده است. Cloudflare Workers برای اجرای مستقیم TensorFlow/Python مناسب نیست. برای اجرای رایگان و زمان‌بندی‌شده، GitHub Actions گزینه ساده‌تری است. برای اجرای واقعا دائمی، بهتر است از VPS کوچک یا Container پایتون استفاده شود.

### نکات معامله

- با پول واقعی فقط بعد از Paper Trading طولانی‌مدت کار کن.
- نتیجه مثبت چند روزه کافی نیست؛ باید روی داده خارج از آموزش هم پایدار باشد.
- همیشه کارمزد، اسلیپیج، Stop Loss و Drawdown را حساب کن.
- حجم معامله را کوچک نگه دار.
- اگر مدل چند بار پشت سر هم خطا داد، معامله را متوقف کن و گزارش‌ها را بررسی کن.

---

## English

This project is a research, alerting, and paper-trading tool for short-horizon crypto price forecasting. The current setup focuses on `BTC/USDT` on the `1h` timeframe and uses multi-horizon models, technical features, futures market data, a range-market filter, risk controls, and Telegram notifications.

> Important warning: no AI model can predict crypto prices perfectly. This project is for research, testing, alerting, and paper trading. Before using real money, validate results out of sample with fees, slippage, and risk management.

### Main Features

- Fetches and updates fresh market candles
- Trains on continuous data with gap checks
- Multi-horizon forecasts for `1,3,6` future candles
- Technical features plus futures fundamentals such as funding rate, open interest, long/short crowding, taker buy/sell pressure, leverage pressure, squeeze risk, and open-interest/price divergence
- Correlation-based feature selection on the training window only
- `LONG`, `SHORT`, and `HOLD` signals
- Paper trading with a virtual 100 USD account
- Hybrid strategy: model-driven trend trades plus range-market mean reversion
- Risk controls that reduce size, block entries, or allow small recovery probes after drawdown based on signal quality
- Dynamic position sizing based on signal quality
- More realistic execution simulation with spread and slippage
- Telegram alerts, including `HOLD` messages
- Persian Telegram messages for alerts, paper trading, health checks, retraining, and performance reports
- Signal explanation in Telegram messages and paper-trading state
- GitHub Actions automation
- Nightly retraining at `02:00 UTC`
- Safe staging retraining with quality gate and Telegram report
- Daily performance report and scheduled health checks
- Separate 5m staging paper trading with its own Telegram report and paper state
- Signal-performance report grouped by model decision reasons

### Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

TensorFlow/Keras versions are pinned in `requirements.txt` because saved `.keras` files are sensitive to Keras serialization changes.

### `.env`

For local Telegram messages, create a `.env` file:

```text
TELEGRAM_TOKEN=telegram_bot_token
TELEGRAM_CHAT_ID=telegram_chat_id
```

Optional exchange API values:

```text
Access_ID=coinex_api_key
Secret_Key=coinex_secret
```

For GitHub Actions, add the same Telegram values as repository secrets:

```text
Settings -> Secrets and variables -> Actions -> Repository secrets
```

Secret names must be exactly:

```text
TELEGRAM_TOKEN
TELEGRAM_CHAT_ID
```

### Train a 5-Minute Model

```powershell
python crypto_predictor.py train --data dataset/5m_btc_history.csv --sequence-length 96 --horizon 1 --epochs 40 --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

Fetch fresh candles before training:

```powershell
python crypto_predictor.py train --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

### Train a 1-Hour Model with Futures Fundamentals

Fetch futures fundamentals:

```powershell
python fundamental_data.py --symbol BTC/USDT --timeframe 1h --market-data dataset/1h-btc_history.csv --output dataset/1h-btc_fundamentals.csv
```

Train multi-horizon models:

```powershell
python crypto_predictor.py train-multi --update --update-fundamentals --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --sequence-length 96 --epochs 40 --max-train-rows 5000 --feature-set advanced-fundamental --feature-selection correlation --max-selected-features 32 --min-selected-features 12 --target-mode classification --threshold 0.0015 --min-confidence 0.50
```

Fundamental data is joined with a backward as-of merge, so each candle only sees values published at or before that candle. The fundamental file is aligned to market candles, and `fundamental_source_age_hours` tells the model how old the real source data is. The derived futures features are designed to capture the main short-horizon forces behind BTC moves: leverage expansion, crowded longs/shorts, aggressive taker flow, funding extremes, squeeze risk, and cases where open interest rises while price fails to confirm.

### Model Smoke Test

```powershell
python smoke_test_models.py --symbol BTC/USDT --timeframe 1h --horizons 1,3,6 --sanitize
```

The `--sanitize` option removes unsupported Keras config keys from saved model archives before loading.

### Predict Once

```powershell
python crypto_predictor.py predict --data dataset/5m_btc_history.csv
```

With fresh exchange data and Telegram:

```powershell
python crypto_predictor.py predict --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --telegram
```

### Telegram Alert Bot

Run one alert check with the saved 1h multi-horizon models:

```powershell
python alert_bot.py --update --update-fundamentals --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --min-agree 1 --min-confidence 0.45 --strategy hybrid
```

Run continuously on a Python server:

```powershell
python alert_bot.py --mode loop --update --update-fundamentals --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --min-agree 1 --min-confidence 0.45 --strategy hybrid --sleep-seconds 300
```

The `.github/workflows/telegram-alert.yml` workflow runs every 15 minutes and sends `HOLD` updates too.

Telegram messages include a `Why` section with horizon votes, confidence, range/trend regime context, and the reason behind the final `LONG`, `SHORT`, or `HOLD` signal.

### Paper Trading

Backtest the current 1h setup with a virtual 100 USD futures account:

```powershell
python paper_trader.py --mode backtest --data dataset/1h-btc_history.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100 --days 7 --fee-rate 0.001 --spread-bps 2 --slippage-bps 2
```

Run one live paper-trading step:

```powershell
python paper_trader.py --mode single --update --update-fundamentals --telegram --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100 --min-agree 1 --min-confidence 0.45 --strategy hybrid --risk-enabled --risk-max-drawdown-pct 5 --risk-max-loss-streak 3 --dynamic-position-sizing --dynamic-min-position-size-pct 0.25 --dynamic-max-position-size-pct 1.0 --spread-bps 2 --slippage-bps 2
```

The `.github/workflows/paper-trading.yml` workflow runs hourly, updates market/fundamental data, applies the hybrid strategy and risk controls, sends a Telegram status message, and commits the virtual account state to `paper_state.json`.

Dynamic position sizing runs after the risk manager. If risk allows a new entry, signal quality is scored with confidence, agreeing horizon votes, strongest horizon confidence, and market regime, then position size is scaled between the configured minimum and maximum.

Use `--spread-bps` and `--slippage-bps` to make fills more conservative. For example, `--spread-bps 2 --slippage-bps 2` makes each entry/exit roughly 3 bps worse than the mid price.

Paper trading stores the latest signal explanation in `last_signal_reason` and `last_signal_reasons`. When a trade opens or closes, the trade record also keeps a short decision summary.

### Signal Performance Report

```powershell
python signal_report.py --symbol BTC/USDT --state-file paper_state.json --telegram
```

This report reads the paper-trading history and groups results by signal, trade direction, market regime, strategy, and close reason. The `.github/workflows/signal-report.yml` workflow sends it to Telegram every day at `03:30 UTC`.

### Performance Report

```powershell
python performance_report.py --telegram --symbol BTC/USDT --data dataset/1h-btc_history.csv --state-file paper_state.json --initial-capital 100
```

The `.github/workflows/performance-report.yml` workflow sends this report daily at `03:00 UTC`. It reports equity, realized/unrealized PnL, win rate, drawdown, best/worst trade, and the current paper position.

### Health Check

```powershell
python health_check.py --telegram --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --paper-state paper_state.json --alert-state alert_state.json
```

The `.github/workflows/health-check.yml` workflow runs every 6 hours. It checks market-data freshness, fundamental-data freshness, model files, paper/alert state timestamps, dataset gaps, and paper equity.

### Strategy Optimization

Search for better paper-trading settings:

```powershell
python strategy_optimizer.py --data dataset/1h-btc_history.csv --days 14 --initial-capital 100 --top 10 --spread-bps 2 --slippage-bps 2
```

Walk-forward validation:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --train-days 14 --test-days 3 --step-days 3 --walkforward-days 45 --initial-capital 100 --spread-bps 2 --slippage-bps 2
```

Test hybrid and range strategies:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --train-days 7 --test-days 2 --step-days 2 --walkforward-days 16 --initial-capital 100 --horizon-sets "1,3,6;1,6;3,6;6" --confidence-grid "0.45,0.50" --min-agree-grid "1,2" --strategy-grid "model,hybrid,range" --range-lower-grid "0.15,0.20,0.25" --range-upper-grid "0.75,0.80,0.85" --range-atr-max-grid "0.006,0.008,0.012" --range-trend-max-grid "0.002,0.003,0.005"
```

`hybrid` uses the model in trending/unclear markets and switches to range mean reversion when price is near rolling support/resistance inside a low-trend range.

### Automatic Retraining

The `.github/workflows/model-retrain.yml` workflow retrains the 1h models every night at `02:00 UTC`:

1. Refresh market and fundamental data
2. Train horizons `1,3,6` into `models/staging`
3. Smoke test staging models
4. Run walk-forward validation against staging
5. Check the quality gate
6. If accepted, promote staging into production `models/`
7. Send a Telegram report either way and store the latest retrain state in `model_registry.json`

### Free Scheduled Execution

This project is ready for scheduled execution on GitHub Actions. Cloudflare Workers cannot run this Python/TensorFlow model directly. GitHub Actions is the simplest free scheduled runner. For true always-on looping, use a small Python VPS or container.

### Trading Notes

- Use real money only after long enough paper trading.
- A few profitable days are not enough; results must stay stable out of sample.
- Always include fees, slippage, stop loss, and drawdown.
- Keep position size small.
- If the model fails repeatedly, stop trading and inspect the reports.
