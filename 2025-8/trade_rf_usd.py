import os
from dotenv import load_dotenv
import ccxt
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import requests
from logging.handlers import RotatingFileHandler

# تنظیمات لاگ با encoding UTF-8
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('trading.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# تنظیم encoding کنسول (برای ویندوز)
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# بارگذاری متغیرهای محیطی
load_dotenv()

# اطلاعات API
API_KEY = os.getenv("COINEX_API_KEY")
API_SECRET = os.getenv("COINEX_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # توکن ربات تلگرام
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Chat ID تلگرام

# چک کردن env vars در شروع
required_envs = [API_KEY, API_SECRET, TELEGRAM_TOKEN, CHAT_ID]
if any(v is None for v in required_envs):
    logger.error("یکی از متغیرهای محیطی لازم تنظیم نشده است.")
    raise ValueError("متغیرهای محیطی لازم تنظیم نشده‌اند.")

def send_telegram_message(message):
    """ارسال پیام به تلگرام با مدیریت خطا"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"خطا در ارسال به تلگرام: {str(e)}")

def connect_to_coinex():
    """اتصال به API CoinEx"""
    try:
        exchange = ccxt.coinex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
        })
        exchange.load_markets()
        logger.info("اتصال به CoinEx با موفقیت انجام شد.")
        send_telegram_message("اتصال به CoinEx با موفقیت انجام شد.")
        return exchange
    except Exception as e:
        logger.error(f"خطا در اتصال به CoinEx: {str(e)}")
        send_telegram_message(f"خطا در اتصال به CoinEx: {str(e)}")
        return None

def fetch_ohlcv(exchange, symbol, timeframe='5m', limit=100, since=None, retries=7):
    """گرفتن داده‌های قیمتی OHLCV با تایم‌فریم 5 دقیقه"""
    for attempt in range(retries):
        try:
            params = {'limit': limit}
            if since:
                params['since'] = since
                logger.info(f"درخواست API برای {symbol} با since={since} ({pd.to_datetime(since, unit='ms')})")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, params=params)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # بررسی زمان‌های غیرمنطقی
            current_time = int(time.time() * 1000)
            if df['timestamp'].max().timestamp() * 1000 > current_time + 3600 * 1000 or df['timestamp'].min().timestamp() * 1000 < current_time - 30 * 24 * 3600 * 1000:
                logger.warning(f"زمان‌های کندل غیرمنطقی هستند: از {df['timestamp'].min()} تا {df['timestamp'].max()}")
                df['timestamp'] = pd.to_datetime(current_time - ((limit - df.index) * 5 * 60 * 1000), unit='ms')
            if len(df) == 0:
                logger.warning(f"داده‌ای برای {symbol} دریافت نشد.")
                send_telegram_message(f"داده‌ای برای {symbol} دریافت نشد.")
                return None
            logger.info(f"تعداد کندل‌های جدید جمع‌آوری‌شده برای {symbol}: {len(df)}, اولین کندل: {df['timestamp'].iloc[0]}, آخرین کندل: {df['timestamp'].iloc[-1]}")
            send_telegram_message(f"تعداد کندل‌های جدید جمع‌آوری‌شده برای {symbol}: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"خطا در گرفتن داده‌های {symbol} (تلاش {attempt+1}/{retries}): {str(e)}, پاسخ سرور: {e.response if hasattr(e, 'response') else 'نامشخص'}")
            if attempt < retries - 1:
                time.sleep(15 * (2 ** attempt))  # افزایش زمان backoff
            else:
                send_telegram_message(f"خطا در گرفتن داده‌های {symbol} پس از {retries} تلاش: {str(e)}")
                return None

def calculate_rsi(data, periods=14):
    """محاسبه RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss.replace(0, np.nan)  # جلوگیری از division by zero
    rsi = 100 - (100 / (1 + rs.fillna(0)))
    return rsi

def calculate_atr(df, periods=14):
    """محاسبه ATR (Average True Range)"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=periods).mean()
    return atr

def prepare_features(df):
    """آماده‌سازی ویژگی‌ها با lagged values و شاخص‌ها برای پیش‌بینی آینده"""
    df = df.copy()
    df['ma5'] = df['close'].shift(1).rolling(window=5).mean()  # MA بر اساس گذشته
    df['pct_change'] = df['close'].shift(1).pct_change()  # تغییرات گذشته
    df['rsi'] = calculate_rsi(df['close'].shift(1))  # RSI بر اساس گذشته
    df['atr'] = calculate_atr(df)  # ATR
    df['volume_norm'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    df['momentum'] = df['close'].shift(1).diff(5)  # momentum گذشته

    features = ['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'atr', 'momentum']
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag + 1)  # lagged از گذشته

    df = df.dropna()
    X = df[features + [f'close_lag_{lag}' for lag in [1, 2, 3]]].values
    y = df['close'].shift(-1).dropna().values  # پیش‌بینی close بعدی
    X = X[:-1]  # align with y
    timestamps = df['timestamp'][:-1].values
    logger.info(f"تعداد نمونه‌ها بعد از پیش‌پردازش: {len(X)}")
    return df, X, y, timestamps

def train_and_test_model(X, y, timestamps, symbol):
    """آموزش و تست مدل با TimeSeriesSplit و GridSearchCV"""
    if len(X) < 10:
        logger.warning(f"داده ناکافی برای آموزش مدل {symbol}.")
        return None, None, None, None

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
    model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error')

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    logger.info(f"بهترین پارامترها: {model.best_params_}")
    best_model = model.best_estimator_

    # ارزیابی روی آخرین split
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    predicted = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    logger.info(f"MSE (test): {mse}, R2 (test): {r2}")
    send_telegram_message(f"MSE (test) برای {symbol}: {mse}, R2 (test): {r2}")
    logger.info(f"اولین ۵ پیش‌بینی (test): {predicted[:5]}")

    return best_model, scaler, mse, r2

def predict_next_close(model, scaler, last_data):
    """پیش‌بینی قیمت بسته شدن کندل بعدی بدون نویز"""
    if model is None:
        logger.warning("مدل وجود ندارد، پیش‌بینی ممکن نیست.")
        return 0.0

    last_data_scaled = scaler.transform(last_data.reshape(1, -1))
    predicted_close = model.predict(last_data_scaled)[0]
    logger.info(f"پیش‌بینی close بعدی: {predicted_close:.6f}")
    return predicted_close

def wait_for_next_candle(interval=300):
    """صبر تا مرز کندل بعدی (همگام‌سازی)"""
    try:
        current_time = time.time()
        sleep_time = interval - (current_time % interval)
        if sleep_time > 1:  # اگر زمان انتظار کمتر از ۱ ثانیه است، نخواب
            logger.info(f"منتظر {sleep_time:.2f} ثانیه برای کندل بعدی...")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("توقف برنامه توسط کاربر در انتظار کندل.")
        raise

def simulate_live_trading(exchange, symbol, initial_capital=100, fee=0.002):
    """شبیه‌سازی ترید زنده هر 5 دقیقه و محاسبه سود بعد از یک ساعت"""
    try:
        # چک کردن جفت‌ارز
        markets = exchange.load_markets()
        if symbol not in markets:
            logger.error(f"جفت‌ارز {symbol} در CoinEx موجود نیست.")
            send_telegram_message(f"جفت‌ارز {symbol} در CoinEx موجود نیست.")
            return 0

        capital = initial_capital
        position = 0
        buy_price = 0
        profit = 0
        trade_count = 0
        historical_df = pd.DataFrame()
        model, scaler = None, None
        last_predicted_close = None

        # دریافت داده اولیه (۲۰۰ کندل برای بهبود مدل)
        initial_df = fetch_ohlcv(exchange, symbol, timeframe='5m', limit=200)
        if initial_df is None or len(initial_df) < 20:
            logger.error(f"دریافت داده اولیه برای {symbol} ناموفق بود یا ناکافی است: {len(initial_df) if initial_df is not None else 0} کندل")
            send_telegram_message(f"دریافت داده اولیه برای {symbol} ناموفق بود یا ناکافی است.")
            return 0
        historical_df = initial_df
        logger.info(f"داده اولیه برای {symbol}: {len(historical_df)} کندل")

        # تنظیم last_timestamp بر اساس زمان محلی
        current_time = int(time.time() * 1000)
        last_timestamp = current_time - (200 * 5 * 60 * 1000)  # 200 کندل قبل
        logger.info(f"last_timestamp اولیه: {last_timestamp} ({pd.to_datetime(last_timestamp, unit='ms')})")

        for i in range(12):  # 12 کندل × 5 دقیقه = 60 دقیقه
            try:
                wait_for_next_candle()  # همگام‌سازی
                # به‌روزرسانی last_timestamp با زمان محلی
                current_time = int(time.time() * 1000)
                last_timestamp = current_time - (current_time % (5 * 60 * 1000))  # شروع کندل فعلی
                logger.info(f"last_timestamp برای کندل {i+1}: {last_timestamp} ({pd.to_datetime(last_timestamp, unit='ms')})")

                new_df = fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1, since=last_timestamp)
                if new_df is None or len(new_df) == 0:
                    logger.warning(f"داده جدید برای {symbol} نیست.")
                    send_telegram_message(f"داده جدید برای {symbol} نیست.")
                    continue

                historical_df = pd.concat([historical_df, new_df]).drop_duplicates(subset='timestamp').sort_values('timestamp')
                logger.info(f"تعداد کل کندل‌ها در historical_df برای {symbol}: {len(historical_df)}")

                if len(historical_df) < 20:  # حداقل داده برای ویژگی‌ها
                    logger.warning(f"داده کافی برای {symbol} نیست: {len(historical_df)} کندل")
                    send_telegram_message(f"داده کافی برای {symbol} نیست: {len(historical_df)} کندل")
                    continue

                df_processed, X, y, timestamps = prepare_features(historical_df)
                if len(X) < 10:
                    logger.warning(f"داده پردازش‌شده کافی برای {symbol} نیست: {len(X)} نمونه")
                    continue

                if model is None:  # فقط اولین بار train
                    model, scaler, _, _ = train_and_test_model(X, y, timestamps, symbol)
                    if model is None:
                        logger.warning(f"آموزش مدل برای {symbol} ناموفق بود.")
                        continue

                # ویژگی‌ها برای آخرین ردیف (برای پیش‌بینی بعدی)
                last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'atr', 'momentum'] + 
                                        [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values
                predicted_next_close = predict_next_close(model, scaler, last_data)
                current_price = df_processed['close'].iloc[-1]

                # مقایسه پیش‌بینی قبلی با close واقعی
                if last_predicted_close is not None and len(historical_df) > 1:
                    actual_close = historical_df['close'].iloc[-1]
                    logger.info(f"close واقعی کندل قبلی: {actual_close:.6f}, پیش‌بینی قبلی: {last_predicted_close:.6f}, خطا: {abs(actual_close - last_predicted_close):.6f}")
                    send_telegram_message(f"close واقعی کندل قبلی: {actual_close:.6f}, پیش‌بینی قبلی: {last_predicted_close:.6f}")

                if position == 0:
                    if predicted_next_close > current_price * 1.0005:  # آستانه 0.05% برای خرید
                        trade_amount = capital * 0.1 / current_price  # 10% ریسک
                        position += trade_amount
                        capital -= trade_amount * current_price * (1 + fee)
                        buy_price = current_price
                        logger.info(f"کندل {i+1}: خرید {trade_amount:.6f} {symbol.split('/')[0]} در قیمت {current_price:.2f}")
                        send_telegram_message(f"کندل {i+1}: خرید {trade_amount:.6f} {symbol.split('/')[0]} در قیمت {current_price:.2f}")
                        trade_count += 1
                else:
                    if predicted_next_close < current_price * 0.9995:  # آستانه 0.05% برای فروش
                        sell_amount = position
                        capital += sell_amount * current_price * (1 - fee)
                        trade_profit = (current_price - buy_price) * sell_amount - (buy_price * fee + current_price * fee) * sell_amount
                        profit += trade_profit
                        logger.info(f"کندل {i+1}: فروش {sell_amount:.6f} {symbol.split('/')[0]} در قیمت {current_price:.2f}, سود: {trade_profit:.2f}$")
                        send_telegram_message(f"کندل {i+1}: فروش {sell_amount:.6f} {symbol.split('/')[0]} در قیمت {current_price:.2f}, سود: {trade_profit:.2f}$")
                        position = 0
                        trade_count += 1

                last_predicted_close = predicted_next_close

            except KeyboardInterrupt:
                logger.info(f"توقف برنامه توسط کاربر در کندل {i+1} برای {symbol}.")
                send_telegram_message(f"توقف برنامه توسط کاربر در کندل {i+1} برای {symbol}.")
                break
            except Exception as e:
                logger.error(f"خطا در کندل {i+1} برای {symbol}: {str(e)}")
                send_telegram_message(f"خطا در کندل {i+1} برای {symbol}: {str(e)}")
                continue

        # محاسبه سود نهایی و بستن پوزیشن‌های باز
        if position > 0:
            final_price = historical_df['close'].iloc[-1]
            capital += position * final_price * (1 - fee)
            final_profit = (final_price - buy_price) * position - (buy_price * fee + final_price * fee) * position
            profit += final_profit
            logger.info(f"پایان: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
            send_telegram_message(f"پایان: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")

        total_profit = profit
        logger.info(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        send_telegram_message(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        return total_profit

    except KeyboardInterrupt:
        logger.info(f"توقف کلی برنامه توسط کاربر برای {symbol}.")
        send_telegram_message(f"توقف کلی برنامه توسط کاربر برای {symbol}.")
        # بستن پوزیشن‌های باز
        if position > 0 and len(historical_df) > 0:
            final_price = historical_df['close'].iloc[-1]
            capital += position * final_price * (1 - fee)
            final_profit = (final_price - buy_price) * position - (buy_price * fee + final_price * fee) * position
            profit += final_profit
            logger.info(f"توقف: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
            send_telegram_message(f"توقف: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
        total_profit = profit
        logger.info(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        send_telegram_message(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        return total_profit

def main():
    try:
        logger.info("شروع برنامه")
        send_telegram_message("شروع برنامه")
        exchange = connect_to_coinex()
        if not exchange:
            return

        symbols = ["ADA/USDT", "ETH/USDT", "XRP/USDT"]
        for symbol in symbols:
            logger.info(f"تحلیل زنده {symbol}...")
            send_telegram_message(f"تحلیل زنده {symbol} شروع شد...")
            profit = simulate_live_trading(exchange, symbol, initial_capital=100)
            send_telegram_message(f"سود کل برای {symbol} بعد از یک ساعت: {profit:.2f}$")
    except KeyboardInterrupt:
        logger.info("توقف برنامه توسط کاربر در main.")
        send_telegram_message("توقف برنامه توسط کاربر در main.")

if __name__ == "__main__":
    main()