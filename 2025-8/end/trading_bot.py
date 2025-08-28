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

# تنظیمات لاگ
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('trading_bot.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# اطلاعات API
load_dotenv()
API_KEY = os.getenv("COINEX_API_KEY")
API_SECRET = os.getenv("COINEX_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
required_envs = [API_KEY, API_SECRET, TELEGRAM_TOKEN, CHAT_ID]
if any(v is None for v in required_envs):
    logger.error("یکی از متغیرهای محیطی لازم تنظیم نشده است.")
    raise ValueError("یکی از متغیرهای محیطی لازم تنظیم نشده است.")

def connect_to_coinex():
    try:
        exchange = ccxt.coinex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 2000,
        })
        exchange.load_markets()
        logger.info("اتصال به CoinEx با موفقیت انجام شد.")
        send_telegram_message("اتصال به CoinEx با موفقیت انجام شد.")
        return exchange
    except Exception as e:
        logger.error(f"خطا در اتصال به CoinEx: {str(e)}")
        send_telegram_message(f"خطا در اتصال به CoinEx: {str(e)}")
        return None

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=payload)
        response.raise_for_status()
        logger.info("پیام به تلگرام ارسال شد.")
    except Exception as e:
        logger.error(f"خطا در ارسال به تلگرام: {str(e)}")

def fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1000, since=None, retries=7):
    for attempt in range(retries):
        try:
            params = {'limit': limit}
            if since is not None and limit > 1:
                params['since'] = since
                logger.info(f"درخواست API برای {symbol} با since={since} ({pd.to_datetime(since, unit='ms')})")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, params=params)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
                time.sleep(15 * (2 ** attempt))
            else:
                send_telegram_message(f"خطا در گرفتن داده‌های {symbol} پس از {retries} تلاش: {str(e)}")
                return None

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs.fillna(0)))
    return rsi

def prepare_features(df):
    df = df.copy()
    df['ma5'] = df['close'].shift(1).rolling(window=5).mean()
    df['pct_change'] = df['close'].shift(1).pct_change()
    df['rsi'] = calculate_rsi(df['close'].shift(1))
    df['volume_norm'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    df['momentum'] = df['close'].shift(1).diff(5)
    features = ['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum']
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag + 1)
    df = df.dropna()
    X = df[features + [f'close_lag_{lag}' for lag in [1, 2, 3]]].values
    y = df['close'].shift(-1).dropna().values
    X = X[:-1]
    timestamps = df['timestamp'][:-1].values
    logger.info(f"تعداد نمونه‌ها بعد از پیش‌پردازش: {len(X)}")
    return df, X, y, timestamps

def train_and_test_model(X, y, timestamps, symbol):
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 7], 'min_samples_split': [2, 5]}
        model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        best_model = model.best_estimator_
        train_indices, test_indices = list(tscv.split(X))[-1]
        X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"MSE (test) برای {symbol}: {mse}, R2 (test): {r2}")
        send_telegram_message(f"MSE (test) برای {symbol}: {mse}, R2 (test): {r2}")
        return best_model, scaler, X_test, y_test
    except Exception as e:
        logger.error(f"خطا در آموزش مدل برای {symbol}: {str(e)}")
        send_telegram_message(f"خطا در آموزش مدل برای {symbol}: {str(e)}")
        return None, None, None, None

def predict_next_close(model, scaler, last_data):
    last_data_scaled = scaler.transform([last_data])
    return model.predict(last_data_scaled)[0]

def wait_for_next_candle():
    current_time = time.time()
    seconds_to_next_candle = 300 - (current_time % 300)
    if seconds_to_next_candle > 0:
        logger.info(f"منتظر کندل بعدی، {seconds_to_next_candle:.0f} ثانیه باقی مانده")
        time.sleep(seconds_to_next_candle)

def simulate_live_trading(exchange, symbol, initial_capital=100, fee=0.002):
    try:
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

        base_currency = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
        initial_df = fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1000)
        if initial_df is None or len(initial_df) < 20:
            logger.error(f"دریافت داده اولیه برای {symbol} ناموفق بود یا ناکافی است: {len(initial_df) if initial_df is not None else 0} کندل")
            send_telegram_message(f"دریافت داده اولیه برای {symbol} ناموفق بود یا ناکافی است.")
            return 0
        historical_df = initial_df
        logger.info(f"داده اولیه برای {symbol}: {len(historical_df)} کندل")

        current_time = int(time.time() * 1000)
        last_timestamp = current_time - (1000 * 5 * 60 * 1000)
        logger.info(f"last_timestamp اولیه: {last_timestamp} ({pd.to_datetime(last_timestamp, unit='ms')})")

        for i in range(72):  # 72 کندل × 5 دقیقه = 6 ساعت
            try:
                wait_for_next_candle()
                current_time = int(time.time() * 1000)
                last_timestamp = current_time - (current_time % (5 * 60 * 1000))
                logger.info(f"last_timestamp برای کندل {i+1}: {last_timestamp} ({pd.to_datetime(last_timestamp, unit='ms')})")

                new_df = fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1, since=None)
                if new_df is None or len(new_df) == 0:
                    logger.warning(f"داده جدید برای {symbol} نیست.")
                    send_telegram_message(f"داده جدید برای {symbol} نیست.")
                    continue

                historical_df = pd.concat([historical_df, new_df]).drop_duplicates(subset='timestamp').sort_values('timestamp')
                logger.info(f"تعداد کل کندل‌ها در historical_df برای {symbol}: {len(historical_df)}")

                if len(historical_df) < 20:
                    logger.warning(f"داده کافی برای {symbol} نیست: {len(historical_df)} کندل")
                    send_telegram_message(f"داده کافی برای {symbol} نیست: {len(historical_df)} کندل")
                    continue

                df_processed, X, y, timestamps = prepare_features(historical_df)
                if len(X) < 10:
                    logger.warning(f"داده پردازش‌شده کافی برای {symbol} نیست: {len(X)} نمونه")
                    continue

                if model is None:
                    model, scaler, _, _ = train_and_test_model(X, y, timestamps, symbol)
                    if model is None:
                        logger.warning(f"آموزش مدل برای {symbol} ناموفق بود.")
                        continue

                last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum'] + 
                                        [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values
                predicted_next_close = predict_next_close(model, scaler, last_data)
                current_price = df_processed['close'].iloc[-1]

                if last_predicted_close is not None and len(historical_df) > 1:
                    actual_close = historical_df['close'].iloc[-1]
                    logger.info(f"close واقعی کندل قبلی: {actual_close:.6f}, پیش‌بینی قبلی: {last_predicted_close:.6f}, خطا: {abs(actual_close - last_predicted_close):.6f}")
                    send_telegram_message(f"close واقعی کندل قبلی: {actual_close:.6f}, پیش‌بینی قبلی: {last_predicted_close:.6f}")

                if position == 0:
                    if predicted_next_close > current_price * 1.0003:
                        trade_amount = capital * 0.1 / current_price
                        position += trade_amount
                        capital -= trade_amount * current_price * (1 + fee)
                        buy_price = current_price
                        logger.info(f"کندل {i+1}: خرید {trade_amount:.6f} {base_currency} در قیمت {current_price:.2f}")
                        send_telegram_message(f"کندل {i+1}: خرید {trade_amount:.6f} {base_currency} در قیمت {current_price:.2f}")
                        trade_count += 1
                else:
                    if predicted_next_close < current_price * 0.9997:
                        sell_amount = position
                        capital += sell_amount * current_price * (1 - fee)
                        trade_profit = (current_price - buy_price) * sell_amount - (buy_price * fee + current_price * fee) * sell_amount
                        profit += trade_profit
                        logger.info(f"کندل {i+1}: فروش {sell_amount:.6f} {base_currency} در قیمت {current_price:.2f}, سود: {trade_profit:.2f}$")
                        send_telegram_message(f"کندل {i+1}: فروش {sell_amount:.6f} {base_currency} در قیمت {current_price:.2f}, سود: {trade_profit:.2f}$")
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

        if position > 0:
            final_price = historical_df['close'].iloc[-1]
            capital += position * final_price * (1 - fee)
            final_profit = (final_price - buy_price) * position - (buy_price * fee + final_price * fee) * position
            profit += final_profit
            logger.info(f"پایان: فروش باقی‌مونده {position:.6f} {base_currency} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
            send_telegram_message(f"پایان: فروش باقی‌مونده {position:.6f} {base_currency} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")

        total_profit = profit
        logger.info(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        send_telegram_message(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
        return total_profit

    except KeyboardInterrupt:
        logger.info(f"توقف کلی برنامه توسط کاربر برای {symbol}.")
        send_telegram_message(f"توقف کلی برنامه توسط کاربر برای {symbol}.")
        if position > 0 and len(historical_df) > 0:
            final_price = historical_df['close'].iloc[-1]
            capital += position * final_price * (1 - fee)
            final_profit = (final_price - buy_price) * position - (buy_price * fee + final_price * fee) * position
            profit += final_profit
            logger.info(f"توقف: فروش باقی‌مونده {position:.6f} {base_currency} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
            send_telegram_message(f"توقف: فروش باقی‌مونده {position:.6f} {base_currency} در قیمت {final_price:.2f}, سود: {final_profit:.2f}$")
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

        symbols = ["ADAUSDT", "ETHUSDT", "XRPUSDT"]
        for symbol in symbols:
            logger.info(f"تحلیل زنده {symbol}...")
            send_telegram_message(f"تحلیل زنده {symbol} شروع شد...")
            profit = simulate_live_trading(exchange, symbol, initial_capital=100)
            send_telegram_message(f"سود کل برای {symbol} بعد از 6 ساعت: {profit:.2f}$")
    except KeyboardInterrupt:
        logger.info("توقف برنامه توسط کاربر در main.")
        send_telegram_message("توقف برنامه توسط کاربر در main.")

if __name__ == "__main__":
    main()