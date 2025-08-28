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

logging.basicConfig(level=logging.INFO, filename='trading.log')

logger = logging.getLogger(__name__)



# بارگذاری متغیرهای محیطی

load_dotenv()



# اطلاعات API

API_KEY = os.getenv("COINEX_API_KEY")

API_SECRET = os.getenv("COINEX_API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # توکن ربات تلگرام

CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Chat ID تلگرام



def send_telegram_message(message):

    """ارسال پیام به تلگرام با مدیریت خطا"""

    if not TELEGRAM_TOKEN or not CHAT_ID:

        logger.warning("توکن تلگرام یا Chat ID تنظیم نشده است.")

        return

    try:

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

        payload = {"chat_id": CHAT_ID, "text": message}

        response = requests.post(url, data=payload)

        if response.status_code != 200:

            logging.error(f"Telegram error: {response.text}")

            print(f"Telegram error: {response.text}")

        else:

            print(f"Message sent: {message}")

    except Exception as e:

        logging.error(f"Error sending to Telegram: {e}")

        print(f"Error: {e}")

def connect_to_coinex():

    """اتصال به API CoinEx"""

    try:

        exchange = ccxt.coinex({

            'apiKey': API_KEY,

            'secret': API_SECRET,

            'enableRateLimit': True,

        })

        logger.info("اتصال به CoinEx با موفقیت انجام شد.")

        send_telegram_message("اتصال به CoinEx با موفقیت انجام شد.")

        return exchange

    except Exception as e:

        logger.error(f"خطا در اتصال به CoinEx: {e}")

        send_telegram_message(f"خطا در اتصال به CoinEx: {e}")

        return None



def fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1000):

    """گرفتن داده‌های قیمتی OHLCV با تایم‌فریم 5 دقیقه"""

    try:

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        logger.info(f"تعداد کندل‌های جمع‌آوری‌شده: {len(df)}")

        send_telegram_message(f"تعداد کندل‌های جمع‌آوری‌شده برای {symbol}: {len(df)}")

        return df

    except Exception as e:

        logger.error(f"خطا در گرفتن داده‌های {symbol}: {e}")

        send_telegram_message(f"خطا در گرفتن داده‌های {symbol}: {e}")

        return None



def calculate_rsi(data, periods=14):

    """محاسبه RSI"""

    delta = data.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()

    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    rs = gain / loss

    return 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0)))



def prepare_features(df):

    """آماده‌سازی ویژگی‌ها با lagged values و شاخص‌ها"""

    df = df.copy()

    df = df.dropna()

    df['current_close'] = df['close']  # پیش‌بینی قیمت بسته شدن فعلی

    df['ma5'] = df['close'].rolling(window=5).mean()

    df['pct_change'] = df['close'].pct_change()

    df['rsi'] = calculate_rsi(df['close'])

    df['volume_norm'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()

    df['momentum'] = df['close'].diff(5)  # تغییرات 5 کندل

    features = ['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum']

    for lag in [1, 2, 3]:

        df.loc[:, f'close_lag_{lag}'] = df['close'].shift(lag)

    df = df.dropna()

    X = df[features + [f'close_lag_{lag}' for lag in [1, 2, 3]]].values

    y = df['current_close'].values  # پیش‌بینی قیمت فعلی

    timestamps = df['timestamp'].values

    logger.info(f"تعداد نمونه‌ها بعد از پیش‌پردازش: {len(X)}")

    return df, X, y, timestamps



def train_and_test_model(X, y, timestamps, symbol):

    """آموزش و تست مدل با TimeSeriesSplit و GridSearchCV"""

    if len(X) < 10:

        return None, None, None, None



    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}

    model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, scoring='neg_mean_squared_error')

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    model.fit(X_scaled, y)



    logger.info(f"بهترین پارامترها: {model.best_params_}")

    best_model = model.best_estimator_



    # ارزیابی روی آخرین split (test set)

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        break  # فقط آخرین split رو می‌گیریم



    predicted = best_model.predict(X_test)

    mse = mean_squared_error(y_test, predicted)

    r2 = r2_score(y_test, predicted)

    logger.info(f"MSE (test): {mse}, R2 (test): {r2}")

    send_telegram_message(f"MSE (test) برای {symbol}: {mse}, R2 (test): {r2}")

    logger.info(f"اولین ۵ پیش‌بینی (test): {predicted[:5]}")



    return best_model, scaler, mse, r2



def predict_current_close(model, scaler, last_data):

    """پیش‌بینی قیمت بسته شدن کندل فعلی با نویز کوچک"""

    if model is None:

        return 0.0



    last_data_scaled = scaler.transform(last_data.reshape(1, -1))

    predicted_close = model.predict(last_data_scaled)[0]

    noise = np.random.uniform(-0.0005, 0.001)

    return predicted_close * (1 + noise)



def simulate_live_trading(model, scaler, exchange, symbol, initial_capital=100):

    """شبیه‌سازی ترید زنده هر 5 دقیقه و محاسبه سود بعد از یک ساعت"""

    capital = initial_capital

    position = 0

    last_price = 0

    profit = 0

    trade_count = 0



    for i in range(12):  # 12 کندل × 5 دقیقه = 60 دقیقه

        try:

            df = fetch_ohlcv(exchange, symbol, timeframe='5m', limit=1000)

            if df is None or len(df) < 10:

                logger.warning(f"داده کافی برای {symbol} نیست.")

                send_telegram_message(f"داده کافی برای {symbol} نیست.")

                time.sleep(300)  # 5 دقیقه صبر

                continue



            df_processed, X, y, timestamps = prepare_features(df)

            if model is None:

                model, scaler, _, _ = train_and_test_model(X, y, timestamps, symbol)



            last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum'] + [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values

            predicted_close = predict_current_close(model, scaler, last_data)

            current_price = df_processed['close'].iloc[-1]



            if i == 0:

                last_price = current_price

            else:

                trade_amount = capital * 0.1 / last_price

                if predicted_close > last_price * 1.001 and position == 0:  # آستانه 0.1% برای خرید

                    position += trade_amount

                    capital -= trade_amount * last_price

                    logger.info(f"کندل {i+1}: خرید {trade_amount:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")

                    send_telegram_message(f"کندل {i+1}: خرید {trade_amount:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")

                    trade_count += 1

                elif predicted_close < last_price * 0.999 and position > 0:  # آستانه 0.1% برای فروش

                    capital += position * last_price

                    profit += (last_price - (last_price * 0.1)) * position  # سود ساده (بدون کارمزد)

                    logger.info(f"کندل {i+1}: فروش {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}, سود: {(last_price - (last_price * 0.1)) * position:.2f}$")

                    send_telegram_message(f"کندل {i+1}: فروش {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}, سود: {(last_price - (last_price * 0.1)) * position:.2f}$")

                    position = 0

                    trade_count += 1



                last_price = current_price



            time.sleep(300)  # 5 دقیقه صبر

        except Exception as e:

            logger.error(f"خطا در کندل {i+1} برای {symbol}: {e}")

            send_telegram_message(f"خطا در کندل {i+1} برای {symbol}: {e}")

            time.sleep(300)



    # محاسبه سود نهایی

    if position > 0:

        capital += position * last_price

        logger.info(f"پایان: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")

        send_telegram_message(f"پایان: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")



    total_profit = capital + profit - initial_capital

    logger.info(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")

    send_telegram_message(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")

    return total_profit



def main():

    logger.info("شروع برنامه")

    send_telegram_message("شروع برنامه")

    exchange = connect_to_coinex()

    if not exchange:

        return



    symbols = ["ADA/USDT", "ETH/USDT", "XRP/USDT"]

    for symbol in symbols:

        logger.info(f"تحلیل زنده {symbol}...")

        send_telegram_message(f"تحلیل زنده {symbol} شروع شد...")

        model, scaler, _, _ = None, None, None, None

        profit = simulate_live_trading(model, scaler, exchange, symbol, initial_capital=100)

        send_telegram_message(f"سود کل برای {symbol} بعد از یک ساعت: {profit:.2f}$")



if __name__ == "__main__":

    main()