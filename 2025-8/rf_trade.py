import os
from dotenv import load_dotenv
import ccxt
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# بارگذاری متغیرهای محیطی
load_dotenv()

# اطلاعات API
API_KEY = os.getenv("COINEX_API_KEY")
API_SECRET = os.getenv("COINEX_API_SECRET")

def connect_to_coinex():
    """اتصال به API CoinEx"""
    try:
        exchange = ccxt.coinex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
        })
        logger.info("اتصال به CoinEx با موفقیت انجام شد.")
        return exchange
    except Exception as e:
        logger.error(f"خطا در اتصال به CoinEx: {e}")
        return None

def fetch_ohlcv(exchange, symbol, timeframe='1d', limit=90):
    """گرفتن داده‌های قیمتی OHLCV"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"خطا در گرفتن داده‌های {symbol}: {e}")
        return None

def train_and_test_model(df):
    """آموزش و تست مدل با داده‌های واقعی"""
    if df is None or len(df) < 7:
        return None, None

    # آماده‌سازی داده‌ها با ویژگی‌های بیشتر
    features = ['open', 'high', 'low', 'close', 'volume']
    X = df[features].values
    y = df['close'].values

    # تقسیم داده‌ها به آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # آموزش مدل
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)  # 200 درخت، عمق محدود
    model.fit(X_train, y_train)

    # تست مدل
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    logger.info(f"Mean Squared Error: {mse}")

    # رسم نمودار دقت
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'].iloc[-len(y_test):], y_test, label='واقعی', color='blue')
    plt.plot(df['timestamp'].iloc[-len(y_test):], predicted, label='پیش‌بینی‌شده', color='red')
    plt.xlabel('زمان')
    plt.ylabel('قیمت')
    plt.title('دقت پیش‌بینی مدل برای BTC/USDT')
    plt.legend()
    plt.savefig('model_accuracy.png')
    plt.show()

    return model, mse

def predict_growth(model, df):
    """پیش‌بینی رشد با مدل آموزش‌دیده"""
    if model is None:
        return 0.0

    # استفاده از آخرین داده‌ها برای پیش‌بینی
    last_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-1].values.reshape(1, -1)
    future_days = np.array([last_data[0] * np.ones(5)])  # فرض ساده: ویژگی‌ها ثابت می‌مونن
    for _ in range(7):
        pred = model.predict(future_days)
        future_days = np.vstack((future_days, [last_data[0][0], last_data[0][1], last_data[0][2], pred[0], last_data[0][4]]))
    predicted_close = future_days[-1][3]  # قیمت پیش‌بینی‌شده روز هفتم
    last_close = df['close'].iloc[-1]
    predicted_growth = ((predicted_close - last_close) / last_close) * 100
    return predicted_growth

def analyze_growth_potential(df, symbol):
    """تحلیل احتمال رشد با یادگیری ماشین (Random Forest)"""
    if df is None or len(df) < 7:
        logger.warning(f"داده کافی برای {symbol} نیست.")
        return False, 0.0

    model, mse = train_and_test_model(df)
    if model is None:
        return False, 0.0

    predicted_growth = predict_growth(model, df)
    is_potential = predicted_growth > 20
    return is_potential, predicted_growth

def main():
    logger.info("شروع برنامه")
    exchange = connect_to_coinex()
    if not exchange:
        return

    # بررسی فقط یک ارز (BTC/USDT)
    symbol = "BTC/USDT"
    logger.info(f"تحلیل {symbol}...")
    df = fetch_ohlcv(exchange, symbol)
    if df is not None:
        is_potential, growth = analyze_growth_potential(df, symbol)
        if is_potential:
            logger.info(f"{symbol}: پتانسیل رشد {growth:.2f}%")
            with open('potential_coins.csv', 'a') as f:
                f.write(f"{symbol},{growth:.2f}\n")
        else:
            logger.info(f"{symbol} پتانسیل رشد بالای 20% ندارد.")

if __name__ == "__main__":
    main()