import os
from dotenv import load_dotenv
import ccxt
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# تنظیمات لاگ برای دیباگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# بارگذاری متغیرهای محیطی
load_dotenv()

# اطلاعات ورود به API
API_KEY = os.getenv("COINEX_API_KEY")
API_SECRET = os.getenv("COINEX_API_SECRET")

def connect_to_coinex():
    """اتصال به API CoinEx"""
    try:
        exchange = ccxt.coinex({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # برای جلوگیری از محدودیت‌های API
        })
        logger.info("اتصال به CoinEx با موفقیت انجام شد.")
        return exchange
    except Exception as e:
        logger.error(f"خطا در اتصال به CoinEx: {e}")
        return None

def fetch_ohlcv(exchange, symbol, timeframe='1d', limit=30):
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
    """آموزش و تست مدل با داده‌های مصنوعی یا واقعی"""
    if df is None or len(df) < 7:  # حداقل 7 دوره برای پیش‌بینی
        return None, None

    # آماده‌سازی داده‌ها
    df['day'] = np.arange(len(df))  # ویژگی روز
    X = df['day'].values.reshape(-1, 1)
    y = df['close'].values

    # تقسیم داده‌ها به آموزش و تست
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # آموزش مدل
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 درخت
    model.fit(X_train, y_train)

    # تست مدل
    predicted = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted)
    logger.info(f"Mean Squared Error: {mse}")
    # append mse to csv file
    with open('model_performance.csv', 'a') as f:
        f.write(f"{mse}\n")

    return model, mse

def predict_growth(model, df):
    """پیش‌بینی رشد با مدل آموزش‌دیده"""
    if model is None:
        return 0.0

    # پیش‌بینی برای 7 روز آینده
    future_days = np.array([[len(df) + i for i in range(1, 8)]])
    predicted_prices = model.predict(future_days.reshape(-1, 1))
    last_close = df['close'].iloc[-1]
    predicted_close = predicted_prices[-1]
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

    # گرفتن لیست ارزها
    markets = exchange.load_markets()
    logger.info(f"تعداد ارزها: {len(markets)}")

    # بررسی ارزها
    potential_coins = []
    for symbol in list(markets.keys()):
        if '/USDT' not in symbol and '/USDC' not in symbol:
            continue  # فقط جفت‌های USDT/USDC
        logger.info(f"تحلیل {symbol}...")
        with open('model_performance.csv', 'a') as f:
            f.write(f"{symbol},")
        df = fetch_ohlcv(exchange, symbol)
        if df is not None:
            is_potential, growth = analyze_growth_potential(df, symbol)
            if is_potential:
                potential_coins.append((symbol, growth))


    # نمایش نتایج
    if potential_coins:
        logger.info("ارزهای با پتانسیل رشد 20%:")
        # create csv file for potential coins
        for coin, growth in potential_coins:
            logger.info(f"{coin}: پتانسیل رشد {growth:.2f}%")
            with open('potential_coins.csv', 'a') as f:
                f.write(f"{coin},{growth:.2f}\n")
            
    else:
        logger.info("هیچ ارزی با پتانسیل رشد 20% یافت نشد.")

if __name__ == "__main__":
    main()