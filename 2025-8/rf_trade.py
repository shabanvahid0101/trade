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

def fetch_ohlcv(exchange, symbol, timeframe='5m', total_limit=2000):
    """گرفتن داده‌های قیمتی OHLCV با جمع‌آوری تا 2000 تایی"""
    try:
        all_ohlcv = []
        limit = 1000
        since = None

        while len(all_ohlcv) < total_limit:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            if len(ohlcv) < limit:
                break

            logger.info(f"گرفتن {len(all_ohlcv)} کندل...")

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"تعداد کندل‌های جمع‌آوری‌شده: {len(df)}")
        return df.iloc[:total_limit] if len(df) > total_limit else df
    except Exception as e:
        logger.error(f"خطا در گرفتن داده‌های {symbol}: {e}")
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

def train_and_test_model(X, y, timestamps):
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
    logger.info(f"اولین ۵ پیش‌بینی (test): {predicted[:5]}")

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[test_idx], y_test, label='واقعی', color='blue')
    plt.plot(timestamps[test_idx], predicted, label='پیش‌بینی‌شده', color='red')
    plt.xlabel('زمان')
    plt.ylabel('قیمت')
    plt.title('دقت پیش‌بینی مدل برای BTC/USDT (Test Set)')
    plt.legend()
    plt.savefig('model_accuracy_improved.png')
    plt.show()

    return best_model, scaler, mse, r2

def predict_current_close(model, scaler, last_data):
    """پیش‌بینی قیمت بسته شدن کندل فعلی"""
    if model is None:
        return 0.0

    last_data_scaled = scaler.transform(last_data.reshape(1, -1))
    predicted_close = model.predict(last_data_scaled)[0]
    return predicted_close

def analyze_growth_potential(df, symbol):
    """تحلیل احتمال رشد با یادگیری ماشین"""
    if df is None or len(df) < 10:
        logger.warning(f"داده کافی برای {symbol} نیست.")
        return False, 0.0

    df_processed, X, y, timestamps = prepare_features(df)
    model, scaler, mse, r2 = train_and_test_model(X, y, timestamps)
    if model is None:
        return False, 0.0

    last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum'] + [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values
    predicted_close = predict_current_close(model, scaler, last_data)
    last_close = df_processed['close'].iloc[-1]
    predicted_growth = ((predicted_close - last_close) / last_close) * 100
    logger.info(f"پیش‌بینی قیمت بسته شدن کندل فعلی: {predicted_close}, رشد پیش‌بینی‌شده: {predicted_growth:.2f}%")
    is_potential = predicted_growth > 20
    return is_potential, predicted_growth

def main():
    logger.info("شروع برنامه")
    exchange = connect_to_coinex()
    if not exchange:
        return

    symbol = "BTC/USDT"
    logger.info(f"تحلیل {symbol}...")
    df = fetch_ohlcv(exchange, symbol, timeframe='5m', total_limit=2000)
    if df is None:
        return

    is_potential, growth = analyze_growth_potential(df, symbol)
    if is_potential:
        logger.info(f"{symbol}: پتانسیل رشد {growth:.2f}%")
        with open('potential_coins.csv', 'a') as f:
            f.write(f"{symbol},{growth:.2f}\n")
    else:
        logger.info(f"{symbol} پتانسیل رشد بالای 20% ندارد.")

if __name__ == "__main__":
    main()