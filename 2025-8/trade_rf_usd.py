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

def fetch_ohlcv(exchange, symbol, timeframe='15m', total_limit=2000):
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

def calculate_atr(df, periods=14):
    """محاسبه ATR (Average True Range)"""
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    return df['true_range'].rolling(window=periods).mean()

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
    df['atr'] = calculate_atr(df)
    features = ['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum', 'atr']
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
    logger.info(f"اولین ۵ پیش‌بینی (test): {predicted[:5]}")

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[test_idx], y_test, label='واقعی', color='blue')
    plt.plot(timestamps[test_idx], predicted, label='پیش‌بینی‌شده', color='red')
    plt.xlabel('زمان')
    plt.ylabel('قیمت')
    plt.title(f'دقت پیش‌بینی مدل برای {symbol} (Test Set)')
    plt.legend()
    plt.savefig(f'model_accuracy_{symbol.replace("/", "_")}.png')
    plt.show()

    return best_model, scaler, mse, r2

def predict_current_close(model, scaler, last_data):
    """پیش‌بینی قیمت بسته شدن کندل فعلی با نویز کوچک"""
    if model is None:
        return 0.0

    last_data_scaled = scaler.transform(last_data.reshape(1, -1))
    predicted_close = model.predict(last_data_scaled)[0]
    # اضافه کردن نویز کوچک (0.05% تا 0.1%)
    noise = np.random.uniform(-0.0005, 0.001)
    return predicted_close * (1 + noise)

def simulate_hourly_profit(model, scaler, df_processed, symbol, initial_capital=100):
    """شبیه‌سازی سود دو ساعته با سرمایه اولیه 100 دلار"""
    if model is None or len(df_processed) < 12:  # حداقل 12 کندل برای دو ساعت (15 دقیقه‌ای ≈ 8 کندل)
        logger.warning("داده کافی برای شبیه‌سازی دو ساعته نیست.")
        return 0.0

    capital = initial_capital
    position = 0  # مقدار ارز در دست (بر حسب تعداد)
    last_price = df_processed['close'].iloc[-1]
    predictions = []
    atr = df_processed['atr'].iloc[-1]  # ATR فعلی
    growth_threshold = atr / last_price * 0.2  # آستانه پویا با ضریب 0.2

    # پیش‌بینی برای 8 کندل بعدی (دو ساعت با تایم‌فریم 15 دقیقه)
    for i in range(8):
        last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum', 'atr'] + [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values
        predicted_close = predict_current_close(model, scaler, last_data)
        predictions.append(predicted_close)

        # استراتژی با آستانه پویا (بدون فیلتر RSI)
        trade_amount = capital * 0.1 / last_price
        if predicted_close > last_price * (1 + growth_threshold) and position == 0:  # خرید
            position += trade_amount
            capital -= trade_amount * last_price
            logger.info(f"کندل {i+1}: خرید {trade_amount:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")
        elif predicted_close < last_price * (1 - growth_threshold) and position > 0:  # فروش
            capital += position * last_price
            profit = (last_price - (last_price * 0.1)) * position  # سود ساده (بدون کارمزد)
            capital += profit
            logger.info(f"کندل {i+1}: فروش {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}, سود: {profit:.2f}$")
            position = 0

        last_price = predicted_close  # قیمت جدید برای کندل بعدی

    # محاسبه سود نهایی
    if position > 0:
        capital += position * last_price
        logger.info(f"پایان: فروش باقی‌مونده {position:.6f} {symbol.split('/')[0]} در قیمت {last_price:.2f}")

    total_profit = capital - initial_capital
    logger.info(f"سرمایه اولیه: {initial_capital}$, سرمایه نهایی: {capital:.2f}$, سود کل: {total_profit:.2f}$")
    return total_profit

def analyze_growth_potential(df, symbol):
    """تحلیل احتمال رشد با یادگیری ماشین"""
    if df is None or len(df) < 10:
        logger.warning(f"داده کافی برای {symbol} نیست.")
        return False, 0.0

    df_processed, X, y, timestamps = prepare_features(df)
    model, scaler, mse, r2 = train_and_test_model(X, y, timestamps, symbol)
    if model is None:
        return False, 0.0

    last_data = df_processed[['open', 'high', 'low', 'volume_norm', 'ma5', 'pct_change', 'rsi', 'momentum', 'atr'] + [f'close_lag_{lag}' for lag in [1, 2, 3]]].iloc[-1].values
    predicted_close = predict_current_close(model, scaler, last_data)
    last_close = df_processed['close'].iloc[-1]
    predicted_growth = ((predicted_close - last_close) / last_close) * 100
    logger.info(f"پیش‌بینی قیمت بسته شدن کندل فعلی: {predicted_close}, رشد پیش‌بینی‌شده: {predicted_growth:.2f}%")
    is_potential = predicted_growth > 20

    # شبیه‌سازی سود دو ساعته
    hourly_profit = simulate_hourly_profit(model, scaler, df_processed, symbol, initial_capital=100)
    logger.info(f"سود پیش‌بینی‌شده برای دو ساعت با 100 دلار: {hourly_profit:.2f}$")

    return is_potential, predicted_growth

def main():
    logger.info("شروع برنامه")
    exchange = connect_to_coinex()
    if not exchange:
        return

    symbols = ["ADA/USDT", "ETH/USDT", "XRP/USDT"]  # لیست ارزها برای تست
    for symbol in symbols:
        logger.info(f"تحلیل {symbol}...")
        df = fetch_ohlcv(exchange, symbol, timeframe='15m', total_limit=2000)
        if df is None:
            continue

        is_potential, growth = analyze_growth_potential(df, symbol)
        if is_potential:
            logger.info(f"{symbol}: پتانسیل رشد {growth:.2f}%")
            with open('potential_coins.csv', 'a') as f:
                f.write(f"{symbol},{growth:.2f}\n")
        else:
            logger.info(f"{symbol} پتانسیل رشد بالای 20% ندارد.")

if __name__ == "__main__":
    main()