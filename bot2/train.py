import ccxt
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')  # Use mixed precision for speed on M1/M2

import joblib  # For saving/loading scaler (برای ذخیره/بارگذاری اسکیلر)
from logging.handlers import RotatingFileHandler  # For rotating logs (برای چرخش لاگ‌ها)
import requests  # For Telegram (برای تلگرام)

# Setup rotating logging (تنظیم لاگ گردشی)
handler = RotatingFileHandler('train_log.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

load_dotenv()
ACCESS_ID = os.getenv('Access_ID')
SECRET_KEY = os.getenv('Secret_Key')
def send_telegram_message(message):
    """
    Send a message to your Telegram bot (ارسال پیام به ربات تلگرام)
    """
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        logging.warning("Telegram TOKEN or CHAT_ID not set in .env - message not sent")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'  # برای فرمت بهتر (بولد، ایموجی و ...)
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            logging.info(f"Telegram message sent: {message}")
        else:
            logging.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")
def send_plot_to_telegram(plot_path, caption=""):
    """
    Send a saved plot (PNG) to Telegram with caption (ارسال پلات ذخیره‌شده به تلگرام با کپشن)
    """
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        logging.warning("Telegram TOKEN or CHAT_ID not set - plot not sent")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    try:
        with open(plot_path, 'rb') as photo:
            files = {'photo': photo}
            payload = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'}
            response = requests.post(url, data=payload, files=files, timeout=15)
        
        if response.status_code == 200:
            logging.info(f"Plot sent to Telegram: {plot_path}")
            print(f"📊 Plot sent to Telegram!")
        else:
            logging.error(f"Telegram plot error: {response.text}")
    except Exception as e:
        logging.error(f"Error sending plot to Telegram: {e}")
def fetch_and_update_data(symbol='BTC/USDT', timeframe='5m', batch_limit=1000, file='dataset/5m_btc_history.csv', retries=3):
    exchange = ccxt.coinex({'apiKey': ACCESS_ID, 'secret': SECRET_KEY, 'enableRateLimit': True})
    
    try:
        old_df = pd.read_csv(file)
        old_df['timestamp'] = pd.to_datetime(old_df['timestamp'])
        last_timestamp = old_df['timestamp'].max().value // 10**6
        since = last_timestamp + 1
        logging.info(f"Existing dataset loaded: {len(old_df)} candles, fetching from since {since}")
    except FileNotFoundError:
        old_df = pd.DataFrame()
        since = None
        logging.info("No existing dataset - fetching new")
    
    new_data = []
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
            if not ohlcv:
                logging.info("No new data available")
                break
            new_data.extend(ohlcv)
            logging.info(f"Fetched {len(ohlcv)} new candles")
            break
        except Exception as e:
            logging.error(f"Retry {attempt+1}/{retries}: {e}")
            time.sleep(5)
    
    if new_data:
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        combined.to_csv(file, index=False)
        logging.info(f"Updated dataset: {len(combined)} candles (added {len(new_df)} new)")
        return combined
    else:
        logging.info("No new data - returning existing")
        return old_df
def add_ichimoku_features(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    """
    Manual Ichimoku Cloud calculation + flat level detection + reaction assessment
    (محاسبه دستی ابر ایچیموکو + تشخیص سطح صاف + سنجش واکنش قیمت)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 1. Conversion Line (Tenkan-sen) - خط تبدیل
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
    
    # 2. Base Line (Kijun-sen) - خط پایه
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # 3. Leading Span A (Senkou Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
    
    # 4. Leading Span B (Senkou Span B) - سنکو اسپن B
    span_b_high = high.rolling(window=senkou_period).max()
    span_b_low = low.rolling(window=senkou_period).min()
    df['senkou_span_b'] = ((span_b_high + span_b_low) / 2).shift(kijun_period)
    
    # 5. Lagging Span (Chikou Span) - اختیاری، برای تأیید
    df['chikou_span'] = close.shift(-kijun_period)
    
    # تشخیص سطح صاف (Flat level detection between Tenkan and Senkou Span B)
    window = 5  # پنجره برای چک صاف بودن
    df['tenkan_variance'] = df['tenkan_sen'].rolling(window=window).var()
    df['span_b_variance'] = df['senkou_span_b'].rolling(window=window).var()
    df['tenkan_span_b_diff'] = abs(df['tenkan_sen'] - df['senkou_span_b'])
    
    variance_threshold = df['close'].mean() * 0.0005  # آستانه پویا بر اساس قیمت (0.05%)
    diff_threshold = df['close'].mean() * 0.005       # تفاوت کمتر از 0.5%
    
    df['is_flat_ichimoku_level'] = (
        (df['tenkan_variance'] < variance_threshold) &
        (df['span_b_variance'] < variance_threshold) &
        (df['tenkan_span_b_diff'] < diff_threshold)
    )
    
    # سنجش واکنش قیمت به سطح صاف (Reaction assessment)
    reaction_window = 10
    df['ichimoku_reaction'] = 0.0
    # بعد از محاسبه is_flat_level
    df['ichimoku_reaction'] = 0.0

    # Find indices where flat level exists in last reaction_window
    flat_mask = df['is_flat_ichimoku_level'].rolling(window=reaction_window).sum() > 0
    indices = flat_mask[flat_mask].index

    for i in indices:
        level_slice = df['senkou_span_b'].iloc[i - reaction_window:i]
        level = level_slice.mean()
        price_change = (df['close'].iloc[i] - df['close'].iloc[i - reaction_window]) / df['close'].iloc[i - reaction_window]
        volume_factor = df['volume'].iloc[i] / df['volume'].iloc[i - reaction_window:i].mean()
        df.loc[i, 'ichimoku_reaction'] = price_change * volume_factor
    return df
def calculate_fibonacci_levels(df):
    # Find swing high/low (نوسان بالا/پایین - simple method: rolling max/min)
    df['swing_high'] = df['high'].rolling(window=20).max().shift(1)  # Recent high (بالای اخیر)
    df['swing_low'] = df['low'].rolling(window=20).min().shift(1)  # Recent low (پایین اخیر)
    
    fib_ratios = [-0.13, 0, 0.13, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.13]  # Standard + custom (استاندارد + سفارشی)
    
    for ratio in fib_ratios:
        df[f'fib_{ratio}'] = df['swing_low'] + (df['swing_high'] - df['swing_low']) * ratio
    
    # Measure reaction (سنجش واکنش): proximity to nearest fib level (نزدیکی به نزدیک‌ترین سطح)
    fib_cols = [col for col in df.columns if col.startswith('fib_')]
    df['nearest_fib'] = df.apply(lambda row: min(fib_cols, key=lambda col: abs(row['close'] - row[col])), axis=1)
    df['fib_reaction'] = df.apply(lambda row: abs(row['close'] - row[row['nearest_fib']]) / row['close'], axis=1)  # Relative proximity (نزدیکی نسبی - 0 = on level, small = strong reaction)
    
    # Drop NaN and add to features (حذف NaN و اضافه به ویژگی‌ها)
    df = df.dropna()
    return df
def preprocess_data(df):
    """
    Preprocess data: add indicators, lags, volatility, and Fibonacci levels.
    """
    # Simple Moving Average and EMA
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan).fillna(1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Lagged close prices
    for lag in [1, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    # Volatility
    df['volatility'] = df['high'] - df['low']
    df = add_ichimoku_features(df)  # اضافه کردن ایچیموکو
    df = df.dropna().reset_index(drop=True)
    df = calculate_fibonacci_levels(df)  # اضافه کردن سطوح فیبوناچی
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_50', 'ema_20', 'rsi_14', 
                'macd', 'macd_signal', 'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 
                'volatility', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                'ichimoku_reaction', 'is_flat_ichimoku_level', 'fib_reaction']
    
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    df_scaled['timestamp'] = df['timestamp'].values
    
    return df_scaled, scaler, df
def eda(df_original, df_processed):
    plt.figure(figsize=(14, 6))
    plt.plot(df_original['timestamp'], df_original['close'], label='Close Price')
    plt.plot(df_original['timestamp'], df_original['sma_50'], label='SMA 50', alpha=0.7)
    plt.plot(df_original['timestamp'], df_original['ema_20'], label='EMA 20', alpha=0.7)
    plt.title('BTC Price with Moving Averages')
    plt.legend()
    plt.savefig('pic/new/5m_btc_price_ma.png')
    plot_path = 'pic/new/5m_btc_price_ma.png'
    send_plot_to_telegram(plot_path, caption="📊 5m_BTC Price with Moving Averages")
    plt.close()
    # send telegram this plot
    

    plt.figure(figsize=(14, 4))
    plt.plot(df_original['timestamp'], df_original['rsi_14'])
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(30, color='g', linestyle='--', alpha=0.5)
    plt.title('RSI Indicator')
    plt.savefig('pic/new/5m_rsi_indicator.png')
    plot_path = 'pic/new/5m_rsi_indicator.png'
    send_plot_to_telegram(plot_path, caption="📈 5m_RSI Indicator")
    plt.close()
    
    plt.figure(figsize=(12, 10))
    corr = df_processed.drop(columns=['timestamp']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.savefig('pic/new/5m_feature_correlation.png')
    plot_path = 'pic/new/5m_feature_correlation.png'
    send_plot_to_telegram(plot_path, caption="📉 5m_Feature Correlation Heatmap")
    plt.close()
def create_sequences(df_scaled, sequence_length=60):
    X, y = [], []
    data_values = df_scaled.drop(columns=['timestamp']).values
    for i in range(sequence_length, len(data_values)):
        X.append(data_values[i-sequence_length:i])
        y.append(data_values[i, 0])  # close index
    X = np.array(X)
    y = np.array(y)
    logging.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    return X, y

def build_and_train_model(X, y, sequence_length=60):    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    model = Sequential()
    model.add(Input(shape=(sequence_length, X.shape[2])))  # Input layer اول (هشدار رفع می‌شه)
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(50, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=64, callbacks=[early_stop], verbose=1)
    
    return model, X_test, y_test, history

def evaluate_model(model, X_test, y_test, scaler, df_original=None):
    y_pred_scaled = model.predict(X_test)
    
    dummy_pred = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
    dummy_pred[:, 0] = y_pred_scaled.flatten()  # close index
    y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_test = np.zeros((len(y_test), scaler.scale_.shape[0]))
    dummy_test[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test)[:, 0]
    
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    
    logging.info(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    print(f"\nModel Evaluation in 5m Results:")
    print(f"MAE: {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"R² Score: {r2:.4f}")
    send_telegram_message(f"Model Evaluation in 5m Results:\nMAE: {mae:.2f} USD\nRMSE: {rmse:.2f} USD\nR² Score: {r2:.4f}")
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_actual[-200:], label='Actual Price', alpha=0.8)
    plt.plot(y_pred[-200:], label='Predicted Price', alpha=0.8)
    plt.title('5m_Actual vs Predicted BTC Price (Test Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('pic/new/5m_actual_vs_predicted.png')
    plot_path = 'pic/new/5m_actual_vs_predicted.png'
    send_plot_to_telegram(plot_path, caption="📈 5m Actual vs Predicted BTC Price")
    plt.close()
    
    errors = y_test_actual - y_pred
    plt.figure(figsize=(14, 4))
    plt.plot(errors[-200:])
    plt.title('Prediction Errors')
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel('Error (USD)')
    plt.savefig('pic/new/5m_prediction_errors.png')
    plot_path = 'pic/new/5m_prediction_errors.png'
    send_plot_to_telegram(plot_path, caption="📉 5m Prediction Errors")
    plt.close()
    
    return mae, rmse, r2

def backtest(data, model, scaler, sequence_length=60, initial_capital=10000, threshold=0.3):
    df_processed, _, _ = preprocess_data(data.copy())
    X, _ = create_sequences(df_processed, sequence_length)
    
    predictions = model.predict(X)
    
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy[:, 0] = predictions.flatten()
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]
    
    actual_prices = data['close'].iloc[sequence_length:].values
    
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(predicted_prices)):
        current_price = actual_prices[i]
        predicted = predicted_prices[i]
        change_pct = (predicted - current_price) / current_price * 100
        
        if change_pct > threshold and position == 0:
            position = 1
            buy_price = current_price
            trades.append(f"BUY at {current_price:.2f}")
        elif change_pct < -threshold and position == 1:
            position = 0
            sell_price = current_price
            profit = (sell_price - buy_price) / buy_price * capital
            capital += profit
            trades.append(f"SELL at {sell_price:.2f}, Profit: {profit:.2f} USD")
    if position == 1 and (current_price - buy_price) / buy_price < -0.01:  # -1% stop-loss
        position = 0
        sell_price = current_price
        profit = (sell_price - buy_price) / buy_price * capital
        capital += profit
        trades.append(f"STOP-LOSS SELL at {sell_price:.2f}, Loss: {profit:.2f} USD")                            
    if position == 1:
        sell_price = actual_prices[-1]
        profit = (sell_price - buy_price) / buy_price * capital
        capital += profit
        trades.append(f"FINAL SELL at {sell_price:.2f}, Profit: {profit:.2f} USD")
    
    total_return = (capital - initial_capital) / initial_capital * 100
    print(f"\nBacktesting Results:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)//2 if position == 0 else len(trades)//2 + 1}")
    
    return capital, total_return

def train_and_backtest(symbol='BTC/USDT', timeframe='5m'):
    data = fetch_and_update_data(symbol=symbol, timeframe=timeframe, batch_limit=1000)
    if data is not None and len(data) > 60:
        df_processed, scaler, df_original = preprocess_data(data)
        X, y = create_sequences(df_processed)
        eda(df_original, df_processed)

        model, X_test, y_test, history = build_and_train_model(X, y)  # scaler حذف از فراخوانی اگر در تابع نیاز نیست
        
        model.save('btc_lstm_model.keras')
        joblib.dump(scaler, 'scaler.pkl')
        print("Model retrained and saved!")
        
        mae, rmse, r2 = evaluate_model(model, X_test, y_test, scaler, df_original)  # scaler از main
        
        capital, return_pct = backtest(data, model, scaler)
        
        msg = f"Retraining Done!\nR²: {r2:.4f}\nMAE: {mae:.2f} USD\nBacktest Return: {return_pct:.2f}%"
        send_telegram_message(msg)
if __name__ == "__main__":
    train_and_backtest()