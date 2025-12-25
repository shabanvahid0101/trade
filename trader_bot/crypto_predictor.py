import ccxt  # Library for exchange connection (کتابخانه اتصال به صرافی)
import pandas as pd  # For dataframes (برای دیتافریم‌ها)
import time  # For delays (برای تأخیرها)
import logging  # For logs (برای لاگ‌ها)
from dotenv import load_dotenv  # For env variables (برای متغیرهای محیطی)
import os  # For OS operations (برای عملیات سیستم‌عامل)
import numpy as np  # For calculations (برای محاسبات عددی)
import matplotlib.pyplot as plt  # For plotting (برای رسم نمودار)
import seaborn as sns  # For better plots (برای نمودارهای زیباتر)
from sklearn.preprocessing import MinMaxScaler  # For scaling (برای نرمال‌سازی)
from sklearn.model_selection import train_test_split  # For splitting (برای تقسیم داده)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Metrics (معیارهای ارزیابی)
from tensorflow.keras.models import Sequential  # Keras model (مدل کراس)
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input  # Layers (لایه‌ها)
from tensorflow.keras.callbacks import EarlyStopping  # To stop early (توقف زودهنگام)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')  # Speed up on M1 (سرعت روی M1)

# Import for rotation (وارد کردن برای چرخش)
from logging.handlers import RotatingFileHandler  
# تنظیم rotating log (Setup rotating log)
handler = RotatingFileHandler('logging.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

load_dotenv()
ACCESS_ID = os.getenv('Access_ID')
SECRET_KEY = os.getenv('Secret_Key')

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
def preprocess_data(df):
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan).fillna(1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    for lag in [1, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    df['volatility'] = df['high'] - df['low']
    
    df = df.dropna().reset_index(drop=True)
    
    scaler = MinMaxScaler()
    # Dropped high/low/open/lag_3/5/10 for multicollinearity (برای چندخطی)
    features = ['close', 'volume', 'sma_50', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'close_lag_1', 'volatility']  
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
    plt.savefig('pic/new/btc_price_ma.png')
    plt.close()
    
    plt.figure(figsize=(14, 4))
    plt.plot(df_original['timestamp'], df_original['rsi_14'])
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(30, color='g', linestyle='--', alpha=0.5)
    plt.title('RSI Indicator')
    plt.savefig('pic/new/rsi_indicator.png')
    plt.close()
    
    plt.figure(figsize=(12, 10))
    corr = df_processed.drop(columns=['timestamp']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.savefig('pic/new/feature_correlation.png')
    plt.close()

def create_sequences(df_scaled, sequence_length=60):
    X, y = [], []
    data_values = df_scaled.drop(columns=['timestamp']).values
    for i in range(sequence_length, len(data_values)):
        X.append(data_values[i-sequence_length:i])
        y.append(data_values[i, 3])  # close index
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
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=32, callbacks=[early_stop], verbose=1)
    
    return model, X_test, y_test, history, scaler

def evaluate_model(model, X_test, y_test, scaler, df_original=None):
    y_pred_scaled = model.predict(X_test)
    
    dummy_pred = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
    dummy_pred[:, 3] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy_pred)[:, 3]
    
    dummy_test = np.zeros((len(y_test), scaler.scale_.shape[0]))
    dummy_test[:, 3] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test)[:, 3]
    
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    
    logging.info(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    print(f"\nModel Evaluation Results:")
    print(f"MAE: {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"R² Score: {r2:.4f}")
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_actual[-200:], label='Actual Price', alpha=0.8)
    plt.plot(y_pred[-200:], label='Predicted Price', alpha=0.8)
    plt.title('Actual vs Predicted BTC Price (Test Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('pic/new/actual_vs_predicted.png')
    plt.close()
    
    errors = y_test_actual - y_pred
    plt.figure(figsize=(14, 4))
    plt.plot(errors[-200:])
    plt.title('Prediction Errors')
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel('Error (USD)')
    plt.savefig('pic/new/prediction_errors.png')
    plt.close()
    
    return mae, rmse, r2

def predict_next_price(model, df_processed, scaler, sequence_length=60):
    last_sequence = df_processed.drop(columns=['timestamp']).values[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
    
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    dummy = np.zeros((1, scaler.scale_.shape[0]))
    dummy[0, 3] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy)[0, 3]
    
    current_price = df_processed['close'].iloc[-1] * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]
    
    print(f"\nNext Candle Prediction:")
    print(f"Current Price: {current_price:.2f} USD")
    print(f"Predicted Next Close: {pred_price:.2f} USD")
    print(f"Expected Change: {pred_price - current_price:.2f} USD ({((pred_price/current_price)-1)*100:.2f}%)")
    
    return pred_price

def live_trading_loop(model, scaler, sequence_length=60):

    print("Live bot started! Press Ctrl+C to stop.")
    while True:
        try:
            new_data = fetch_and_update_data(symbol='BTC/USDT', timeframe='5m', batch_limit=200, file='dataset/5m_btc_history.csv', retries=3)
            if new_data is not None:
                df_processed, scaler_new, df_original = preprocess_data(new_data)
                predicted = predict_next_price(model, df_processed, scaler, sequence_length)
                current = df_original['close'].iloc[-1]
                
                change_pct = ((predicted / current) - 1) * 100
                if change_pct > 0.3:
                    print(f"BUY SIGNAL! Expected +{change_pct:.2f}%")
                elif change_pct < -0.3:
                    print(f"SELL SIGNAL! Expected {change_pct:.2f}%")
                else:
                    print(f"HOLD - Change: {change_pct:.2f}%")
                
            time.sleep(300)
        except Exception as e:
            logging.error(f"Live loop error: {e}")
            time.sleep(60)

def backtest(data, model, scaler, sequence_length=60, initial_capital=10000, threshold=0.3):
    df_processed, _, _ = preprocess_data(data.copy())
    X, _ = create_sequences(df_processed, sequence_length)
    
    predictions = model.predict(X)
    
    # Inverse scale predictions (معکوس نرمال‌سازی پیش‌بینی‌ها)
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy[:, 3] = predictions.flatten()
    predicted_prices = scaler.inverse_transform(dummy)[:, 3]
    
    actual_prices = data['close'].iloc[sequence_length:].values  # Actual next close (قیمت واقعی بعدی)
    
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long (خرید)
    trades = []  # List of trades (لیست تریدها)
    
    for i in range(len(predicted_prices)):
        current_price = actual_prices[i]
        predicted = predicted_prices[i]
        change_pct = (predicted - current_price) / current_price * 100
        
        if change_pct > threshold and position == 0:
            position = 1
            buy_price = current_price
            trades.append(f"BUY at {current_price:.2f} on {data['timestamp'].iloc[sequence_length + i]}")
        elif change_pct < -threshold and position == 1:
            position = 0
            sell_price = current_price
            profit = (sell_price - buy_price) / buy_price * capital
            capital += profit
            trades.append(f"SELL at {sell_price:.2f}, Profit: {profit:.2f} USD")
    
    # Final sell if holding (فروش نهایی اگر نگه داشته)
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
    print("\nTrades:")
    for trade in trades:
        print(trade)
    
    return capital, total_return

if __name__ == "__main__":

    data = fetch_and_update_data(symbol='BTC/USDT', timeframe='5m')
    if data is not None:
        print("Raw data head:")
        print(data.head())
        
        df_processed, scaler, df_original = preprocess_data(data)
        print("\nPreprocessed data head:")
        print(df_processed.head())
        df_processed.to_csv('dataset/btc_preprocessed.csv')
        print("\nData saved to dataset/btc_preprocessed.csv")
        
        X, y = create_sequences(df_processed, sequence_length=60)
        print(f"\nSequences ready!")
        print(f"X shape: {X.shape}  -> (samples, timesteps, features)")
        print(f"y shape: {y.shape}  -> target close prices (normalized)")
        
        eda(df_original, df_processed)
        
        model, X_test, y_test, history, scaler = build_and_train_model(X, y)
        model.save('btc_lstm_model.keras')  # Use Keras format (فرمت جدید)

        print("\nModel trained and saved as btc_lstm_model.keras")
        capital, return_pct = backtest(data, model, scaler)
        
        mae, rmse, r2 = evaluate_model(model, X_test, y_test, scaler, df_original)
        
        next_price = predict_next_price(model, df_processed, scaler)
        
        live_trading_loop(model, scaler)