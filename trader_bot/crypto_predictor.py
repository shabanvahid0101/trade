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
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Layers (لایه‌ها)
from tensorflow.keras.callbacks import EarlyStopping  # To stop early (توقف زودهنگام)

load_dotenv()  # Load .env file (بارگذاری فایل .env)
ACCESS_ID = os.getenv('Access_ID')  # API key (کلید API)
SECRET_KEY = os.getenv('Secret_Key')  # API secret (راز API)

logging.basicConfig(filename='logging.log',level=logging.INFO,format='%(asctime)s - %(message)s',force=True)  # Set log level (تنظیم سطح لاگ)

def fetch_data(symbol='BTC/USDT', timeframe='5m', limit=1000, retries=3):  # Function to get data (تابع برای گرفتن داده)
    exchange = ccxt.coinex({'apiKey': ACCESS_ID, 'secret': SECRET_KEY, 'enableRateLimit': True})  # Connect to exchange (اتصال به صرافی)
    for attempt in range(retries):  # Loop for retries (حلقه برای تلاش مجدد)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Fetch OHLCV (گرفتن OHLCV)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # Create dataframe (ساخت دیتافریم)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp (تبدیل زمان)
            logging.info(f"Fetched {len(df)} rows for {symbol}")  # Log success (لاگ موفقیت)
            return df  # Return data (بازگشت داده)
        except Exception as e:  # Catch error (گرفتن خطا)
            logging.error(f"Retry {attempt+1}/{retries}: {e}")  # Log error (لاگ خطا)
            time.sleep(5)  # Wait (انتظار)
    return None  # If failed (اگر شکست خورد)
def preprocess_data(df):  # Function for data preparation (تابع آماده‌سازی داده)
    # Manual indicators (اندیکاتورهای دستی – بدون pandas_ta)
    df['sma_50'] = df['close'].rolling(window=50).mean()  # Simple Moving Average (میانگین متحرک ساده)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()  # Exponential Moving Average (میانگین متحرک نمایی)
    
    # RSI manual (شاخص قدرت نسبی دستی)
    delta = df['close'].diff()  # Price change (تغییر قیمت)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()  # Average gain (میانگین سود)
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()  # Average loss (میانگین ضرر)
    rs = gain / loss  # Relative strength (قدرت نسبی)
    df['rsi_14'] = 100 - (100 / (1 + rs))  # RSI formula (فرمول RSI)
    
    # Lagged features (ویژگی‌های تأخیری) – خیلی مهم برای time series
    for lag in [1, 3, 5, 10]:  # Previous prices (قیمت‌های قبلی)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    # Volatility feature (نوسان قیمت – مفید برای پیش‌بینی)
    df['volatility'] = df['high'] - df['low']
    
    df = df.dropna().reset_index(drop=True)  # Drop rows with NaN (حذف ردیف‌های ناقص)
    
    # Normalization (نرمال‌سازی به 0-1)
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'volume', 
                'sma_50', 'ema_20', 'rsi_14', 
                'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10',
                'volatility']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), 
                             columns=features, 
                             index=df.index)
    df_scaled['timestamp'] = df['timestamp'].values  # Keep time (نگه داشتن زمان)
    
    logging.info(f"Preprocessed data shape: {df_scaled.shape}")  # Log shape (لاگ اندازه)
    return df_scaled, scaler, df  # Return scaled, scaler, original (بازگشت داده اسکیل‌شده، اسکیلر، اصلی)

def eda(df_original, df_processed):
    # Price plot (نمودار قیمت)
    plt.figure(figsize=(14, 6))
    plt.plot(df_original['timestamp'], df_original['close'], label='Close Price')
    plt.plot(df_original['timestamp'], df_original['sma_50'], label='SMA 50', alpha=0.7)
    plt.plot(df_original['timestamp'], df_original['ema_20'], label='EMA 20', alpha=0.7)
    plt.title('BTC Price with Moving Averages')
    plt.legend()
    plt.show()
    
    # RSI plot
    plt.figure(figsize=(14, 4))
    plt.plot(df_original['timestamp'], df_original['rsi_14'])
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)  # Overbought (اشباع خرید)
    plt.axhline(30, color='g', linestyle='--', alpha=0.5)  # Oversold (اشباع فروش)
    plt.title('RSI Indicator')
    plt.show()
    
    # Correlation heatmap (نقشه حرارتی همبستگی)
    plt.figure(figsize=(12, 10))
    corr = df_processed.drop(columns=['timestamp']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.show()
def create_sequences(df_scaled, sequence_length=60):  # Function to create sequences (تابع ساخت توالی)
    X, y = [], []  # X: input sequences, y: target price (X: توالی‌های ورودی، y: قیمت هدف)
    
    # Drop timestamp for modeling (حذف زمان برای مدل‌سازی)
    data_values = df_scaled.drop(columns=['timestamp']).values
    
    for i in range(sequence_length, len(data_values)):  # Loop over data (حلقه روی داده)
        X.append(data_values[i-sequence_length:i])  # Last 60 rows as input (۶۰ ردیف قبلی)
        y.append(data_values[i, 3])  # Index 3 = 'close' column (ستون close هدف پیش‌بینی)
    
    X = np.array(X)  # Convert to numpy array (تبدیل به آرایه نامپای)
    y = np.array(y)
    
    logging.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    # مثال: X shape = (900, 60, 12) یعنی ۹۰۰ نمونه، هر کدام ۶۰ timestep، ۱۲ feature
    return X, y
    # Train/Test split - NO shuffle! (تقسیم بدون بهم زدن)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build LSTM model (ساخت مدل LSTM)
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(sequence_length, X.shape[2])))  # First layer (لایه اول)
    model.add(Dropout(0.2))  # Prevent overfitting (جلوگیری از اورفیتینگ)
    model.add(LSTM(50, return_sequences=False))  # Second layer (لایه دوم)
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))  # Fully connected (لایه تمام‌متصل)
    model.add(Dense(1))  # Output: one price (خروجی: یک قیمت)
    
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile (کامپایل)
    model.summary()  # Print model structure (چاپ ساختار مدل)
    
    # Early stopping (توقف اگر بهتر نشد)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Train (آموزش مدل)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,  # Max epochs (حداکثر اپوک)
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=1)
    
    return model, X_test, y_test, history, scaler
def build_and_train_model(X, y, sequence_length=60):
    # Train/Test split - NO shuffle for time series (تقسیم بدون بهم زدن برای سری زمانی)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Build improved LSTM model (ساخت مدل بهبودیافته LSTM)
    model = Sequential()  # <--- این خط فراموش شده بود! حالا اضافه شد
    
    # لایه‌های بیشتر برای دقت بالاتر
    model.add(LSTM(150, return_sequences=True, input_shape=(sequence_length, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # خروجی: قیمت بعدی
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # اضافه کردن mae به metrics
    model.summary()  # چاپ ساختار مدل
    
    # Early stopping برای جلوگیری از اورفیتینگ
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # Train مدل
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=150,  # بیشتر برای یادگیری بهتر
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=1)
    
    return model, X_test, y_test, history, scaler
def evaluate_model(model, X_test, y_test, scaler, df_original):
    # Predict on test set (پیش‌بینی روی داده تست)
    y_pred_scaled = model.predict(X_test)  # Scaled predictions (پیش‌بینی نرمال‌شده)
    
    # Inverse transform only 'close' column (معکوس نرمال‌سازی فقط برای ستون close)
    # Create dummy array with same shape as features
    dummy_pred = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
    dummy_pred[:, 3] = y_pred_scaled.flatten()  # Index 3 = close column
    y_pred = scaler.inverse_transform(dummy_pred)[:, 3]
    
    dummy_test = np.zeros((len(y_test), scaler.scale_.shape[0]))
    dummy_test[:, 3] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test)[:, 3]
    
    # Calculate metrics (محاسبه معیارها)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    
    logging.info(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    print(f"\nModel Evaluation Results:")
    print(f"MAE (Mean Absolute Error): {mae:.2f} USD")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} USD")
    print(f"R² Score: {r2:.4f} (closer to 1 = better)")
    
    # Plot actual vs predicted (نمودار واقعی در مقابل پیش‌بینی)
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_actual[-200:], label='Actual Price', alpha=0.8)  # Last 200 points (۲۰۰ نقطه آخر)
    plt.plot(y_pred[-200:], label='Predicted Price', alpha=0.8)
    plt.title('Actual vs Predicted BTC Price (Test Set)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    
    # Plot prediction error (نمودار خطا)
    errors = y_test_actual - y_pred
    plt.figure(figsize=(14, 4))
    plt.plot(errors[-200:])
    plt.title('Prediction Errors (Actual - Predicted)')
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel('Error (USD)')
    plt.show()
    
    return mae, rmse, r2

def predict_next_price(model, df_processed, scaler, sequence_length=60):
    # Take last sequence (آخرین توالی)
    last_sequence = df_processed.drop(columns=['timestamp']).values[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
    
    pred_scaled = model.predict(last_sequence, verbose=0)
    
    # Inverse scale
    dummy = np.zeros((1, scaler.scale_.shape[0]))
    dummy[0, 3] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy)[0, 3]
    
    current_price = df_processed['close'].iloc[-1] * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]  # Approximate current
    # Better: use df_original
    current_price = df_original['close'].iloc[-1]
    
    print(f"\nNext Candle Prediction:")
    print(f"Current Price: {current_price:.2f} USD")
    print(f"Predicted Next Close: {pred_price:.2f} USD")
    print(f"Expected Change: {pred_price - current_price:.2f} USD ({((pred_price/current_price)-1)*100:.2f}%)")
    
    return pred_price

# Test in main (تست در بخش اصلی)
if __name__ == "__main__":
    data = fetch_data(symbol='BTC/USDT', timeframe='5m', limit=1000)
    if data is not None:
        print("Raw data head:")
        print(data.head())
        
        df_processed, scaler, df_original = preprocess_data(data)
        print("\nPreprocessed data head:")
        print(df_processed.head())
        # Save to CSV for checking (ذخیره برای بررسی)
        df_processed.to_csv('btc_preprocessed.csv')
        print("\nData saved to btc_preprocessed.csv")
        df_processed, scaler, df_original = preprocess_data(data)
        
        X, y = create_sequences(df_processed, sequence_length=60)
        print(f"\nSequences ready!")
        print(f"X shape: {X.shape}  -> (samples, timesteps, features)")
        print(f"y shape: {y.shape}  -> target close prices (normalized)")
        # eda(df_original, df_processed)
        model, X_test, y_test, history, scaler = build_and_train_model(X, y)
        model.save('btc_lstm_model.h5')  # Save model (ذخیره مدل)
        print("\nModel trained and saved as btc_lstm_model.h5")
        # ... preprocess, sequences, train ...
        mae, rmse, r2 = evaluate_model(model, X_test, y_test, scaler, df_original)
        next_price = predict_next_price(model, df_processed, scaler)
