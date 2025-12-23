import ccxt  # Library for exchange connection (کتابخانه اتصال به صرافی)
import pandas as pd  # For dataframes (برای دیتافریم‌ها)
import time  # For delays (برای تأخیرها)
import logging  # For logs (برای لاگ‌ها)
from dotenv import load_dotenv  # For env variables (برای متغیرهای محیطی)
import os  # For OS operations (برای عملیات سیستم‌عامل)
import numpy as np  # For calculations (برای محاسبات عددی)
from sklearn.preprocessing import MinMaxScaler  # For scaling (برای نرمال‌سازی)
import matplotlib.pyplot as plt  # For plotting (برای رسم نمودار)
import seaborn as sns  # For better plots (برای نمودارهای زیباتر)

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
        eda(df_original, df_processed)  # قبلی
        
        X, y = create_sequences(df_processed, sequence_length=60)
        print(f"\nSequences ready!")
        print(f"X shape: {X.shape}  -> (samples, timesteps, features)")
        print(f"y shape: {y.shape}  -> target close prices (normalized)")
        eda(df_original, df_processed)