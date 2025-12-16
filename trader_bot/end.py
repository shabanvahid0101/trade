import ccxt  # Library for exchange connection (کتابخانه اتصال به صرافی)
import pandas as pd  # For dataframes (برای دیتافریم‌ها)
import time  # For delays (برای تأخیرها)
import logging  # For logs (برای لاگ‌ها)
from dotenv import load_dotenv  # For env variables (برای متغیرهای محیطی)
import os  # For OS operations (برای عملیات سیستم‌عامل)

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

# Test (تست)
df = fetch_data()  # Call function (فراخوانی تابع)
if df is not None:  # Check if data exists (چک اگر داده وجود داره)
    logging.info("Data fetch successful.")  # Log success (لاگ موفقیت)
else:
    logging.error("Data fetch failed after retries.")  # Log failure (لاگ شکست)
