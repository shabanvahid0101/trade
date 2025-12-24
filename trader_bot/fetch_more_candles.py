import ccxt
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import os

load_dotenv()
ACCESS_ID = os.getenv('Access_ID')
SECRET_KEY = os.getenv('Secret_Key')

logging.basicConfig(level=logging.INFO)

def fetch_data(symbol='BTC/USDT', timeframe='5m', total_limit=5000, batch_limit=1000, retries=3):
    exchange = ccxt.binance()  # بدون API key برای public data (داده عمومی)
    all_data = []
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000  # Timeframe in ms (زمان تایم‌فریم در میلی‌ثانیه)
    
    # Start since from current - total duration (شروع since از فعلی منهای مدت کل)
    since = exchange.milliseconds() - (total_limit * timeframe_ms)
    
    while len(all_data) < total_limit:
        current_limit = min(batch_limit, total_limit - len(all_data))
        for attempt in range(retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=current_limit)
                if not ohlcv or len(ohlcv) == 0:
                    logging.info("No more historical data available")
                    break
                all_data.extend(ohlcv)
                # Update since to before the first candle of this batch (به‌روزرسانی since به قبل از اولین کندل این بچ)
                since = ohlcv[0][0] - timeframe_ms  # Go back one timeframe for next older batch (به عقب یک تایم‌فریم برای بچ قدیمی‌تر)
                logging.info(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}, since updated to {since}")
                break
            except Exception as e:
                logging.error(f"Retry {attempt+1}/{retries}: {e}")
                time.sleep(5)
        else:
            logging.error("Failed after retries")
            break
        
        time.sleep(exchange.rateLimit / 1000 + 1)  # Extra sleep (خواب اضافی)
    
    if all_data:
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)  # Remove duplicates & sort (حذف تکراری‌ها و مرتب)
        logging.info(f"Final unique historical data: {len(df)} candles")
        return df
    return None

if __name__ == "__main__":
    df = fetch_data(total_limit=5000)  # Test (تست)
    if df is not None:
        print(df.head())  # First (اولین)
        print(df.tail())  # Last (آخرین)
        df.to_csv('btc_5000_candles.csv', index=False)
        print("Saved to btc_5000_candles.csv - No duplicates!")  # Saved without duplicates (ذخیره بدون تکراری)