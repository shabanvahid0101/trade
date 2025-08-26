import os
from dotenv import load_dotenv
import ccxt
import logging
import pandas as pd
import pandas_ta as ta

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
def analyze_growth_potential(df, symbol):
    if df is None or len(df) < 26:
        logger.warning(f"داده کافی برای {symbol} نیست (کمتر از 26 دوره).")
        return False, 0.0

    try:
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd_result = ta.macd(df['close'])
        if macd_result is None or 'MACD_12_26_9' not in macd_result:
            logger.warning(f"MACD برای {symbol} محاسبه نشد یا داده کافی نیست.")
            return False, 0.0
        df['macd'] = macd_result['MACD_12_26_9']
        df['signal'] = macd_result['MACDs_12_26_9']
    except Exception as e:
        logger.error(f"خطا در محاسبه اندیکاتورها برای {symbol}: {e}")
        return False, 0.0

    last_rsi = df['rsi'].iloc[-1]
    last_macd = df['macd'].iloc[-1]
    last_signal = df['signal'].iloc[-1]
    last_close = df['close'].iloc[-1]
    first_close = df['close'].iloc[0]  # قیمت 7 روز پیش

    weekly_growth = ((last_close - first_close) / first_close) * 100
    potential_growth = weekly_growth + (last_rsi / 100) * 20

    # شرایط جدید: RSI < 50 و رشد پتانسیل > 10
    is_potential = last_rsi < 50 and last_macd > last_signal and potential_growth > 10
    return is_potential, potential_growth

def main():
    logger.info("شروع برنامه")
    exchange = connect_to_coinex()
    if not exchange:
        return

    # گرفتن لیست ارزها
    markets = exchange.load_markets()
    logger.info(f"تعداد ارزها: {len(markets)}")

    # بررسی 10 ارز اول برای تست (می‌تونی همه رو ببینی)
    potential_coins = []
    for symbol in list(markets.keys()):  # برای تست، 10 تای اول
        if '/USDT' not in symbol and '/USDC' not in symbol:
            continue  # فقط جفت‌های USDT/USDC
        logger.info(f"تحلیل {symbol}...")
        df = fetch_ohlcv(exchange, symbol)
        if df is not None:
            is_potential, growth = analyze_growth_potential(df, symbol)
            if is_potential:
                potential_coins.append((symbol, growth))

    # نمایش نتایج
    if potential_coins:
        logger.info("ارزهای با پتانسیل رشد 20%:")
        for coin, growth in potential_coins:
            logger.info(f"{coin}: پتانسیل رشد {growth:.2f}%")
    else:
        logger.info("هیچ ارزی با پتانسیل رشد 20% یافت نشد.")

if __name__ == "__main__":
    main()