import os
from dotenv import load_dotenv
import ccxt
import logging

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

def main():
    logger.info("شروع برنامه")
    exchange = connect_to_coinex()
    if exchange:
        # تست گرفتن لیست ارزها
        markets = exchange.load_markets()
        logger.info(f"تعداد ارزها: {len(markets)}")
        # مثال لیست ۱۰ ارز
        logger.info(list(markets.keys())[:10])

if __name__ == "__main__":
    main()