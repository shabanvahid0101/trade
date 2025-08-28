import os
from dotenv import load_dotenv
import telegram
import time

# بارگذاری متغیرهای محیطی
load_dotenv()

# اطلاعات تلگرام
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

def get_chat_id():
    """دریافت Chat ID با ارسال پیام تست"""
    if not TELEGRAM_TOKEN:
        print("خطا: توکن تلگرام تنظیم نشده است. لطفاً فایل .env را چک کنید.")
        return

    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        # ارسال یه پیام به خودت برای گرفتن Chat ID
        chat_id = None
        updates = bot.get_updates(timeout=10)  # صبر 10 ثانیه برای دریافت به‌روزرسانی
        if updates:
            for update in updates:
                if update.message:
                    chat_id = update.message.chat_id
                    print(f"Chat ID پیدا شد: {chat_id}")
                    break
        if not chat_id:
            print("هیچ چت ID‌ای پیدا نشد. لطفاً اول به ربات پیام بده.")
    except telegram.error.TelegramError as e:
        print(f"خطا: {e}")
    except Exception as e:
        print(f"خطای غیرمنتظره: {e}")

if __name__ == "__main__":
    print("لطفاً قبل از اجرا، یه پیام به رباتت بفرست.")
    time.sleep(2)  # زمان برای ارسال پیام
    get_chat_id()