import requests
import time
import logging
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, urlencode

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Telegram error: {response.text}")
            print(f"Telegram error: {response.text}")
        else:
            print(f"Message sent: {message}")
    except Exception as e:
        logging.error(f"Error sending to Telegram: {e}")
        print(f"Error: {e}")
send_telegram_message("Bot started successfully!")