import ccxt
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
from logging.handlers import RotatingFileHandler
# Logging setup (لاگ ستاپ)
handler = RotatingFileHandler('live_log.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

load_dotenv()
ACCESS_ID = os.getenv('COINEX_API_KEY')
SECRET_KEY = os.getenv('COINEX_API_SECRET')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Load model and scaler (بارگذاری مدل و اسکیلر)
model = load_model('btc_lstm_model.keras')
scaler = joblib.load('scaler.pkl')
def send_telegram_message(message):
    """
    Send a message to your Telegram bot (ارسال پیام به ربات تلگرام)
    """
    if os.getenv("TELEGRAM_DISABLED", "1").strip().lower() not in {"0", "false", "no", "off"}:
        logging.info("Telegram notifications are disabled by TELEGRAM_DISABLED kill switch.")
        return

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
def calculate_fibonacci_levels(df):
    # Find swing high/low (نوسان بالا/پایین - simple method: rolling max/min)
    df['swing_high'] = df['high'].rolling(window=30).max().shift(1)  # Recent high (بالای اخیر)
    df['swing_low'] = df['low'].rolling(window=30).min().shift(1)  # Recent low (پایین اخیر)
    
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
def add_advanced_features(df):
    """
    Add new features for higher accuracy (اضافه کردن ویژگی‌های جدید برای دقت بالاتر)
    - Sentiment score from X (امتیاز احساسات از X)
    - On-chain hash rate (نرخ هش آن‌چین)
    - Volume momentum (شتاب حجم - OBV)
    """
    # 1. Sentiment Score from X (امتیاز احساسات از X - using tool)
    # Use x_semantic_search for recent BTC sentiment (جستجوی معنایی برای احساسات اخیر BTC)
    # In practice, call tool and average scores (در عمل ابزار رو صدا بزن و میانگین بگیر)
    # For demo, assume score from -1 (negative) to 1 (positive) (برای دمو فرض کن امتیاز از -۱ تا ۱)
    sentiment_scores = np.random.uniform(-1, 1, len(df))  # Placeholder - replace with real tool call (جایگزین با تماس واقعی ابزار)
    df['sentiment_score'] = sentiment_scores
    
    # 2. On-Chain Hash Rate (نرخ هش آن‌چین - from code_execution with coingecko)
    # Example tool call: code_execution with "from coingecko import CoinGeckoAPI; api = CoinGeckoAPI(); print(api.get_coin_by_id('bitcoin')['hashing_algorithm'])"
    # For demo, simulate (برای دمو شبیه‌سازی کن)
    hash_rates = np.random.uniform(100e6, 200e6, len(df))  # TH/s (تراهش در ثانیه)
    df['hash_rate'] = hash_rates
    
    # 3. Volume Momentum - OBV (شتاب حجم - On-Balance Volume)
    df['price_change_sign'] = np.sign(df['close'].diff())
    df['obv'] = (df['volume'] * df['price_change_sign']).cumsum()
    
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
    
    df = df.dropna().reset_index(drop=True)
    df = calculate_fibonacci_levels(df)  # اضافه کردن سطوح فیبوناچی
    df = add_advanced_features(df)  # اضافه کردن ویژگی‌های پیشرفته
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_50', 'ema_20', 'rsi_14', 
                'macd', 'macd_signal', 'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 
                'volatility', 'fib_reaction', 'obv']

    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    df_scaled['timestamp'] = df['timestamp'].values
    
    return df_scaled, scaler, df
def predict_next_price(model, df_processed, scaler, sequence_length=60, num_dropout_samples=10):
    """
    Improved next candle prediction with guards, logging, and optional confidence interval (پیش‌بینی بهبودیافته کندل بعدی با گاردها، لاگینگ، و بازه اطمینان اختیاری)
    - num_dropout_samples: For Monte Carlo dropout to estimate uncertainty (برای تخمین عدم اطمینان با دراپ‌آوت مونت کارلو)
    """
    try:
        # Guard: Check length (گارد: چک طول)
        if len(df_processed) < sequence_length:
            logging.warning(f"Insufficient data for prediction: {len(df_processed)} < {sequence_length}")
            send_telegram_message(f"Prediction skipped: insufficient data ({len(df_processed)} < {sequence_length})")
            return None  # Return None if too short (اگر کوتاه باشه None برگردون)
        
        # Drop timestamp if exists (حذف timestamp اگر وجود داره)
        features = df_processed.columns[df_processed.columns != 'timestamp']
        last_sequence = df_processed[features].values[-sequence_length:]
        
        # Dynamic close index (اندیس بسته‌شدن پویا - assuming 'close' is in features)
        close_index = features.tolist().index('close') if 'close' in features else None
        if close_index is None:
            logging.error("No 'close' column in features - cannot predict")
            send_telegram_message("Prediction Error: 'close' column missing in features")
            return None
        
        last_sequence = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
        
        # Prediction with optional uncertainty (پیش‌بینی با عدم اطمینان اختیاری)
        if num_dropout_samples > 1:
            model.trainable = True  # Enable dropout for inference (فعال کردن دراپ‌آوت برای استنتاج)
            predictions = []
            for _ in range(num_dropout_samples):
                pred_scaled = model.predict(last_sequence, verbose=0)
                predictions.append(pred_scaled[0, 0])
            pred_scaled_mean = np.mean(predictions)
            pred_std = np.std(predictions)  # Uncertainty (عدم اطمینان)
            logging.info(f"Prediction uncertainty: std = {pred_std:.4f}")
        else:
            pred_scaled_mean = model.predict(last_sequence, verbose=0)[0, 0]
            pred_std = 0
        
        # Inverse scaling (معکوس مقیاس‌بندی)
        dummy = np.zeros((1, len(scaler.scale_)))
        dummy[0, close_index] = pred_scaled_mean
        pred_price = scaler.inverse_transform(dummy)[0, close_index]
        
        logging.info(f"Predicted next close: ${pred_price:.2f} (uncertainty std: {pred_std:.2f})")
        send_telegram_message(f"Predicted next close: ${pred_price:.2f} (uncertainty std: {pred_std:.2f})")
        return pred_price, pred_std  # Return price and uncertainty (قیمت و عدم اطمینان رو برگردون)
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        send_telegram_message(f"Prediction Error: {str(e)}")
        return None, 0

def get_balance(exchange, asset='USDT'):
    """
    Fetch free balance (گرفتن موجودی آزاد)
    """
    try:
        balance = exchange.fetch_balance()
        return balance['free'].get(asset, 0)
    except Exception as e:
        logging.error(f"Balance fetch error: {e}")
        send_telegram_message(f"Balance Error: {e}")
        return 0

def spot_trade_on_signal(exchange, symbol='BTC/USDT', signal='HOLD', risk_pct=0.3, current_state=None):
    """
    مدیریت ترید اسپات: 
    - اگر سیگنال BUY باشه → خرید کنه (اگر پوزیشن باز نداشته باشه)
    - اگر سیگنال SELL باشه → اگر پوزیشن باز داره، بفروشه و ببنده
    - اگر HOLD باشه یا شرایط مناسب نباشه → کاری نکنه
    """
    if current_state is None:
        current_state = {'position_open': False, 'buy_price': None, 'amount': 0}

    try:
        # گرفتن موجودی
        usdt_free = get_balance(exchange, 'USDT')
        btc_free = get_balance(exchange, 'BTC')

        # قیمت فعلی
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']

        if signal == 'BUY' and not current_state['position_open']:
            # محاسبه مقدار خرید (پویا بر اساس ریسک)
            risk_amount = usdt_free * risk_pct
            buy_amount = risk_amount / current_price  # در BTC
            

            # حداقل سفارش کوینکس اسپات BTC/USDT ≈ 0.0001 BTC
            min_order = 0.0001
            if buy_amount < min_order:
                msg = f"Buy amount too small: {buy_amount:.6f} BTC < min {min_order} - skipping BUY"
                logging.warning(msg)
                send_telegram_message(msg)
                return current_state

            # گرد کردن به دقت کوینکس
            buy_amount = round(buy_amount, 6)

            # اجرای خرید مارکت
            order = exchange.create_order(symbol, 'market', 'buy', buy_amount, current_price)
            msg = (
               
                f"Amount: {buy_amount:.6f} BTC\n"
                f"Price: ${current_price:.2f}\n"
                f"Risk: ${risk_amount:.2f} ({risk_pct*100:.1f}%)"
            )
            logging.info(msg)
            send_telegram_message(msg)

            # به‌روزرسانی حالت
            current_state['position_open'] = True
            current_state['buy_price'] = current_price
            current_state['amount'] = buy_amount

        elif signal == 'SELL' and current_state['position_open']:
            # فروش همه مقدار باز
            sell_amount = current_state['amount']

            if sell_amount < 0.0001:
                msg = "No significant position to sell - skipping"
                logging.info(msg)
                send_telegram_message(msg)
                return current_state
            # اجرای فروش مارکت
            order = exchange.create_order('BTC/USDT', 'market', 'sell', btc_free, current_price)
            # محاسبه سود/زیان تقریبی
            profit_pct = ((current_price - current_state['buy_price']) / current_state['buy_price']) * 100
            profit_usd = (current_price - current_state['buy_price']) * sell_amount

            msg = (
                f"Amount: {sell_amount:.6f} BTC\n"
                f"Price: ${current_price:.2f}\n"
                f"Profit/Loss: {profit_pct:+.2f}% (${profit_usd:+.2f})"
            )
            logging.info(msg)
            send_telegram_message(msg)

            # ریست حالت
            current_state['position_open'] = False
            current_state['buy_price'] = None
            current_state['amount'] = 0

        else:
            # HOLD یا شرایط نامناسب
            logging.info(f"HOLD or no action - Position: {'Open' if current_state['position_open'] else 'Closed'}")

        return current_state

    except Exception as e:
        error_msg = f"Spot Trade Error: {str(e)}"
        logging.error(error_msg)
        send_telegram_message(error_msg)
        return current_state

def paper_trading_loop(symbol='BTC/USDT', timeframe='5m', sequence_length=60, sleeptime=300, initial_capital=100, risk_per_trade=0.1):
    """
    Paper Trading: Simulate real trades with virtual capital (تریدینگ کاغذی: شبیه‌سازی ترید واقعی با سرمایه مجازی)
    risk_per_trade: 2% of capital per trade (2% سرمایه در هر ترید)
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long (خرید)
    buy_price = 0
    total_trades = 0
    winning_trades = 0
    
    print(f"Paper Trading Bot Started! Virtual Capital: ${initial_capital:.2f}")
    send_telegram_message(f"Paper Trading Started!\nVirtual Capital: ${initial_capital:.2f}")
    
    while True:
        try:
            new_data = fetch_and_update_data(symbol, timeframe, batch_limit=sequence_length + 50)
            if new_data is not None and len(new_data) >= sequence_length:
                logging.info(f"\nbefore preprocess :{len(new_data)} candles for analysis.")
                df_processed, _, df_original = preprocess_data(new_data)
                logging.info(f"after Preprocess :{len(df_original)} rows.")
                # چک مهم بعد از preprocess
                if len(df_original) < sequence_length:
                    logging.warning(f"After preprocessing, only {len(df_original)} rows left – skipping")
                    time.sleep(sleeptime)
                    continue
                
                predicted, pred_std = predict_next_price(model, df_processed, scaler, sequence_length)
                if predicted is None:
                    logging.warning("Prediction failed - skipping iteration")
                    continue

            current = df_original['close'].iloc[-1]
            change_pct = ((predicted - current) / current) * 100

            # Add uncertainty to message (اضافه عدم اطمینان به پیام)
            uncertainty_msg = f"Uncertainty: ±${pred_std:.2f}"

            if change_pct > 0.3 and pred_std < 100:  # Only trade if certain (فقط اگر مطمئن باشه ترید کن)
                if position == 0:
                    risk_amount = capital * risk_per_trade
                    buy_amount = risk_amount / current
                    buy_amount = round(buy_amount, 6)  # Round to 6 decimal places
                    buy_price = current
                    position = buy_amount
                    total_trades += 1
                    msg = (f"current price: ${current:.2f}\n"
                        f"🟢 PAPER BUY - Amount: {buy_amount:.6f} BTC at ${buy_price:.2f}\n"
                            f"Change: {change_pct:+.2f}% | {uncertainty_msg}\n"
                            f"Risked: ${risk_amount:.2f} ({risk_per_trade*100:.1f}%)\n"
                            f"predicted: ${predicted:.2f}")
                    print(msg)
                    send_telegram_message(msg)
            elif change_pct < -0.3 and pred_std < 100:
                if position > 0:
                    sell_price = current
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    profit_usd = (sell_price - buy_price) * position
                    capital += profit_usd
                    if profit_usd > 0:
                        winning_trades += 1
                    msg = (f"current price: ${current:.2f}\n"
                        f"🔴 PAPER SELL - Sold {position:.6f} BTC at ${sell_price:.2f}\n"
                            f"Profit/Loss: {profit_pct:+.2f}% (${profit_usd:+.2f}) | New Capital: ${capital:.2f}\n"
                            f"Change: {change_pct:+.2f}% | {uncertainty_msg}\n"
                            f"predicted: ${predicted:.2f}")
                    print(msg)
                    send_telegram_message(msg)
                    position = 0

            else:
                msg = (f"current price: ${current:.2f}\n"
                f"⚪ HOLD - Change: {change_pct:+.2f}% | {uncertainty_msg}\npredicted: ${predicted:.2f}")
                print(msg)
                send_telegram_message(msg)
            # Daily summary (خلاصه روزانه)
            timestamp = df_original['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')
            if timestamp.endswith('00:00'):
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                msg = f"Daily Paper Trading Summary\nCapital: ${capital:.2f}\nTotal Trades: {total_trades}\nWin Rate: {win_rate:.1f}%"
                print(msg)
                send_telegram_message(msg)
        
        except Exception as e:
            error_msg = f"Paper Trading Error: {e}"
            logging.error(error_msg)
            print(error_msg)
            send_telegram_message(error_msg)
            time.sleep(60)
        
        time.sleep(sleeptime)
def live_trading_loop(symbol='BTC/USDT', timeframe='5m', sequence_length=60, sleeptime=300):
    """
    Live Trading: Execute real trades based on model predictions (تریدینگ زنده: اجرای ترید واقعی بر اساس پیش‌بینی مدل)
    """
    print("Live Trading Bot Started!")
    send_telegram_message("Live Trading Bot Started!")
    # در ابتدای فایل یا قبل از حلقه
    current_state = {'position_open': False, 'buy_price': None, 'amount': 0}

    while True:
        try:
            new_data = fetch_and_update_data(symbol, timeframe, batch_limit=sequence_length + 50)
            if new_data is not None and len(new_data) >= sequence_length:
                logging.info(f"before preprocess :{len(new_data)} candles for analysis.")
                df_processed, _, df_original = preprocess_data(new_data)
                logging.info(f"after Preprocess :{len(df_original)} rows.")
                # چک مهم بعد از preprocess
                if len(df_original) < sequence_length:
                    logging.warning(f"After preprocessing, only {len(df_original)} rows left – skipping")
                    time.sleep(sleeptime)
                    continue
                
                predicted, pred_std = predict_next_price(model, df_processed, scaler, sequence_length)
                if predicted is None:
                    continue
            
                current = df_original['close'].iloc[-1]
                
                change_pct = ((predicted - current) / current) * 100
                timestamp = df_original['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')
                
                exchange = ccxt.coinex({
                    'apiKey': ACCESS_ID,
                    'secret': SECRET_KEY,
                    'options': {
                        'defaultType': 'spot',
                    },
                })
                # داخل حلقه live_trading_loop
                if change_pct > 0.3 and not current_state['position_open']:
                    current_state = spot_trade_on_signal(exchange, symbol, 'BUY', risk_pct=0.3, current_state=current_state)
                    msg = (
                    f"🟢 SPOT BUY Executed!\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${current:.2f}\n"
                    )
                    logging.info(msg)
                    send_telegram_message(msg)
                elif change_pct < -0.3 and current_state['position_open']:
                    current_state = spot_trade_on_signal(exchange, symbol, 'SELL', risk_pct=0.3, current_state=current_state)
                    msg = (
                    f"🔴 SPOT SELL (Close) Executed!\n"
                    f"Symbol: {symbol}\n"
                    f"Price: ${current:.2f}\n"
                    )
                    logging.info(msg)
                    send_telegram_message(msg)
                else:

                    msg = f"⚪ HOLD - Current Price: ${current:.2f}, Predicted: ${predicted:.2f}"
                    print(msg)
                    send_telegram_message(msg)
        except Exception as e:
            error_msg = f"live trading loop Error: {e}"
            logging.error(error_msg)
            send_telegram_message(error_msg)
            time.sleep(60)
        
        time.sleep(sleeptime)
if __name__ == "__main__":
    # paper_trading_loop()
    live_trading_loop()
