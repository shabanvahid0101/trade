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
ACCESS_ID = os.getenv('Access_ID')
SECRET_KEY = os.getenv('Secret_Key')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Load model and scaler (بارگذاری مدل و اسکیلر)
model = load_model('btc_lstm_model.keras')
scaler = joblib.load('scaler.pkl')
def send_telegram_message(message):
    """
    Send a message to your Telegram bot (ارسال پیام به ربات تلگرام)
    """
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
def add_ichimoku_features(df, tenkan_period=9, kijun_period=26, senkou_period=52):
    """
    Manual Ichimoku Cloud calculation + flat level detection + reaction assessment
    (محاسبه دستی ابر ایچیموکو + تشخیص سطح صاف + سنجش واکنش قیمت)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 1. Conversion Line (Tenkan-sen) - خط تبدیل
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
    
    # 2. Base Line (Kijun-sen) - خط پایه
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # 3. Leading Span A (Senkou Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)
    
    # 4. Leading Span B (Senkou Span B) - سنکو اسپن B
    span_b_high = high.rolling(window=senkou_period).max()
    span_b_low = low.rolling(window=senkou_period).min()
    df['senkou_span_b'] = ((span_b_high + span_b_low) / 2).shift(kijun_period)
    
    # 5. Lagging Span (Chikou Span) - اختیاری، برای تأیید
    df['chikou_span'] = close.shift(-kijun_period)
    
    # تشخیص سطح صاف (Flat level detection between Tenkan and Senkou Span B)
    window = 5  # پنجره برای چک صاف بودن
    df['tenkan_variance'] = df['tenkan_sen'].rolling(window=window).var()
    df['span_b_variance'] = df['senkou_span_b'].rolling(window=window).var()
    df['tenkan_span_b_diff'] = abs(df['tenkan_sen'] - df['senkou_span_b'])
    
    variance_threshold = df['close'].mean() * 0.0005  # آستانه پویا بر اساس قیمت (0.05%)
    diff_threshold = df['close'].mean() * 0.005       # تفاوت کمتر از 0.5%
    
    df['is_flat_ichimoku_level'] = (
        (df['tenkan_variance'] < variance_threshold) &
        (df['span_b_variance'] < variance_threshold) &
        (df['tenkan_span_b_diff'] < diff_threshold)
    )
    
    # سنجش واکنش قیمت به سطح صاف (Reaction assessment)
    reaction_window = 10
    df['ichimoku_reaction'] = 0.0
    # بعد از محاسبه is_flat_level
    df['ichimoku_reaction'] = 0.0

    # Find indices where flat level exists in last reaction_window
    flat_mask = df['is_flat_ichimoku_level'].rolling(window=reaction_window).sum() > 0
    indices = flat_mask[flat_mask].index

    for i in indices:
        level_slice = df['senkou_span_b'].iloc[i - reaction_window:i]
        level = level_slice.mean()
        price_change = (df['close'].iloc[i] - df['close'].iloc[i - reaction_window]) / df['close'].iloc[i - reaction_window]
        volume_factor = df['volume'].iloc[i] / df['volume'].iloc[i - reaction_window:i].mean()
        df.loc[i, 'ichimoku_reaction'] = price_change * volume_factor
    return df
def calculate_fibonacci_levels(df):
    # Find swing high/low (نوسان بالا/پایین - simple method: rolling max/min)
    df['swing_high'] = df['high'].rolling(window=20).max().shift(1)  # Recent high (بالای اخیر)
    df['swing_low'] = df['low'].rolling(window=20).min().shift(1)  # Recent low (پایین اخیر)
    
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
    df = add_ichimoku_features(df)  # اضافه کردن ایچیموکو
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_50', 'ema_20', 'rsi_14', 
                'macd', 'macd_signal', 'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10', 
                'volatility', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 
                'ichimoku_reaction', 'is_flat_ichimoku_level', 'fib_reaction']
    
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    df_scaled['timestamp'] = df['timestamp'].values
    
    return df_scaled, scaler, df
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
    
    print("Paper Trading Bot Started! Virtual Capital: $10,000")
    send_telegram_message(f"Paper Trading Started!\nVirtual Capital: ${initial_capital:.2f}")
    
    while True:
        try:
            new_data = fetch_and_update_data(symbol, timeframe, batch_limit=sequence_length + 50)
            if new_data is not None and len(new_data) >= sequence_length:
                df_processed, _, df_original = preprocess_data(new_data)
                predicted = predict_next_price(model, df_processed, scaler, sequence_length)
                current = df_original['close'].iloc[-1]
                
                change_pct = ((predicted - current) / current) * 100
                timestamp = df_original['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')
                
                if change_pct > 0.3 and position == 0:
                    # BUY
                    position_size = capital * risk_per_trade / (current * 0.01)  # Approx position (تقریبی)
                    buy_price = current
                    position = 1
                    total_trades += 1
                    msg = f"🟢 PAPER BUY 🟢\nTime: {timestamp}\nPrice: ${current:.2f}\nCapital: ${capital:.2f}"
                    print(msg)
                    send_telegram_message(msg)
                
                elif change_pct < -0.3 and position == 1:
                    # SELL
                    profit_pct = (current - buy_price) / buy_price * 100
                    profit_usd = capital * risk_per_trade * (profit_pct / 100)
                    capital += profit_usd
                    if profit_pct > 0:
                        winning_trades += 1
                    
                    position = 0
                    msg = f"🔴 PAPER SELL 🔴\nTime: {timestamp}\nPrice: ${current:.2f}\nProfit: {profit_pct:+.2f}% (${profit_usd:+.2f})\nNew Capital: ${capital:.2f}"
                    print(msg)
                    send_telegram_message(msg)
                
                else:
                    print(f"HOLD - Change: {change_pct:+.2f}% | Capital: ${capital:.2f}")
                    send_telegram_message(f"current price is ${current:.2f}\npredict price is ${predicted:.2f}\nHOLD\nTime: {timestamp}\nPredicted Change: {change_pct:+.2f}%\nCapital: ${capital:.2f}")
                
                # Daily summary (خلاصه روزانه)
                if timestamp.endswith('00:00'):
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    msg = f"Daily Paper Trading Summary\nCapital: ${capital:.2f}\nTotal Trades: {total_trades}\nWin Rate: {win_rate:.1f}%"
                    send_telegram_message(msg)
        
        except Exception as e:
            error_msg = f"Paper Trading Error: {e}"
            logging.error(error_msg)
            send_telegram_message(error_msg)
            time.sleep(60)
        
        time.sleep(sleeptime)
if __name__ == "__main__":
    paper_trading_loop()
