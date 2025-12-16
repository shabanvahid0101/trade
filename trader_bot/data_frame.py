import numpy as np  # برای محاسبات
import data
import pandas as pd  # برای دیتافریم‌ها

df = data.fetch_data()  # گرفتن داده‌ها با استفاده از تابع موجود
def preprocess_data(df):  # Function for preprocessing (تابع پیش‌پردازش)
    # Manual SMA (میانگین متحرک ساده)
    df['sma_50'] = df['close'].rolling(window=50).mean()  # Rolling mean (میانگین غلتان)
    
    # Manual EMA (میانگین متحرک نمایی)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()  # Exponential weighted mean
    
    # Manual RSI (شاخص قدرت نسبی) - قدم به قدم
    delta = df['close'].diff()  # Change in price (تغییر قیمت)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gain (میانگین سود)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average loss (میانگین ضرر)
    rs = gain / loss  # Relative strength (قدرت نسبی)
    df['rsi_14'] = 100 - (100 / (1 + rs))  # RSI formula (فرمول RSI)
    
    # Lagged features (ویژگی‌های تأخیری)
    for lag in [1, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    df = df.dropna()  # Drop missing values (حذف مقادیر گم‌شده)
    
    # Normalization (نرمال‌سازی)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = ['open', 'high', 'low', 'close', 'volume', 'sma_50', 'ema_20', 'rsi_14', 'close_lag_1', 'close_lag_5', 'close_lag_10']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    df_scaled['timestamp'] = df['timestamp']
    
    return df_scaled, scaler