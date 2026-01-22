import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import logging
# فقط یک بار دانلود می‌کنیم (اگر قبلاً دانلود نشده)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
logging.basicConfig(level=logging.INFO,filename='futures.log',filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
def get_vader_sentiment(text):
    """محاسبه امتیاز VADER برای یک متن"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound']  # از -1 (منفی) تا +1 (مثبت)

def add_sentiment_scores(df, posts_df=None):
    """
    اضافه کردن ویژگی sentiment_score به دیتافریم OHLCV
    
    پارامترها:
    df: دیتافریم اصلی با ستون timestamp (pd.DatetimeIndex)
    posts_df: (اختیاری) دیتافریم پست‌های X با ستون‌های:
        - created_at (زمان پست)
        - text (متن پست)
        - اگر نداریم، مقدار پیش‌فرض می‌ذاریم
    
    خروجی: دیتافریم با ستون جدید sentiment_score
    """
    if posts_df is None or 'created_at' not in posts_df.columns or 'text' not in posts_df.columns:
        logging.warning("No X posts data provided → using placeholder sentiment (0.0)")
        df['sentiment_score'] = 0.0
        return df
    
    # تبدیل زمان‌ها به timezone-aware اگر لازم باشه
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    posts_df = posts_df.copy()
    posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
    
    # برای هر ردیف در df، میانگین sentiment پست‌های ۲۴ ساعت قبلش رو محاسبه می‌کنیم
    df['sentiment_score'] = 0.0
    
    for idx, row in df.iterrows():
        time_end = row['timestamp']
        time_start = time_end - pd.Timedelta(hours=24)
        
        recent_posts = posts_df[
            (posts_df['created_at'] >= time_start) & 
            (posts_df['created_at'] <= time_end)
        ]
        
        if len(recent_posts) > 0:
            scores = recent_posts['text'].apply(get_vader_sentiment)
            df.at[idx, 'sentiment_score'] = scores.mean()
        else:
            df.at[idx, 'sentiment_score'] = 0.0  # یا NaN یا مقدار قبلی
    
    # پر کردن مقادیر اولیه که پست ندارند (forward fill یا میانگین)
    df['sentiment_score'] = df['sentiment_score'].replace(0, np.nan).ffill().fillna(0)
    
    logging.info(f"Sentiment score added. Mean: {df['sentiment_score'].mean():.4f}")
    return df