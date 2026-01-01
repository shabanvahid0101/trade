import joblib  # برای ذخیره scaler
from tensorflow.keras.models import load_model
from crypto_predictor import fetch_and_update_data, preprocess_data, predict_next_price,send_telegram_message
import time

# Load model and scaler (بارگذاری مدل و اسکیلر)
model = load_model('btc_lstm_model.keras')
scaler = joblib.load('scaler.pkl')  # اول scaler رو ذخیره کن (در main اضافه کن: joblib.dump(scaler, 'scaler.pkl'))

def live_predict_only():
    while True:
        new_data = fetch_and_update_data(batch_limit=200)  # فقط جدیدها
        if len(new_data) > 60:
            df_processed, _, _ = preprocess_data(new_data)
            predicted = predict_next_price(model, df_processed, scaler)
            current = new_data['close'].iloc[-1]
            change_pct = ((predicted - current) / current) * 100
            
            if change_pct > 0.3:
                send_telegram_message(f"BUY! +{change_pct:.2f}%")
            elif change_pct < -0.3:
                send_telegram_message(f"SELL! {change_pct:.2f}%")
        
        time.sleep(300)  # هر 5 دقیقه

live_predict_only()