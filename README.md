# Crypto Price Predictor

This project trains a time-series model for short-horizon crypto price forecasting.
It is designed for research and decision support, not guaranteed profit.

## What changed

- Predicts future return/price, not the current candle or an EMA by mistake.
- Uses chronological train/validation/test splits.
- Fits scalers only on the training window to avoid data leakage.
- Saves a full model artifact with feature scalers, target scaler, metrics, and configuration.
- Evaluates against a naive baseline and includes a fee-aware backtest.
- Live prediction reports signal, expected return, and confidence.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional `.env` values:

```text
Access_ID=coinex_api_key
Secret_Key=coinex_secret
TELEGRAM_TOKEN=telegram_bot_token
TELEGRAM_CHAT_ID=telegram_chat_id
```

## Train

```powershell
python crypto_predictor.py train --data dataset/5m_btc_history.csv --sequence-length 96 --horizon 1 --epochs 40 --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

To fetch new candles before training:

```powershell
python crypto_predictor.py train --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50
```

Futures-style backtests can take both long and short signals:

```powershell
python crypto_predictor.py train --update --exchange binance --market-mode futures --leverage 1 --threshold 0.0015
```

Use `--feature-selection correlation` to rank indicators on the training window only and keep the strongest non-redundant features. The chosen features are saved in `model_artifacts.pkl` and printed in the training metrics.

## Multi-Horizon Training

Train separate models for several future windows and only trade when they agree:

```powershell
python crypto_predictor.py train-multi --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --horizons 1,3,6,12 --max-train-rows 5000 --feature-set advanced --feature-selection correlation --max-selected-features 18 --target-mode classification --threshold 0.0002 --min-confidence 0.50 --market-mode futures --leverage 1 --training-verbose 2
```

Multi-horizon prediction:

```powershell
python crypto_predictor.py predict-multi --data dataset/5m_btc_history.csv --horizons 1,3,6,12 --min-agree 2 --min-confidence 0.50
```

## Predict Once

```powershell
python crypto_predictor.py predict --data dataset/5m_btc_history.csv
```

With fresh exchange data and Telegram:

```powershell
python crypto_predictor.py predict --update --symbol BTC/USDT --timeframe 5m --data dataset/5m_btc_history.csv --telegram
```

## Live Loop

```powershell
python live_predict.py --symbol BTC/USDT --timeframe 5m --sleep-seconds 300 --telegram
```

## Trading Notes

No model can predict crypto prices perfectly. Before using real money, require a stable out-of-sample edge after fees, slippage, and drawdown limits. Use small position sizing, stop-loss rules, and paper trading first.
