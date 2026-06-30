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

## Futures Fundamentals

Fetch Binance Futures market data such as funding rate, open interest, global long/short ratio, and taker buy/sell ratio:

```powershell
python fundamental_data.py --symbol BTC/USDT --timeframe 1h --market-data dataset/1h-btc_history.csv --output dataset/1h-btc_fundamentals.csv
```

Train with the technical + futures fundamentals feature set:

```powershell
python crypto_predictor.py train-multi --update --update-fundamentals --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --fundamental-data dataset/1h-btc_fundamentals.csv --horizons 1,3,6 --sequence-length 96 --epochs 40 --max-train-rows 5000 --feature-set advanced-fundamental --feature-selection correlation --max-selected-features 24 --min-selected-features 10 --target-mode classification --threshold 0.0015 --min-confidence 0.50
```

Prediction, alerting, paper trading, and strategy optimization also accept `--fundamental-data` and `--update-fundamentals`. Data is joined with a backward as-of merge so each candle only sees fundamental values already published at or before that candle.

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

## Telegram Alert Bot

Run one alert check with the saved 1h multi-horizon models:

```powershell
python alert_bot.py --update --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --horizons 1,3,6 --min-agree 2 --min-confidence 0.50
```

Run continuously on a Python server:

```powershell
python alert_bot.py --mode loop --update --telegram --send-hold --symbol BTC/USDT --timeframe 1h --data dataset/1h-btc_history.csv --horizons 1,3,6 --sleep-seconds 300
```

The included GitHub Actions workflow `.github/workflows/telegram-alert.yml` runs every 15 minutes and sends HOLD updates too. Add these repository secrets before enabling it:

```text
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_channel_or_chat_id
```

Cloudflare Workers cannot run this Python/TensorFlow model directly. Use GitHub Actions for a free scheduled runner, or a small Python VPS/container for true always-on looping.

## Paper Trading

Backtest the current 1h multi-horizon setup with a virtual 100 USD futures account:

```powershell
python paper_trader.py --mode backtest --data dataset/1h-btc_history.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100 --days 7
```

Run one live paper-trading step and save the virtual account state in `paper_state.json`:

```powershell
python paper_trader.py --mode single --update --telegram --data dataset/1h-btc_history.csv --timeframe 1h --horizons 1,3,6 --initial-capital 100
```

The included GitHub Actions workflow `.github/workflows/paper-trading.yml` runs every hour, updates `paper_state.json`, sends a Telegram status message, and commits the virtual account state back to the repository.

## Strategy Optimization

Search for better 1h paper-trading settings across confidence, horizon agreement, stop loss, take profit, and market regime filters:

```powershell
python strategy_optimizer.py --data dataset/1h-btc_history.csv --days 14 --initial-capital 100 --top 10
```

Regime filter grids can skip low-quality candles. `--atr-min-grid` requires minimum ATR percent, `--trend-min-grid` requires minimum 20/50 SMA trend distance, `--volatility-min-grid` requires minimum rolling volatility, and `--trend-filter-grid off,follow` tests whether long trades should follow positive trend and short trades should follow negative trend.

The optimizer writes `strategy_optimization.json` locally and prints the top configurations. Review results out of sample before changing live paper-trading settings.

Walk-forward validation chooses the best config on a rolling training window and tests it on the next unseen window:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --train-days 14 --test-days 3 --step-days 3 --walkforward-days 45 --initial-capital 100
```

Example walk-forward run with explicit regime filter grids:

```powershell
python strategy_optimizer.py --mode walk-forward --data dataset/1h-btc_history.csv --train-days 14 --test-days 3 --step-days 3 --walkforward-days 45 --initial-capital 100 --horizon-sets "1,3,6;1,6;3,6;6" --confidence-grid "0.45,0.50" --min-agree-grid "1,2" --stop-loss-grid "0" --take-profit-grid "0,0.02" --atr-min-grid "0,0.003,0.006" --trend-min-grid "0,0.003,0.006" --trend-filter-grid "off,follow"
```

## Trading Notes

No model can predict crypto prices perfectly. Before using real money, require a stable out-of-sample edge after fees, slippage, and drawdown limits. Use small position sizing, stop-loss rules, and paper trading first.
