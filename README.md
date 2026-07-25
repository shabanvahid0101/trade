# Cryptocurrency Trading Experiments

A collection of Python notebooks and scripts for market-data analysis,
price-prediction experiments, and simulated cryptocurrency trading. The code
uses CoinEx-compatible market data through CCXT and explores linear regression,
random forests, and an LSTM/random-forest hybrid.

## Highlights

- Fetches OHLCV candle data with `ccxt`
- Builds features such as RSI and ATR
- Trains and evaluates regression models on time-series data
- Simulates trading with fees and starting capital
- Produces model-accuracy charts and log files
- Sends optional status messages through Telegram

The most complete implementation is:

```text
2025-8/end/trading_bot.py
```

## Setup

```bash
cd 2025-8/end
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the expected configuration keys into a local `.env` file. Keep exchange
keys, Telegram tokens, and chat IDs out of version control.

## Run

```bash
python trading_bot.py
```

Review the symbols, timeframe, fee assumptions, and simulation settings in the
script before running it.

## Disclaimer

This repository is for research and educational experimentation. Model
predictions and simulated results do not guarantee future performance and are
not financial advice. Start with paper trading and verify every assumption
before connecting real funds.

