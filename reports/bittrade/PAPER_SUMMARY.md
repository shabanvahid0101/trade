# BitTrade forward paper trading

2026-10-01T09:09:36.232972+00:00

Mode: **experimental**. Virtual JPY only. Live trading: disabled.

Initial: 100,000.00 JPY | Equity: 99,968.18 JPY | Return: -0.03%
Realized: -31.82 JPY | Unrealized (estimated liquidation): +0.00 JPY
Closed trades: 1 | Open positions: 0 | Max observed drawdown: 0.03%
Paused: False | Fresh research: True

Book snapshots approximate full fills, including visible spread, 5 bps extra slippage and 0.1% fees per side.
Stops and exits execute only on polling at the then-current book; missed historical prices are never assumed.
Backtest results and this forward portfolio are separate; repeated retraining is part of this paper policy.

## Open positions

| Strategy | Entry JPY | Quantity | Cost JPY | Opened UTC |
|---|---:|---:|---:|---|

## Recent exits

| Strategy | Net PnL JPY | Net return | Reason | Closed UTC |
|---|---:|---:|---|---|
| btcjpy_1h | -31.82 | -0.32% | model_exit | 2026-10-01T00:51:01.241784+00:00 |

## Last decisions

- btcjpy_15m: probability_below_threshold
- btcjpy_1h: probability_below_threshold
- btcjpy_5m: probability_below_threshold
- xrpjpy_1h: probability_below_threshold
- xrpjpy_5m: probability_below_threshold
