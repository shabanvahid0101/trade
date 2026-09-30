# BitTrade forward paper trading

2026-09-30T21:25:52.896674+00:00

Mode: **experimental**. Virtual JPY only. Live trading: disabled.

Initial: 100,000.00 JPY | Equity: 99,967.84 JPY | Return: -0.03%
Realized: +0.00 JPY | Unrealized (estimated liquidation): -32.16 JPY
Closed trades: 0 | Open positions: 1 | Max observed drawdown: 0.03%
Paused: False | Fresh research: True

Book snapshots approximate full fills, including visible spread, 5 bps extra slippage and 0.1% fees per side.
Stops and exits execute only on polling at the then-current book; missed historical prices are never assumed.
Backtest results and this forward portfolio are separate; repeated retraining is part of this paper policy.

## Open positions

| Strategy | Entry JPY | Quantity | Cost JPY | Opened UTC |
|---|---:|---:|---:|---|
| btcjpy_1h | 13188656.1926 | 0.00075 | 9901.38 | 2026-09-30T21:25:52.896674+00:00 |

## Recent exits

| Strategy | Net PnL JPY | Net return | Reason | Closed UTC |
|---|---:|---:|---|---|

## Last decisions

- btcjpy_15m: probability_below_threshold
- btcjpy_1h: paper_opened
- btcjpy_5m: probability_below_threshold
- xrpjpy_1h: probability_below_threshold
- xrpjpy_5m: probability_below_threshold
