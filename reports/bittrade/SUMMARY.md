# BitTrade spot research

2026-09-30T15:45:50.711628+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 21.2 | 2 | -0.34% | -0.54% | 0.00% | HOLD |
| btcjpy_1h | 83.7 | 10 | -0.20% | 9.53% | 0.00% | HOLD |
| btcjpy_5m | 7.4 | 1 | -0.51% | 0.47% | 0.00% | HOLD |
| xrpjpy_15m | 21.2 | 2 | -0.66% | -3.60% | 0.00% | HOLD |
| xrpjpy_1h | 83.7 | 15 | -5.43% | 8.83% | 0.00% | HOLD |
| xrpjpy_5m | 7.4 | 0 | 0.00% | 0.07% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.
