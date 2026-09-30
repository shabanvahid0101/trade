# BitTrade spot research

2026-09-30T21:25:50.691292+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 21.5 | 1 | -0.64% | -0.81% | 0.00% | HOLD |
| btcjpy_1h | 83.9 | 9 | -0.38% | 9.11% | 0.00% | HOLD |
| btcjpy_5m | 7.6 | 1 | -0.51% | -0.90% | 0.00% | HOLD |
| xrpjpy_1h | 83.9 | 16 | -14.05% | 8.16% | 0.00% | HOLD |
| xrpjpy_5m | 7.6 | 0 | 0.00% | -2.01% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
