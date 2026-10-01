# BitTrade spot research

2026-10-01T00:50:59.238766+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 21.6 | 1 | -0.87% | -0.66% | 0.00% | HOLD |
| btcjpy_1h | 84.0 | 8 | 0.41% | 8.11% | 0.00% | HOLD |
| btcjpy_5m | 7.7 | 1 | -0.51% | -0.82% | 0.00% | HOLD |
| xrpjpy_1h | 84.0 | 18 | -9.58% | 7.63% | 0.00% | HOLD |
| xrpjpy_5m | 7.7 | 0 | 0.00% | -2.64% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
