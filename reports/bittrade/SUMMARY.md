# BitTrade spot research

2026-09-30T17:00:35.557062+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 21.3 | 1 | -0.64% | -0.15% | 0.00% | HOLD |
| btcjpy_1h | 83.8 | 8 | 0.41% | 9.98% | 0.00% | HOLD |
| btcjpy_5m | 7.4 | 1 | -0.51% | 0.03% | 0.00% | HOLD |
| xrpjpy_1h | 83.8 | 15 | -5.26% | 10.12% | 0.00% | HOLD |
| xrpjpy_5m | 7.4 | 0 | 0.00% | -0.07% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
