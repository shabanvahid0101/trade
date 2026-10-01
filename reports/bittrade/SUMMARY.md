# BitTrade spot research

2026-10-01T14:27:56.021952+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 22.2 | 1 | -0.64% | -0.80% | 0.00% | HOLD |
| btcjpy_1h | 84.6 | 10 | 0.06% | 9.57% | 0.00% | HOLD |
| btcjpy_5m | 8.3 | 1 | -0.92% | 0.36% | 0.00% | HOLD |
| xrpjpy_1h | 84.6 | 14 | -3.80% | 6.07% | 0.00% | HOLD |
| xrpjpy_5m | 8.3 | 0 | 0.00% | -0.96% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
