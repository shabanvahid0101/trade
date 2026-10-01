# BitTrade spot research

2026-10-01T07:11:55.930052+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 21.9 | 1 | -0.64% | -0.03% | 0.00% | HOLD |
| btcjpy_1h | 84.3 | 9 | 1.22% | 8.85% | 0.00% | HOLD |
| btcjpy_5m | 8.0 | 1 | -0.92% | 1.38% | 0.00% | HOLD |
| xrpjpy_1h | 84.3 | 17 | -8.17% | 3.81% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked xrpjpy_5m: Cannot assume fills on zero-volume candles. Signal: HOLD.
