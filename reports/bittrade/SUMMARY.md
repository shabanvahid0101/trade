# BitTrade spot research

2026-10-01T09:09:34.044963+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 22.0 | 1 | -0.64% | -0.71% | 0.00% | HOLD |
| btcjpy_1h | 84.4 | 8 | 0.41% | 8.52% | 0.00% | HOLD |
| btcjpy_5m | 8.1 | 1 | -0.92% | 0.30% | 0.00% | HOLD |
| xrpjpy_1h | 84.4 | 17 | -8.35% | 4.78% | 0.00% | HOLD |
| xrpjpy_5m | 8.1 | 0 | 0.00% | -0.73% | 0.00% | HOLD |

Blocked ethjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
