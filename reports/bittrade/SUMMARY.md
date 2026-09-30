# BitTrade spot research

2026-09-30T05:45:05.053210+00:00

Exploratory daily expanding-window research, not independent repeated tests. Costs assumed, not reconstructed from historical order books.

Candidate return is diagnostic. The gated policy stays in cash if validation fails.

| Market / timeframe | Days | Test trades | Candidate net return | Buy & hold | Gated policy | Signal |
|---|---:|---:|---:|---:|---:|---|
| btcjpy_15m | 20.8 | 1 | -0.64% | -1.15% | 0.00% | HOLD |
| btcjpy_1h | 83.2 | 9 | 1.30% | 10.65% | 0.00% | HOLD |
| btcjpy_5m | 6.9 | 0 | 0.00% | -0.67% | 0.00% | HOLD |
| ethjpy_15m | 20.8 | 0 | 0.00% | -1.72% | 0.00% | HOLD |
| xrpjpy_1h | 83.2 | 14 | -4.10% | 13.48% | 0.00% | HOLD |
| xrpjpy_5m | 6.9 | 0 | 0.00% | -0.26% | 0.00% | HOLD |

Blocked ethjpy_1h: Cannot assume fills on zero-volume candles. Signal: HOLD.

Blocked ethjpy_5m: Missing features; insufficient trading activity. Signal: HOLD.

Blocked xrpjpy_15m: Cannot assume fills on zero-volume candles. Signal: HOLD.
