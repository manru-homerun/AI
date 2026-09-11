# Conditional GRU Course Decoder

This experiment generates a fixed-length CONTENT_ID sequence from user/trip features.

## Metrics

| model | teacher recall@10 | teacher mrr@10 | route overlap | position accuracy | duplicate rate | avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| h=48 | 0.2534 | 0.1394 | 0.0985 | 0.0329 | 0.0000 | 6.28 |
| h=64 | 0.2553 | 0.1462 | 0.1168 | 0.0445 | 0.0000 | 5.86 |

## Notes

- Teacher forcing metrics are step-level next-token ranking metrics.
- Greedy route metrics use autoregressive generation with duplicate masking.
- The v1 decoder controls length with desired_poi_count and does not use EOS.
