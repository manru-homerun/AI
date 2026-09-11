# Conditional GRU Course Decoder

This experiment generates a fixed-length CONTENT_ID sequence from user/trip features.

## Metrics

| model | teacher recall@10 | teacher mrr@10 | route overlap | position accuracy | duplicate rate | avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| h=48 | 0.0309 | 0.0128 | 0.0103 | 0.0062 | 0.0000 | 5.59 |

## Notes

- Teacher forcing metrics are step-level next-token ranking metrics.
- Greedy route metrics use autoregressive generation with duplicate masking.
- The v1 decoder controls length with desired_poi_count and does not use EOS.
