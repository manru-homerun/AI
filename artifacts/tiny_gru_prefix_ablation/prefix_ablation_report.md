# Tiny GRU Prefix Length Ablation

This evaluation freezes the trained Tiny GRU checkpoint and changes only the test-time prefix length.

## Metadata

- checkpoint: `artifacts\tiny_gru_onnx_experiment\best_tiny_gru_contentid.pt`
- input_path: `data\processed\total_input.csv`
- sequence_path: `data\processed\total_travel_seq_with_contentid.csv`
- batch_size: `64`

## Metrics

| condition | samples | avg prefix len | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 354 | 2.08 | 0.1525 | 0.2401 | 0.2740 | 0.3418 | 0.2066 | 0.2384 |
| last-3 | 354 | 1.80 | 0.1525 | 0.2373 | 0.2740 | 0.3418 | 0.2053 | 0.2374 |
| last-2 | 354 | 1.53 | 0.1497 | 0.2401 | 0.2797 | 0.3277 | 0.2032 | 0.2328 |
| last-1 | 354 | 1.00 | 0.1356 | 0.2316 | 0.2712 | 0.3249 | 0.1937 | 0.2249 |
| no-seq | 354 | 1.00 | 0.1441 | 0.2260 | 0.2712 | 0.3164 | 0.1974 | 0.2258 |

## Interpretation

- `full` is less than 5 percentage points above `last-1` on Recall@10, so the long-prefix effect may be small.
- `full` is within about 1-2 percentage points of `last-2` or `last-3`, suggesting the model mostly uses recent POIs.
- `last-1` and `no-seq` are close, so user/trip features may dominate over sequence history.

## Notes

- `no-seq` uses one `<UNK>` token instead of an empty sequence to keep GRU inference stable.
- `top_k` is a ranking metric cutoff, not a model input feature.
