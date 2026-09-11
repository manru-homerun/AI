# Tiny GRU + ONNX Runtime Experiment

## Setup

- Baseline GRU artifact: `artifacts/gru_model_experiments/best_gru_contentid.pt`
- Tiny artifact: `artifacts/tiny_gru_onnx_experiment/best_tiny_gru_contentid.pt`
- ONNX artifact: `artifacts/tiny_gru_onnx_experiment/model.onnx`
- Data split seed: `42`
- Tiny candidates:
  - `embedding_dim=16`, `hidden_dim=32`, `num_layers=1`
  - `embedding_dim=16`, `hidden_dim=48`, `num_layers=1`
  - `embedding_dim=16`, `hidden_dim=64`, `num_layers=1`

## Result

Best Tiny config:

```json
{
  "embedding_dim": 16,
  "hidden_dim": 48,
  "dropout": 0.2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 8,
  "weight_decay": 1e-05
}
```

Test metrics:

| model | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|
| popular baseline | 0.0989 | 0.1582 | 0.2062 | 0.2684 | 0.1448 | 0.1737 |
| Tiny GRU best | 0.1525 | 0.2401 | 0.2740 | 0.3418 | 0.2066 | 0.2384 |
| previous GRU hidden=96 | - | - | 0.3390 | 0.4237 | 0.2493 | 0.2903 |

Model sizes:

| artifact | bytes |
|---|---:|
| previous GRU `.pt` | 1,464,615 |
| Tiny GRU `.pt` | 614,181 |
| Tiny GRU `.onnx` | 622,549 |

ONNX validation:

- PyTorch vs ONNX Runtime max absolute difference: `9.5367431640625e-07`
- ONNX Runtime batch checks passed for batch size `1` and `3`.
- FastAPI test client returned `200` for `/health` and `/recommend`.

## Decision

Tiny GRU is deployable and removes the PyTorch runtime requirement from production, but this run does not fully match the previous `hidden_dim=96` GRU quality.

It is good enough if the priority is EC2 footprint and installation simplicity. If recommendation quality is the priority, run one more middle-ground experiment with either `embedding_dim=32, hidden_dim=64` or `embedding_dim=32, hidden_dim=96`, then export the winner to ONNX.

