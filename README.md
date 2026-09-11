# Korean Travel Recommendation

This project uses `uv` for Python environment and dependency management.

## Setup

Install production dependencies only:

```bash
uv sync --no-dev --link-mode=copy
```

Install full local development dependencies, including notebooks, training, PyTorch, and ONNX export:

```bash
uv sync --group dev --link-mode=copy
```

The project targets Python `>=3.9,<3.13`. The local `.python-version` is set to Python `3.9`.

Close running Jupyter kernels before syncing, because Windows can lock packages such as `pyzmq` while a kernel is active.

## Common Commands

Run the Tiny GRU experiment:

```bash
uv run python -m src.training.tiny_gru_experiment --artifact-dir artifacts\tiny_gru_onnx_experiment
```

Export the trained Tiny GRU checkpoint to ONNX:

```bash
uv run python -m src.inference.export_tiny_gru_onnx --artifact-dir artifacts\tiny_gru_onnx_experiment
```

Start Jupyter:

```bash
uv run jupyter notebook
```

Serve the ONNX model with FastAPI:

```bash
uv run uvicorn src.api.tiny_gru_app:app --host 0.0.0.0 --port 8000
```

## Dependency Layout

- Project dependencies are production-serving dependencies and do not include PyTorch.
- The `dev` dependency group contains training, notebook, CatBoost, PyTorch, and ONNX export dependencies.
- Existing `requirements.txt` and `requirements-production.txt` are kept as legacy fallback files for now.
