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

Run the Conditional GRU course decoder experiment:

```bash
uv run python -m src.training.conditional_gru_decoder_experiment --artifact-dir artifacts\conditional_gru_decoder_experiment
```

Export the Conditional GRU course decoder to ONNX:

```bash
uv run python -m src.inference.export_conditional_gru_decoder_onnx --artifact-dir artifacts\conditional_gru_decoder_experiment
```

Summarize the Conditional GRU course decoder artifacts:

```bash
uv run python -m src.evaluation.conditional_gru_decoder_eval --artifact-dir artifacts\conditional_gru_decoder_experiment
```

Start Jupyter:

```bash
uv run jupyter notebook
```

Serve the ONNX model with FastAPI:

```bash
uv run uvicorn src.api.tiny_gru_app:app --host 0.0.0.0 --port 8000
```

When `artifacts\conditional_gru_decoder_experiment` contains the exported ONNX files, the same FastAPI app also serves `/generate-course`.

## Dependency Layout

- Project dependencies are production-serving dependencies and do not include PyTorch.
- The `dev` dependency group contains training, notebook, CatBoost, PyTorch, and ONNX export dependencies.
- Existing `requirements.txt` and `requirements-production.txt` are kept as legacy fallback files for now.
