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
uv run --no-dev uvicorn src.api.tiny_gru_app:app --host 0.0.0.0 --port 8000
```

When `artifacts\conditional_gru_decoder_experiment` contains the exported ONNX files, the same FastAPI app also serves `/generate-course`.

## EC2 Docker Deployment

The production deployment uses Nginx, Docker Compose, FastAPI, and ONNX Runtime. PyTorch is not installed in the production image.

Target EC2 layout:

```text
/opt/travel-ai/
  releases/
  shared/
  current -> releases/<commit-sha>
```

Manual first deployment on EC2:

```bash
sudo mkdir -p /opt/travel-ai/releases /opt/travel-ai/shared
sudo chown -R "$USER":"$USER" /opt/travel-ai
cd /opt/travel-ai
```

Copy or unpack a release into `/opt/travel-ai/releases/<commit-sha>`, then:

```bash
ln -sfn /opt/travel-ai/releases/<commit-sha> /opt/travel-ai/current
cd /opt/travel-ai/current
docker compose build ai-api
docker compose up -d
docker compose ps
```

Only Nginx publishes a host port:

```text
443 -> nginx
ai-api:8000 -> Docker internal network only
```

TLS certificates are read from the EC2 host and must never be committed:

```text
/etc/letsencrypt/live/13.125.237.207/fullchain.pem
/etc/letsencrypt/live/13.125.237.207/privkey.pem
```

After certificate renewal, reload Nginx:

```bash
docker exec travel-ai-nginx nginx -s reload
```

GitHub Actions deployment does not use a Deploy Key. The workflow checks out the repo inside GitHub Actions, creates a source bundle, uploads it to EC2 over SSH/SCP, switches `/opt/travel-ai/current`, runs Docker Compose, and checks `https://<EC2_HOST>/health`.

Required GitHub Secrets:

```text
EC2_HOST
EC2_USER
EC2_SSH_KEY
EC2_APP_DIR
```

`EC2_APP_DIR` is optional and defaults to `/opt/travel-ai`.

Security Group baseline:

```text
443: Oracle Backend Public IP/32
22: admin IP/32
8000: no inbound rule
80: only if the active certificate renewal flow needs HTTP-01
```

Deployment verification:

```bash
docker compose ps
docker compose exec nginx nginx -t
curl --fail https://13.125.237.207/health
```

## Dependency Layout

- Project dependencies are production-serving dependencies and do not include PyTorch.
- The `dev` dependency group contains training, notebook, CatBoost, PyTorch, and ONNX export dependencies.
- Existing `requirements.txt` and `requirements-production.txt` are kept as legacy fallback files for now.
