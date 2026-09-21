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

Run the shared Next-POI GRU experiment in the notebook, then export the trained checkpoint to ONNX:

```bash
uv run python -m src.inference.export_shared_next_poi_gru_onnx --artifact-dir artifacts\shared_next_poi_gru_experiment\shared-next-poi-gru-v1
```

Legacy Tiny GRU and Conditional GRU experiments are kept for comparison only:

```bash
uv run python -m src.training.tiny_gru_experiment --artifact-dir artifacts\tiny_gru_onnx_experiment
uv run python -m src.inference.export_tiny_gru_onnx --artifact-dir artifacts\tiny_gru_onnx_experiment
uv run python -m src.training.conditional_gru_decoder_experiment --artifact-dir artifacts\conditional_gru_decoder_experiment
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

Serve the shared ONNX model with FastAPI:

```bash
uv run --no-dev uvicorn src.api.tiny_gru_app:app --host 0.0.0.0 --port 8000
```

When `artifacts\shared_next_poi_gru_experiment\shared-next-poi-gru-v1` contains the exported shared ONNX files, the same FastAPI app serves both `/recommend` and `/generate-course`.
The backend-facing `/generate-course` and `/recommend` endpoints automatically return test data if model artifacts are missing or inference fails.

Generate a travel course with the backend contract body:

```bash
curl -X POST http://127.0.0.1:8000/generate-course \
  -H "Content-Type: application/json" \
  -d '{
    "areaCode": "11000",
    "contentIdList": ["2815426", "2773265"],
    "travelDuration": "2",
    "travelPersona": 3,
    "ageGroup": "30",
    "gender": "남",
    "travelerStyle": "4",
    "preferredArea": ["50110", "26350"],
    "residenceArea": "11000",
    "hasChild": 0,
    "hasElderly": 0,
    "hasDisabled": 0,
    "companionCount": 1
  }'
```

Recommend additional spots with the backend contract body:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "contentIdSequence": ["2815426", "2773265"],
    "areaCode": "11000",
    "travelDuration": "2",
    "travelPersona": 3,
    "ageGroup": "30",
    "gender": "남",
    "travelerStyle": "4",
    "preferredArea": ["50110", "26350"],
    "residenceArea": "11",
    "hasChild": false,
    "hasElderly": false,
    "hasDisabled": false,
    "companionCount": 1
  }'
```

`areaCode` must be one of the supported backend region codes: `11000` Seoul, `41110` Suwon, `28000` Incheon, `30000` Daejeon, `27000` Daegu, `12000` Gwangju, `26000` Busan, or `48120` Changwon. Backend-facing model responses are constrained to the requested region's POI master candidates, with static regional fallback used when the shared runtime is unavailable or inference fails.

`travelPersona` must be an integer from 1 to 7. `ageGroup` is temporarily normalized on the AI server to the model buckets: values below 20 use 20, values from 20 to 59 are floored to their 10-year bucket, and values 60 or above use 60. `preferredArea` must contain one to three 5-digit string codes. `contentIdList` is required for `/generate-course` and is treated as a required POI set, not an ordered prefix, when the shared model is available. If course inference fails, the existing regional fallback still places the validated content IDs first and fills the remainder from the requested `areaCode` region. `/generate-course` creates `travelDuration * 6` POIs, up to 6 POIs per day, and `/recommend` always uses top 4 recommendations on the AI server side.

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

When switching releases or changing `nginx/nginx.conf`, recreate Nginx so the bind-mounted config points at the current release:

```bash
docker compose up -d --no-deps --force-recreate nginx
docker compose exec nginx nginx -t
```

GitHub Actions deployment does not use a Deploy Key. The workflow checks out the repo inside GitHub Actions, creates a source bundle, uploads it to EC2 over SSH/SCP, switches `/opt/travel-ai/current`, runs Docker Compose, and checks `https://<EC2_HOST>/ready`.

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
curl --fail https://13.125.237.207/ready
```

## Dependency Layout

- Project dependencies are production-serving dependencies and do not include PyTorch.
- The `dev` dependency group contains training, notebook, CatBoost, PyTorch, and ONNX export dependencies.
- Existing `requirements.txt` and `requirements-production.txt` are kept as legacy fallback files for now.
