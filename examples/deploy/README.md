# Docker Deployment Example

This example demonstrates deploying a thalamus-serve model as a Docker container.

## Files

- `Dockerfile` - Multi-stage Docker build with uv for fast dependency installation
- `pyproject.toml` - Project dependencies
- `src/main.py` - Sentiment analysis model using DistilBERT
- `docker-compose.yml` - Docker Compose configuration

## Building

```bash
# Generate uv.lock file first
uv lock

# Build the Docker image
docker build -t thalamus-sentiment:latest .
```

## Running

### With Docker

```bash
docker run -p 8000:8000 \
  -e THALAMUS_API_KEY=your-secret-key \
  -e HF_TOKEN=your-hf-token \
  thalamus-sentiment:latest
```

### With Docker Compose

```bash
# Set environment variables
export THALAMUS_API_KEY=your-secret-key
export HF_TOKEN=your-hf-token  # Optional, for private models

# Run the service
docker compose up
```

## Testing

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"inputs": [{"text": "I love this product!"}]}'
```

Expected response:

```json
{
  "outputs": [
    {"label": "POSITIVE", "score": 0.9998}
  ],
  "meta": {
    "model": "sentiment",
    "version": "1.0.0",
    "latency_ms": 45.2,
    "batch_size": 1
  }
}
```

## Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/status

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Configuration

| Environment Variable | Description |
|---------------------|-------------|
| `THALAMUS_API_KEY` | API key for protected endpoints (required) |
| `THALAMUS_LOG_LEVEL` | Logging level (default: INFO) |
| `THALAMUS_CACHE_DIR` | Weight cache directory (default: /tmp/thalamus) |
| `THALAMUS_CACHE_MAX_GB` | Maximum cache size in GB (default: 50) |
| `HF_TOKEN` | HuggingFace token for private models |

## Production Considerations

1. **Persistent Cache**: Mount a volume for `/app/.cache` to persist downloaded weights across container restarts.

2. **Resource Limits**: Set appropriate CPU/memory limits based on your model requirements.

3. **Health Checks**: Configure Kubernetes liveness and readiness probes using `/health` and `/ready` endpoints.

4. **Scaling**: The service is stateless and can be horizontally scaled behind a load balancer.

5. **GPU Support**: For GPU inference, use an NVIDIA base image and add `--gpus all` to the docker run command.
