# thalamus-serve

[![PyPI version](https://badge.fury.io/py/thalamus-serve.svg)](https://badge.fury.io/py/thalamus-serve)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/brick-so/thalamus-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/brick-so/thalamus-serve/actions/workflows/ci.yml)

A Python ML model serving framework built on FastAPI with built-in observability, caching, and GPU management.

## Features

- **Simple Decorator API** - Register models with `@app.model()` decorator
- **Explicit Schema Types** - Input/output types specified via `input_type` and `output_type` parameters
- **Built-in Observability** - Prometheus metrics, structured logging
- **Weight Management** - Automatic fetching from S3, HuggingFace Hub, or HTTP
- **GPU Management** - Automatic device detection and allocation (CUDA/MPS/CPU)
- **Caching** - LRU cache for model weights with configurable size
- **Batch Processing** - Native batch inference support
- **Health Checks** - `/health`, `/ready`, `/status` endpoints
- **Capacity Reporting** - `/capacity` advertises free slots and ideal batch size
- **SageMaker BYOC** - `sagemaker_app()` serves `/ping` + `/invocations` on port 8080
- **API Key Authentication** - Protect endpoints with API key middleware

## Installation

```bash
pip install thalamus-serve

# With GPU/PyTorch support
pip install thalamus-serve[gpu]

# For development
pip install thalamus-serve[dev]
```

## Quick Start

```python
from pathlib import Path
from pydantic import BaseModel
from thalamus_serve import Thalamus


class TextInput(BaseModel):
    text: str


class SentimentOutput(BaseModel):
    label: str
    confidence: float


app = Thalamus()


@app.model(
    model_id="sentiment",
    version="1.0.0",
    description="Sentiment analysis model",
    default=True,
    input_type=TextInput,
    output_type=SentimentOutput,
)
class SentimentModel:
    def load(self, weights: dict[str, Path], device: str) -> None:
        # Load your model weights here
        pass

    def predict(self, inputs: list[TextInput]) -> list[SentimentOutput]:
        return [
            SentimentOutput(label="positive", confidence=0.95)
            for _ in inputs
        ]


if __name__ == "__main__":
    app.serve(host="0.0.0.0", port=8000)
```

Run the server:

```bash
export THALAMUS_API_KEY=your-secret-key
python app.py
```

Test the endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"inputs": [{"text": "I love this!"}]}'
```

## Examples

See the [examples/](examples/) directory for complete implementations:

- **[vanilla/](examples/vanilla/)** - Basic example with minimal setup
- **[torch/](examples/torch/)** - PyTorch image classification model
- **[hf_model/](examples/hf_model/)** - HuggingFace Transformers integration
- **[sklearn/](examples/sklearn/)** - scikit-learn model serving
- **[deploy/](examples/deploy/)** - Docker deployment with Docker Compose

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Service health status |
| `/ready` | GET | No | Readiness check (critical models loaded) |
| `/status` | GET | No | Detailed status with cache/GPU info |
| `/metrics` | GET | No | Prometheus metrics |
| `/capacity` | GET | Yes | Remaining request slots and ideal batch size |
| `/schema` | GET | Yes | List all model schemas |
| `/schema/{model_id}` | GET | Yes | Get specific model schema |
| `/predict` | POST | Yes | Run inference |
| `/cache/clear` | POST | Yes | Clear weight cache |
| `/models/{model_id}/unload` | POST | Yes | Unload model from memory |

## Model Interface

```python
class MyModel:
    def load(self, weights: dict[str, Path], device: str) -> None:
        """Called during startup to load weights."""
        pass

    def predict(self, inputs: list[InputType]) -> list[OutputType]:
        """Required. Runs inference on a batch."""
        pass

    # Optional hooks
    def preprocess(self, inputs: list[InputType]) -> list[Any]: ...
    def postprocess(self, outputs: list[Any]) -> list[OutputType]: ...

    @property
    def is_ready(self) -> bool:
        """Optional. Used by /ready endpoint."""
        return True

    def capacity(self) -> ModelCapacity:
        """Optional. Used by /capacity endpoint."""
        ...
```

## Capacity

`/capacity` tells a caller how much work this deployment will accept right now, so a
batching client can size its next dispatch instead of guessing. Declare the numbers on
the decorator:

```python
@app.model(
    model_id="image-classifier",
    input_type=Input,
    output_type=Output,
    max_batch_size=32,
    ideal_batch_size=16,
    max_concurrent_requests=2,
)
class ImageClassifier:
    ...
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `max_batch_size` | `1` | Hard cap on inputs accepted in one `/predict` call |
| `ideal_batch_size` | `max_batch_size` | Throughput sweet spot |
| `max_concurrent_requests` | `1` | Parallel `/predict` calls the pod tolerates |

The defaults are deliberately conservative — a model that has not declared anything
advertises a batch size of 1. `ideal_batch_size` must be between 1 and `max_batch_size`,
and both must be at least 1; violating that raises `ValueError` at import time.

A model that can measure its own headroom overrides the static numbers with a `capacity()`
method:

```python
from thalamus_serve import ModelCapacity, get_gpu_memory

class ImageClassifier:
    def capacity(self) -> ModelCapacity:
        used_mb, total_mb = get_gpu_memory(self.device) or (0.0, 1.0)
        headroom = 1.0 - (used_mb / total_mb)
        return ModelCapacity(
            accepting=headroom > 0.1,
            remaining_requests=2 if headroom > 0.4 else 1,
            ideal_batch_size=16 if headroom > 0.4 else 4,
            max_batch_size=32,
            reason=None if headroom > 0.1 else "vram_headroom_low",
        )
```

The hook is polled before every dispatch, so it must be O(1) — read a cached gauge, never
run inference. A hook that raises is reported as `accepting=false` with
`reason="capacity_hook_error"`; it never fails the request.

The response aggregates across models. Top-level `accepting` is the AND over **critical**
models, so an unloaded non-critical model does not mark the pod unavailable. Top-level
`remaining_requests` is the minimum across accepting models — the bottleneck — and is `0`
whenever `accepting` is false, so the two can never disagree. Per-model slot counts stay
visible in `models` regardless. Querying
`/capacity` never triggers a lazy model load; an unloaded model reports
`reason="model_not_loaded"`.

```json
{
  "accepting": true,
  "remaining_requests": 2,
  "models": {
    "image-classifier@2.1.0": {
      "accepting": true,
      "remaining_requests": 2,
      "ideal_batch_size": 16,
      "max_batch_size": 32,
      "reason": null
    }
  },
  "inflight_requests": 0,
  "uptime_seconds": 128.4
}
```

## SageMaker (BYOC)

AWS SageMaker hosting requires a container that answers `GET /ping` and
`POST /invocations` on port 8080 — a different contract from `/predict`. `sagemaker_app()`
builds that app from a model you have already registered, reusing the same weights,
device allocation and hooks:

```python
# src/main.py
app = Thalamus()

@app.model(
    model_id="image-classifier", default=True, device="cuda",
    input_type=Input, output_type=Output,
    weights={"checkpoint": S3Weight(bucket="...", key="...")},
)
class ImageClassifier: ...

def create_app():
    return app.sagemaker_app()
```

```dockerfile
ENTRYPOINT ["uvicorn", "src.main:create_app", "--factory", \
            "--host", "0.0.0.0", "--port", "8080"]
```

SageMaker endpoints serve one model, so `sagemaker_app()` resolves the registered default
(or an explicit `model_id=` / `version=`) and loads **only** that model — siblings
registered on the same app are left untouched, so a multi-model app can ship a
single-model image.

| Behavior | Detail |
|----------|--------|
| `GET /ping` | `200` once the model reports ready, else `503` |
| `POST /invocations` | One `Input` in, one `Output` out |
| Response shape | Bare `Output` by default; `envelope="predict_response"` wraps it in `PredictResponse` |
| Bad input | `400` for malformed JSON or schema violations — SageMaker reads `500` as a ModelError |
| Model failure | `500` with a generic message; details go to the log, never the response |
| Empty output | `500` — returning no output for one input violates the contract |
| Auth | **None.** SageMaker sends no credentials; access is controlled by AWS IAM |

Two details are load-bearing and covered by regression tests, because both have bitten
this contract before:

- `is_ready` is read as a **property**, never called. Invoking it as `is_ready()` raises
  `TypeError: 'bool' object is not callable`, so the container never goes healthy.
- `predict` runs in a worker thread via `asyncio.to_thread`. Calling it inline would block
  the event loop for the whole inference, so SageMaker's periodic `/ping` goes unanswered
  and it restarts the container mid-prediction.

Batching and `/capacity` do not apply here: the contract is single-in/single-out, and
SageMaker does its own request-level scaling.

## Configuration

### Weight Sources

Models can load weights from multiple sources by specifying them directly in the decorator:

```python
from pydantic import BaseModel
from thalamus_serve import Thalamus, HFWeight, S3Weight, HTTPWeight

app = Thalamus()

class MyInput(BaseModel):
    text: str

class MyOutput(BaseModel):
    embedding: list[float]

@app.model(
    model_id="my-model",
    version="1.0.0",
    device="cuda:0",
    input_type=MyInput,
    output_type=MyOutput,
    weights={
        "model": HFWeight(repo="bert-base-uncased"),
        "tokenizer": S3Weight(bucket="my-bucket", key="models/tokenizer.json"),
    },
)
class MyModel:
    def load(self, weights: dict[str, Path], device: str) -> None:
        # weights["model"] is a directory path (full repo)
        # weights["tokenizer"] is a file path
        pass
```

**Supported weight sources:**

| Source | Single File | Directory/Sharded |
|--------|-------------|-------------------|
| **HuggingFace** | `HFWeight(repo="...", filename="model.bin")` | `HFWeight(repo="...")` |
| **S3** | `S3Weight(bucket="...", key="path/model.pt")` | `S3Weight(bucket="...", prefix="path/shards/")` |
| **HTTP** | `HTTPWeight(urls=["https://.../model.pt"])` | `HTTPWeight(urls=["https://.../shard1.pt", "https://.../shard2.pt"])` |

**Examples:**

```python
# HuggingFace - single file
HFWeight(repo="bert-base-uncased", filename="pytorch_model.bin", revision="main")

# HuggingFace - full repo snapshot (for sharded models)
HFWeight(repo="meta-llama/Llama-2-7b-hf")

# S3 - single file
S3Weight(bucket="my-models", key="bert/model.pt", region="us-east-1")

# S3 - directory prefix (downloads all files under prefix)
S3Weight(bucket="my-models", prefix="llama/shards/")

# HTTP - single file
HTTPWeight(urls=["https://example.com/model.pt"])

# HTTP - multiple files (sharded)
HTTPWeight(urls=[
    "https://example.com/model-00001.pt",
    "https://example.com/model-00002.pt",
])
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `THALAMUS_API_KEY` | - | API key for protected endpoints |
| `THALAMUS_LOG_LEVEL` | INFO | Logging level |
| `THALAMUS_CACHE_DIR` | /tmp/thalamus | Weight cache directory |
| `THALAMUS_CACHE_MAX_GB` | 50 | Maximum cache size in GB |
| `HF_TOKEN` | - | HuggingFace authentication token |
| `AWS_ACCESS_KEY_ID` | - | AWS credentials for S3 |
| `AWS_SECRET_ACCESS_KEY` | - | AWS credentials for S3 |

## Built-in Schemas

Common ML types available from the package:

```python
from thalamus_serve import (
    Base64Data,  # Base64-encoded binary data
    BBox,        # Bounding box (x1, y1, x2, y2)
    Label,       # Classification label with confidence
    Vector,      # Embedding vector
    Span,        # Text span with optional label
    Prob,        # Probability value (0-1)
    S3Ref,       # S3 reference (bucket, key)
    Url,         # URL type
)
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
