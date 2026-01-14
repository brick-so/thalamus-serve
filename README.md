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
```

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
