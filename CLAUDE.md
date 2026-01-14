# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

thalamus-serve is a Python ML model serving framework built on FastAPI. It provides a standardized way to deploy machine learning models with built-in observability, caching, and GPU management.

## Commands

```bash
# Install dependencies (using uv)
uv sync

# Install with GPU/PyTorch support
uv sync --extra gpu

# Run tests
uv run pytest thalamus_serve/tests/

# Run a single test
uv run pytest thalamus_serve/tests/test_service_contract.py::TestHealthContract::test_response_schema -v

# Run the example server
uv run python examples/vanilla/vanilla.py

# Start a model server programmatically
# The Thalamus app is an ASGI app that can be run with uvicorn
uv run uvicorn examples.vanilla.vanilla:app --host 0.0.0.0 --port 8000
```

## Architecture

### Core Components

- **`Thalamus`** (`core/app.py`): Main application class. Acts as both a decorator factory for registering models and an ASGI application. Use `@app.model()` decorator to register model classes.

- **`ModelSpec` / `ModelRegistry`** (`core/model.py`): Model metadata and registry system. Input/output types are inferred from the `predict()` method signature or can be explicitly provided. Models can have optional `load()`, `preprocess()`, and `postprocess()` methods.

- **Routes** (`core/routes.py`): FastAPI router with endpoints:
  - `/health`, `/ready`, `/status` - Health checks (public)
  - `/metrics` - Prometheus metrics (public)
  - `/schema`, `/schema/{model_id}` - Model schemas (requires auth)
  - `/predict` - Batch inference (requires auth)
  - `/cache/clear`, `/models/{model_id}/unload` - Management (requires auth)

### Infrastructure

- **Weight Fetching** (`storage/fetch.py`): Downloads model weights from S3, HuggingFace Hub, or HTTP URLs. Uses an LRU cache with configurable max size.

- **GPU Management** (`infra/gpu.py`): Device detection (CUDA/MPS/CPU), memory tracking, and allocation. `GPUAllocator` distributes models across available devices.

- **Caching** (`infra/cache.py`): `WeightCache` provides content-addressed caching with automatic LRU eviction.

### Configuration

- **Weight Sources**: Model weights are configured directly in the `@app.model()` decorator using `S3Weight`, `HFWeight`, or `HTTPWeight` types.

- **Environment Variables**:
  - `THALAMUS_API_KEY` - Required for protected endpoints
  - `THALAMUS_LOG_LEVEL` - Logging level (default: INFO)
  - `THALAMUS_CACHE_DIR` - Weight cache directory (default: /tmp/thalamus)
  - `THALAMUS_CACHE_MAX_GB` - Max cache size in GB (default: 50)
  - `HF_TOKEN` - HuggingFace authentication token
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - S3 credentials

## Model Interface

Models registered with `@app.model()` should implement:

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

## Schemas

The `schemas/` directory provides Pydantic models for common ML types:
- `Base64Data`, `BBox`, `Label`, `Vector`, `Span`, `Prob` (common.py)
- `S3Ref`, `S3PresignedUrl`, `Url` (storage.py)
- API request/response types (api.py)
