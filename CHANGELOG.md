# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-XX-XX

### Added

- Initial release of thalamus-serve
- Core `Thalamus` application class with `@app.model()` decorator
- FastAPI-based HTTP server with configurable host/port
- Model registry with version support
- Automatic input/output schema generation from type hints
- Built-in endpoints:
  - `/health` - Service health check
  - `/ready` - Readiness check (critical models)
  - `/status` - Detailed status with cache/GPU info
  - `/metrics` - Prometheus metrics
  - `/schema` - Model schema listing
  - `/predict` - Batch inference endpoint
  - `/cache/clear` - Cache management
  - `/models/{id}/unload` - Model unloading
- Weight fetching from S3, HuggingFace Hub, and HTTP URLs
- LRU weight cache with configurable size
- GPU management with CUDA and MPS support
- Prometheus metrics for requests, latency, batch size
- Structured logging with structlog
- API key authentication middleware
- Common ML schema types (BBox, Label, Vector, Span, etc.)
- Example implementations for vanilla, PyTorch, HuggingFace, and scikit-learn

### Security

- API key required for protected endpoints
- Configurable via `THALAMUS_API_KEY` environment variable

[Unreleased]: https://github.com/brick-so/thalamus-serve/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/brick-so/thalamus-serve/releases/tag/v0.1.0
