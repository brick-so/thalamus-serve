# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-20

### Added

- `GET /capacity` endpoint reporting, per model, how many more requests the deployment
  will accept and its ideal batch size. Requires authentication. Aggregates to a
  top-level `accepting` (AND over critical models) and `remaining_requests` (the
  minimum across accepting models, forced to 0 whenever `accepting` is false). Querying
  it never triggers a lazy model load.
- `max_batch_size`, `ideal_batch_size`, and `max_concurrent_requests` parameters on the
  `@app.model()` decorator, stored on `ModelSpec`. All default conservatively to 1 and
  are validated at import time — `ideal_batch_size` must fall within
  `[1, max_batch_size]`, and both limits must be at least 1.
- Optional `capacity()` hook on model classes, overriding the static numbers when the
  model is loaded. A hook that raises or returns an unparseable value is reported as
  `accepting=false` with `reason="capacity_hook_error"` rather than failing the request.
- `thalamus_inflight_requests` gauge tracking `/predict` calls currently in flight.
- `ModelCapacity` and `CapacityResponse` exported from the package root.

## [0.2.0] - 2026-01-14

### Changed

- **BREAKING**: `input_type` and `output_type` are now required parameters in `@app.model()` decorator
- Removed automatic type inference from `predict()` method signatures
- Weight sources now specified directly in decorator (removed `thalamus-deploy.json` dependency)

### Added

- Docker deployment example with Dockerfile, docker-compose.yml, and documentation
- Medical cost prediction example demonstrating multi-weight loading (model + preprocessor)
- S3 prefix/directory download support for sharded models (`S3Weight(prefix="...")`)
- HTTP multi-URL download support for sharded models (`HTTPWeight(urls=[...])`)
- HuggingFace cache integration with unified LRU eviction
- PyPI publishing workflow in GitHub Actions (triggered by version tags)
- Full mypy type checking compliance

### Fixed

- `/cache/clear` endpoint now properly clears all cache subdirectories (HuggingFace, S3 prefixes, HTTP URLs)
- CI workflow compatibility with Ubuntu 24.04+ externally managed Python

### Removed

- Iris classifier example (replaced with medical cost example)
- Automatic type inference from `predict()` signatures (explicit types now required)

## [0.1.0] - 2026-01-14

### Added

- Initial release of thalamus-serve
- Core `Thalamus` application class with `@app.model()` decorator
- FastAPI-based HTTP server with configurable host/port
- Model registry with version support
- Input/output schema validation via `input_type` and `output_type` parameters
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

[Unreleased]: https://github.com/brick-so/thalamus-serve/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/brick-so/thalamus-serve/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/brick-so/thalamus-serve/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brick-so/thalamus-serve/releases/tag/v0.1.0
