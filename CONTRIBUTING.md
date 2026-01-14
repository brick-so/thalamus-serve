# Contributing to thalamus-serve

Thank you for your interest in contributing to thalamus-serve! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/brick-so/thalamus-serve.git
cd thalamus-serve

# Install dependencies with dev extras
uv sync --extra dev

# Install with GPU support (optional)
uv sync --extra gpu --extra dev
```

### Running Tests

```bash
# Run all tests
uv run pytest thalamus_serve/tests/

# Run with coverage
uv run pytest thalamus_serve/tests/ -v --cov=thalamus_serve

# Run a specific test
uv run pytest thalamus_serve/tests/test_service_contract.py::TestHealthContract -v
```

### Code Quality

We use the following tools to maintain code quality:

```bash
# Linting with Ruff
uv run ruff check .
uv run ruff format .

# Type checking with mypy
uv run mypy thalamus_serve/
```

## Code Style

- We follow PEP 8 with a line length of 88 characters
- Use type hints for all function signatures
- Write docstrings for public APIs
- Keep functions focused and small

### Import Order

Imports should be organized in this order:
1. Standard library imports
2. Third-party imports
3. Local imports

Ruff will automatically sort imports for you.

## Pull Request Guidelines

### Before Submitting

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with descriptive messages

3. Ensure all tests pass:
   ```bash
   uv run pytest thalamus_serve/tests/
   ```

4. Ensure code passes linting:
   ```bash
   uv run ruff check .
   uv run mypy thalamus_serve/
   ```

### PR Requirements

- Clear description of changes
- Tests for new functionality
- Documentation updates if needed
- Passing CI checks

### Commit Messages

Use clear, descriptive commit messages:

- `feat: add support for custom preprocessing hooks`
- `fix: resolve memory leak in weight cache`
- `docs: update installation instructions`
- `refactor: simplify GPU allocation logic`
- `test: add tests for batch processing`

## Project Structure

```
thalamus_serve/
├── core/           # Core application logic
│   ├── app.py      # Thalamus main class
│   ├── model.py    # Model registry and specs
│   ├── routes.py   # FastAPI routes
│   └── middleware.py
├── infra/          # Infrastructure
│   ├── cache.py    # Weight caching
│   └── gpu.py      # GPU management
├── observability/  # Logging and metrics
│   ├── logging.py
│   ├── metrics.py
│   └── middleware.py
├── schemas/        # Pydantic schemas
│   ├── api.py      # API request/response
│   ├── common.py   # Common ML types
│   └── storage.py  # Storage types
├── storage/        # Weight fetching
│   └── fetch.py
├── tests/          # Test suite
└── __init__.py     # Public API exports
```

## Adding a New Feature

1. **Discuss first**: Open an issue to discuss the feature before implementing
2. **Write tests**: Add tests for your feature
3. **Update docs**: Update README if needed
4. **Follow patterns**: Look at existing code for patterns to follow

## Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Version information

## Questions?

Feel free to open an issue for questions or discussions.
