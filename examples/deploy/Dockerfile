FROM python:3.12-slim@sha256:a75662dfec8d90bd7161c91050be2e0a9b21d284f3b7a7253d5db25f7d583fb3

RUN groupadd -r thalamus && useradd -r -g thalamus thalamus

WORKDIR /app
ENV XDG_CACHE_HOME=/app/.cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.22@sha256:2320e6c239737dc73cccce393a8bb89eba2383d17018ee91a59773df802c20e6 /uv /usr/local/bin/uv

RUN chown thalamus:thalamus /app
USER thalamus

COPY --chown=thalamus:thalamus pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=thalamus:thalamus src ./src

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
