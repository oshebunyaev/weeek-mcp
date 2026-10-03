# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm AS runtime

ARG UV_VERSION=0.12.21
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/app/.venv/bin:$PATH

RUN groupadd --system --gid 10001 weeek \
    && useradd --system --uid 10001 --gid weeek --home-dir /app --shell /usr/sbin/nologin weeek \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY weeek_mcp ./weeek_mcp
RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p /data/session /data/pending /data/logs \
    && chown -R weeek:weeek /app /data

USER weeek
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["weeek-mcp"]

FROM runtime AS runtime-kb
USER root
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN uv sync --frozen --no-dev --no-editable --extra kb \
    && mkdir -p /ms-playwright \
    && playwright install --with-deps chromium \
    && chown -R weeek:weeek /app /ms-playwright
USER weeek
