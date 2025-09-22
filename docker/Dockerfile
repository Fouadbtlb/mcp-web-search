# Stage 1: Base image with Python and essential build tools
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PATH="/root/.local/bin:$PATH"

# Install build essentials and Chromium dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       # Chromium dependencies for Playwright/Pyppeteer
       libnss3 \
       libnspr4 \
       libdbus-1-3 \
       libatk1.0-0 \
       libatk-bridge2.0-0 \
       libcups2 \
       libdrm2 \
       libxkbcommon0 \
       libxcomposite1 \
       libxdamage1 \
       libxfixes3 \
       libxrandr2 \
       libgbm1 \
       libasound2 \
       # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# ---

# Stage 2: Builder image for installing dependencies
FROM base AS builder

# Copy only files needed for dependency installation
COPY requirements.txt pyproject.toml poetry.lock* ./

# Install dependencies
# This layer is cached and only re-runs if dependencies change
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --no-root --without dev

# Install Pyppeteer and download Chromium
# This is a separate layer to cache the large download
RUN python -m pip install pyppeteer \
    && python -c "from pyppeteer.chromium_downloader import download_chromium; download_chromium()"

# ---

# Stage 3: Final application image
FROM base AS final

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.local /root/.local

# Copy application source code
COPY src/ ./src/

# Set the entrypoint
ENTRYPOINT ["python", "-m", "src.server"]
