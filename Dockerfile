# Multi-stage build for minimal MCP Web Search Server
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libxml2-dev \
    libxslt-dev \
    linux-headers

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-alpine

# Install runtime dependencies only
RUN apk add --no-cache \
    libxml2 \
    libxslt \
    curl

# Copy virtual environment from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Create non-root user
RUN adduser -D -s /bin/sh mcp-user

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/
COPY .env.production /app/.env

# Set ownership to non-root user
RUN chown -R mcp-user:mcp-user /app

# Switch to non-root user
USER mcp-user

# Environment variables with defaults
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    OLLAMA_URL="" \
    EMBEDDING_MODEL="all-minilm" \
    MAX_CHUNK_SIZE=512 \
    SEARXNG_URL="https://searx.be" \
    REDIS_URL="" \
    LOG_LEVEL=INFO \
    MCP_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$MCP_PORT/health || exit 1

# Expose port
EXPOSE $MCP_PORT

# Labels for metadata
LABEL org.opencontainers.image.title="MCP Web Search Server" \
      org.opencontainers.image.description="Lightweight MCP server for web search with optional Ollama integration" \
      org.opencontainers.image.source="https://github.com/your-username/mcp-web-search" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="2.0.0"

# Default command (MCP server mode)
# Default to MCP stdio mode, set SERVER_MODE=http for HTTP API
ENV SERVER_MODE=stdio

# Expose port 8000 for HTTP mode (optional)
EXPOSE 8000

# Default command (MCP stdio mode)
CMD ["python", "-m", "src.server"]
