# MCP Web Search Server 🔎

[![Docker Pulls](https://img.shields.io/docker/pulls/boutalebfouad/mcp-web-search)](https://hub.docker.com/r/boutalebfouad/mcp-web-search)  
[![Image Size](https://img.shields.io/docker/image-size/boutalebfouad/mcp-web-search/latest)](https://hub.docker.com/r/boutalebfouad/mcp-web-search)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight **Model Context Protocol (MCP)** server for intelligent web search — multi-source aggregation, full content extraction and AI-powered semantic ranking. Designed for easy integration with AI assistants (OpenWebUI, Claude, custom MCP clients) and for deployment in production using Docker.

---

## ✨ Quick highlights

- 🚀 Multi-engine search (SearxNG + DuckDuckGo fallback)
- 🧠 AI-powered semantic ranking (embeddings via Ollama or local fallback)
- 📄 Full article extraction (not only snippets)
- ⚡ Lightweight: ~150MB Alpine image, fast startup
- 🔗 Dual modes: **stdio (MCP)** and **HTTP (REST)**
- 🩺 Production features: health checks, optional Redis cache, monitoring hooks

---

## 🧰 Features

- **Multi-Source Aggregation** — query SearxNG first and fallback to other engines when needed.
- **Semantic Ranking** — embed results and score by similarity to query.
- **Full Content Extraction** — pull article text, metadata (author, publish_date, language).
- **Freshness & Quality** — freshness weighting, reading-time estimation, deduplication.
- **MCP Tool** — exposes a `search_web` tool for MCP clients.
- **HTTP API** — `POST /search`, `GET /health`, `GET /docs` (Swagger UI).
- **Docker-first** — healthcheck-ready container and simple compose examples.

---

## 🚀 Quick start

### Run (HTTP mode)

```bash
docker run -d --name mcp-web-search \
  -p 8001:8001 \
  -e SERVER_MODE=http \
  -e SEARXNG_URL=https://searx.be \
  boutalebfouad/mcp-web-search:latest
```

Check health:

```bash
curl http://localhost:8001/health
```

### Docker Compose (recommended for local dev)

```yaml
version: '3.8'
services:
  mcp-web-search:
    image: boutalebfouad/mcp-web-search:latest
    ports:
      - "8001:8001"
    environment:
      - SERVER_MODE=http
      - SEARXNG_URL=https://searx.be
      - OLLAMA_URL=http://ollama:11434
    restart: unless-stopped

  # Optional: local Ollama and Redis
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb

volumes:
  ollama_data:
```

---

## ⚙️ Configuration (environment variables)

| Variable | Default | Description |
|---|---:|---|
| `SERVER_MODE` | `stdio` | `stdio` (MCP) or `http` (REST) |
| `SEARXNG_URL` | `https://searx.be` | SearxNG instance URL |
| `OLLAMA_URL` | `` | Ollama server for embeddings (optional) |
| `EMBEDDING_MODEL` | `all-minilm` | Embedding model name |
| `REDIS_URL` | `` | Redis for caching (optional) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CACHE_ENABLED` | `false` | Enable result caching |

---

## 📡 HTTP API

### `POST /search`

**Request** (JSON):

```json
{
  "q": "AI news 2025",
  "n_results": 5,
  "fresh_only": false,
  "require_full_fetch": true
}
```

**Response** (excerpt):

```json
{
  "query": "AI news 2025",
  "results": [
    {
      "url": "https://example.com/article",
      "title": "Article Title",
      "snippet": "Short summary...",
      "content": "Full article text...",
      "author": "Author Name",
      "publish_date": "2025-01-15",
      "language": "en",
      "semantic_score": 0.89,
      "combined_score": 0.92
    }
  ],
  "intent": { "detected": "news", "confidence": 0.85 }
}
```

### `GET /health`

Returns service status (health, version, dependencies availability).

### `GET /docs`

Interactive Swagger UI for the HTTP API.

---

## 🔌 MCP integration (stdio)

When running in `stdio` mode the server exposes a tool named `search_web` that MCP clients can call. Example tool schema:

```json
{
  "name": "search_web",
  "description": "Search the web with AI-powered ranking",
  "inputSchema": {
    "type": "object",
    "properties": {
      "q": { "type": "string", "description": "Search query" },
      "n_results": { "type": "integer", "description": "Number of results (1-20)", "minimum": 1, "maximum": 20 }
    },
    "required": ["q"]
  }
}
```

Example (stdio JSON-RPC call):

```bash
echo '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": { "name": "search_web", "arguments": { "q": "python machine learning tutorial", "n_results": 5 } }
}' | docker run --rm -i boutalebfouad/mcp-web-search:latest
```

---

## 🐳 Docker tags

- `latest` — latest stable release
- `v1.3.0-stable` — pinned stable release
- `develop` — development build

---

## ⚠️ Troubleshooting

**No results**
- Verify `SEARXNG_URL` is reachable from the container.
- Check network/DNS settings.

**High memory usage**
- Use CPU-only mode by leaving `OLLAMA_URL` empty.
- Disable caching or lower concurrency.

**Status shows `unhealthy`**
- `docker logs mcp-web-search` and inspect startup errors.
- Ensure `curl -f http://localhost:8001/health` from inside container succeeds.

---

## 🧪 Development

Local dev quickstart:

```bash
git clone https://github.com/your-repo/mcp-web-search.git
cd mcp-web-search
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.server
```

Build the container locally:

```bash
docker build -t mcp-web-search:dev .
```

---

## 📄 Contributing

Contributions are welcome. Please read `CONTRIBUTING.md` and add tests for behavior changes. Use feature branches and open a PR with a clear description.

---

## ⚖️ License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

Built on top of: SearxNG, Ollama, FastAPI and the Model Context Protocol ecosystem.

---

**Made with ❤️ for the MCP ecosystem**

