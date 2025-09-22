# MCP Web Search Server V2 �

[![GitHub Stars](https://img.shields.io/github/stars/Fouadbtlb/mcp-web-search?style=social)](https://github.com/Fouadbtlb/mcp-web-search)
[![GitHub Release](https://img.shields.io/github/v/release/Fouadbtlb/mcp-web-search)](https://github.com/Fouadbtlb/mcp-web-search/releases)
[![Docker Pulls](https://img.shields.io/docker/pulls/boutalebfouad/mcp-web-search)](https://hub.docker.com/r/boutalebfouad/mcp-web-search)
[![Image Size](https://img.shields.io/docker/image-size/boutalebfouad/mcp-web-search/latest)](https://hub.docker.com/r/boutalebfouad/mcp-web-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MCP Web Search V2** is a complete overhaul, transforming the service into a multi-stage, AI-powered search pipeline. It's designed for deep integration with LLMs and AI Agents, providing highly relevant, content-rich results by orchestrating fetching, extraction, semantic ranking, and reranking.

---

## ✨ V2 Highlights

- 🧠 **Multi-Stage AI Pipeline**: Fetch -> Extract -> Rank -> Rerank for state-of-the-art relevance.
- 🏆 **Advanced AI Models**: Utilizes powerful embedding (`nomic-ai/stella_en_1.5B_v5`) and reranker (`BAAI/bge-reranker-v2-m3`) models.
- 📄 **Firecrawl-Inspired Extractor**: Advanced content extraction that finds the main article on a page and converts it to clean, LLM-optimized Markdown.
- 🌐 **JavaScript Rendering**: Uses Pyppeteer (Chromium) to render dynamic web pages, ensuring content from SPAs is fully captured.
- ⚙️ **Flexible Search Modes**: Choose between `full` (all stages), `semantic` (fetch + embedding), or `fast` (quick fetch) search modes.
- 🐳 **Optimized Docker Build**: Multi-stage Dockerfile for a smaller, more efficient production image.
- 🔧 **Simplified Configuration**: All settings managed through a single `.env` file and Pydantic settings.

---

##  архитектура V2

The V2 architecture is a sequential pipeline designed to maximize relevance and content quality.

1.  **Fetch**: Gathers initial search results from multiple sources (`searxng`, `duckduckgo`).
2.  **Extract**: For each URL, fetches the full page content. It uses multiple extraction techniques and JS rendering to find the main content and converts it to clean Markdown.
3.  **Semantic Rank**: Generates embeddings for the query and all extracted content, then calculates cosine similarity to perform an initial semantic ranking.
4.  **Rerank**: Takes the top N semantically similar documents and uses a powerful cross-encoder model to perform a final, highly accurate reranking.

---

## 🚀 Quick Start

### 1. Create Environment File

Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

Review and edit the `.env` file to suit your needs. You can enable/disable services like Ollama or change the models used.

### 2. Run with Docker Compose

The recommended way to run the V2 server is with Docker Compose.

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

This will build the image and start the server, which will be accessible on `http://localhost:8000` by default.

### 3. Test the Server

Check the status endpoint to see if all services are running:

```bash
curl http://localhost:8000/status
```

Perform your first search:

```bash
curl -X POST http://localhost:8000/search \
-H "Content-Type: application/json" \
-d '{
  "query": "Latest news on AI in 2025",
  "max_results": 5,
  "search_mode": "full"
}'
```

---

## ⚙️ Configuration (`.env` file)

All configuration is now handled via an `.env` file. See `.env.example` for all available options.

| Variable | Default | Description |
|---|---|---|
| `HTTP_PORT` | `8000` | Port for the HTTP server. |
| `MCP_PORT` | `8001` | Port for the MCP server (if run with `--mcp`). |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `SEARXNG_INSTANCES` | `https://searx.be,...` | Comma-separated list of SearxNG instances. |
| `EMBEDDING_MODEL` | `nomic-ai/stella_en_1.5B_v5` | HuggingFace model for embeddings. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace model for reranking. |
| `RERANKING_ENABLED`| `true` | Whether to use the reranking stage. |
| `OLLAMA_URL` | `http://localhost:11434` | URL for an Ollama instance (optional). |
| `EXTRACTION_MODE` | `readability` | Content extraction strategy. |
| `LLM_OPTIMIZED_MARKDOWN` | `true` | Convert extracted content to clean Markdown. |
| `ENABLE_JS_RENDERING` | `true` | Use Chromium to render JavaScript on pages. |
| `CACHE_ENABLED` | `true` | Enable caching for search results. |
| `CACHE_TTL` | `3600` | Cache Time-To-Live in seconds. |

---

## 📡 HTTP API (V2)

### `POST /search`

**Request**:

```json
{
  "query": "What is the Model Context Protocol?",
  "max_results": 10,
  "search_mode": "full",
  "use_cache": true
}
```

- `search_mode`: Can be `full`, `semantic`, or `fast`.

**Response**:

```json
{
  "query": "What is the Model Context Protocol?",
  "message": "Successfully found 10 results.",
  "results": [
    {
      "url": "https://modelcontextprotocol.io/",
      "title": "Model Context Protocol",
      "snippet": "An open standard for AI model interaction...",
      "content": "# Model Context Protocol\n\nAn open standard for...",
      "markdown": "# Model Context Protocol...",
      "semantic_score": 0.95,
      "rerank_score": 0.998
    }
  ],
  "metadata": {
    "result_count": 10,
    "embedding_model": "nomic-ai/stella_en_1.5B_v5",
    "reranker_model": "BAAI/bge-reranker-v2-m3"
  }
}
```

### `GET /status`

Returns the status of all internal services, including loaded models and cache status.

---

## 🔌 MCP Integration (V2)

The MCP `search` tool now supports the `search_mode` parameter.

Example MCP call:

```bash
echo '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "python machine learning tutorial",
      "max_results": 5,
      "search_mode": "semantic"
    }
  }
}' | docker run --rm -i mcp-web-search:2.0.0 --mcp
```

---

## 🧪 Development

The project now uses **Poetry** for dependency management.

```bash
git clone https://github.com/Fouadbtlb/mcp-web-search.git
cd mcp-web-search
poetry install
poetry shell
python -m src.server
```

Build the container locally:

```bash
docker-compose -f docker/docker-compose.yml build
```

---

## 🤝 Contributing

Contributions to V2 are highly encouraged! The new architecture is more modular and easier to extend.

- 🐛 Report bugs or request features via [Issues](https://github.com/Fouadbtlb/mcp-web-search/issues).
- 🔀 Submit pull requests with improvements.
- ⭐ Star the repo if you find it useful!

---

## ⚖️ License

This project is released under the **MIT License**. See [`LICENSE`](https://github.com/Fouadbtlb/mcp-web-search/blob/main/LICENSE) for details.

