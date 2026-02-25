# CrawlPrime

**Web RAG — crawl, index, and query live web content.**

CrawlPrime is a standalone web RAG built on top of **ContextPrime** shared utilities.
It transforms dynamic websites into structured, queryable knowledge — using the same
DocTags hierarchy, vector storage, and hybrid retrieval that powers ContextPrime
document RAG.

## Architecture

```
URL ──► WebCrawler (crawl4ai + Playwright)
              │
              ▼
        WebDocTagsMapper          ← ContextPrime shared utility
              │
              ▼
        WebIngestionPipeline      ← ContextPrime shared utility
         │            │
         ▼            ▼
      Neo4j        Qdrant (vectors)
                      │
                      ▼
              HybridRetriever     ← ContextPrime shared utility
                      │
                      ▼
              AgenticPipeline     ← ContextPrime shared utility
                      │
                      ▼
               LLM synthesis → Answer
```

CrawlPrime owns: `pipeline.py` (orchestrator), `planner.py` (URL-aware step planning),
`api.py` (FastAPI app), `main.py` (CLI).

ContextPrime provides: WebCrawler, WebDocTagsMapper, WebIngestionPipeline,
HybridRetriever, AgenticPipeline, QdrantManager, Neo4jManager.

## Installation

```bash
# Option A — standalone install (fetches ContextPrime from GitHub)
pip install -e .
playwright install chromium

# Option B — dev mode (ContextPrime cloned as sibling directory)
pip install -r requirements.txt
playwright install chromium

# Start services (Qdrant + Neo4j)
docker-compose -f ../docker-compose.yml up -d
```

## Quick Start

### Python API

```python
import asyncio
from crawl_prime.pipeline import CrawlPrimePipeline

async def main():
    cp = CrawlPrimePipeline(collection="my_web_kb", enable_synthesis=True)

    # Crawl and index a site
    report = await cp.ingest("https://example.com")
    print(f"Indexed {report.chunks_ingested} chunks")

    # Query it
    result = await cp.query("What services does the site offer?")
    print(result.answer)
    cp.close()

asyncio.run(main())
```

### CLI

```bash
python -m crawl_prime.main --url "https://example.com" --output data/output
```

### REST API

```bash
uvicorn crawl_prime.api:app --reload --port 8001
```

```bash
# Ingest a URL
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What services does the site offer?"}'
```

## Constructor Parameters

| Parameter | Default | Description |
|---|---|---|
| `collection` | `"crawlprime_default"` | Qdrant collection name |
| `qdrant_host` | `"localhost"` | Qdrant host |
| `qdrant_port` | `6333` | Qdrant port |
| `neo4j_host` | `"localhost"` | Neo4j host (graph retrieval disabled gracefully if unreachable) |
| `neo4j_port` | `7687` | Neo4j bolt port |
| `neo4j_user` | `"neo4j"` | Neo4j username |
| `neo4j_password` | `"password"` | Neo4j password |
| `vector_weight` | `0.6` | RRF weight for vector search (60%) |
| `graph_weight` | `0.3` | RRF weight for graph traversal (30%; drops to 0.0 if Neo4j unavailable) |
| `lexical_weight` | `0.1` | RRF weight for BM25 lexical search (10%) |
| `enable_synthesis` | `True` | Enable LLM answer synthesis (requires `OPENAI_API_KEY`) |
| `storage_path` | `Path("data/crawlprime")` | Directory for RL Q-table and memory storage |
| `raptor_pipeline` | `None` | Optional RAPTOR hierarchical summarisation pipeline |
| `community_pipeline` | `None` | Optional community detection pipeline |

**Retrieval weight rationale:** 60% vector captures semantic similarity; 30% graph
leverages `(:Page)-[:LINKS_TO]->(:Page)` edges written by the ingestion pipeline to
surface contextually linked pages; 10% lexical (BM25) catches exact-match terms that
dense embeddings can miss. Neo4j is optional — if unreachable, `graph_weight`
automatically drops to 0.0 and the remaining weight shifts to vector.

## Environment Variables

```bash
OPENAI_API_KEY=sk-...             # Required for LLM synthesis
QDRANT_HOST=localhost              # Qdrant host (default: localhost)
QDRANT_PORT=6333                   # Qdrant port (default: 6333)
NEO4J_URI=bolt://localhost:7687    # Neo4j URI
NEO4J_USERNAME=neo4j               # Neo4j username
NEO4J_PASSWORD=yourpassword        # Neo4j password

# Optional: route LLM calls through OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Optional: enable LLM query decomposition for complex queries
DOCTAGS_LLM_DECOMPOSITION=true
```

## crawl4ai 0.8.x API

CrawlPrime requires `crawl4ai>=0.8.0`. The 0.8.x release changed the
`AsyncWebCrawler` constructor — the `BrowserConfig` object must be passed
as `config=`, not `browser_config=`:

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
    result = await crawler.arun(url=url, config=CrawlerRunConfig())
```

## Testing

```bash
# Unit tests (no Docker needed)
.venv/bin/python -m pytest tests/test_processing.py -v

# Integration tests (requires Docker + Playwright + OPENAI_API_KEY)
.venv/bin/python -m pytest tests/integration/test_pipeline_e2e.py -v -m integration

# All tests except real-web
.venv/bin/python -m pytest tests/ -m "not real_web"

# Real-web smoke test (requires live internet + Docker + OPENAI_API_KEY)
.venv/bin/python -m pytest tests/integration/test_real_web.py -v -m real_web
```

The integration conftest automatically loads `.env` from the sibling `doctags_rag/`
directory and overrides `QDRANT_HOST=localhost` so tests work against local Docker
containers regardless of what the `.env` specifies for production.

## Relationship to ContextPrime

| Concern | ContextPrime | CrawlPrime |
|---|---|---|
| PDF / DOCX / HTML file ingestion | ✓ | — |
| Document query pipeline | ✓ | — |
| Web crawling + indexing | utility only | ✓ (orchestrates) |
| Web query pipeline | — | ✓ |
| URL detection in planner | — | ✓ |
| WebCrawler, WebDocTagsMapper | ✓ shared | imports |
| WebIngestionPipeline | ✓ shared | imports |
| HybridRetriever, AgenticPipeline | ✓ owns | imports |
