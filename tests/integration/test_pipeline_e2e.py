"""
CrawlPrime end-to-end integration test.

Exercises the full web RAG path via CrawlPrimePipeline:
  CrawlPrimePipeline.ingest(url) → Qdrant
  CrawlPrimePipeline.query(text) → grounded answer

Requirements:
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - pytest-httpserver (for serving the local test page)
  - OPENAI_API_KEY set

Run with:
  pytest tests/integration/test_pipeline_e2e.py -v -m integration
"""

import os
import sys
from pathlib import Path

try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))

import pytest
from .conftest import requires_services
from crawl_prime.pipeline import CrawlPrimePipeline

pytestmark = pytest.mark.integration

_requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping synthesis test",
)

_TEST_PAGE_HTML = """<!DOCTYPE html>
<html>
<head><title>Acme Widget Manual</title></head>
<body>
<h1>Acme Widget Safety Manual</h1>
<h2>Safety Precautions</h2>
<p>Do not operate the Acme Widget near open water.</p>
<p>Always wear protective gloves when handling the cutting blade.</p>
<p>In case of emergency, press the red STOP button immediately.</p>
</body>
</html>"""


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_ingest_then_query_returns_grounded_answer(
    test_collection_name, cleanup_test_collection, httpserver
):
    """
    Full pipeline: serve local HTML → ingest → query → grounded answer.
    """
    httpserver.expect_request("/test").respond_with_data(
        _TEST_PAGE_HTML, content_type="text/html"
    )
    test_url = httpserver.url_for("/test")

    pipeline = CrawlPrimePipeline(
        collection=test_collection_name,
        enable_synthesis=True,
        neo4j_password=os.getenv("NEO4J_PASSWORD", "replace_with_strong_neo4j_password"),
    )

    # Ingest
    report = await pipeline.ingest(test_url)
    assert report.chunks_ingested > 0, (
        f"No chunks ingested from {test_url}. Failures: {report.failed_documents}"
    )

    # Query
    result = await pipeline.query(
        "What safety precautions are required when using the Acme Widget?"
    )

    assert result.answer and len(result.answer) > 30
    assert "Retrieved content" not in result.answer

    page_keywords = {"acme", "safety", "protective", "glove", "stop",
                     "blade", "water", "emergency", "widget"}
    matched = page_keywords & set(result.answer.lower().split())
    assert matched, (
        f"Answer does not reference page content.\n"
        f"Expected one of: {page_keywords}\nAnswer: {result.answer[:400]}"
    )

    pipeline.close()
