"""
CrawlPrime real-web smoke test against worldwidecloud.io.

Exercises the full pipeline against a live public URL via CrawlPrimePipeline.

Requirements:
  - Live internet access
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - OPENAI_API_KEY set

Run with:
  pytest tests/integration/test_real_web.py -v -m real_web
"""

import os
import sys
import uuid
from pathlib import Path

try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))

import pytest
from .conftest import requires_services, _qdrant_reachable
from crawl_prime.pipeline import CrawlPrimePipeline

pytestmark = pytest.mark.real_web

_SITE_URL = "https://worldwidecloud.io"

_requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-web synthesis test",
)


@pytest.fixture(scope="module")
def real_collection_name():
    return f"crawlprime_real_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def cleanup_real_collection(real_collection_name):
    yield
    if not _qdrant_reachable():
        return
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
        cols = [c.name for c in client.get_collections().collections]
        if real_collection_name in cols:
            client.delete_collection(real_collection_name)
    except Exception:
        pass


@pytest.fixture(scope="module")
def ingested_pipeline(real_collection_name, cleanup_real_collection):
    """
    Module-scoped: crawl worldwidecloud.io once, return the pipeline for
    all query tests in this module to share.
    """
    import asyncio

    pipeline = CrawlPrimePipeline(
        collection=real_collection_name,
        enable_synthesis=True,
        neo4j_password=os.getenv("NEO4J_PASSWORD", "replace_with_strong_neo4j_password"),
    )
    report = asyncio.get_event_loop().run_until_complete(
        pipeline.ingest(_SITE_URL)
    )
    assert report.chunks_ingested > 0, (
        f"Ingestion of {_SITE_URL} produced no chunks. "
        f"Failed docs: {report.failed_documents}"
    )
    return pipeline


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_service_area_query(ingested_pipeline):
    """Answer should name London boroughs served by the site."""
    result = await ingested_pipeline.query(
        "Which London boroughs and areas does World Wide Cloud serve?"
    )
    assert result.answer and len(result.answer) > 30
    assert "Retrieved content" not in result.answer

    answer_lower = result.answer.lower()
    london_boroughs = {"westminster", "camden", "southwark", "lambeth",
                       "wandsworth", "london", "islington"}
    matched = london_boroughs & set(answer_lower.split())
    assert matched, (
        f"Answer does not mention any London boroughs.\n"
        f"Expected one of: {london_boroughs}\nAnswer: {result.answer[:400]}"
    )


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_technology_history_query(ingested_pipeline):
    """Answer should contain the stable year '2007'."""
    result = await ingested_pipeline.query(
        "When did World Wide Cloud become an Amazon developer?"
    )
    assert result.answer and len(result.answer) > 20
    assert "Retrieved content" not in result.answer
    assert "2007" in result.answer, (
        f"Answer should contain '2007'.\nAnswer: {result.answer[:400]}"
    )


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_faq_differentiator_query(ingested_pipeline):
    """Answer should reference at least 2 differentiating terms from the page."""
    result = await ingested_pipeline.query(
        "What makes World Wide Cloud's AI consulting different from other providers?"
    )
    assert result.answer and len(result.answer) > 30
    assert "Retrieved content" not in result.answer

    answer_lower = result.answer.lower()
    differentiators = {"2008", "ibm", "quantum", "amazon", "apple",
                       "automation", "small", "enterprise", "london"}
    matched = differentiators & set(answer_lower.split())
    assert len(matched) >= 2, (
        f"Answer should reference at least 2 differentiating terms.\n"
        f"Expected 2+ of: {differentiators}\nMatched: {matched}\n"
        f"Answer: {result.answer[:400]}"
    )
