"""
CrawlPrime FastAPI application.

Provides:
  POST /ingest  — crawl a URL and store in Qdrant
  POST /query   — query indexed web content

Run with:
  uvicorn src.crawl_prime.api:app --reload --port 8001
"""

import sys
from pathlib import Path

_DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
if str(_DOCTAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCTAGS_ROOT))

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import CrawlPrimePipeline

app = FastAPI(
    title="CrawlPrime",
    description="Web RAG — crawl, index, and query live web content.",
    version="1.0.0",
)

# Single shared pipeline instance (lazy — created on first request)
_pipeline: Optional[CrawlPrimePipeline] = None


def _get_pipeline() -> CrawlPrimePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CrawlPrimePipeline()
    return _pipeline


# ── Request / response models ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str
    collection: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    url: str
    chunks_ingested: int
    failed: List[str]


class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    num_results: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, summary="Crawl a URL and index it")
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Crawl the given URL, convert to DocTags, chunk, embed, and store in Qdrant.
    """
    try:
        pipeline = _get_pipeline()
        report = await pipeline.ingest(request.url)
        return IngestResponse(
            status="ok",
            url=request.url,
            chunks_ingested=report.chunks_ingested,
            failed=report.failed_documents,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse, summary="Query indexed web content")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the web knowledge base and return a synthesised answer.
    """
    try:
        pipeline = _get_pipeline()
        result = await pipeline.query(request.query)
        return QueryResponse(
            answer=result.answer,
            num_results=len(result.results),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}
