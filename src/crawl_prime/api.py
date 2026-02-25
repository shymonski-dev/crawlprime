"""
CrawlPrime FastAPI application.

Provides:
  POST /ingest           — crawl a URL in the background; returns a job_id
  GET  /ingest/{job_id}  — poll ingest job status
  POST /query            — query indexed web content
  GET  /health           — health check

Run with:
  uvicorn crawl_prime.api:app --reload --port 8001
"""

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Dev-mode fallback: if contextprime is not installed as a package,
# add the sibling doctags_rag directory to sys.path.
try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))

from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import CrawlPrimePipeline


# ── Lifespan — create pipeline at startup, close cleanly at shutdown ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise the shared CrawlPrimePipeline before the first request and
    release all database connections when the server shuts down.

    Using lifespan instead of @app.on_event("startup") / "shutdown":
      - Avoids the deprecated on_event API
      - Guarantees close() is called even if startup raises mid-way
      - Is safe across all uvicorn worker configurations
    """
    app.state.pipeline = CrawlPrimePipeline()
    yield
    app.state.pipeline.close()


app = FastAPI(
    title="CrawlPrime",
    description="Web RAG — crawl, index, and query live web content.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── In-memory job store for background ingest tasks ─────────────────────────
# Keyed by job_id (hex UUID).  In a multi-worker deployment this should be
# replaced with a shared store (Redis, database, etc.).

_jobs: Dict[str, dict] = {}


# ── Request / response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    url: str
    collection: Optional[str] = None


class IngestJobResponse(BaseModel):
    job_id: str
    status: str               # "pending" | "running" | "done" | "error"
    url: Optional[str] = None
    chunks_ingested: Optional[int] = None
    failed: Optional[List[str]] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    num_results: int


# ── Background ingest task ───────────────────────────────────────────────────

async def _run_ingest(job_id: str, url: str) -> None:
    """Run pipeline.ingest() in the background and record the result."""
    _jobs[job_id]["status"] = "running"
    try:
        pipeline: CrawlPrimePipeline = app.state.pipeline
        report = await pipeline.ingest(url)
        _jobs[job_id].update(
            status="done",
            chunks_ingested=report.chunks_ingested,
            failed=report.failed_documents,
        )
    except Exception as exc:
        _jobs[job_id].update(status="error", error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=202,
    summary="Crawl a URL and index it (async)",
)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestJobResponse:
    """
    Submit a URL for crawling and indexing.  Returns immediately with a
    ``job_id``; poll ``GET /ingest/{job_id}`` to check progress.

    Crawling is offloaded to a background task to avoid proxy timeout
    (504) on large pages.
    """
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending", "url": request.url}
    background_tasks.add_task(_run_ingest, job_id, request.url)
    return IngestJobResponse(job_id=job_id, status="pending", url=request.url)


@app.get(
    "/ingest/{job_id}",
    response_model=IngestJobResponse,
    summary="Poll ingest job status",
)
async def get_ingest_status(job_id: str) -> IngestJobResponse:
    """
    Poll the status of a background ingest job submitted via ``POST /ingest``.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return IngestJobResponse(job_id=job_id, **job)


@app.post("/query", response_model=QueryResponse, summary="Query indexed web content")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the web knowledge base and return a synthesised answer.
    """
    try:
        pipeline: CrawlPrimePipeline = app.state.pipeline
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
