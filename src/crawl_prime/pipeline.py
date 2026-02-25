"""
CrawlPrimePipeline — the central orchestrator for CrawlPrime web RAG.

Composes ContextPrime shared utilities:
  WebIngestionPipeline  — crawl + DocTags + chunk + embed + store
  HybridRetriever       — vector + graph + lexical retrieval with RRF fusion
  AgenticPipeline       — multi-agent query processing with LLM synthesis

Usage::

    import asyncio
    from crawl_prime.pipeline import CrawlPrimePipeline

    async def main():
        with CrawlPrimePipeline(collection="my_web_kb") as cp:
            await cp.ingest("https://example.com")
            result = await cp.query("What services does the site offer?")
            print(result.answer)
        # Neo4j and Qdrant connections closed automatically on exit

    asyncio.run(main())
"""

import sys
from pathlib import Path

# Dev-mode fallback: if contextprime is not installed as a package,
# add the sibling doctags_rag directory to sys.path.
try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))

from typing import Optional, Any
from loguru import logger

from contextprime.pipelines.web_ingestion import WebIngestionPipeline
from contextprime.pipelines.document_ingestion import IngestionReport as WebIngestionReport
from contextprime.pipelines.document_ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
from contextprime.retrieval.hybrid_retriever import HybridRetriever
from contextprime.retrieval.qdrant_manager import QdrantManager
from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticResult
from contextprime.core.config import QdrantConfig, Neo4jConfig
from contextprime.knowledge_graph.neo4j_manager import Neo4jManager
from contextprime.knowledge_graph.graph_queries import GraphQueryInterface
from contextprime.knowledge_graph.graph_ingestor import GraphIngestionManager


class CrawlPrimePipeline:
    """
    End-to-end web RAG pipeline.

    Wraps ContextPrime's shared utilities into a single, easy-to-use
    interface for crawling, indexing, and querying web content.
    """

    def __init__(
        self,
        collection: str = "crawlprime_default",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        neo4j_host: str = "localhost",
        neo4j_port: int = 7687,
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        vector_weight: float = 0.6,
        graph_weight: float = 0.3,
        lexical_weight: float = 0.1,
        enable_synthesis: bool = True,
        storage_path: Optional[Path] = None,
        raptor_pipeline: Optional[Any] = None,
        community_pipeline: Optional[Any] = None,
    ):
        """
        Args:
            collection:         Qdrant collection name for web content.
            qdrant_host:        Qdrant host.
            qdrant_port:        Qdrant port.
            neo4j_host:         Neo4j host (optional; graph retrieval disabled on failure).
            neo4j_port:         Neo4j bolt port.
            neo4j_user:         Neo4j username.
            neo4j_password:     Neo4j password.
            vector_weight:      RRF weight for vector results (default 0.6).
            graph_weight:       RRF weight for graph results (default 0.3; drops to 0.0 if Neo4j unavailable).
            lexical_weight:     RRF weight for BM25 lexical results (default 0.1).
            enable_synthesis:   Enable LLM answer synthesis (requires OPENAI_API_KEY).
            storage_path:       Directory for RL Q-table and memory storage.
            raptor_pipeline:    Optional RAPTOR pipeline forwarded to AgenticPipeline.
            community_pipeline: Optional community detection pipeline forwarded to AgenticPipeline.
        """
        self.collection = collection

        # Persistent storage path for RL and memory
        self._storage_path = storage_path or Path("data/crawlprime")
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Shared Qdrant config — used for both ingestion and retrieval so that
        # the explicit qdrant_host/qdrant_port params are always honoured
        # (DocumentIngestionPipeline would otherwise lazily create QdrantManager()
        # from get_settings() which may point to a different host).
        qdrant_cfg = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection,
        )

        # Neo4j (graceful fallback if unreachable).
        # Two separate Neo4jManager instances are created from the same config:
        # one for retrieval (self._neo4j) and one for ingestion (self._graph_ingestor).
        # This avoids shared-close issues while ensuring both use the caller's params.
        self._neo4j = None
        self._graph_queries = None
        self._graph_ingestor = None
        _graph_weight = 0.0
        try:
            neo4j_cfg = Neo4jConfig(
                uri=f"bolt://{neo4j_host}:{neo4j_port}",
                username=neo4j_user,
                password=neo4j_password,
            )
            self._neo4j = Neo4jManager(config=neo4j_cfg)
            self._graph_queries = GraphQueryInterface(neo4j_manager=self._neo4j)
            self._graph_ingestor = GraphIngestionManager(
                neo4j_manager=Neo4jManager(config=neo4j_cfg)
            )
            _graph_weight = graph_weight
        except Exception as err:
            logger.warning("Neo4j unavailable, graph retrieval disabled: %s", err)

        # Ingestion pipeline — pass explicit managers so ingestion uses the
        # same Qdrant host and Neo4j credentials as the caller specified.
        ingestion_cfg = DocumentIngestionConfig(
            qdrant_collection=collection,
            create_qdrant_collection=True,
        )
        self._storage = DocumentIngestionPipeline(
            config=ingestion_cfg,
            qdrant_manager=QdrantManager(config=qdrant_cfg),
            graph_ingestor=self._graph_ingestor,
        )
        self._web_ingestion = WebIngestionPipeline(
            document_ingestion_pipeline=self._storage,
        )

        # Retrieval pipeline — store the Qdrant manager so close() can release it
        # (HybridRetriever.close() skips Qdrant when _owns_qdrant=False).
        self._retrieval_qdrant = QdrantManager(config=qdrant_cfg)
        self._retriever = HybridRetriever(
            qdrant_manager=self._retrieval_qdrant,
            neo4j_manager=self._neo4j,
            vector_weight=vector_weight,
            graph_weight=_graph_weight,
        )

        # BM25 lexical retrieval
        if lexical_weight > 0:
            self._retriever.lexical_enabled = True
            self._retriever.lexical_weight = lexical_weight

        # Agentic query pipeline
        self._agentic = AgenticPipeline(
            retrieval_pipeline=self._retriever,
            graph_queries=self._graph_queries,
            raptor_pipeline=raptor_pipeline,
            community_pipeline=community_pipeline,
            enable_synthesis=enable_synthesis,
            storage_path=self._storage_path,
        )

        logger.info(
            "CrawlPrimePipeline initialised (collection=%s, synthesis=%s, "
            "graph=%s, lexical=%s)",
            collection,
            enable_synthesis,
            self._neo4j is not None,
            lexical_weight > 0,
        )

    async def ingest(self, url: str) -> WebIngestionReport:
        """
        Crawl a URL and index its content into Qdrant.

        Args:
            url: The URL to crawl and ingest.

        Returns:
            WebIngestionReport with chunks_ingested count and any failures.
        """
        logger.info("CrawlPrime ingesting: %s", url)
        report = await self._web_ingestion.ingest_url(url)
        logger.info(
            "Ingestion complete — %d chunks stored, %d failures",
            report.chunks_ingested,
            len(report.failed_documents),
        )
        return report

    async def query(
        self,
        text: str,
        max_iterations: int = 2,
        min_quality_threshold: float = 0.5,
    ) -> AgenticResult:
        """
        Query the indexed web content.

        Args:
            text:                  Natural language question.
            max_iterations:        Max agentic improvement iterations.
            min_quality_threshold: Minimum quality score to accept answer.

        Returns:
            AgenticResult with .answer and .results.
        """
        return await self._agentic.process_query(
            text,
            max_iterations=max_iterations,
            min_quality_threshold=min_quality_threshold,
        )

    def __enter__(self) -> "CrawlPrimePipeline":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release resources."""
        # Close ingestion pipeline (closes its own QdrantManager; leaves
        # graph_ingestor open because _owns_graph_ingestor=False).
        try:
            self._storage.close()
        except Exception:
            pass
        # Close retrieval Qdrant (not closed by HybridRetriever because
        # _owns_qdrant=False when an explicit manager is passed).
        for qdrant in (self._retrieval_qdrant,):
            try:
                if qdrant is not None:
                    qdrant.close()
            except Exception:
                pass
        # Close both Neo4j connections (retrieval + ingestion).
        for neo4j in (self._neo4j, getattr(self._graph_ingestor, "_neo4j_manager", None)):
            try:
                if neo4j is not None:
                    neo4j.close()
            except Exception:
                pass
