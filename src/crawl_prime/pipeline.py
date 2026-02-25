"""
CrawlPrimePipeline — the central orchestrator for CrawlPrime web RAG.

Composes ContextPrime shared utilities:
  WebIngestionPipeline  — crawl + DocTags + chunk + embed + store
  HybridRetriever       — vector + graph + lexical retrieval with RRF fusion
  AgenticPipeline       — multi-agent query processing with LLM synthesis

Usage::

    import asyncio
    from src.crawl_prime.pipeline import CrawlPrimePipeline

    async def main():
        cp = CrawlPrimePipeline(collection="my_web_kb")
        await cp.ingest("https://example.com")
        result = await cp.query("What services does the site offer?")
        print(result.answer)

    asyncio.run(main())
"""

import sys
from pathlib import Path

# Ensure ContextPrime is importable
_DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
if str(_DOCTAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCTAGS_ROOT))

from typing import Optional, Any
from loguru import logger

from src.pipelines.web_ingestion import WebIngestionPipeline
from src.pipelines.document_ingestion import IngestionReport as WebIngestionReport
from src.pipelines.document_ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_manager import QdrantManager
from src.agents.agentic_pipeline import AgenticPipeline, AgenticResult
from src.core.config import QdrantConfig, Neo4jConfig
from src.knowledge_graph.neo4j_manager import Neo4jManager
from src.knowledge_graph.graph_queries import GraphQueryInterface


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

        # Ingestion pipeline
        ingestion_cfg = DocumentIngestionConfig(
            qdrant_collection=collection,
            create_qdrant_collection=True,
        )
        self._storage = DocumentIngestionPipeline(config=ingestion_cfg)
        self._web_ingestion = WebIngestionPipeline(
            document_ingestion_pipeline=self._storage,
        )

        # Neo4j (graceful fallback if unreachable)
        try:
            neo4j_cfg = Neo4jConfig(
                uri=f"bolt://{neo4j_host}:{neo4j_port}",
                username=neo4j_user,
                password=neo4j_password,
            )
            self._neo4j = Neo4jManager(config=neo4j_cfg)
            self._graph_queries = GraphQueryInterface(neo4j_manager=self._neo4j)
            _graph_weight = graph_weight
        except Exception as err:
            logger.warning("Neo4j unavailable, graph retrieval disabled: %s", err)
            self._neo4j = None
            self._graph_queries = None
            _graph_weight = 0.0

        # Retrieval pipeline
        qdrant_cfg = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection,
        )
        self._retriever = HybridRetriever(
            qdrant_manager=QdrantManager(config=qdrant_cfg),
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

    def close(self) -> None:
        """Release resources."""
        try:
            self._storage.close()
        except Exception:
            pass
        if self._neo4j:
            self._neo4j.close()
