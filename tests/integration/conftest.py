"""
Shared fixtures for CrawlPrime integration tests.

Requires:
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - playwright chromium installed
"""

import sys
import uuid
from pathlib import Path

_DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
if str(_DOCTAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCTAGS_ROOT))

import pytest


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
        client.get_collections()
        return True
    except Exception:
        return False


def _neo4j_reachable() -> bool:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


requires_services = pytest.mark.skipif(
    not (_qdrant_reachable() and _neo4j_reachable()),
    reason="Qdrant or Neo4j not reachable — skipping integration test",
)


@pytest.fixture(scope="module")
def test_collection_name():
    return f"crawlprime_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def cleanup_test_collection(test_collection_name):
    yield
    if not _qdrant_reachable():
        return
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
        cols = [c.name for c in client.get_collections().collections]
        if test_collection_name in cols:
            client.delete_collection(test_collection_name)
    except Exception:
        pass
