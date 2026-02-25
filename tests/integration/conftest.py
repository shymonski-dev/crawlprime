"""
Shared fixtures for CrawlPrime integration tests.

Requires:
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - playwright chromium installed
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

# Load .env so NEO4J_PASSWORD, OPENAI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    _ENV = Path(__file__).resolve().parents[3] / "doctags_rag" / ".env"
    if _ENV.exists():
        load_dotenv(_ENV, override=False)
except ImportError:
    pass

# Force host-side overrides BEFORE get_settings() caches anything.
# The .env has QDRANT_HOST=qdrant (Docker service name); tests run against localhost.
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"

# Reset cached settings singleton so the next call picks up the overrides.
try:
    from contextprime.core.config import reset_settings
    reset_settings()
except Exception:
    pass

import pytest

_NEO4J_URI = "bolt://localhost:7687"
_NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
_NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "replace_with_strong_neo4j_password")


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
        driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
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
