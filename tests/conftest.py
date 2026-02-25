"""
CrawlPrime test configuration.

Stubs out ContextPrime's heavy optional document-processing dependencies
(cv2, paddleocr, etc.) so that the shared web utilities can be imported
from doctags_rag without requiring the full ContextPrime installation.

CrawlPrime only needs the web processing modules from ContextPrime.
The document parsing dependencies are not required.
"""

import sys
from types import ModuleType
from pathlib import Path

# ── Stub heavy ContextPrime-only dependencies ──────────────────────────────
_STUBS = [
    "cv2",
    "paddleocr",
    "paddlepaddle",
    "pdfplumber",
    "docx",
    "python_docx",
    "docling",
    "docling_core",
    "magic",
    "fitz",          # PyMuPDF
    # PIL/Pillow is NOT stubbed — crawl4ai depends on it and it is installed
]
for _name in _STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = ModuleType(_name)

# ── Add doctags_rag to sys.path (dev-mode fallback) ───────────────────────
try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[2] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))
