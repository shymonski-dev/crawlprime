"""
CrawlPrime URL-aware planner.

Extends ContextPrime's document planning with web-specific step generation.
When a query contains a URL, a WEB_INGESTION step is prepended and retrieval
steps are made to depend on it.

This module owns:
  - WEB_INGESTION step type (a CrawlPrime concept)
  - URL detection regex
  - Dependency ordering: WEB_INGESTION → RETRIEVAL → downstream steps
"""

import sys
import re
from pathlib import Path
from enum import Enum
from typing import Optional

# Dev-mode fallback: if contextprime is not installed as a package,
# add the sibling doctags_rag directory to sys.path.
try:
    import contextprime  # noqa: F401 — check if installed
except ImportError:
    _DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
    if _DOCTAGS_ROOT.exists() and str(_DOCTAGS_ROOT) not in sys.path:
        sys.path.insert(0, str(_DOCTAGS_ROOT))

from contextprime.agents.planning_agent import PlanStep, StepType, ExecutionMode

# First pass: capture everything that looks like a URL (any non-whitespace).
_URL_RE = re.compile(r'https?://\S+')
# Second pass: strip trailing punctuation that is almost never part of a URL
# but frequently terminates one in prose ("see https://example.com.").
# Periods, commas and closing brackets are stripped from the right only,
# so mid-URL punctuation (domain dots, path slashes) is preserved.
_TRAILING_PUNCT_RE = re.compile(r'[.,;:!?)\]\'"<>]+$')


class WebStepType(Enum):
    """Step types that extend ContextPrime's StepType for CrawlPrime."""
    WEB_INGESTION = "web_ingestion"


def prepend_web_ingestion_step(
    query: str,
    steps: list,
    step_counter_start: int = 0,
) -> tuple:
    """
    If the query contains a URL, prepend a WEB_INGESTION step and update
    retrieval step dependencies so they wait for ingestion to complete.

    Args:
        query:               The user query, potentially containing a URL.
        steps:               Existing list of PlanStep objects (RETRIEVAL etc.)
        step_counter_start:  The step counter value before any steps were added.

    Returns:
        (steps, web_step_id) where web_step_id is None if no URL was detected.
    """
    url_match = _URL_RE.search(query)
    if not url_match:
        return steps, None

    # Strip trailing punctuation that commonly follows URLs in prose.
    url = _TRAILING_PUNCT_RE.sub("", url_match.group())

    web_step_id = f"step_{step_counter_start}"

    web_step = PlanStep(
        step_id=web_step_id,
        step_type=WebStepType.WEB_INGESTION,  # type: ignore[arg-type]
        description=f"Ingest web content from {url}",
        parameters={"url": url},
        dependencies=[],
        execution_mode=ExecutionMode.SEQUENTIAL,
    )

    def _bump(step_id: str) -> str:
        """Increment the numeric suffix of a step_id by 1."""
        return f"step_{int(step_id.split('_')[-1]) + 1}"

    # Renumber existing steps by +1 and remap ALL dependency references so
    # the DAG remains consistent.  RETRIEVAL steps that had no dependencies
    # gain a new dependency on the web-ingestion step.
    renumbered = []
    for step in steps:
        new_id = _bump(step.step_id)
        if step.step_type == StepType.RETRIEVAL and not step.dependencies:
            deps = [web_step_id]
        else:
            # Remap every dependency to its new (bumped) ID.
            deps = [_bump(d) for d in step.dependencies]
        renumbered.append(PlanStep(
            step_id=new_id,
            step_type=step.step_type,
            description=step.description,
            parameters=step.parameters,
            dependencies=deps,
            execution_mode=step.execution_mode,
            estimated_time_ms=step.estimated_time_ms,
            estimated_cost=step.estimated_cost,
            priority=step.priority,
            required=step.required,
            metadata=step.metadata,
        ))

    return [web_step] + renumbered, web_step_id
