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

_DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
if str(_DOCTAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCTAGS_ROOT))

from src.agents.planning_agent import PlanStep, StepType, ExecutionMode

_URL_RE = re.compile(r'https?://\S+')


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

    web_step_id = f"step_{step_counter_start}"

    web_step = PlanStep(
        step_id=web_step_id,
        step_type=WebStepType.WEB_INGESTION,  # type: ignore[arg-type]
        description=f"Ingest web content from {url_match.group()}",
        parameters={"url": url_match.group()},
        dependencies=[],
        execution_mode=ExecutionMode.SEQUENTIAL,
    )

    # Renumber existing steps by +1 and add dependency on web ingestion
    # for all RETRIEVAL steps that have no dependencies yet.
    renumbered = []
    for step in steps:
        old_id = step.step_id
        # Bump the numeric suffix
        num = int(old_id.split("_")[-1]) + 1
        new_id = f"step_{num}"
        deps = step.dependencies
        if step.step_type == StepType.RETRIEVAL and not deps:
            deps = [web_step_id]
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
