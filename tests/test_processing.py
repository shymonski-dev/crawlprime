"""
Unit tests for CrawlPrime processing layer.

Uses ContextPrime's shared WebDocTagsMapper and WebCrawlResult directly,
since CrawlPrime no longer maintains its own duplicate mapper/engine.
"""

import pytest
from src.processing.web.mapper import WebDocTagsMapper
from src.processing.web.crawler import WebCrawlResult


class TestProcessing:
    def test_markdown_mapping(self):
        """Test converting markdown to DocTags structure via shared WebDocTagsMapper."""

        mock_result = WebCrawlResult(
            url="https://example.com",
            title="Test Page",
            markdown="# Main Title\n\n## Section 1\n\nSome text.\n\n### Subsection A\n\n- List item",
            html="<html>...</html>",
            crawled_at="2024-01-01",
            links=[],
            metadata={},
            success=True,
        )

        mapper = WebDocTagsMapper()
        doctags = mapper.map_to_doctags(mock_result)

        assert doctags.title == "Test Page"
        assert len(doctags.tags) >= 4  # title + section + paragraph + list minimum

    def test_failed_crawl_result(self):
        """A failed WebCrawlResult should be detectable before mapping."""
        result = WebCrawlResult(
            url="https://example.com",
            title="",
            markdown="",
            html="",
            crawled_at="",
            links=[],
            metadata={},
            success=False,
            error="Connection refused",
        )
        assert not result.success
        assert result.error == "Connection refused"

    def test_empty_markdown_produces_minimal_doctags(self):
        """Empty markdown should still produce a valid DocTagsDocument with a title."""
        mock_result = WebCrawlResult(
            url="https://example.com",
            title="Empty Page",
            markdown="",
            html="",
            crawled_at="2024-01-01",
            links=[],
            metadata={},
            success=True,
        )
        mapper = WebDocTagsMapper()
        doctags = mapper.map_to_doctags(mock_result)
        assert doctags.title == "Empty Page"
