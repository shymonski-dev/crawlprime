"""
CLI Entry point for CrawlPrime.

Delegates to the shared web modules in doctags_rag to avoid code duplication.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "doctags_rag"))

from src.processing.web.crawler import WebCrawler
from src.processing.web.mapper import WebDocTagsMapper


async def main():
    parser = argparse.ArgumentParser(description="CrawlPrime: Web to DocTags Crawler")
    parser.add_argument("--url", required=True, help="URL to crawl")
    parser.add_argument("--output", default="data/output", help="Output directory")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    crawler = WebCrawler()
    result = await crawler.crawl_url(args.url)
    if not result.success:
        print(f"Crawl failed: {result.error}")
        return

    mapper = WebDocTagsMapper()
    doctags = mapper.map_to_doctags(result)

    safe = "".join(x for x in result.title if x.isalnum() or x in "._- ") or "doc"
    out = Path(args.output) / f"{safe[:50]}.json"
    out.write_text(
        json.dumps(
            doctags.__dict__ if hasattr(doctags, "__dict__") else str(doctags),
            indent=2,
        )
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
