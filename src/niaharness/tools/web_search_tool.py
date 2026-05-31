"""Web search tool with multi-provider support."""

from __future__ import annotations

import asyncio
import html
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class WebSearchToolInput(BaseModel):
    """Arguments for a web search."""

    query: str = Field(description="The search query to use", min_length=2)
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results")
    allowed_domains: list[str] | None = Field(
        default=None,
        description="Only include search results from these domains",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="Never include search results from these domains",
    )


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str = ""


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        """Perform a search and return results."""


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo HTML search provider."""

    name = "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        endpoint = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
                response = await client.get(
                    endpoint,
                    params={"q": query},
                    headers={"User-Agent": "NiaHarness/0.1"},
                )
                response.raise_for_status()
        except httpx.HTTPError:
            return []

        results = self._parse_results(response.text, limit=max_results)

        # Apply domain filters
        if allowed_domains:
            results = [r for r in results if any(d in r.url for d in allowed_domains)]
        if blocked_domains:
            results = [r for r in results if not any(d in r.url for d in blocked_domains)]

        return results[:max_results]

    def _parse_results(self, body: str, *, limit: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML search results."""
        snippets = [
            self._clean_html(match.group("snippet"))
            for match in re.finditer(
                r'<(?:a|div|span)[^>]+class="[^"]*(?:result__snippet|result-snippet)[^"]*"[^>]*>(?P<snippet>.*?)</(?:a|div|span)>',
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]

        results: list[SearchResult] = []
        anchor_matches = re.finditer(
            r"<a(?P<attrs>[^>]+)>(?P<title>.*?)</a>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for index, match in enumerate(anchor_matches):
            attrs = match.group("attrs")
            class_match = re.search(
                r'class="(?P<class>[^"]+)"', attrs, flags=re.IGNORECASE
            )
            if class_match is None:
                continue
            class_names = class_match.group("class")
            if "result__a" not in class_names and "result-link" not in class_names:
                continue
            href_match = re.search(
                r'href="(?P<href>[^"]+)"', attrs, flags=re.IGNORECASE
            )
            if href_match is None:
                continue
            title = self._clean_html(match.group("title"))
            url = self._normalize_url(href_match.group("href"))
            snippet = snippets[index] if index < len(snippets) else ""
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _normalize_url(raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(target) if target else raw_url
        return raw_url

    @staticmethod
    def _clean_html(fragment: str) -> str:
        text = re.sub(r"(?s)<[^>]+>", " ", fragment)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""

    name = "brave"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            return []

        params: dict[str, Any] = {
            "q": query,
            "count": max_results,
        }
        if allowed_domains:
            params["search_lang"] = "en"
            params["freshness"] = "pw"  # Past week

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params=params,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "X-Subscription-Token": api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                )
            )

        # Apply domain filters
        if blocked_domains:
            results = [r for r in results if not any(d in r.url for d in blocked_domains)]

        return results[:max_results]


class TavilyProvider(SearchProvider):
    """Tavily Search API provider."""

    name = "tavily"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> list[SearchResult]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return []

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )

        return results


class WebSearchTool(BaseTool):
    """Run a web search using multiple providers with fallback support."""

    name = "web_search"
    description = "Search the web and return compact top results with titles, URLs, and snippets."
    input_model = WebSearchToolInput

    def __init__(self) -> None:
        self._providers = self._init_providers()

    def _init_providers(self) -> list[SearchProvider]:
        """Initialize available search providers based on environment."""
        providers: list[SearchProvider] = []

        # DuckDuckGo is always available (fallback)
        providers.append(DuckDuckGoProvider())

        # Brave Search
        if os.environ.get("BRAVE_API_KEY"):
            providers.append(BraveSearchProvider())

        # Tavily
        if os.environ.get("TAVILY_API_KEY"):
            providers.append(TavilyProvider())

        return providers

    def is_read_only(self, arguments: WebSearchToolInput) -> bool:
        return True

    def get_user_facing_name(self, arguments: WebSearchToolInput | None = None) -> str:
        return "Web Search"

    async def execute(
        self,
        arguments: WebSearchToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        start_time = time.monotonic()

        # Try providers in order with fallback
        last_error: str | None = None
        for provider in self._providers:
            try:
                results = await provider.search(
                    query=arguments.query,
                    max_results=arguments.max_results,
                    allowed_domains=arguments.allowed_domains,
                    blocked_domains=arguments.blocked_domains,
                )
                if results:
                    duration = time.monotonic() - start_time
                    return self._format_results(
                        results, arguments.query, duration, provider.name
                    )
            except Exception as e:
                last_error = str(e)
                continue

        # All providers failed
        error_msg = "No search results found."
        if last_error:
            error_msg += f" Last error: {last_error}"

        return ToolResult(output=error_msg, is_error=True)

    def _format_results(
        self,
        results: list[SearchResult],
        query: str,
        duration: float,
        provider_name: str,
    ) -> ToolResult:
        """Format search results into a ToolResult."""
        lines = [f'Search results for: "{query}"']
        lines.append(f"(via {provider_name} in {duration:.1f}s)")
        lines.append("")

        for i, result in enumerate(results, start=1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   URL: {result.url}")
            if result.snippet:
                lines.append(f"   {result.snippet}")
            lines.append("")

        return ToolResult(
            output="\n".join(lines),
            metadata={
                "query": query,
                "num_results": len(results),
                "duration_seconds": duration,
                "provider": provider_name,
            },
        )
