"""TARA Web Search Unit - Internet Search Tools.

Provides tools for:
- Web search via DuckDuckGo
- Real-time information retrieval

Dependencies:
    pip install duckduckgo-search
"""
from __future__ import annotations

from tara.protocols import tara_tool

# =============================================================================
# Optional Dependencies (graceful fallback)
# =============================================================================

try:
    from duckduckgo_search import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False
    DDGS = None  # type: ignore


# =============================================================================
# Web Search Tools
# =============================================================================

@tara_tool(
    name="web_search",
    category="web",
    description="Search the web using DuckDuckGo. Use for current events, news, weather, facts, or anything requiring up-to-date information."
)
def web_search(query: str) -> str:
    """Search the web and return summarized results.
    
    Args:
        query: The search query to look up.
    """
    if not _HAS_DDGS:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    
    try:
        # Initialize DuckDuckGo search
        ddgs = DDGS()
        
        # Get top 3 results
        results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return f"No results found for: {query}"
        
        # Format results
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            # Truncate body if too long
            if len(body) > 200:
                body = body[:200] + "..."
            formatted.append(f"{i}. {title}\n   {body}")
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        return f"Search error: {e}"


@tara_tool(
    name="web_news",
    category="web",
    description="Get recent news headlines on a topic. Use for news queries."
)
def web_news(topic: str) -> str:
    """Get recent news about a topic.
    
    Args:
        topic: The news topic to search for.
    """
    if not _HAS_DDGS:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    
    try:
        ddgs = DDGS()
        
        # Get news results
        results = list(ddgs.news(topic, max_results=3))
        
        if not results:
            return f"No news found for: {topic}"
        
        # Format results
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            source = result.get("source", "Unknown")
            date = result.get("date", "")
            formatted.append(f"{i}. [{source}] {title}" + (f" ({date})" if date else ""))
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"News search error: {e}"
