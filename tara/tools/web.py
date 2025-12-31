"""Web search tools for TARA.

Provides DuckDuckGo search for real-time information.
"""
from __future__ import annotations

from typing import Any, Type

# Try to import LangChain DuckDuckGo tool
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    _HAS_DUCKDUCKGO = True
except ImportError:
    _HAS_DUCKDUCKGO = False
    DuckDuckGoSearchRun = None  # type: ignore

# Try to import LangChain base tools
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    BaseTool = object  # type: ignore
    BaseModel = object  # type: ignore


class WebSearchInput(BaseModel):
    """Input for WebSearchTool."""
    query: str = Field(description="The search query to look up on the web")


class WebSearchTool(BaseTool):
    """Tool to search the web using DuckDuckGo.
    
    Use this for real-time information like:
    - Current events and news
    - Weather information
    - Stock prices
    - Sports scores
    - Facts that may have changed recently
    """
    
    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo for real-time information. "
        "Use for current events, news, weather, facts, or anything that requires up-to-date information. "
        "Returns a summary of search results."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    
    _search_tool: Any = None
    
    def __init__(self, **kwargs: Any):
        """Initialize the web search tool."""
        super().__init__(**kwargs)
        if _HAS_DUCKDUCKGO:
            self._search_tool = DuckDuckGoSearchRun()
    
    def _run(self, query: str) -> str:
        """Run web search."""
        if not _HAS_DUCKDUCKGO:
            return "Error: langchain-community not installed. Run: pip install langchain-community duckduckgo-search"
        
        if not self._search_tool:
            self._search_tool = DuckDuckGoSearchRun()
        
        try:
            result = self._search_tool.run(query)
            return result if result else "No results found."
        except Exception as exc:
            return f"Search error: {exc}"
    
    async def _arun(self, query: str) -> str:
        """Async version - just calls sync for now."""
        return self._run(query)
