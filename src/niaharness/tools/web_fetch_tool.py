"""Fetch and extract content from web pages with enhanced content processing."""

from __future__ import annotations

import re
import time

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class WebFetchToolInput(BaseModel):
    """Arguments for fetching one web page."""

    url: str = Field(description="The URL to fetch content from")
    prompt: str = Field(
        default="Extract and summarize the main content from this page.",
        description="The prompt to run on the fetched content",
    )
    max_chars: int = Field(
        default=12000,
        ge=500,
        le=50000,
        description="Maximum characters to return",
    )
    extract_mode: str = Field(
        default="text",
        description='Extraction mode: "text", "markdown", or "html"',
    )


class WebFetchTool(BaseTool):
    """Fetch one web page and extract content with enhanced processing."""

    name = "web_fetch"
    description = "Fetch one web page and extract readable content."
    input_model = WebFetchToolInput

    async def execute(
        self, arguments: WebFetchToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        start = time.monotonic()

        # Validate URL
        try:
            parsed_url = arguments.url
            if not parsed_url.startswith(("http://", "https://")):
                return ToolResult(
                    output=f"Invalid URL: must start with http:// or https://",
                    is_error=True,
                )
        except Exception as e:
            return ToolResult(
                output=f"Invalid URL: {e}",
                is_error=True,
            )

        # Fetch the URL
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=20.0,
                max_redirects=5,
            ) as client:
                response = await client.get(
                    arguments.url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; NiaHarness/0.1; "
                            "+https://github.com/niaharness)"
                        )
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return ToolResult(
                output=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                is_error=True,
            )
        except httpx.RequestError as e:
            return ToolResult(
                output=f"Failed to fetch URL: {e}",
                is_error=True,
            )

        # Get content type and body
        content_type = response.headers.get("content-type", "")
        body = response.text

        # Handle redirects to different host
        if str(response.url) != arguments.url:
            original_host = _extract_host(arguments.url)
            redirect_host = _extract_host(str(response.url))
            if original_host != redirect_host:
                redirect_info = (
                    f"REDIRECT DETECTED: The URL redirects to a different host.\n\n"
                    f"Original URL: {arguments.url}\n"
                    f"Redirect URL: {response.url}\n\n"
                    f"To complete your request, use WebFetch with the redirect URL."
                )
                return ToolResult(
                    output=redirect_info,
                    metadata={
                        "redirected": True,
                        "original_url": arguments.url,
                        "redirect_url": str(response.url),
                    },
                )

        # Extract content based on type
        if "html" in content_type:
            extracted = _extract_html_content(body, arguments.extract_mode)
        elif "json" in content_type:
            extracted = _extract_json_content(body)
        elif "xml" in content_type or "rss" in content_type:
            extracted = _extract_xml_content(body)
        else:
            extracted = body

        # Apply character limit
        if len(extracted) > arguments.max_chars:
            extracted = extracted[: arguments.max_chars].rstrip()
            extracted += "\n\n...[content truncated]..."

        duration = time.monotonic() - start

        # Build output
        output = (
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n"
            f"Content-Type: {content_type}\n"
            f"Duration: {duration:.1f}s\n\n"
            f"{extracted}"
        )

        return ToolResult(
            output=output,
            metadata={
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(extracted),
                "duration_ms": int(duration * 1000),
            },
        )


def _extract_host(url: str) -> str:
    """Extract hostname from URL."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _extract_html_content(html_text: str, mode: str = "text") -> str:
    """Extract content from HTML."""
    if mode == "html":
        return html_text

    # Remove script and style elements
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_text)

    # Remove HTML tags
    text = re.sub(r"(?s)<[^>]+>", " ", text)

    # Decode HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Clean up whitespace
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_json_content(json_text: str) -> str:
    """Extract readable content from JSON."""
    import json

    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            # Try common content fields
            for field in ["content", "text", "body", "description", "summary"]:
                if field in data and isinstance(data[field], str):
                    return data[field]
            # Pretty print the JSON
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif isinstance(data, list):
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            return str(data)
    except json.JSONDecodeError:
        return json_text


def _extract_xml_content(xml_text: str) -> str:
    """Extract readable content from XML/RSS."""
    # Simple XML to text conversion
    text = re.sub(r"<[^>]+>", " ", xml_text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
