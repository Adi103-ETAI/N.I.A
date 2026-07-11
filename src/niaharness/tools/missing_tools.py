"""P1 Missing tools — web_extract, video_analyze, video_generate, clarify,
project_create, project_list, project_switch, memory (batched ops), search_files,
read_terminal, close_terminal, x_search, text_to_speech.

These are the high-value missing tools from AUDIT.md. Each is a standalone
BaseTool subclass that can be registered in the tool registry.

Most are "thin" tools that wrap existing NIA functionality or provide
stubs that return helpful messages when the underlying dependency (e.g.
FFmpeg for video, a specific API key for X/Twitter search) isn't available.
The agent gets a clear error instead of an "unknown tool" response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from niaharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. web_extract — extract page content as markdown
# ---------------------------------------------------------------------------


class WebExtractInput(BaseModel):
    urls: list[str] = Field(description="List of URLs to extract content from")
    format: str = Field(
        default="markdown", description='Output format: "markdown", "text", or "html"'
    )
    max_chars: int = Field(
        default=12000, ge=500, le=50000,
        description="Maximum characters per URL",
    )


class WebExtractTool(BaseTool):
    """Extract readable content from web pages as markdown."""

    name = "web_extract"
    description = (
        "Extract readable content from one or more web pages. "
        "Returns the content as markdown, text, or HTML. "
        "Useful for reading articles, documentation, or any web page."
    )
    input_model = WebExtractInput

    async def execute(
        self, arguments: WebExtractInput, context: ToolExecutionContext
    ) -> ToolResult:
        results: list[str] = []
        for url in arguments.urls[:5]:  # cap at 5 URLs
            if not url.startswith(("http://", "https://")):
                results.append(f"[{url}] Error: URL must start with http:// or https://")
                continue
            try:
                async with httpx.AsyncClient(
                    timeout=30, follow_redirects=True
                ) as client:
                    response = await client.get(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; NIA-Bot/1.0)"},
                    )
                    response.raise_for_status()
                    html = response.text

                content = self._extract_content(html, arguments.format, arguments.max_chars)
                results.append(f"[{url}]\n{content}")
            except Exception as exc:
                results.append(f"[{url}] Error: {exc}")

        return ToolResult(output="\n\n---\n\n".join(results))

    def _extract_content(self, html: str, format: str, max_chars: int) -> str:
        """Extract readable content from HTML."""
        # Strip script/style tags.
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)

        if format == "html":
            return html[:max_chars]

        # Extract text from HTML.
        # Replace block elements with newlines.
        text = re.sub(r"<(?:p|div|br|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Strip remaining tags.
        text = re.sub(r"<[^>]+>", "", text)
        # Decode HTML entities.
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
        # Collapse whitespace.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        if format == "markdown":
            # Basic markdown conversion: find headings.
            text = re.sub(r"^#\s*(.+)$", r"# \1", text, flags=re.MULTILINE)

        return text.strip()[:max_chars]

    def is_read_only(self, arguments: WebExtractInput) -> bool:
        return True


# ---------------------------------------------------------------------------
# 2. video_analyze — analyze video content
# ---------------------------------------------------------------------------


class VideoAnalyzeInput(BaseModel):
    video_path: str = Field(description="Path to the video file to analyze")
    question: str = Field(
        default="Describe what happens in this video.",
        description="Question to answer about the video",
    )
    max_frames: int = Field(
        default=10, ge=1, le=30,
        description="Maximum number of frames to extract for analysis",
    )


class VideoAnalyzeTool(BaseTool):
    """Analyze video content by extracting frames and sending to a vision model."""

    name = "video_analyze"
    description = (
        "Analyze a video file by extracting key frames and answering "
        "questions about the content. Requires FFmpeg for frame extraction."
    )
    input_model = VideoAnalyzeInput

    async def execute(
        self, arguments: VideoAnalyzeInput, context: ToolExecutionContext
    ) -> ToolResult:
        video_path = Path(arguments.video_path)
        if not video_path.is_absolute():
            video_path = context.cwd / video_path
        if not video_path.exists():
            return ToolResult(
                output=f"Video file not found: {video_path}", is_error=True
            )

        # Check FFmpeg.
        if not shutil.which("ffmpeg"):
            return ToolResult(
                output="FFmpeg is required for video analysis but was not found. "
                "Install it with: apt install ffmpeg (or brew install ffmpeg)",
                is_error=True,
            )

        # Extract frames.
        frames_dir = video_path.parent / f".{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            # Get video duration.
            duration_result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            if duration_result.returncode != 0:
                return ToolResult(
                    output=f"Could not determine video duration: {duration_result.stderr}",
                    is_error=True,
                )
            duration = float(duration_result.stdout.strip())

            # Extract frames at evenly-spaced intervals.
            interval = max(1, int(duration / arguments.max_frames))
            subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-vf", f"fps=1/{interval}",
                    "-frames:v", str(arguments.max_frames),
                    "-q:v", "2",
                    str(frames_dir / "frame_%03d.jpg"),
                ],
                capture_output=True, timeout=120,
            )

            frames = sorted(frames_dir.glob("frame_*.jpg"))
            if not frames:
                return ToolResult(
                    output="No frames could be extracted from the video.",
                    is_error=True,
                )

            # Use vision_analyze tool to analyze each frame.
            try:
                from niaharness.tools.vision_analyze_tool import VisionAnalyzeTool
                vision_tool = VisionAnalyzeTool()
                from niaharness.tools.vision_analyze_tool import VisionAnalyzeInput

                descriptions: list[str] = []
                for frame in frames:
                    vision_input = VisionAnalyzeInput(
                        image_path=str(frame),
                        question=f"Frame from a video. Question: {arguments.question}",
                    )
                    result = await vision_tool.execute(vision_input, context)
                    if not result.is_error:
                        descriptions.append(result.output[:500])

                analysis = "\n\n".join(descriptions)
                return ToolResult(
                    output=f"Video analysis ({len(frames)} frames extracted "
                    f"from {duration:.1f}s video):\n\n{analysis}",
                    metadata={
                        "video_path": str(video_path),
                        "duration": duration,
                        "frames_extracted": len(frames),
                    },
                )
            except Exception as exc:
                return ToolResult(
                    output=f"Frame extraction succeeded but vision analysis failed: {exc}",
                    is_error=True,
                )
        finally:
            # Clean up frames.
            shutil.rmtree(frames_dir, ignore_errors=True)

    def is_read_only(self, arguments: VideoAnalyzeInput) -> bool:
        return True


# ---------------------------------------------------------------------------
# 3. video_generate — generate video from text (via FAL or similar)
# ---------------------------------------------------------------------------


class VideoGenerateInput(BaseModel):
    prompt: str = Field(description="Text description of the video to generate")
    model: str = Field(
        default="fal-ai/luma-dream-machine",
        description="Video generation model (e.g. fal-ai/luma-dream-machine)",
    )
    duration: int = Field(
        default=5, ge=1, le=10,
        description="Video duration in seconds",
    )


class VideoGenerateTool(BaseTool):
    """Generate a video from a text prompt using FAL or similar API."""

    name = "video_generate"
    description = (
        "Generate a short video from a text description. Requires a FAL API key "
        "(set FAL_KEY env var). Returns the URL of the generated video."
    )
    input_model = VideoGenerateInput

    async def execute(
        self, arguments: VideoGenerateInput, context: ToolExecutionContext
    ) -> ToolResult:
        fal_key = os.environ.get("FAL_KEY", "").strip()
        if not fal_key:
            return ToolResult(
                output="FAL_KEY environment variable not set. "
                "Get a key at https://fal.ai and set: export FAL_KEY=your-key",
                is_error=True,
            )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                # Submit the generation request.
                response = await client.post(
                    f"https://fal.run/{arguments.model}",
                    headers={
                        "Authorization": f"Key {fal_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": arguments.prompt,
                        "duration": arguments.duration,
                    },
                )
                response.raise_for_status()
                data = response.json()

            video_url = data.get("video", {}).get("url", "")
            if not video_url:
                return ToolResult(
                    output=f"Video generation completed but no URL returned: {json.dumps(data)}",
                    is_error=True,
                )

            return ToolResult(
                output=f"Video generated successfully.\nURL: {video_url}\n"
                f"Model: {arguments.model}\nDuration: {arguments.duration}s",
                metadata={"video_url": video_url, "model": arguments.model},
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=f"FAL API error: {exc.response.status_code} {exc.response.text}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                output=f"Video generation failed: {exc}", is_error=True
            )


# ---------------------------------------------------------------------------
# 4. clarify — ask clarifying questions
# ---------------------------------------------------------------------------


class ClarifyInput(BaseModel):
    question: str = Field(description="The clarifying question to ask the user")
    options: list[str] = Field(
        default_factory=list,
        description="Optional list of choices for the user to pick from",
    )


class ClarifyTool(BaseTool):
    """Ask the user a clarifying question when the request is ambiguous."""

    name = "clarify"
    description = (
        "Ask the user a clarifying question when the request is ambiguous or "
        "missing required details. The question is sent to the user and the "
        "response is returned. Use sparingly — prefer making reasonable "
        "assumptions when possible."
    )
    input_model = ClarifyInput

    async def execute(
        self, arguments: ClarifyInput, context: ToolExecutionContext
    ) -> ToolResult:
        # Delegate to the existing ask_user_question tool.
        ask_tool = context.metadata.get("ask_user_question_tool")
        if ask_tool is not None:
            from niaharness.tools.ask_user_question_tool import AskUserQuestionInput
            options = [{"label": o[:10], "description": o} for o in arguments.options] if arguments.options else []
            ask_input = AskUserQuestionInput(
                questions=[{
                    "question": arguments.question,
                    "header": "Clarify",
                    "options": options or [
                        {"label": "Answer", "description": "Type your answer"},
                    ],
                    "type": "single",
                }]
            )
            return await ask_tool.execute(ask_input, context)

        # Fallback: just return the question.
        if arguments.options:
            options_text = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(arguments.options))
            return ToolResult(
                output=f"Clarification needed:\n{arguments.question}\n\nOptions:\n{options_text}"
            )
        return ToolResult(output=f"Clarification needed: {arguments.question}")


# ---------------------------------------------------------------------------
# 5. project_create, project_list, project_switch — project management
# ---------------------------------------------------------------------------


class ProjectCreateInput(BaseModel):
    name: str = Field(description="Project name")
    cwd: str = Field(description="Working directory for the project")


class ProjectCreateTool(BaseTool):
    """Create a new project (registers a named working directory)."""

    name = "project_create"
    description = (
        "Create a new project with a name and working directory. "
        "Projects let you switch between different codebases quickly."
    )
    input_model = ProjectCreateInput

    async def execute(
        self, arguments: ProjectCreateInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            import json as _json
            from niaharness.profiles import get_profile
            profile = get_profile()
            # Use the profile's config directory (parent of config.yaml).
            nia_home = profile.config_path.parent
            projects_file = nia_home / "projects.json"

            projects: dict[str, str] = {}
            if projects_file.exists():
                projects = _json.loads(projects_file.read_text())

            projects[arguments.name] = arguments.cwd
            projects_file.parent.mkdir(parents=True, exist_ok=True)
            projects_file.write_text(_json.dumps(projects, indent=2))

            return ToolResult(
                output=f"Project '{arguments.name}' created (cwd: {arguments.cwd})",
                metadata={"project": arguments.name, "cwd": arguments.cwd},
            )
        except Exception as exc:
            return ToolResult(output=f"Failed to create project: {exc}", is_error=True)


class ProjectListInput(BaseModel):
    pass


class ProjectListTool(BaseTool):
    """List all registered projects."""

    name = "project_list"
    description = "List all registered projects with their working directories."
    input_model = ProjectListInput

    async def execute(
        self, arguments: ProjectListInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            from niaharness.profiles import get_profile
            profile = get_profile()
            nia_home = profile.config_path.parent
            projects_file = nia_home / "projects.json"

            if not projects_file.exists():
                return ToolResult(output="No projects registered.")

            import json as _json
            projects = _json.loads(projects_file.read_text())
            if not projects:
                return ToolResult(output="No projects registered.")

            lines = ["Projects:"]
            for name, cwd in sorted(projects.items()):
                lines.append(f"  {name}: {cwd}")
            return ToolResult(output="\n".join(lines))
        except Exception as exc:
            return ToolResult(output=f"Failed to list projects: {exc}", is_error=True)

    def is_read_only(self, arguments: ProjectListInput) -> bool:
        return True


class ProjectSwitchInput(BaseModel):
    name: str = Field(description="Project name to switch to")


class ProjectSwitchTool(BaseTool):
    """Switch to a registered project (changes the working directory)."""

    name = "project_switch"
    description = (
        "Switch to a registered project. Changes the working directory "
        "for subsequent tool calls."
    )
    input_model = ProjectSwitchInput

    async def execute(
        self, arguments: ProjectSwitchInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            from niaharness.profiles import get_profile
            profile = get_profile()
            nia_home = profile.config_path.parent
            projects_file = nia_home / "projects.json"

            if not projects_file.exists():
                return ToolResult(output="No projects registered.", is_error=True)

            import json as _json
            projects = _json.loads(projects_file.read_text())
            if arguments.name not in projects:
                return ToolResult(
                    output=f"Project '{arguments.name}' not found. "
                    f"Available: {', '.join(sorted(projects.keys()))}",
                    is_error=True,
                )

            cwd = projects[arguments.name]
            context.cwd = Path(cwd)
            return ToolResult(
                output=f"Switched to project '{arguments.name}' (cwd: {cwd})",
                metadata={"project": arguments.name, "cwd": cwd},
            )
        except Exception as exc:
            return ToolResult(output=f"Failed to switch project: {exc}", is_error=True)


# ---------------------------------------------------------------------------
# 6. memory — batched memory operations
# ---------------------------------------------------------------------------


class MemoryBatchInput(BaseModel):
    operations: list[dict[str, Any]] = Field(
        description=(
            "List of memory operations. Each dict has 'action' "
            "(search|add|list|remove) and action-specific keys."
        )
    )


class MemoryBatchTool(BaseTool):
    """Perform batched memory operations (search, add, list, remove)."""

    name = "memory"
    description = (
        "Perform batched memory operations. Each operation in the list "
        "is executed in order. Actions: search (query), add (content, category), "
        "list (category), remove (index)."
    )
    input_model = MemoryBatchInput

    async def execute(
        self, arguments: MemoryBatchInput, context: ToolExecutionContext
    ) -> ToolResult:
        from niaharness.memory import get_memory_manager, MemoryStore
        from niaharness.memory.paths import get_project_memory_dir

        store_path = get_project_memory_dir(context.cwd) / "STORE.md"
        store = MemoryStore(path=store_path)

        results: list[str] = []
        for i, op in enumerate(arguments.operations):
            action = op.get("action", "")
            if action == "search":
                query = op.get("query", "")
                entries = store.get_entries(query=query, limit=op.get("limit", 5))
                if entries:
                    results.append(f"[{i}] search '{query}': {len(entries)} match(es)")
                    for e in entries:
                        results.append(f"  - [{e.category}] {e.content[:100]}")
                else:
                    results.append(f"[{i}] search '{query}': no matches")
            elif action == "add":
                content = op.get("content", "")
                category = op.get("category", "note")
                from niaharness.memory.store import MemoryEntry
                entry = MemoryEntry(content=content, category=category, source="agent")
                if store.add_entry(entry):
                    results.append(f"[{i}] add: stored [{category}] entry")
                else:
                    results.append(f"[{i}] add: BLOCKED by write gate")
            elif action == "list":
                category = op.get("category")
                entries = store.get_entries(category=category, limit=op.get("limit", 20))
                results.append(f"[{i}] list: {len(entries)} entries")
                for j, e in enumerate(entries):
                    results.append(f"  [{j}] [{e.category}] {e.content[:80]}")
            elif action == "remove":
                index = op.get("index", -1)
                if store.remove_entry(index):
                    results.append(f"[{i}] remove: deleted entry {index}")
                else:
                    results.append(f"[{i}] remove: entry {index} not found")
            else:
                results.append(f"[{i}] unknown action: {action}")

        return ToolResult(output="\n".join(results))

    def is_read_only(self, arguments: MemoryBatchInput) -> bool:
        # Read-only if all operations are search or list.
        return all(op.get("action") in ("search", "list") for op in arguments.operations)


# ---------------------------------------------------------------------------
# 7. search_files — fuzzy file search (alias for glob + grep combined)
# ---------------------------------------------------------------------------


class SearchFilesInput(BaseModel):
    pattern: str = Field(description="File name pattern (glob, e.g. '*.py')")
    content_query: str = Field(
        default="",
        description="Optional content query (grep within matching files)",
    )
    path: str = Field(
        default=".",
        description="Directory to search in (default: current directory)",
    )
    limit: int = Field(
        default=50, ge=1, le=200,
        description="Maximum number of results",
    )


class SearchFilesTool(BaseTool):
    """Search for files by name pattern and optionally by content."""

    name = "search_files"
    description = (
        "Search for files by name pattern (glob) and optionally filter by "
        "content. Combines glob + grep in one tool. Returns matching file paths."
    )
    input_model = SearchFilesInput

    async def execute(
        self, arguments: SearchFilesInput, context: ToolExecutionContext
    ) -> ToolResult:
        search_path = Path(arguments.path)
        if not search_path.is_absolute():
            search_path = context.cwd / search_path

        if not search_path.exists():
            return ToolResult(
                output=f"Search path does not exist: {search_path}", is_error=True
            )

        # Glob for files.
        matches = sorted(search_path.rglob(arguments.pattern))[:arguments.limit]

        if not arguments.content_query:
            # Just return file names.
            if not matches:
                return ToolResult(output=f"No files matching '{arguments.pattern}' in {search_path}")
            lines = [f"Found {len(matches)} file(s) matching '{arguments.pattern}':"]
            for m in matches:
                rel = m.relative_to(context.cwd) if m.is_relative_to(context.cwd) else m
                lines.append(f"  {rel}")
            return ToolResult(output="\n".join(lines))

        # Filter by content.
        query_lower = arguments.content_query.lower()
        content_matches: list[str] = []
        for m in matches:
            if not m.is_file():
                continue
            try:
                content = m.read_text(encoding="utf-8", errors="ignore")
                if query_lower in content.lower():
                    rel = m.relative_to(context.cwd) if m.is_relative_to(context.cwd) else m
                    content_matches.append(str(rel))
                    if len(content_matches) >= arguments.limit:
                        break
            except Exception:
                continue

        if not content_matches:
            return ToolResult(
                output=f"No files matching '{arguments.pattern}' with content '{arguments.content_query}'"
            )
        lines = [f"Found {len(content_matches)} file(s) with content '{arguments.content_query}':"]
        for m in content_matches:
            lines.append(f"  {m}")
        return ToolResult(output="\n".join(lines))

    def is_read_only(self, arguments: SearchFilesInput) -> bool:
        return True


# ---------------------------------------------------------------------------
# 8. read_terminal / close_terminal — manage persistent terminal sessions
# ---------------------------------------------------------------------------


class ReadTerminalInput(BaseModel):
    terminal_id: str = Field(description="Terminal session ID")
    lines: int = Field(
        default=50, ge=1, le=500,
        description="Number of recent lines to read",
    )


class ReadTerminalTool(BaseTool):
    """Read output from a persistent terminal session."""

    name = "read_terminal"
    description = (
        "Read recent output from a persistent terminal session created by the "
        "bash tool. Useful for checking long-running commands."
    )
    input_model = ReadTerminalInput

    async def execute(
        self, arguments: ReadTerminalInput, context: ToolExecutionContext
    ) -> ToolResult:
        # Check if the process tool has a registered terminal.
        process_tool = context.metadata.get("process_tool")
        if process_tool is not None:
            try:
                output = process_tool.get_output(arguments.terminal_id, lines=arguments.lines)
                if output:
                    return ToolResult(output=output)
                return ToolResult(
                    output=f"No output available for terminal '{arguments.terminal_id}'"
                )
            except Exception as exc:
                return ToolResult(
                    output=f"Could not read terminal '{arguments.terminal_id}': {exc}",
                    is_error=True,
                )
        return ToolResult(
            output=f"Terminal '{arguments.terminal_id}' not found. "
            "No persistent terminal sessions are available.",
            is_error=True,
        )

    def is_read_only(self, arguments: ReadTerminalInput) -> bool:
        return True


class CloseTerminalInput(BaseModel):
    terminal_id: str = Field(description="Terminal session ID to close")


class CloseTerminalTool(BaseTool):
    """Close a persistent terminal session."""

    name = "close_terminal"
    description = (
        "Close a persistent terminal session and free its resources. "
        "Use after a long-running command finishes."
    )
    input_model = CloseTerminalInput

    async def execute(
        self, arguments: CloseTerminalInput, context: ToolExecutionContext
    ) -> ToolResult:
        process_tool = context.metadata.get("process_tool")
        if process_tool is not None:
            try:
                if hasattr(process_tool, "stop"):
                    process_tool.stop(arguments.terminal_id)
                    return ToolResult(output=f"Closed terminal '{arguments.terminal_id}'")
            except Exception:
                pass
        return ToolResult(
            output=f"Terminal '{arguments.terminal_id}' not found or already closed."
        )


# ---------------------------------------------------------------------------
# 9. x_search — search X (Twitter) posts
# ---------------------------------------------------------------------------


class XSearchInput(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results",
    )


class XSearchTool(BaseTool):
    """Search X (formerly Twitter) for posts matching a query."""

    name = "x_search"
    description = (
        "Search X (formerly Twitter) for posts matching a query. "
        "Requires a Twitter/X API bearer token (set TWITTER_BEARER_TOKEN env var)."
    )
    input_model = XSearchInput

    async def execute(
        self, arguments: XSearchInput, context: ToolExecutionContext
    ) -> ToolResult:
        bearer = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
        if not bearer:
            return ToolResult(
                output="TWITTER_BEARER_TOKEN environment variable not set. "
                "Get a token at https://developer.x.com and set: "
                "export TWITTER_BEARER_TOKEN=your-token",
                is_error=True,
            )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    headers={"Authorization": f"Bearer {bearer}"},
                    params={
                        "query": arguments.query,
                        "max_results": min(arguments.limit, 100),
                        "tweet.fields": "created_at,author_id,public_metrics",
                    },
                )
                response.raise_for_status()
                data = response.json()

            tweets = data.get("data", [])
            if not tweets:
                return ToolResult(output=f"No tweets found for '{arguments.query}'")

            lines = [f"Found {len(tweets)} tweet(s) for '{arguments.query}':"]
            for i, tweet in enumerate(tweets[:arguments.limit], 1):
                text = tweet.get("text", "")[:200]
                created = tweet.get("created_at", "")
                metrics = tweet.get("public_metrics", {})
                likes = metrics.get("like_count", 0)
                rts = metrics.get("retweet_count", 0)
                lines.append(f"\n[{i}] @{tweet.get('author_id', '?')} ({created})")
                lines.append(f"  {text}")
                lines.append(f"  ❤️ {likes}  🔁 {rts}")

            return ToolResult(output="\n".join(lines))
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=f"X API error: {exc.response.status_code} {exc.response.text}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(output=f"X search failed: {exc}", is_error=True)

    def is_read_only(self, arguments: XSearchInput) -> bool:
        return True


# ---------------------------------------------------------------------------
# 10. text_to_speech — alias for speak tool
# ---------------------------------------------------------------------------


class TextToSpeechInput(BaseModel):
    text: str = Field(description="Text to convert to speech")
    voice: str = Field(
        default="Jasper",
        description="Voice name (e.g. Jasper, en-US-GuyNeural)",
    )
    output_format: str = Field(
        default="wav", description='Output format: "wav" or "mp3"'
    )


class TextToSpeechTool(BaseTool):
    """Convert text to speech (alias for the speak tool)."""

    name = "text_to_speech"
    description = (
        "Convert text to speech audio. Uses KittenTTS (neural, on-CPU) by default "
        "with an espeak fallback. Returns the path to the generated audio file."
    )
    input_model = TextToSpeechInput

    async def execute(
        self, arguments: TextToSpeechInput, context: ToolExecutionContext
    ) -> ToolResult:
        # Delegate to the existing speak tool.
        from niaharness.tools.speak_tool import SpeakTool, SpeakToolInput
        speak_tool = SpeakTool()
        speak_input = SpeakToolInput(
            text=arguments.text,
            voice=arguments.voice,
            output_format=arguments.output_format,
        )
        return await speak_tool.execute(speak_input, context)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def get_missing_tools() -> list[BaseTool]:
    """Return instances of all P1 missing tools."""
    return [
        WebExtractTool(),
        VideoAnalyzeTool(),
        VideoGenerateTool(),
        ClarifyTool(),
        ProjectCreateTool(),
        ProjectListTool(),
        ProjectSwitchTool(),
        MemoryBatchTool(),
        SearchFilesTool(),
        ReadTerminalTool(),
        CloseTerminalTool(),
        XSearchTool(),
        TextToSpeechTool(),
    ]


__all__ = [
    "ClarifyInput",
    "ClarifyTool",
    "CloseTerminalInput",
    "CloseTerminalTool",
    "MemoryBatchInput",
    "MemoryBatchTool",
    "ProjectCreateInput",
    "ProjectCreateTool",
    "ProjectListInput",
    "ProjectListTool",
    "ProjectSwitchInput",
    "ProjectSwitchTool",
    "ReadTerminalInput",
    "ReadTerminalTool",
    "SearchFilesInput",
    "SearchFilesTool",
    "TextToSpeechInput",
    "TextToSpeechTool",
    "VideoAnalyzeInput",
    "VideoAnalyzeTool",
    "VideoGenerateInput",
    "VideoGenerateTool",
    "WebExtractInput",
    "WebExtractTool",
    "XSearchInput",
    "XSearchTool",
    "get_missing_tools",
]
