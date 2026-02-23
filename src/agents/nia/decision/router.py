"""N.I.A. Decision Core — AI-Powered Request Router.

Replaces the legacy regex-based Gatekeeper with an LLM-driven intent
classifier.  The ``DecisionCore`` sends each user query to the smart model
and parses a structured JSON routing decision, with a keyword fallback for
cases where JSON parsing fails.

Routing Targets:

    ``chat``
        General conversation, Q&A, brainstorming.  Handled by the
        NIA supervisor's chat path.

    ``swarm``
        Code generation, file creation/editing, script execution.
        Routed to the Docker Swarm agent.  The ``skill`` field selects
        the specific worker (e.g. ``"coding-agent"``).

    ``system``
        NIA self-management commands: shutdown, settings changes,
        voice overrides.

Design Notes:
    - Uses ``ainvoke()`` (not ``with_structured_output()``) for maximum
      LLM compatibility (Ollama, Llama 3.1, NVIDIA endpoints).
    - Manual JSON parsing + Pydantic validation for safety.
    - Keyword fallback if the LLM response cannot be parsed.

Usage::

    from src.agents.nia.decision.router import DecisionCore

    core = DecisionCore()
    decision = await core.aroute("write a Python web scraper")
    print(decision.target)  # "swarm"
    print(decision.skill)   # "coding-agent"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import settings

logger = logging.getLogger("NIA.Decision")

# --- Structured Output Schema ---
class RoutingDecision(BaseModel):
    """The structured decision output from the Router."""
    target: Literal["chat", "swarm", "system"] = Field(
        ..., 
        description="The target system to handle the request."
    )
    skill: Optional[str] = Field(
        None, 
        description="The specific skill needed (e.g., 'coding-agent') if target is 'swarm'."
    )
    reasoning: str = Field(
        ..., 
        description="Brief justification for the routing decision."
    )

# --- Router Logic ---
class DecisionCore:
    def __init__(self):
        self._llm = None
        
    @property
    def llm(self):
        """Lazy load the smart model."""
        if not self._llm:
            from src.models.manager import get_smart_model
            # Use smart model for reasoning accuracy
            self._llm = get_smart_model(temperature=0.0) 
        return self._llm

    async def aroute(self, user_query: str) -> RoutingDecision:
        """
        Analyze user query and determine the best route.
        Uses manual JSON parsing instead of with_structured_output() 
        for maximum LLM compatibility.
        """
        if not user_query:
            return RoutingDecision(target="chat", reasoning="Empty input.")

        system_prompt = """You are N.I.A.'s Decision Core. 
Your ONLY job is to classify the user's request and respond with a JSON object.

SYSTEMS:
1. "swarm": For creating, editing, debugging, or executing code/files/scripts.
   - skill: "coding-agent" (Default for any code/file work).
   - skill: "web-browsing-agent" (Only if explicitly asked to browse).
   
2. "chat": For general conversation, questions, brainstorming, greetings, or personality.
   - Example: "Why is the sky blue?", "Plan a project", "Hello".

3. "system": For modifying N.I.A.'s own settings, voice overrides, or shutdown.

You MUST respond with ONLY valid JSON matching this exact schema:
{"target": "chat" | "swarm" | "system", "skill": "string or null", "reasoning": "string"}

Do NOT include markdown code blocks, backticks, or any text outside the JSON object.
Respond with the raw JSON object only."""
        
        try:
            # Use standard ainvoke() for maximum compatibility
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ])
            
            raw_content = response.content.strip()
            
            # Strip markdown code fences if present (```json ... ```)
            raw_content = re.sub(r'^```(?:json)?\s*', '', raw_content)
            raw_content = re.sub(r'\s*```$', '', raw_content)
            raw_content = raw_content.strip()
            
            # Parse JSON
            parsed = json.loads(raw_content)
            
            # Validate with Pydantic
            decision = RoutingDecision(**parsed)
            
            logger.info(f"🧭 Router Decision: {decision.target} ({decision.reasoning})")
            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Router JSON parse failed: {e} | Raw: {raw_content[:200]}")
            # Attempt keyword fallback
            return self._keyword_fallback(user_query, raw_content)
            
        except Exception as e:
            logger.error(f"Router LLM failed: {e}")
            # Fallback to chat if routing fails
            return RoutingDecision(target="chat", reasoning=f"Routing Error: {e}")
    
    def _keyword_fallback(self, user_query: str, raw_response: str = "") -> RoutingDecision:
        """Last-resort keyword-based routing when JSON parsing fails."""
        combined = f"{user_query} {raw_response}".lower()
        
        code_keywords = ["code", "script", "python", "create a file", "run", "execute", 
                         "build", "debug", "program", "function", "class", "compile"]
        system_keywords = ["shutdown", "restart", "settings", "volume", "voice"]
        
        for kw in code_keywords:
            if kw in combined:
                return RoutingDecision(
                    target="swarm", 
                    skill="coding-agent",
                    reasoning=f"Keyword fallback: matched '{kw}'"
                )
        
        for kw in system_keywords:
            if kw in combined:
                return RoutingDecision(
                    target="system",
                    reasoning=f"Keyword fallback: matched '{kw}'"
                )
        
        return RoutingDecision(target="chat", reasoning="Fallback: no routing signals detected")
