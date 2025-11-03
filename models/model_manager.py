"""Model manager for NIA.

This module provides a backend-agnostic `ModelManager` that acts as the
unified interface between NIA's reasoning system and underlying language
models (cloud providers or local runtimes).

Design goals implemented in this skeleton:
- No hardcoded API keys or endpoints: environment variables or config are used.
- Backend-agnostic public API: `reason`, `embed` and provider-specific
  private helpers (`_call_openai`, `_call_local`).
- Robust error handling with exponential backoff and logging.

This file intentionally provides production-quality wiring and clear
extension points (TODOs) rather than tight coupling to a specific SDK.
"""
from typing import Any, Dict, List, Optional
import os
import time
import logging
import random

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub, Ollama

    has_langchain = True
except ImportError:
    has_langchain = False

try:
    from openai import OpenAI  # optional; if present we'll prefer it for OpenAI calls
except Exception:
    OpenAI = None  # type: ignore


class ModelManager:
    """Unified model manager for reasoning and embedding operations.

    Parameters
    - provider: str, one of {'openai', 'local'} (backend selection)
    - model_name: str, model identifier used by the selected provider
    - config: optional dict with provider-specific overrides
    - logger: optional logger

    The implementation below is safe to import even if provider SDKs are
    not installed; methods will raise descriptive NotImplementedError when
    a backend is requested but not available.
    """

    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        self.provider = (provider or "openai").lower()
        self.model_name = model_name
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # read configuration from environment as necessary
        self._openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        # example for local runtimes
        self._local_host = os.environ.get("OLLAMA_HOST") or os.environ.get("LOCAL_LLM_HOST")

        # Initialize Langchain models if available
        self._init_langchain_models()

        self.logger.info("ModelManager initialized (provider=%s, model=%s)", self.provider, self.model_name)
        
    def _init_langchain_models(self) -> None:
        """Initialize Langchain models based on provider."""
        if not has_langchain:
            self.logger.warning("Langchain not available - some features will be limited")
            return
            
        try:
            if self.provider == "openai":
                self.embedding_model = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=self._openai_api_key
                )
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.config.get("temperature", 0.2),
                    max_tokens=self.config.get("max_tokens", 512),
                    openai_api_key=self._openai_api_key
                )
            elif self.provider == "huggingface":
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    api_key=self._hf_api_key
                )
                self.llm = HuggingFaceHub(
                    repo_id=self.model_name,
                    api_key=self._hf_api_key
                )
            elif self.provider == "ollama":
                # Ollama embeddings not directly supported yet
                self.embedding_model = None
                self.llm = Ollama(base_url=self._local_host, model=self.model_name)
            else:
                self.embedding_model = None
                self.llm = None
                
        except Exception as exc:
            self.logger.warning("Failed to initialize Langchain models: %s", exc)
            self.embedding_model = None
            self.llm = None

    # -------------------- Public API --------------------
    def reason(self, prompt: str, mode: str = "default") -> str:
        """Primary reasoning entrypoint.

        Dispatches to the configured backend and returns the model's text
        response. The `mode` param is a soft hint that can select different
        prompts, system messages, or providers.
        """
        self.logger.debug("reason called (mode=%s) prompt=%s", mode, prompt)
        
        # Try Langchain first if available
        if has_langchain and self.llm is not None:
            try:
                self.logger.debug("Using Langchain for reasoning")
                # Apply mode-specific configurations
                if mode == "summarize":
                    prompt = f"Please provide a concise summary: {prompt}"
                elif mode != "default":
                    self.logger.debug("Using custom mode: %s", mode)
                    
                response = self._with_backoff(self.llm.predict, prompt)
                if response:
                    return response
                self.logger.warning("Langchain response was empty, falling back")
            except Exception as exc:
                self.logger.warning("Langchain reasoning failed, falling back: %s", exc)
        
        # Legacy reasoning methods as fallback
        if self.provider == "openai":
            return self._with_backoff(self._call_openai, prompt)
        elif self.provider in ("local", "ollama", "llamacpp"):
            return self._with_backoff(self._call_local, prompt)
        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not supported")

    def embed(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """Return vector embeddings for the provided text.
        
        Args:
            text: The text to embed
            model_name: Optional override for embedding model (e.g. text-embedding-ada-002)
                      If not provided, will use a suitable default based on provider
        
        Returns:
            List[float]: The embedding vector
            
        Raises:
            ImportError: If required embedding SDK not installed
            NotImplementedError: If provider doesn't support embeddings
            RuntimeError: For embedding API errors (rate limits etc)
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
            
        self.logger.debug("embed called text=%s model=%s", text[:100], model_name)
        
        # Try Langchain embeddings first if available
        if has_langchain and self.embedding_model is not None:
            try:
                self.logger.debug("Using Langchain embeddings")
                vec = self.embedding_model.embed_query(text.strip())
                self.logger.debug("Got Langchain embedding dim=%d", len(vec))
                return vec
            except Exception as exc:
                self.logger.warning("Langchain embedding failed, falling back: %s", exc)
                # Fall through to legacy methods
        
        # Legacy embedding methods as fallback
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("OpenAI SDK required for embeddings but not installed")
                
            embed_model = model_name or "text-embedding-ada-002"
            try:
                resp = OpenAI.embeddings.create(
                    input=text.strip(),
                    model=embed_model,
                    encoding_format="float" 
                )
                vec = resp["data"][0]["embedding"]
                self.logger.debug("Got embedding dim=%d model=%s", len(vec), embed_model)
                return vec
                
            except Exception as exc:
                self.logger.exception("OpenAI embedding failed: %s", exc)
                raise RuntimeError(f"Embedding failed: {exc}") from exc

        elif self.provider in ("local", "ollama"):
            try:
                # Example of local embedding via Ollama API
                import requests
                if not self._local_host:
                    raise RuntimeError("LOCAL_LLM_HOST not configured for embeddings")
                    
                resp = requests.post(
                    f"{self._local_host}/api/embeddings",
                    json={"model": model_name or "llama2", "prompt": text}
                )
                resp.raise_for_status()
                data = resp.json()
                return data["embedding"]
                
            except ImportError:
                raise ImportError("requests library required for local embeddings")
            except Exception as exc:
                self.logger.exception("Local embedding failed: %s", exc)
                raise RuntimeError(f"Local embedding failed: {exc}") from exc

        raise NotImplementedError(f"Embeddings not supported for provider: {self.provider}")

    # -------------------- Backend helpers --------------------
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI-compatible API to obtain a text completion.

        This wrapper tries to avoid hard SDK coupling: if the `openai` SDK
        is present we use it; otherwise we raise. All network calls are
        wrapped by the backoff wrapper used by `reason`.
        """
        if OpenAI is None:
            raise NotImplementedError("OpenAI SDK is not installed in this environment")

        if not self._openai_api_key:
            self.logger.warning("OPENAI_API_KEY is not set; OpenAI requests may fail")

        self.logger.debug("_call_openai: model=%s prompt=%s", self.model_name, prompt)
        try:
            # NOTE: this is a simple example call. Integrators should adapt
            # to the latest API (chat/completions) and control system/user
            # messages appropriately.
            client = OpenAI(api_key=self._openai_api_key)
            resp = client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "user", "content": prompt}]
            )
            # extract text safely
            text = ""
            if resp and getattr(resp, "choices", None):
                # support both dict and SDK object shapes
                choice = resp.choices[0]
                text = getattr(choice, "message", {}).get("content") if isinstance(choice, dict) else (choice.message.get("content") if hasattr(choice, "message") else "")
            elif isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                text = resp["choices"][0]["message"]["content"]

            return text or ""
        except Exception as exc:
            self.logger.exception("OpenAI call failed: %s", exc)
            raise

    def _call_local(self, prompt: str) -> str:
        """Call a local LLM runtime.

        This is intentionally a minimal shim. Real implementations may call
        into Ollama, LlamaCPP, or another local runtime via subprocesses or
        a dedicated client library.
        """
        self.logger.debug("_call_local: host=%s prompt=%s", self._local_host, prompt)

        # If there's a configured local host we could attempt an HTTP call
        # or integrate with a local client. Keep this a stub for now.
        # TODO: implement Ollama / local LLM HTTP client or subprocess runner.
        raise NotImplementedError("Local model calling is not implemented yet")

    # -------------------- Utility helpers --------------------
    def _with_backoff(self, func, *args, retries: int = 3, base_delay: float = 0.5, **kwargs):
        """Run func(*args, **kwargs) with exponential backoff on exceptions.

        Returns the function's result or raises the last exception after
        exhausting retries.
        """
        attempt = 0
        while True:
            try:
                attempt += 1
                self.logger.debug("Attempt %d for function %s", attempt, getattr(func, "__name__", str(func)))
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt > retries:
                    self.logger.error("Function %s failed after %d attempts", getattr(func, "__name__", str(func)), attempt)
                    raise
                # exponential backoff with jitter
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, delay * 0.1)
                wait = delay + jitter
                self.logger.warning("Transient error calling model backend: %s; retrying in %.2fs (attempt %d/%d)", exc, wait, attempt, retries)
                time.sleep(wait)

    # -------------------- Convenience / compatibility --------------------
    def render_response(self, action_result: Dict[str, Any]) -> Optional[str]:
        """Optional helper: summarize action_result using the model.

        Returns a short human-friendly string or None if not implemented.
        """
        # Default implementation: attempt to call the model to produce a
        # concise summary. Integrators may override or extend this.
        summary_prompt = "Summarize the following action results for a user:\n" + str(action_result)
        try:
            return self.reason(summary_prompt, mode="summarize")
        except Exception as exc:
            self.logger.debug("render_response failed: %s", exc)
            return None


