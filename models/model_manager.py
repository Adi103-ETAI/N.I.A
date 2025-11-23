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

DEFAULT_PROVIDER_MODELS: Dict[str, str] = {
    "nvidia": "meta/llama-4-maverick-17b-128e-instruct",
    "openai": "gpt-4o",
    "ollama": "llama3",
}

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub

    has_langchain = True
except ImportError:
    has_langchain = False

try:
    from langchain_ollama import OllamaLLM
    has_ollama = True
except ImportError:
    OllamaLLM = None  # type: ignore
    has_ollama = False

try:
    from openai import OpenAI  # optional; if present we'll prefer it for OpenAI calls
except Exception:
    OpenAI = None  # type: ignore


class ModelManager:
    """Unified model manager for reasoning and embedding operations.

    Parameters
    - provider: str, one of {'openai', 'nvidia', 'local', 'ollama', 'huggingface'} (backend selection)
    - model_name: str, model identifier used by the selected provider
    - config: optional dict with provider-specific overrides
    - logger: optional logger

    The implementation below is safe to import even if provider SDKs are
    not installed; methods will raise descriptive NotImplementedError when
    a backend is requested but not available.
    """

    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> None:
        self.provider = (provider or "openai").lower()
        self._primary_provider = self.provider
        self._primary_model_name = model_name
        self.model_name = model_name
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self._provider_models: Dict[str, str] = dict(self.config.get("provider_models", {}))

        # read configuration from environment as necessary
        self._openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
        self._nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
        # example for local runtimes
        self._local_host = os.environ.get("OLLAMA_HOST") or os.environ.get("LOCAL_LLM_HOST")

        # Configure optional fallback providers (list). Can be provided via
        # config['fallback_providers'] or env var MODEL_PROVIDER_FALLBACKS="ollama,huggingface"
        fallbacks = self.config.get("fallback_providers")
        if not fallbacks:
            env_fb = os.environ.get("MODEL_PROVIDER_FALLBACKS", "")
            fallbacks = [p.strip().lower() for p in env_fb.split(",") if p.strip()] if env_fb else []
        # store as list excluding the primary provider
        self._fallback_providers = [p for p in (fallbacks or []) if p and p.lower() != self.provider]

        # Ensure model name matches the currently selected provider
        self._apply_model_for_provider(self.provider)

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
                use_langchain_ollama = bool(self.config.get("use_langchain_ollama"))
                if use_langchain_ollama:
                    if has_ollama and OllamaLLM is not None:
                        self.llm = OllamaLLM(base_url=self._local_host, model=self.model_name)
                    else:
                        self.logger.warning("langchain-ollama not installed; Ollama provider unavailable")
                        self.llm = None
                else:
                    # Default to the local HTTP shim to avoid surprise network calls during tests.
                    self.llm = None
            else:
                self.embedding_model = None
                self.llm = None
                
        except Exception as exc:
            self.logger.warning("Failed to initialize Langchain models: %s", exc)
            self.embedding_model = None
            self.llm = None

    def _apply_model_for_provider(self, provider: str) -> None:
        """Pick the correct model for the active provider."""
        selected_model: Optional[str] = None
        if provider == self._primary_provider:
            selected_model = self._primary_model_name
        elif provider in self._provider_models:
            selected_model = self._provider_models[provider]
        elif provider in DEFAULT_PROVIDER_MODELS:
            selected_model = DEFAULT_PROVIDER_MODELS[provider]

        if selected_model:
            self.model_name = selected_model

    def _call_langchain_llm(self, prompt: str) -> str:
        """Best-effort call wrapper that normalizes Langchain LLM outputs."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return ""

        call_chain = []
        for attr in ("predict", "invoke"):
            method = getattr(llm, attr, None)
            if callable(method):
                call_chain.append(method)

        if not call_chain and callable(llm):
            call_chain.append(llm)

        if not call_chain:
            raise AttributeError(f"{type(llm).__name__!r} object has no supported inference interface")

        last_error: Optional[Exception] = None
        for call in call_chain:
            try:
                result = call(prompt)
                text = self._normalize_langchain_response(result)
                if text:
                    return text
            except Exception as exc:
                last_error = exc
                self.logger.debug("Langchain LLM call via %s failed: %s", getattr(call, "__name__", call), exc)

        if last_error:
            raise last_error

        return ""

    def _normalize_langchain_response(self, result: Any) -> str:
        """Extract a string response from diverse Langchain output types."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()

        # AIMessage / BaseMessage style
        for attr in ("content", "text"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if isinstance(value, str):
                    return value.strip()

        if isinstance(result, dict):
            for key in ("content", "text", "output", "message"):
                value = result.get(key)
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, dict):
                    nested = self._normalize_langchain_response(value)
                    if nested:
                        return nested
            # look for first string value
            for value in result.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(result, list):
            for item in result:
                normalized = self._normalize_langchain_response(item)
                if normalized:
                    return normalized

        # Fallback to string conversion
        text = str(result).strip()
        return text

    # -------------------- Public API --------------------
    def reason(self, prompt: str, mode: str = "default") -> str:
        """Primary reasoning entrypoint.

        Dispatches to the configured backend and returns the model's text
        response. The `mode` param is a soft hint that can select different
        prompts, system messages, or providers.
        """
        self.logger.debug("reason called (mode=%s) prompt=%s", mode, prompt)
        
        # Try primary provider then configured fallbacks in order
        providers_to_try = [self.provider] + list(self._fallback_providers)
        last_exc: Optional[Exception] = None
        original_provider = self.provider

        for idx, prov in enumerate(providers_to_try):
            try:
                self.logger.debug("Attempting reasoning with provider=%s (try %d/%d)", prov, idx + 1, len(providers_to_try))

                # If switching provider, update and re-init langchain clients
                if prov != self.provider:
                    self.provider = prov
                    # refresh local host in case env differs
                    self._local_host = os.environ.get("OLLAMA_HOST") or os.environ.get("LOCAL_LLM_HOST")
                    self._apply_model_for_provider(self.provider)
                    self._init_langchain_models()

                # Prepare adjusted prompt for mode
                adj_prompt = prompt
                if mode == "summarize":
                    adj_prompt = f"Please provide a concise summary: {prompt}"
                elif mode != "default":
                    self.logger.debug("Using custom mode: %s", mode)

                # Allow persona/system prompt injection
                persona_preamble = self.config.get("persona_prompt")
                mode_prompts = self.config.get("mode_prompts", {})
                mode_preamble = mode_prompts.get(mode) if isinstance(mode_prompts, dict) else None
                preamble_parts = [txt for txt in (persona_preamble, mode_preamble) if txt]
                if preamble_parts:
                    adj_prompt = "\n\n".join(preamble_parts + [adj_prompt])

                # Prefer Langchain client if available
                if has_langchain and getattr(self, "llm", None) is not None:
                    try:
                        response = self._with_backoff(self._call_langchain_llm, adj_prompt)
                    except AttributeError as missing_method_exc:
                        self.logger.debug("Langchain LLM missing compatible interface: %s", missing_method_exc)
                        response = ""

                    if response:
                        # restore original provider
                        if original_provider != prov:
                            self.provider = original_provider
                            self._apply_model_for_provider(self.provider)
                            self._init_langchain_models()
                        return response
                    else:
                        self.logger.debug("Provider %s produced no response via langchain shim; trying native call", prov)

                # Provider-specific calls
                if prov == "openai":
                    resp = self._with_backoff(self._call_openai, adj_prompt)
                elif prov == "nvidia":
                    resp = self._with_backoff(self._call_nvidia, adj_prompt)
                elif prov in ("local", "ollama", "llamacpp"):
                    resp = self._with_backoff(self._call_local, adj_prompt)
                else:
                    raise NotImplementedError(f"Provider '{prov}' is not supported")

                if resp:
                    if original_provider != prov:
                        self.provider = original_provider
                        self._apply_model_for_provider(self.provider)
                        self._init_langchain_models()
                    return resp

                self.logger.warning("Provider %s returned empty response, trying next provider", prov)

            except Exception as exc:
                last_exc = exc
                self.logger.warning("Provider %s failed during reasoning: %s; trying next provider", prov, exc)
                continue

        # All providers failed or returned empty
        if last_exc:
            raise last_exc
        raise RuntimeError("All configured providers returned empty responses")

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

        # Use Langchain for embeddings
        if has_langchain and self.embedding_model is not None:
            try:
                self.logger.debug("Using Langchain embeddings")
                vec = self.embedding_model.embed_query(text.strip())
                self.logger.debug("Got Langchain embedding dim=%d", len(vec))
                return vec
            except Exception as exc:
                self.logger.warning("Langchain embedding failed: %s", exc)
                raise RuntimeError("Failed to generate embeddings") from exc
        
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

    def _call_nvidia(self, prompt: str) -> str:
        """Call NVIDIA API to obtain a text completion.

        Uses the NVIDIA Integrate API endpoint for chat completions.
        All network calls are wrapped by the backoff wrapper used by `reason`.
        """
        # Try to import requests (required for NVIDIA API calls)
        try:
            import requests
        except Exception:
            raise ImportError("The 'requests' library is required for NVIDIA API calls. Add 'requests' to your requirements.")

        if not self._nvidia_api_key:
            self.logger.warning("NVIDIA_API_KEY is not set; NVIDIA requests may fail")

        self.logger.debug("_call_nvidia: model=%s prompt=%s", self.model_name, prompt)

        # NVIDIA API endpoint
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        stream = False

        headers = {
            "Authorization": f"Bearer {self._nvidia_api_key}",
            "Accept": "text/event-stream" if stream else "application/json"
        }

        # Build payload with config overrides
        payload = {
            "model": self.model_name if self.model_name else "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 1.00),
            "top_p": self.config.get("top_p", 1.00),
            "frequency_penalty": self.config.get("frequency_penalty", 0.00),
            "presence_penalty": self.config.get("presence_penalty", 0.00),
            "stream": stream
        }

        try:
            self.logger.debug("Calling NVIDIA API at %s with model=%s", invoke_url, payload["model"])
            response = requests.post(
                invoke_url,
                headers=headers,
                json=payload,
                timeout=self.config.get("timeout", 30)
            )
            response.raise_for_status()

            # Handle streaming response (if enabled in future)
            if stream:
                text_parts = []
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        # Parse SSE format if needed (data: {...})
                        if decoded_line.startswith("data: "):
                            import json
                            try:
                                data = json.loads(decoded_line[6:])  # Remove "data: " prefix
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        text_parts.append(delta["content"])
                            except json.JSONDecodeError:
                                continue
                return "".join(text_parts) if text_parts else ""
            else:
                # Handle non-streaming response
                data = response.json()
                
                # Extract text from NVIDIA API response format
                # Expected format: {"choices": [{"message": {"content": "..."}}]}
                if isinstance(data, dict) and "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if isinstance(choice, dict) and "message" in choice:
                        message = choice["message"]
                        if isinstance(message, dict) and "content" in message:
                            text = message["content"]
                            if text:
                                return text
                    
                    # Fallback: try direct content field
                    if "content" in choice:
                        return str(choice["content"])
                
                # Last resort: look for any text-like field
                if isinstance(data, dict):
                    for key in ["content", "text", "response", "output"]:
                        if key in data and isinstance(data[key], str):
                            return data[key]
                    
                    # Try nested structures
                    if "message" in data and isinstance(data["message"], dict):
                        if "content" in data["message"]:
                            return str(data["message"]["content"])

                self.logger.warning("NVIDIA API returned unexpected response format")
                return ""

        except requests.exceptions.RequestException as exc:
            self.logger.exception("NVIDIA API call failed: %s", exc)
            raise RuntimeError(f"NVIDIA API request failed: {exc}") from exc
        except Exception as exc:
            self.logger.exception("NVIDIA call failed: %s", exc)
            raise

    def _call_local(self, prompt: str) -> str:
        """Call a local LLM runtime.

        This is intentionally a minimal shim. Real implementations may call
        into Ollama, LlamaCPP, or another local runtime via subprocesses or
        a dedicated client library.
        """
        self.logger.debug("_call_local: host=%s prompt=%s", self._local_host, prompt)

        # Try to import requests (preferred) otherwise raise an informative error
        try:
            import requests
        except Exception:
            raise ImportError("The 'requests' library is required for local Ollama calls. Add 'requests' to your requirements.")

        # Determine host: prefer configured value, then env var, then default Ollama host
        host = self._local_host or os.environ.get("OLLAMA_HOST") or os.environ.get("LOCAL_LLM_HOST") or "http://localhost:11434"
        host = host.rstrip("/")

        url = f"{host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
        }

        # Forward some config options if present
        if "max_tokens" in self.config:
            payload["max_tokens"] = int(self.config["max_tokens"])
        if "temperature" in self.config:
            payload["temperature"] = float(self.config["temperature"])

        try:
            self.logger.debug("Calling Ollama at %s payload=%s", url, {k: payload[k] for k in ("model",)})
            resp = requests.post(url, json=payload, timeout=self.config.get("timeout", 30))
            resp.raise_for_status()
            data = resp.json()

            # Ollama and local runtimes may return a few shapes. Try common ones.
            # 1) {'output': 'text...'}
            if isinstance(data, dict) and "output" in data and isinstance(data["output"], str):
                return data["output"]

            # 2) {'text': '...'}
            if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                return data["text"]

            # 3) {'results': [{'id':..., 'content': '...'}]}
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list) and data["results"]:
                first = data["results"][0]
                if isinstance(first, dict):
                    for key in ("content", "output", "text"):
                        if key in first and isinstance(first[key], str):
                            return first[key]
                elif isinstance(first, str):
                    return first

            # 4) {'choices': [{'text': '...'}]}
            if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    return choice.get("text", "") or choice.get("message", "") or ""

            # Fallback: try to extract any top-level string value
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str) and v.strip():
                        return v

            # As a last resort, return empty string (caller will try next provider)
            self.logger.debug("_call_local: unexpected response shape: %s", data)
            return ""

        except Exception as exc:
            self.logger.exception("Local/Ollama call failed: %s", exc)
            raise

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
        persona_intro = self.config.get("persona_prompt") or (
            "You are the friendly voice of the NIA assistant. "
            "Always introduce yourself as NIA when asked who you are."
        )
        summary_prompt = (
            persona_intro
            + "\nA JSON payload describing internal execution details is provided below. "
            "Respond ONLY with the short, user-facing message you would say back to the user. "
            "Do not mention conversation IDs, tools, goals, providers, or any metadata. "
            "Do not include bullet points or summaries of the execution. "
            "Just speak to the user naturally.\n\n"
            f"Action result:\n{action_result}"
        )
        try:
            return self.reason(summary_prompt, mode="summarize")
        except Exception as exc:
            self.logger.debug("render_response failed: %s", exc)
            return None


