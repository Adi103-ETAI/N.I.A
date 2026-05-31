"""N.I.A Configuration - JSON-based config management.

Stores provider credentials and settings in ~/.nia/config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_DIR = Path.home() / ".nia"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] = field(default_factory=list)
    active_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NIAConfig:
    """Root configuration for N.I.A."""
    active_provider: str = ""
    active_model: str = ""
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    theme: str = "default"
    auto_connect: bool = True  # Auto-detect providers from env vars

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "providers": {
                name: {
                    "api_key": pc.api_key,
                    "base_url": pc.base_url,
                    "models": pc.models,
                    "active_model": pc.active_model,
                    "extra": pc.extra,
                }
                for name, pc in self.providers.items()
            },
            "theme": self.theme,
            "auto_connect": self.auto_connect,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NIAConfig:
        """Create from dictionary."""
        providers = {}
        for name, pdata in data.get("providers", {}).items():
            providers[name] = ProviderConfig(
                api_key=pdata.get("api_key"),
                base_url=pdata.get("base_url"),
                models=pdata.get("models", []),
                active_model=pdata.get("active_model"),
                extra=pdata.get("extra", {}),
            )

        return cls(
            active_provider=data.get("active_provider", ""),
            active_model=data.get("active_model", ""),
            providers=providers,
            theme=data.get("theme", "default"),
            auto_connect=data.get("auto_connect", True),
        )


class ConfigManager:
    """Manages N.I.A configuration.

    Config stored in ~/.nia/config.json
    Credentials can also come from environment variables.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or DEFAULT_CONFIG_FILE
        self._config: NIAConfig | None = None

    @property
    def config(self) -> NIAConfig:
        """Get or load configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config

    def load(self) -> NIAConfig:
        """Load config from disk, falling back to env vars."""
        config = NIAConfig()

        # Load from file if exists
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                config = NIAConfig.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass

        # Auto-detect from environment variables
        if config.auto_connect:
            self._apply_env_detection(config)

        return config

    def save(self, config: NIAConfig | None = None) -> None:
        """Save configuration to disk."""
        config = config or self.config
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def add_provider(self, name: str, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
        """Add or update a provider configuration."""
        self.config.providers[name] = ProviderConfig(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self.save()

    def remove_provider(self, name: str) -> bool:
        """Remove a provider configuration."""
        if name in self.config.providers:
            del self.config.providers[name]
            if self.config.active_provider == name:
                self.config.active_provider = ""
            self.save()
            return True
        return False

    def set_active_provider(self, provider_id: str, model: str | None = None) -> None:
        """Set the active provider and optionally model."""
        self.config.active_provider = provider_id
        if model:
            self.config.active_model = model
            if provider_id in self.config.providers:
                self.config.providers[provider_id].active_model = model
        self.save()

    def get_provider_config(self, name: str) -> ProviderConfig | None:
        """Get configuration for a specific provider."""
        return self.config.providers.get(name)

    def _apply_env_detection(self, config: NIAConfig) -> None:
        """Auto-detect providers from environment variables.

        Only detects providers that have API keys set in env vars.
        Does NOT auto-detect Ollama (user must configure explicitly).
        """
        env_map = {
            "anthropic": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "base_url_env": "ANTHROPIC_BASE_URL",
            },
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "base_url_env": "OPENAI_BASE_URL",
            },
            "groq": {
                "api_key_env": "GROQ_API_KEY",
            },
            "together": {
                "api_key_env": "TOGETHER_API_KEY",
            },
            "deepseek": {
                "api_key_env": "DEEPSEEK_API_KEY",
            },
            "google": {
                "api_key_env": "GOOGLE_API_KEY",
            },
            "nvidia": {
                "api_key_env": "NVIDIA_API_KEY",
                "base_url_default": "https://integrate.api.nvidia.com/v1",
            },
            "cerebras": {
                "api_key_env": "CEREBRAS_API_KEY",
            },
            "fireworks": {
                "api_key_env": "FIREWORKS_API_KEY",
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
            },
        }

        for provider_id, env_config in env_map.items():
            api_key = os.environ.get(env_config.get("api_key_env", ""), "")
            base_url = os.environ.get(env_config.get("base_url_env", ""), "")
            base_url_default = env_config.get("base_url_default", "")

            if api_key or base_url:
                if provider_id not in config.providers:
                    config.providers[provider_id] = ProviderConfig()

                pc = config.providers[provider_id]
                if api_key:
                    pc.api_key = api_key
                if base_url:
                    pc.base_url = base_url
                elif base_url_default and not pc.base_url:
                    pc.base_url = base_url_default

        # Auto-select first configured provider if none set
        if not config.active_provider and config.providers:
            config.active_provider = next(iter(config.providers))
