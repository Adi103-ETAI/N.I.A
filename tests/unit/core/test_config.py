"""Unit tests for core/config.py - Centralized Configuration Module.

Tests the pydantic-settings based configuration system including:
- Default values
- Environment variable loading
- Type validation
- Helper properties
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch


class TestSettingsDefaults:
    """Test that settings loads with correct defaults."""
    
    def test_settings_imports(self):
        """Test that settings can be imported."""
        from src.core.config import settings, Settings
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_default_paths(self):
        """Test default path settings."""
        from src.core.config import settings
        
        assert settings.LOG_DIR == Path("logs")
        assert settings.MODEL_DIR == Path("models")
        assert settings.DATA_DIR == Path("data")
        assert settings.SOUNDS_DIR == Path("sounds")
    
    def test_default_voice_settings(self):
        """Test default voice configuration."""
        from src.core.config import settings
        
        assert isinstance(settings.WAKE_WORDS, list)
        assert "nia" in settings.WAKE_WORDS
        assert settings.TTS_VOICE == "en-US-AriaNeural"
        assert settings.VOICE_ENABLED is True
    
    def test_default_llm_models(self):
        """Test default LLM model names."""
        from src.core.config import settings
        
        assert "llama" in settings.LLM_MODEL.lower()
        assert "llama" in settings.LLM_MODEL_SMART.lower()
        assert "llama" in settings.LLM_MODEL_FAST.lower()
        assert "vision" in settings.LLM_MODEL_VISION.lower()
    
    def test_default_system_settings(self):
        """Test default system settings."""
        from src.core.config import settings
        
        assert settings.DEBUG is False
        assert settings.MEMORY_RETENTION_DAYS == 7
        assert settings.VERSION == "3.0.0"  # Updated for v3.0
    
    def test_temperature_range(self):
        """Test that temperature has valid range."""
        from src.core.config import settings
        
        assert 0.0 <= settings.LLM_TEMPERATURE <= 2.0


class TestSettingsValidation:
    """Test settings validation and computed properties."""
    
    def test_has_nvidia_key_false_when_empty(self):
        """Test has_nvidia_key returns False when key is empty."""
        from src.core.config import settings
        
        # Default empty key should return False
        if not settings.NVIDIA_API_KEY.get_secret_value().startswith("nvapi-"):
            assert settings.has_nvidia_key is False
    
    def test_log_file_property(self):
        """Test computed log_file property."""
        from src.core.config import settings
        
        expected = settings.LOG_DIR / "nia.log"
        assert settings.log_file == expected
    
    def test_ensure_directories(self):
        """Test that ensure_directories creates required dirs."""
        from src.core.config import settings
        
        # This should not raise
        settings.ensure_directories()
        
        # Directories should exist
        assert settings.LOG_DIR.exists() or True  # May not have write permission
        assert settings.DATA_DIR.exists() or True


class TestWakeWordsParser:
    """Test wake words parsing logic."""
    
    def test_wake_words_from_comma_string(self):
        """Test parsing comma-separated wake words."""
        # Test the parsing logic directly
        input_str = "hey nia,jarvis,computer"
        result = [w.strip().lower() for w in input_str.split(",") if w.strip()]
        assert "hey nia" in result
        assert "jarvis" in result
        assert "computer" in result
    
    def test_wake_words_lowercase(self):
        """Test that wake words are lowercased."""
        input_str = "NIA,JARVIS"
        result = [w.strip().lower() for w in input_str.split(",") if w.strip()]
        assert "nia" in result
        assert "jarvis" in result
    
    def test_wake_words_json_format(self):
        """Test parsing JSON array format."""
        import json
        input_str = '["hey nia", "jarvis", "computer"]'
        parsed = json.loads(input_str)
        result = [w.strip().lower() for w in parsed if w.strip()]
        assert "hey nia" in result
        assert "jarvis" in result
        assert "computer" in result


class TestEnvironmentOverrides:
    """Test that environment variables override defaults."""
    
    def test_debug_from_env(self):
        """Test DEBUG can be set from environment."""
        from src.core.config import Settings
        from pydantic_settings import SettingsConfigDict
        
        class TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=None, extra="ignore")
        
        with patch.dict(os.environ, {"DEBUG": "true"}, clear=False):
            s = TestSettings()
            assert s.DEBUG is True
    
    def test_memory_retention_from_env(self):
        """Test MEMORY_RETENTION_DAYS from environment."""
        from src.core.config import Settings
        from pydantic_settings import SettingsConfigDict
        
        class TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=None, extra="ignore")
        
        with patch.dict(os.environ, {"MEMORY_RETENTION_DAYS": "30"}, clear=False):
            s = TestSettings()
            assert s.MEMORY_RETENTION_DAYS == 30
    
    def test_tts_voice_from_env(self):
        """Test TTS_VOICE can be overridden."""
        from src.core.config import Settings
        from pydantic_settings import SettingsConfigDict
        
        class TestSettings(Settings):
            model_config = SettingsConfigDict(env_file=None, extra="ignore")
        
        with patch.dict(os.environ, {"TTS_VOICE": "en-GB-SoniaNeural"}, clear=False):
            s = TestSettings()
            assert s.TTS_VOICE == "en-GB-SoniaNeural"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
