"""Unit tests for models/model_manager.py - LLM Provider Factory.

Tests the ModelManager and ModelFactory classes including:
- Initialization and configuration
- Provider detection
- Model preset selection
- Response generation (mocked)
"""
import pytest
from unittest.mock import patch, MagicMock


class TestModelManagerInit:
    """Test ModelManager initialization."""
    
    def test_model_manager_imports(self):
        """Test that ModelManager can be imported."""
        from models.model_manager import ModelManager, ModelFactory, ModelConfig
        assert ModelManager is not None
        assert ModelFactory is not None
        assert ModelConfig is not None
    
    def test_model_manager_default_init(self):
        """Test ModelManager initializes with defaults."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert mm is not None
        assert mm.config is not None
    
    def test_model_manager_with_provider(self):
        """Test ModelManager with explicit provider."""
        from models.model_manager import ModelManager
        
        mm = ModelManager(provider="nvidia")
        assert mm.provider == "nvidia"
    
    def test_model_manager_with_model_name(self):
        """Test ModelManager with explicit model name."""
        from models.model_manager import ModelManager
        
        mm = ModelManager(model_name="meta/llama-3.1-70b-instruct")
        assert mm.model_name == "meta/llama-3.1-70b-instruct"


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        from models.model_manager import ModelConfig
        
        config = ModelConfig()
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048
        assert config.default_timeout == 60
    
    def test_model_config_preferred_providers(self):
        """Test default provider preferences."""
        from models.model_manager import ModelConfig
        
        config = ModelConfig()
        assert isinstance(config.preferred_providers, list)
        assert len(config.preferred_providers) > 0
    
    def test_model_config_preset_models(self):
        """Test preset model selections."""
        from models.model_manager import ModelConfig
        
        config = ModelConfig()
        assert "llama" in config.smart_model.lower()
        assert "llama" in config.fast_model.lower()


class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_factory_initialization(self):
        """Test ModelFactory can be initialized."""
        from models.model_manager import ModelFactory
        
        factory = ModelFactory()
        assert factory is not None
    
    def test_factory_available_providers(self):
        """Test getting list of available providers."""
        from models.model_manager import ModelFactory
        
        factory = ModelFactory()
        providers = factory.get_available_providers()
        assert isinstance(providers, (list, set))
    
    def test_provider_check(self):
        """Test checking if provider is available."""
        from models.model_manager import ModelFactory
        
        factory = ModelFactory()
        # At least one of these should work
        result = factory.is_provider_available("nvidia") or \
                 factory.is_provider_available("openai") or \
                 factory.is_provider_available("ollama")
        # This just checks the method exists and returns bool
        assert isinstance(result, bool)


class TestModelManagerMethods:
    """Test ModelManager public methods."""
    
    def test_get_smart_model_exists(self):
        """Test that get_smart_model method exists."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "get_smart_model")
        assert callable(mm.get_smart_model)
    
    def test_get_fast_model_exists(self):
        """Test that get_fast_model method exists."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "get_fast_model")
        assert callable(mm.get_fast_model)
    
    def test_get_vision_model_exists(self):
        """Test that get_vision_model method exists."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "get_vision_model")
        assert callable(mm.get_vision_model)
    
    def test_invoke_method_exists(self):
        """Test that invoke method exists."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "invoke")
        assert callable(mm.invoke)
    
    def test_generate_response_exists(self):
        """Test that generate_response method exists."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "generate_response")
        assert callable(mm.generate_response)
    
    def test_reason_legacy_method(self):
        """Test legacy reason method exists for backwards compatibility."""
        from models.model_manager import ModelManager
        
        mm = ModelManager()
        assert hasattr(mm, "reason")
        assert callable(mm.reason)


class TestModelCatalog:
    """Test MODEL_CATALOG definitions."""
    
    def test_model_catalog_exists(self):
        """Test MODEL_CATALOG is defined."""
        from models.model_manager import MODEL_CATALOG
        assert MODEL_CATALOG is not None
        assert isinstance(MODEL_CATALOG, dict)
        assert len(MODEL_CATALOG) > 0
    
    def test_model_spec_structure(self):
        """Test ModelSpec has required fields."""
        from models.model_manager import MODEL_CATALOG, ModelSpec
        
        # Get first model spec
        first_key = list(MODEL_CATALOG.keys())[0]
        spec = MODEL_CATALOG[first_key]
        
        assert isinstance(spec, ModelSpec)
        assert hasattr(spec, "provider")
        assert hasattr(spec, "model_name")
        assert hasattr(spec, "display_name")


class TestProviderEnum:
    """Test Provider enum."""
    
    def test_provider_enum_values(self):
        """Test Provider enum has expected values."""
        from models.model_manager import Provider
        
        assert Provider.NVIDIA.value == "nvidia"
        assert Provider.OPENAI.value == "openai"
        assert Provider.OLLAMA.value == "ollama"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
