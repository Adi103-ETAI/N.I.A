"""Test script for new configuration system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.loader import load_settings


def test_config_loading():
    """Test that configuration loads correctly."""
    print("=" * 60)
    print("N.I.A. v4.0.0 Configuration Test")
    print("=" * 60)
    
    try:
        settings = load_settings()
        print("\n[OK] Configuration loaded successfully!\n")
        
        # Test NIA config
        print("[NIA] Configuration:")
        print(f"   Name: {settings.nia.name}")
        print(f"   Version: {settings.nia.version}")
        print(f"   Log Level: {settings.nia.log_level}")
        print(f"   Routing Mode: {settings.nia.routing_mode}")
        print(f"   Gatekeeper Enabled: {settings.nia.gatekeeper.enabled}")
        print(f"   Graph Max Iterations: {settings.nia.graph.max_iterations}")
        
        # Test TARA config
        print("\n[TARA] Configuration:")
        print(f"   Name: {settings.tara.name}")
        print(f"   Max Tool Retries: {settings.tara.max_tool_retries}")
        print(f"   Tool Timeout: {settings.tara.tool_timeout}s")
        
        # Test IRIS config
        print("\n[IRIS] Configuration:")
        print(f"   Name: {settings.iris.name}")
        print(f"   Vision Model: {settings.iris.vision_model}")
        print(f"   Sentry Scan Interval: {settings.iris.sentry.scan_interval}s")
        
        # Test NOLA config
        print("\n[NOLA] Configuration:")
        print(f"   Name: {settings.nola.name}")
        print(f"   Wake Word: {settings.nola.wake_word}")
        print(f"   TTS Voice: {settings.nola.tts_voice}")
        
        # Test model config
        print("\n[MODELS] Configuration:")
        print(f"   Default Provider: {settings.default_provider}")
        print(f"   Valid Providers: {settings.valid_providers}")
        print(f"   Models Loaded: {len(settings.models)}")
        print(f"   Providers Loaded: {len(settings.providers)}")
        print(f"   Fallback Chain: {settings.fallback_chain}")
        
        # Test desktop config
        print("\n[DESKTOP] Configuration:")
        print(f"   System Apps: {settings.desktop.system_apps[:5]}...")
        print(f"   UIA Max Elements: {settings.uia.max_elements}")
        
        print("\n" + "=" * 60)
        print("[OK] All configuration tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config_loading()
    sys.exit(0 if success else 1)

