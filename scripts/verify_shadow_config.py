
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("Testing Shadow Config Loading...")
print("-" * 50)

# 1. Test UIA Ops (TARA)
try:
    from tara.tools.uia_ops import ACTIONABLE_TYPES, SKIP_TYPES, MAX_ELEMENTS
    print(f"[OK] UIA: Loaded {len(ACTIONABLE_TYPES)} actionable types (Type: {type(ACTIONABLE_TYPES)})")
    print(f"[OK] UIA: Loaded {len(SKIP_TYPES)} skip types")
    print(f"[OK] UIA: MAX_ELEMENTS = {MAX_ELEMENTS}")
    
    # Verify set conversion
    if isinstance(ACTIONABLE_TYPES, set):
        print("[OK] UIA: ACTIONABLE_TYPES is correctly a set")
    else:
        print(f"[FAIL] UIA: ACTIONABLE_TYPES is {type(ACTIONABLE_TYPES)}, expected set")
        
except Exception as e:
    print(f"[FAIL] UIA Ops Failed: {e}")

# 2. Test Plugins (TARA)
try:
    from tara.plugin_system.loader import DEFAULT_PLUGINS_DIR, LOAD_TIMEOUT_WARNING
    from tara.plugin_system.watcher import DEBOUNCE_DELAY
    print(f"[OK] Plugins: Dir='{DEFAULT_PLUGINS_DIR}', Timeout={LOAD_TIMEOUT_WARNING}")
    print(f"[OK] Watcher: Debounce={DEBOUNCE_DELAY}")
except Exception as e:
    print(f"[FAIL] Plugin System Failed: {e}")

# 3. Test Sentry (IRIS)
try:
    from iris.sentry import TRIGGERS, SCAN_INTERVAL
    print(f"[OK] Sentry: Interval={SCAN_INTERVAL}")
    print(f"[OK] Sentry: Triggers has {len(TRIGGERS)} categories")
except Exception as e:
    print(f"[FAIL] Sentry Failed: {e}")

# 4. Test Model Manager (NIA)
try:
    from models.model_manager import DEFAULT_PROVIDER, VALID_PROVIDERS
    print(f"[OK] NIA: Default Provider='{DEFAULT_PROVIDER}'")
    print(f"[OK] NIA: Valid Providers ({len(VALID_PROVIDERS)}) = {VALID_PROVIDERS}")
    
    # Verify frozenset conversion
    if isinstance(VALID_PROVIDERS, frozenset):
        print("[OK] NIA: VALID_PROVIDERS is correctly a frozenset")
    else:
        print(f"[FAIL] NIA: VALID_PROVIDERS is {type(VALID_PROVIDERS)}, expected frozenset")

except Exception as e:
    print(f"[FAIL] Model Manager Failed: {e}")

# 5. Test NOLA (Voice)
try:
    from nola.io.speech import PLAYBACK_POLL_INTERVAL, PIPER_TIMEOUT_SEC
    from nola.io.hearing import VOSK_MODEL_PATH
    
    print(f"[OK] NOLA Speech: Poll={PLAYBACK_POLL_INTERVAL}, Timeout={PIPER_TIMEOUT_SEC}")
    
    # Check if VOSK path ends with 'vosk_model' (from JSON)
    if "vosk_model" in str(VOSK_MODEL_PATH):
        print(f"[OK] NOLA Hearing: VOSK_MODEL_PATH correctly points to vosk_model")
    else:
        print(f"[FAIL] NOLA Hearing: VOSK_MODEL_PATH = {VOSK_MODEL_PATH}")

except Exception as e:
    print(f"[FAIL] NOLA Failed: {e}")

print("-" * 50)
