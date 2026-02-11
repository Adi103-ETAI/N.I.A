
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.getcwd())

print("Testing Configuration Loading...")
print("-" * 50)

# 1. Test Model Catalog (NIA)
try:
    from src.models.manager import ModelManager
    MODEL_CATALOG = ModelManager()._catalog
    print(f"[OK] NIA Models: Loaded {len(MODEL_CATALOG)} models")
except Exception as e:
    print(f"[FAIL] NIA Models Failed: {e}")

# 2. Test Routing Map (NIA)
try:
    from src.agents.nia.graph.nodes import get_routing_keywords
    tara_kw, iris_kw = get_routing_keywords()
    print(f"[OK] NIA Routing: Loaded {len(tara_kw)} TARA keywords, {len(iris_kw)} IRIS keywords")
except Exception as e:
    print(f"[FAIL] NIA Routing Failed: {e}")

# 3. Test Vision Triggers (IRIS)
try:
    from src.agents.iris.agent import _load_iris_config
    config = _load_iris_config()
    print(f"[OK] IRIS Vision: Loaded {len(config.get('screen_keywords', []))} screen keywords")
except Exception as e:
    print(f"[FAIL] IRIS Vision Failed: {e}")

# 4. Test Commands (TARA)
try:
    from src.core.engine import _COMMANDS
    print(f"[OK] TARA Commands: Loaded {len(_COMMANDS)} command categories")
except Exception as e:
    print(f"[FAIL] TARA Commands Failed: {e}")

print("-" * 50)
