class MyPlugin:
    name = "my_plugin"

    def run(self, params):
        # A simple plugin returning params and a friendly message
        return {"ok": True, "message": "Hello from my_plugin (in-process)", "params": params}

# For CLI convenience: allow running directly to return JSON on stdout
if __name__ == "__main__":
    import sys
    import json
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    # Provide a stable JSON reply so subprocess runner can parse output
    print(json.dumps({"ok": True, "message": "Hello from my_plugin (subprocess)", "params": payload}))
