from models.model_manager import ModelManager


def test_failover_from_openai_to_ollama(monkeypatch):
    # Primary openai raises, fallback ollama returns a valid response
    mm = ModelManager(provider="openai", config={"fallback_providers": ["ollama"]})

    def bad_openai(prompt: str):
        raise RuntimeError("OpenAI quota exceeded")

    def good_ollama(prompt: str):
        return "ollama response"

    monkeypatch.setattr(mm, "_call_openai", bad_openai)
    monkeypatch.setattr(mm, "_call_local", good_ollama)

    resp = mm.reason("Hello world")
    assert resp == "ollama response"


def test_env_fallbacks(monkeypatch):
    # Use env var to set fallback providers
    monkeypatch.setenv("MODEL_PROVIDER_FALLBACKS", "ollama")

    mm = ModelManager(provider="openai")

    def bad_openai(prompt: str):
        raise RuntimeError("OpenAI down")

    def good_ollama(prompt: str):
        return "ok"

    monkeypatch.setattr(mm, "_call_openai", bad_openai)
    monkeypatch.setattr(mm, "_call_local", good_ollama)

    resp = mm.reason("test")
    assert resp == "ok"
