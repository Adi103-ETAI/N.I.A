import builtins
import threading


from interface import chat


def test_main_ignores_eof_then_exits(monkeypatch):
    # Simulate input() first raising EOFError (spurious click) then returning 'exit'
    inputs = [EOFError(), 'exit']

    def fake_input(prompt=''):
        v = inputs.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(builtins, 'input', fake_input)

    # Run main in a thread so we don't block the test process
    t = threading.Thread(target=lambda: chat.main(['--no-plugins'] if False else []))
    t.start()
    # Let the thread make progress; it should return after receiving 'exit'
    t.join(timeout=5)
    assert not t.is_alive()
