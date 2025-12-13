import time
import queue
from types import SimpleNamespace

from core.voice_manager import BackgroundListener, normalize_listen_result


class FakeVoiceManager:
    def __init__(self):
        self.calls = 0

    def listen(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {'text': 'hello-from-asr'}
        return None


def test_background_listener_polls_and_enqueues():
    q = queue.Queue()
    vm = FakeVoiceManager()
    bl = BackgroundListener(vm, output_queue=q, poll_interval=0.01, timeout=0.1)
    bl.start()
    time.sleep(0.05)
    bl.stop()
    assert q.get_nowait() == 'hello-from-asr'
