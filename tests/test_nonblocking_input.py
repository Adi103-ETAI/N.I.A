import sys
import time
import queue
from types import SimpleNamespace

from core.nonblocking_input import read_input_with_queue


def test_prefers_queue_item():
    q = queue.Queue()
    q.put_nowait('spoken')
    assert read_input_with_queue('> ', q, timeout=0.1) == 'spoken'


def test_windows_path_simulated():
    q = queue.Queue()
    seq = list('abc\n')

    class FakeM:
        def __init__(self):
            self.i = 0

        def kbhit(self):
            return self.i < len(seq)

        def getwche(self):
            ch = seq[self.i]
            self.i += 1
            return ch

    fake = FakeM()
    result = read_input_with_queue('> ', q, timeout=1, msvcrt_module=fake)
    assert result == 'abc'


def test_posix_select_simulated(monkeypatch):
    q = queue.Queue()

    def fake_select(r, w, x, timeout):
        # indicate stdin is ready
        return (r, w, x)

    fake_sel = SimpleNamespace(select=fake_select)
    fake_stdin = SimpleNamespace(readline=lambda: 'typed\n')
    res = read_input_with_queue('> ', q, timeout=1, select_module=fake_sel, stdin=fake_stdin, msvcrt_module=False)
    assert res == 'typed'


def test_timeout_returns_empty():
    q = queue.Queue()
    # No msvcrt or select; input() would block, so we rely on timeout
    res = read_input_with_queue('> ', q, timeout=0.01, select_module=None, msvcrt_module=False)
    assert res == ''
