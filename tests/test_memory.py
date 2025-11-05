from core.memory import MemoryManager, InMemoryMemory
import tempfile
import os
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_inmemory_basic_crud():
    mem = InMemoryMemory()
    mem.store('test', 'key1', {'val': 1})
    assert mem.retrieve('test', 'key1')['val'] == 1
    assert mem.delete('test', 'key1') is True
    assert mem.retrieve('test', 'key1') is None

def test_sqlite_memorymanager_crud():
    tf = tempfile.NamedTemporaryFile(delete=False)
    path = tf.name
    tf.close()
    try:
        mgr = MemoryManager(path, model_dim=8)
        mgr.store_memory('col', 'k', {'foo': 'bar'})
        item = mgr.get_memory('col', 'k')
        assert item['foo'] == 'bar'
        mgr._remove_memory('col', 'k')
        assert mgr.get_memory('col', 'k') is None
    finally:
        import gc
        gc.collect()
        if os.path.exists(path):
            os.unlink(path)

def test_ttl_expiry_and_cleanup():
    tf = tempfile.NamedTemporaryFile(delete=False)
    path = tf.name
    tf.close()
    try:
        mgr = MemoryManager(path, model_dim=8, memory_ttl=1)
        mgr.store_memory('col', 'old', {'foo': 1})
        time.sleep(1.2)
        assert mgr.get_memory('col', 'old') is None
    finally:
        import gc
        gc.collect()
        if os.path.exists(path):
            os.unlink(path)

def test_stats_and_clear():
    tf = tempfile.NamedTemporaryFile(delete=False)
    path = tf.name
    tf.close()
    try:
        mgr = MemoryManager(path, model_dim=8)
        for i in range(5):
            mgr.store_memory('bucket', f'k{i}', {"num": i})
        stats = mgr.get_stats()
        assert stats['total_memories'] >= 5
        count = mgr.clear_collection('bucket')
        assert count == 5
    finally:
        import gc
        gc.collect()
        if os.path.exists(path):
            os.unlink(path)
