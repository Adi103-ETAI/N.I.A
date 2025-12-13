"""Memory system for NIA - single corrected implementation.

Optional vector features (FAISS) are supported but not required to run.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Avoid heavy or platform-specific imports at module import time. VectorStore
# will perform lazy imports of numpy and faiss when instantiated.


logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    @abstractmethod
    def store(self, collection: str, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def search(self, collection: str, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def delete(self, collection: str, key: str) -> bool:
        raise NotImplementedError()


class SqliteBackend(MemoryBackend):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    collection TEXT,
                    key TEXT,
                    value TEXT,
                    created_at TIMESTAMP,
                    PRIMARY KEY (collection, key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON memory(collection)")

    def store(self, collection: str, key: str, value: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory (collection, key, value, created_at) VALUES (?, ?, ?, ?)",
                (collection, key, json.dumps(value), datetime.now().isoformat()),
            )

    def retrieve(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM memory WHERE collection = ? AND key = ?", (collection, key)).fetchone()
            return json.loads(row[0]) if row else None

    def search(self, collection: str, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        text = query.get("text", "") or ""
        pattern = f"%{text}%"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT value FROM memory WHERE collection = ? AND value LIKE ? LIMIT ?", (collection, pattern, limit)).fetchall()
            return [json.loads(r[0]) for r in rows]

    def delete(self, collection: str, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM memory WHERE collection = ? AND key = ?", (collection, key))
            return cur.rowcount > 0


class VectorStore:
    def __init__(self, dimension: int, index_path: Optional[str] = None, cache_size: int = 10000, save_interval: int = 300, vector_ttl: Optional[int] = None) -> None:
        # Lazy import numpy and faiss to avoid heavy imports when the module
        # itself is imported but VectorStore is not used.
        try:
            import numpy as np
        except Exception:
            raise ImportError("numpy is required for VectorStore")
        try:
            import faiss
        except Exception:
            raise ImportError("FAISS is required for VectorStore")

        # Save modules on instance to prefer instance-level references rather
        # than relying on module-level variables.
        self.np = np
        self.faiss = faiss

        self.dimension = int(dimension)
        self.index_path = index_path
        self.cache_size = int(cache_size)
        self.save_interval = int(save_interval)
        self.vector_ttl = int(vector_ttl) if vector_ttl is not None else None

        self._lock = threading.RLock()
        self._last_save = time.time()

        if index_path and os.path.exists(index_path):
            self.index = self.faiss.read_index(index_path)
        else:
            self.index = self.faiss.IndexFlatL2(self.dimension)

        self._vector_map: Dict[int, str] = {}
        self._key_map: Dict[str, int] = {}
        self._next_id = 0

        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def add_vector(self, key: str, vector: List[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}")

        with self._lock:
            arr = self.np.asarray([vector], dtype=self.np.float32)
            self.index.add(arr)

            vid = self._next_id
            self._next_id += 1
            self._vector_map[vid] = key
            self._key_map[key] = vid

            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = arr
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

            self._timestamps[key] = time.time()

            if self.index_path and (time.time() - self._last_save) > self.save_interval:
                try:
                    self.faiss.write_index(self.index, self.index_path)
                    self._last_save = time.time()
                except Exception:
                    logger.exception("Failed to auto-save FAISS index")

    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[str, float]]:
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query dimension mismatch: expected {self.dimension}")

        with self._lock:
            now = time.time()
            if self.vector_ttl:
                expired = [kk for kk, ts in self._timestamps.items() if now - ts > self.vector_ttl]
                for key in expired:
                    self._remove_vector(key)

            D, idxs = self.index.search(self.np.asarray([query_vector], dtype=self.np.float32), k)
            out: List[Tuple[str, float]] = []
            for idx, dist in zip(idxs[0], D[0]):
                if idx == -1:
                    continue
                key = self._vector_map.get(int(idx))
                if key is None:
                    continue
                if self.vector_ttl and (now - self._timestamps.get(key, 0) > self.vector_ttl):
                    continue
                out.append((key, float(dist)))
            return out

    def _remove_vector(self, key: str) -> None:
        if key in self._key_map:
            vid = self._key_map.pop(key)
            self._vector_map.pop(vid, None)
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def save(self) -> None:
        if not self.index_path:
            return
        with self._lock:
            faiss.write_index(self.index, self.index_path)
            self._last_save = time.time()


@dataclass
class MemoryQuery:
    collection: str
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 10


class MemoryManager:
    def __init__(self, db_path: str, model_dim: int = 1536, index_path: Optional[str] = None, max_memories: int = 1_000_000, cache_size: int = 10000, save_interval: int = 300, memory_ttl: Optional[int] = None, vector_backend: str = "faiss") -> None:
        self.store = SqliteBackend(db_path)
        self.db_path = db_path
        self.max_memories = int(max_memories)
        self.memory_ttl = int(memory_ttl) if memory_ttl is not None else None

        backend = (vector_backend or "faiss").lower()
        # Attempt to instantiate a VectorStore only when the backend is
        # 'faiss'. VectorStore performs its own lazy import of numpy/faiss
        # and will raise ImportError on missing deps; handle that gracefully.
        if backend == "faiss":
            try:
                self.vectors = VectorStore(dimension=model_dim, index_path=index_path, cache_size=cache_size, save_interval=save_interval, vector_ttl=memory_ttl)
                self._has_vectors = True
            except Exception:
                # Missing optional dependencies or init failure: fall back
                self.vectors = None
                self._has_vectors = False
        elif backend in {"pinecone", "weaviate", "external"}:
            self.vectors = None
            self._has_vectors = False
        else:
            self.vectors = None
            self._has_vectors = False

        self._context: Dict[str, Any] = {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_meta (
                    collection TEXT,
                    key TEXT,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    PRIMARY KEY (collection, key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON memory_meta(expires_at)")

    def store_memory(self, collection: str, key: str, memory: Dict[str, Any], embedding: Optional[List[float]] = None, ttl: Optional[int] = None) -> None:
        stats = self.get_stats()
        if stats.get("total_memories", 0) >= self.max_memories:
            self._cleanup_old_memories()
            stats = self.get_stats()
            if stats.get("total_memories", 0) >= self.max_memories:
                raise RuntimeError("Memory limit reached")

        created_at = datetime.now()
        expires_at = None
        use_ttl = ttl if ttl is not None else self.memory_ttl
        if use_ttl:
            expires_at = created_at + timedelta(seconds=use_ttl)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO memory_meta (collection, key, created_at, expires_at) VALUES (?, ?, ?, ?)", (collection, key, created_at.isoformat(), expires_at.isoformat() if expires_at else None))

        memory = dict(memory)
        memory.setdefault("created_at", created_at.isoformat())
        if expires_at:
            memory["expires_at"] = expires_at.isoformat()

        self.store.store(collection, key, memory)

        if embedding and self._has_vectors and self.vectors is not None:
            self.vectors.add_vector(key, embedding)

    def get_memory(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT expires_at FROM memory_meta WHERE collection = ? AND key = ?", (collection, key)).fetchone()
            if row and row[0]:
                try:
                    expires_at = datetime.fromisoformat(row[0])
                    if datetime.now() > expires_at:
                        self._remove_memory(collection, key)
                        return None
                except Exception:
                    pass

        return self.store.retrieve(collection, key)

    def search_similar(self, collection: str, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        if not self._has_vectors or self.vectors is None:
            raise RuntimeError("Vector search not available (configure `vector_backend` or integrate an external provider)")
        hits = self.vectors.search(query_embedding, k=limit)
        results: List[Dict[str, Any]] = []
        for key, score in hits:
            mem = self.get_memory(collection, key)
            if mem:
                mem = dict(mem)
                mem["similarity_score"] = score
                results.append(mem)
        return results

    def search_text(self, collection: str, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.store.search(collection, {"text": text}, limit)

    def update_context(self, updates: Dict[str, Any]) -> None:
        self._context.update(updates)

    def get_context(self) -> Dict[str, Any]:
        return dict(self._context)

    def search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if query.embedding and self._has_vectors and self.vectors is not None:
            results = self.search_similar(query.collection, query.embedding, query.limit)

        if not results and query.text:
            results = self.search_text(query.collection, query.text, query.limit)

        if (query.start_time or query.end_time) and results:
            filtered: List[Dict[str, Any]] = []
            for m in results:
                try:
                    created = datetime.fromisoformat(m.get("created_at", ""))
                except Exception:
                    created = None
                if query.start_time and created and created < query.start_time:
                    continue
                if query.end_time and created and created > query.end_time:
                    continue
                filtered.append(m)
            results = filtered[: query.limit]

        return results

    def _remove_memory(self, collection: str, key: str) -> None:
        self.store.delete(collection, key)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory_meta WHERE collection = ? AND key = ?", (collection, key))

    def _cleanup_old_memories(self) -> int:
        removed = 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM memory_meta WHERE expires_at IS NOT NULL AND expires_at < ?", (datetime.now().isoformat(),))
            removed += cur.rowcount

            cur_total = conn.execute("SELECT COUNT(*) FROM memory").fetchone()
            total = cur_total[0] if cur_total else 0
            if total >= self.max_memories:
                rows = conn.execute("SELECT collection, key FROM memory_meta ORDER BY created_at ASC LIMIT ?", (min(1000, total),)).fetchall()
                for col, key in rows:
                    self._remove_memory(col, key)
                    removed += 1

        return removed

    def clear_collection(self, collection: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM memory WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM memory_meta WHERE collection = ?", (collection,))
            return cur.rowcount

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"total_memories": 0, "by_category": {}, "vector_enabled": self._has_vectors, "db_size_bytes": 0}
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT collection, COUNT(*) FROM memory GROUP BY collection"):
                stats["by_category"][row[0]] = row[1]
                stats["total_memories"] += row[1]

        try:
            stats["db_size_bytes"] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        except Exception:
            stats["db_size_bytes"] = 0

        if self._has_vectors and self.vectors is not None:
            stats["vector_count"] = len(self.vectors._timestamps)
            if self.vectors.index_path and os.path.exists(self.vectors.index_path):
                stats["vector_size_bytes"] = os.path.getsize(self.vectors.index_path)

        return stats


class InMemoryMemory(MemoryBackend):
    def __init__(self) -> None:
        self._store: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def store(self, collection: str, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._store.append({"collection": collection, "key": key, "value": value})

    def retrieve(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for item in self._store:
                if item["collection"] == collection and item["key"] == key:
                    return item["value"]
            return None

    def search(self, collection: str, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            res: List[Dict[str, Any]] = []
            text = query.get("text", "") or ""
            for item in self._store:
                if item["collection"] != collection:
                    continue
                if text and text not in json.dumps(item["value"]):
                    continue
                res.append(item["value"])
                if len(res) >= limit:
                    break
            return res

    def delete(self, collection: str, key: str) -> bool:
        with self._lock:
            for i, item in enumerate(self._store):
                if item["collection"] == collection and item["key"] == key:
                    self._store.pop(i)
                    return True
            return False
