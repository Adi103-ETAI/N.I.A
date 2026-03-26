"""ChromaDB namespace isolation for subagent memory.

Each subagent gets its own ChromaDB collection (``agent_{uuid}``).
On success the coordinator merges novel documents into the shared
``nia_global`` collection; on failure the namespace is dropped without
polluting the global index.

Usage::

    from src.core.memory.namespaces import get_namespace_manager

    ns = get_namespace_manager()
    ns.create_agent_namespace("abc-123")
    await ns.store_in_namespace("abc-123", "found the config file", role="assistant")
    results = await ns.recall_from_namespace("abc-123", "config file location")
    await ns.merge_namespace("abc-123")   # success path
    ns.drop_namespace("abc-123")          # failure path (alternative)
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.logger import setup_logger
from src.core.config import get_embedding_function

logger = setup_logger("MEMORY.Namespaces")

# ---------------------------------------------------------------------------
# Optional dependency guard (mirrors manager.py)
# ---------------------------------------------------------------------------
try:
    import chromadb  # noqa: F401 — used for type references
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    logger.warning("chromadb not installed. Namespace isolation disabled.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GLOBAL_COLLECTION_NAME = "nia_global"
_AGENT_PREFIX = "agent_"

# Cosine distance below which two documents are considered duplicates.
# ChromaDB cosine distance ranges 0 (identical) to 2 (opposite).
_DEDUP_DISTANCE_THRESHOLD = 0.15


# =============================================================================
# NamespaceManager
# =============================================================================

class NamespaceManager:
    """Manages per-subagent ChromaDB namespaces with isolation and merge."""

    def __init__(self, memory_manager: Any | None = None) -> None:
        """Initialise using the shared ChromaDB client from *memory_manager*.

        Parameters
        ----------
        memory_manager:
            An existing :class:`MemoryManager` instance.  When *None* the
            singleton returned by ``get_memory_manager()`` is used.
        """
        if memory_manager is None:
            from src.core.memory.manager import get_memory_manager
            memory_manager = get_memory_manager()

        self._client: chromadb.ClientAPI | None = getattr(
            memory_manager, "_chroma_client", None
        )
        self._lock = threading.Lock()
        self._embedding_fn = get_embedding_function()

        # Eagerly create / connect to the global collection.
        self._global = self._get_or_create_global()

        logger.debug(
            "NamespaceManager initialised (client=%s, global=%s)",
            "ok" if self._client else "NONE",
            "ok" if self._global else "NONE",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_global(self) -> Any | None:
        """Return the ``nia_global`` collection, creating it if needed."""
        if not _HAS_CHROMADB or self._client is None:
            return None
        try:
            kwargs: Dict[str, Any] = {"name": _GLOBAL_COLLECTION_NAME}
            if self._embedding_fn:
                kwargs["embedding_function"] = self._embedding_fn
            return self._client.get_or_create_collection(**kwargs)
        except Exception as exc:
            logger.error("Failed to init global collection: %s", exc, exc_info=True)
            return None

    def _get_agent_collection(self, agent_id: str) -> Any | None:
        """Return the agent's collection or *None* if it does not exist."""
        if not _HAS_CHROMADB or self._client is None:
            return None
        name = f"{_AGENT_PREFIX}{agent_id}"
        try:
            return self._client.get_collection(
                name=name,
                embedding_function=self._embedding_fn or None,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API — namespace lifecycle
    # ------------------------------------------------------------------

    def create_agent_namespace(self, agent_id: str) -> str:
        """Create an isolated ChromaDB collection for *agent_id*.

        Returns the collection name (``agent_{agent_id}``).
        """
        if not _HAS_CHROMADB or self._client is None:
            logger.warning("ChromaDB unavailable — cannot create namespace")
            return f"{_AGENT_PREFIX}{agent_id}"

        name = f"{_AGENT_PREFIX}{agent_id}"
        with self._lock:
            try:
                kwargs: Dict[str, Any] = {"name": name}
                if self._embedding_fn:
                    kwargs["embedding_function"] = self._embedding_fn
                self._client.get_or_create_collection(**kwargs)
                logger.info("Created agent namespace: %s", name)
            except Exception as exc:
                logger.error(
                    "Failed to create namespace %s: %s", name, exc, exc_info=True,
                )
        return name

    def drop_namespace(self, agent_id: str) -> bool:
        """Delete the agent collection (no-pollution cleanup on failure).

        Returns *True* if the collection was dropped, *False* if it was not
        found or ChromaDB is unavailable.
        """
        if not _HAS_CHROMADB or self._client is None:
            return False

        name = f"{_AGENT_PREFIX}{agent_id}"
        with self._lock:
            try:
                self._client.delete_collection(name=name)
                logger.info("Dropped agent namespace: %s", name)
                return True
            except Exception:
                logger.debug("Namespace %s not found (already dropped?)", name)
                return False

    def list_active_namespaces(self) -> list[str]:
        """Return names of all collections matching ``agent_*``."""
        if not _HAS_CHROMADB or self._client is None:
            return []

        try:
            all_cols = self._client.list_collections()
            # ChromaDB >=0.4 returns Collection objects; older versions
            # return dicts.  Handle both.
            names: list[str] = []
            for col in all_cols:
                col_name = col.name if hasattr(col, "name") else str(col)
                if col_name.startswith(_AGENT_PREFIX):
                    names.append(col_name)
            return sorted(names)
        except Exception as exc:
            logger.error("list_active_namespaces failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Public API — store / recall
    # ------------------------------------------------------------------

    async def store_in_namespace(
        self,
        agent_id: str,
        text: str,
        role: str = "assistant",
        metadata: dict | None = None,
    ) -> bool:
        """Store a document in the agent's namespace collection.

        Uses ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        if not _HAS_CHROMADB or self._client is None:
            return False

        try:
            return await asyncio.to_thread(
                self._store_in_namespace_sync, agent_id, text, role, metadata,
            )
        except Exception as exc:
            logger.error("store_in_namespace failed: %s", exc, exc_info=True)
            return False

    def _store_in_namespace_sync(
        self,
        agent_id: str,
        text: str,
        role: str,
        metadata: dict | None,
    ) -> bool:
        """Sync implementation — runs inside ``to_thread``."""
        collection = self._get_agent_collection(agent_id)
        if collection is None:
            logger.warning("Namespace for %s not found — creating on-the-fly", agent_id)
            self.create_agent_namespace(agent_id)
            collection = self._get_agent_collection(agent_id)
            if collection is None:
                return False

        doc_id = f"{_AGENT_PREFIX}{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        meta = metadata.copy() if metadata else {}
        meta.update({
            "role": role,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
        })

        collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return True

    async def recall_from_namespace(
        self,
        agent_id: str,
        query: str,
        n: int = 5,
    ) -> list[str]:
        """Query the agent namespace; fall back to global if empty."""
        if not _HAS_CHROMADB or self._client is None:
            return []

        try:
            return await asyncio.to_thread(
                self._recall_from_namespace_sync, agent_id, query, n,
            )
        except Exception as exc:
            logger.error("recall_from_namespace failed: %s", exc, exc_info=True)
            return []

    def _recall_from_namespace_sync(
        self,
        agent_id: str,
        query: str,
        n: int,
    ) -> list[str]:
        """Sync implementation — runs inside ``to_thread``."""
        collection = self._get_agent_collection(agent_id)
        if collection is not None and collection.count() > 0:
            results = collection.query(query_texts=[query], n_results=min(n, collection.count()))
            docs = results.get("documents", [[]])[0]
            if docs:
                return docs

        # Fallback to global
        return self._recall_from_global_sync(query, n)

    async def recall_from_global(self, query: str, n: int = 5) -> list[str]:
        """Query the global ``nia_global`` collection."""
        if not _HAS_CHROMADB or self._global is None:
            return []

        try:
            return await asyncio.to_thread(self._recall_from_global_sync, query, n)
        except Exception as exc:
            logger.error("recall_from_global failed: %s", exc, exc_info=True)
            return []

    def _recall_from_global_sync(self, query: str, n: int) -> list[str]:
        """Sync implementation — runs inside ``to_thread``."""
        if self._global is None or self._global.count() == 0:
            return []
        results = self._global.query(
            query_texts=[query], n_results=min(n, self._global.count()),
        )
        return results.get("documents", [[]])[0]

    # ------------------------------------------------------------------
    # Public API — merge
    # ------------------------------------------------------------------

    async def merge_namespace(self, agent_id: str) -> bool:
        """Merge agent namespace into global, deduplicate, then drop.

        Deduplication: for each agent document, query the global collection.
        If the nearest neighbour distance is below ``_DEDUP_DISTANCE_THRESHOLD``
        the document is considered a duplicate and skipped.

        Returns *True* on success, *False* on error.
        """
        if not _HAS_CHROMADB or self._client is None or self._global is None:
            return False

        try:
            return await asyncio.to_thread(self._merge_namespace_sync, agent_id)
        except Exception as exc:
            logger.error("merge_namespace failed: %s", exc, exc_info=True)
            return False

    def _merge_namespace_sync(self, agent_id: str) -> bool:
        """Sync merge — runs inside ``to_thread``."""
        collection = self._get_agent_collection(agent_id)
        if collection is None:
            logger.warning("merge_namespace: collection for %s not found", agent_id)
            return False

        count = collection.count()
        if count == 0:
            # Nothing to merge — just drop.
            self.drop_namespace(agent_id)
            return True

        # Fetch all documents from the agent namespace.
        data = collection.get(include=["documents", "metadatas"])
        documents: list[str] = data.get("documents", [])
        metadatas: list[dict] = data.get("metadatas", [])

        merged = 0
        skipped = 0

        for idx, doc in enumerate(documents):
            if not doc:
                continue

            # --- dedup check against global ---
            is_duplicate = False
            if self._global.count() > 0:
                result = self._global.query(
                    query_texts=[doc],
                    n_results=1,
                    include=["distances"],
                )
                distances = result.get("distances", [[]])[0]
                if distances and distances[0] < _DEDUP_DISTANCE_THRESHOLD:
                    is_duplicate = True

            if is_duplicate:
                skipped += 1
                continue

            # --- write to global ---
            meta = metadatas[idx] if idx < len(metadatas) else {}
            meta = meta.copy() if meta else {}
            meta["merged_from"] = agent_id
            meta["merged_at"] = datetime.now().isoformat()

            doc_id = (
                f"global_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{idx}"
            )
            self._global.add(documents=[doc], metadatas=[meta], ids=[doc_id])
            merged += 1

        # Drop the agent namespace after successful merge.
        self.drop_namespace(agent_id)

        logger.info(
            "Merged namespace %s into global: %d added, %d duplicates skipped",
            agent_id, merged, skipped,
        )
        return True


# =============================================================================
# ServiceRegistry singleton factory
# =============================================================================

_factory_lock = threading.Lock()


def get_namespace_manager(**kwargs: Any) -> NamespaceManager:
    """Get or create the :class:`NamespaceManager` via ServiceRegistry.

    Thread-safe double-checked locking — same pattern as
    ``get_memory_manager()``.
    """
    from src.core.di import ServiceRegistry

    ns = ServiceRegistry.get("namespace_manager")
    if ns is not None:
        return ns

    with _factory_lock:
        ns = ServiceRegistry.get("namespace_manager")
        if ns is None:
            ns = NamespaceManager(**kwargs)
            ServiceRegistry.register("namespace_manager", ns)
            logger.info("NamespaceManager registered in ServiceRegistry")
        return ns


__all__ = ["NamespaceManager", "get_namespace_manager"]
