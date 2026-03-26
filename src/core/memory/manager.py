"""N.I.A. 4-Layer Hybrid Memory System.

Architecture:
    Layer 1 - Episodic (ChromaDB): Semantic search of past conversations
    Layer 2 - Procedural (NetworkX): Skill chains and task sequences
    Layer 3 - Preferences (SQLite): User facts and settings
    Layer 4 - Security (SQLite): Security logs and blocked triggers

Storage (all in data/ directory):
    - data/vectors/   : ChromaDB persistent storage
    - data/skills.gml : NetworkX graph file
    - data/memory.db  : SQLite for preferences and security

Usage:
    from src.core.memory import MemoryManager, get_memory_manager
    
    memory = get_memory_manager()
    
    # Store a conversation episode
    await memory.store_episode("What's the weather?", role="user")
    
    # Recall relevant episodes
    episodes = await memory.recall_episodes("weather forecast", n=5)
    
    # Store a skill path
    memory.add_skill_path("open browser", ["launch app", "navigate to url"])
    
    # Get full context for LLM
    context = await memory.get_full_context("open chrome")
"""
from __future__ import annotations

import os
import sqlite3
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from src.core.logger import setup_logger
from src.core.config import get_settings, get_embedding_function

# Get cached settings singleton
settings = get_settings()

logger = setup_logger("MEMORY")


# =============================================================================
# Optional Dependencies
# =============================================================================

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    logger.warning("chromadb not installed. Episodic memory disabled.")

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False
    logger.warning("networkx not installed. Procedural memory disabled.")


# =============================================================================
# Configuration (All in data/ directory)
# =============================================================================

DATA_DIR = Path("data")
VECTORS_DIR = DATA_DIR / "vectors"
SKILLS_FILE = DATA_DIR / "skills.gml"
MEMORY_DB = DATA_DIR / "memory.db"


# =============================================================================
# Memory Manager (4-Layer Hybrid)
# =============================================================================

class MemoryManager:
    """4-Layer Hybrid Memory System for N.I.A.
    
    Storage:
        - data/vectors/   : ChromaDB (episodic)
        - data/skills.gml : NetworkX (procedural)
        - data/memory.db  : SQLite (preferences + security)
    """
    
    def __init__(
        self,
        vectors_dir: Optional[str] = None,
        skills_file: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> None:
        """Initialize the 4-Layer Memory System.
        
        Args:
            vectors_dir: ChromaDB path (default: data/vectors/).
            skills_file: NetworkX graph path (default: data/skills.gml).
            db_path: SQLite path (default: data/memory.db).
        """
        self.root_dir = Path(__file__).parent.parent
        self._vectors_dir = Path(vectors_dir) if vectors_dir else VECTORS_DIR
        self._skills_file = Path(skills_file) if skills_file else SKILLS_FILE
        self._db_path = Path(db_path) if db_path else MEMORY_DB
        self._lock = threading.Lock()  # Protect shared state (graph)
        
        # Ensure data directory exists
        os.makedirs(str(DATA_DIR), exist_ok=True)
        os.makedirs(str(self._vectors_dir), exist_ok=True)
        
        # Initialize all layers
        self._init_episodic()
        self._init_procedural()
        self._init_sql_sync() # Initial setup needs to be sync or run in loop
        
        logger.debug(f"MemoryManager initialized (Root: {self.root_dir})")
    
    # =========================================================================
    # Layer 1: Episodic Memory (ChromaDB)
    # =========================================================================
    
    def _init_episodic(self) -> None:
        """Initialize ChromaDB with embeddings from centralized config."""
        self._chroma_client = None
        self._episodes = None
        
        if not _HAS_CHROMADB:
            return
        
        try:
            # Connect to DB (Persistent Storage)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self._vectors_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            
            # Get embedding function from centralized config
            embedding_fn = get_embedding_function()
            
            # Create/Connect to collection with appropriate embeddings
            if embedding_fn:
                self._episodes = self._chroma_client.get_or_create_collection(
                    name="episodes",
                    embedding_function=embedding_fn
                )
                logger.debug("🧠 Episodic Memory: Connected (OpenAI Embeddings)")
            else:
                self._episodes = self._chroma_client.get_or_create_collection(
                    name="episodes"
                )
                logger.debug("🧠 Episodic Memory: Connected (Local/Free)")
            
        except Exception as exc:
            logger.error("❌ ChromaDB Init Failed: %s", exc, exc_info=True)
            self._episodes = None
    
    async def store_episode(
        self,
        text: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a conversation episode (Async)."""
        if not self._episodes:
            return False
        
        try:
            return await asyncio.to_thread(self._store_episode_sync, text, role, metadata)
        except Exception as exc:
            logger.error("store_episode failed: %s", exc, exc_info=True)
            return False

    def _store_episode_sync(self, text, role, metadata):
        """Internal sync implementation of store_episode."""
        episode_id = f"{role}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        meta = metadata or {}
        meta.update({"role": role, "timestamp": datetime.now().isoformat()})
        
        self._episodes.add(
            documents=[text],
            metadatas=[meta],
            ids=[episode_id],
        )
        return True
    
    async def recall_episodes(self, query: str, n: int = 5) -> List[str]:
        """Recall relevant past episodes via semantic search (Async)."""
        if not self._episodes:
            return []
        
        try:
            return await asyncio.to_thread(self._recall_episodes_sync, query, n)
        except Exception as exc:
            logger.error("recall_episodes failed: %s", exc, exc_info=True)
            return []

    def _recall_episodes_sync(self, query, n):
        """Internal sync implementation of recall_episodes."""
        results = self._episodes.query(query_texts=[query], n_results=n)
        return results.get("documents", [[]])[0]
    
    # =========================================================================
    # Layer 2: Procedural Memory (NetworkX)
    # =========================================================================
    
    def _init_procedural(self) -> None:
        """Initialize NetworkX graph for skill chains."""
        self._graph = None
        
        if not _HAS_NETWORKX:
            return
        
        try:
            if self._skills_file.exists():
                self._graph = nx.read_gml(str(self._skills_file))
                logger.debug("Loaded skills: %d nodes", self._graph.number_of_nodes())
            else:
                self._graph = nx.DiGraph()
                logger.debug("Created new skills graph")
        except Exception as exc:
            logger.error("Graph init failed: %s", exc, exc_info=True)
            self._graph = nx.DiGraph() if _HAS_NETWORKX else None
    
    def _save_graph(self) -> bool:
        """Persist the skills graph to disk."""
        if not self._graph:
            return False
        try:
            # Graph writes already protected by callers using the lock
            nx.write_gml(self._graph, str(self._skills_file))
            return True
        except Exception as exc:
            logger.error("_save_graph failed: %s", exc, exc_info=True)
            return False
    
    def add_skill_path(self, goal: str, steps: List[str]) -> bool:
        """Add a skill path: goal -> step1 -> step2 -> ..."""
        if not self._graph:
            return False
        
        with self._lock:  # Thread-safe mutation
            try:
                self._graph.add_node(goal, type="goal")
                prev = goal
                for i, step in enumerate(steps):
                    step_id = f"{goal}__step_{i}"
                    self._graph.add_node(step_id, type="step", label=step)
                    self._graph.add_edge(prev, step_id)
                    prev = step_id

                self._save_graph()
                logger.debug("Added skill: %s (%d steps)", goal, len(steps))
                return True
            except Exception as exc:
                logger.error("add_skill_path failed: %s", exc, exc_info=True)
                return False
    
    def get_skill_path(self, goal: str) -> List[str]:
        """Get ordered steps for a goal."""
        if not self._graph or goal not in self._graph:
            return []
        
        try:
            steps = []
            for _, successors in nx.bfs_successors(self._graph, goal):
                for s in successors:
                    steps.append(self._graph.nodes[s].get("label", s))
            return steps
        except Exception as exc:
            logger.error("get_skill_path failed: %s", exc, exc_info=True)
            return []
    
    def find_similar_goal(self, query: str) -> Optional[str]:
        """Find a goal matching the query (substring match)."""
        if not self._graph:
            return None
        
        q = query.lower()
        for node in self._graph.nodes:
            if self._graph.nodes[node].get("type") == "goal":
                if node.lower() in q or q in node.lower():
                    return node
        return None
    
    # =========================================================================
    # Layer 3: Preferences & Layer 4: Security (Async SQLite)
    # =========================================================================
    
    def _init_sql_sync(self) -> None:
        """Initialize SQLite tables (Sync for startup)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS preferences (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        category TEXT DEFAULT 'general'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS security_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        trigger TEXT NOT NULL,
                        action TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS skill_stats (
                        tool_name TEXT PRIMARY KEY,
                        usage_count INTEGER DEFAULT 1,
                        last_used TEXT
                    )
                """)
            logger.debug("SQL tables ready: %s", self._db_path)
        except Exception as exc:
            logger.error("SQL init failed: %s", exc)

    async def set_preference(self, key: str, value: str, category: str = "general") -> bool:
        """Set a user preference (Async)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO preferences (key, value, category) VALUES (?, ?, ?)",
                    (key, value, category),
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error("set_preference failed: %s", e, exc_info=True)
            return False
    
    def set_preference_sync(self, key: str, value: str, category: str = "general") -> bool:
        """Set a user preference (Sync wrapper)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO preferences (key, value, category) VALUES (?, ?, ?)",
                    (key, value, category),
                )
            return True
        except Exception as e:
            logger.error("set_preference_sync failed: %s", e, exc_info=True)
            return False
    
    async def get_preference(self, key: str) -> Optional[str]:
        """Get a user preference (Async)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("SELECT value FROM preferences WHERE key = ?", (key,)) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else None
        except Exception as e:
            logger.error("get_preference failed: %s", e, exc_info=True)
            return None

    def get_preference_sync(self, key: str) -> Optional[str]:
        """Get a user preference (Sync wrapper)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT value FROM preferences WHERE key = ?", (key,)
                ).fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error("get_preference_sync failed: %s", e, exc_info=True)
            return None
    
    async def get_all_preferences(self) -> Dict[str, str]:
        """Get all preferences as key-value dict (Async)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("SELECT key, value FROM preferences") as cursor:
                    rows = await cursor.fetchall()
                    return {r[0]: r[1] for r in rows}
        except Exception as e:
            logger.error("get_all_preferences failed: %s", e, exc_info=True)
            return {}

    def get_all_preferences_sync(self) -> Dict[str, str]:
        """Get all preferences (Sync wrapper)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                rows = conn.execute("SELECT key, value FROM preferences").fetchall()
                return {r[0]: r[1] for r in rows}
        except Exception as e:
            logger.error("get_all_preferences_sync failed: %s", e, exc_info=True)
            return {}
    
    async def record_skill_usage(self, tool_name: str) -> bool:
        """Increment usage count for a successful tool execution (Async)."""
        try:
            timestamp = datetime.now().isoformat()
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    INSERT INTO skill_stats (tool_name, usage_count, last_used)
                    VALUES (?, 1, ?)
                    ON CONFLICT(tool_name) DO UPDATE SET
                    usage_count = usage_count + 1,
                    last_used = ?
                """, (tool_name, timestamp, timestamp))
                await db.commit()
            logger.debug("📈 Skill Reinforced: %s", tool_name)
            return True
        except Exception as e:
            logger.error("Failed to record skill: %s", e, exc_info=True)
            return False
    
    async def get_skill_stats(self) -> Dict[str, Any]:
        """Get all skill usage statistics (Async)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT tool_name, usage_count, last_used FROM skill_stats ORDER BY usage_count DESC"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return {
                        r[0]: {"usage_count": r[1], "last_used": r[2]}
                        for r in rows
                    }
        except Exception as e:
            logger.error("get_skill_stats failed: %s", e, exc_info=True)
            return {}
    
    def log_security_event(self, trigger: str, action: str) -> bool:
        """Log a security event (Sync - Fire and Forget).

        Security logging often happens in sync contexts (e.g., Warden).
        For async, wrap in asyncio.to_thread if needed, or use aiosqlite.
        Given it's a log, sync is acceptable for safety/reliability.
        """
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute(
                    "INSERT INTO security_logs (timestamp, trigger, action) VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), trigger, action),
                )
            logger.info("Security: %s -> %s", trigger, action)
            return True
        except Exception as e:
            logger.error("log_security_event failed: %s", e, exc_info=True)
            return False
    
    def is_blocked(self, trigger: str) -> bool:
        """Check if a trigger is blocked (Sync)."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT 1 FROM security_logs WHERE trigger = ? AND action = 'blocked' LIMIT 1",
                    (trigger,),
                ).fetchone()
                return row is not None
        except Exception as e:
            logger.error("is_blocked failed: %s", e, exc_info=True)
            return False
    
    # =========================================================================
    # Context Assembler
    # =========================================================================
    
    async def get_full_context(self, query: str) -> Dict[str, Any]:
        """Assemble full context from all memory layers (Async).
        
        Returns:
            Dict with preferences, relevant_episodes, relevant_skills, is_blocked.
        """
        # Fetch parallelizable data
        # Note: ChromaDB recall is heaviest
        episodes_task = self.recall_episodes(query, n=5)
        prefs_task = self.get_all_preferences()
        
        episodes, prefs = await asyncio.gather(episodes_task, prefs_task)

        # Sync check for blocking (fast enough)
        blocked = self.is_blocked(query)
        
        context = {
            "preferences": prefs,
            "relevant_episodes": episodes,
            "relevant_skills": [],
            "is_blocked": blocked,
        }
        
        # Check for matching skill (in-memory graph check, fast)
        goal = self.find_similar_goal(query)
        if goal:
            context["relevant_skills"] = self.get_skill_path(goal)
            context["matched_goal"] = goal
        
        return context
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    async def vacuum_memory_db(self) -> None:
        """Vacuum SQLite database to reclaim space and optimize."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("VACUUM")
            logger.debug("Memory database vacuumed")
        except Exception as exc:
            logger.debug("Vacuum failed: %s", exc)
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics (Sync for dashboard)."""
        stats = {"episodic": 0, "skills": 0, "preferences": 0, "security": 0}
        
        if self._episodes:
            try:
                stats["episodic"] = self._episodes.count()
            except Exception:
                pass
        
        if self._graph:
            stats["skills"] = self._graph.number_of_nodes()
        
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                stats["preferences"] = conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
                stats["security"] = conn.execute("SELECT COUNT(*) FROM security_logs").fetchone()[0]
        except Exception:
            pass
        
        return stats

    def check_db_health(self) -> Dict[str, bool]:
        """Check health of all memory layers."""
        health = {
            "episodic": False,
            "procedural": False,
            "sql": False
        }
        
        if self._episodes:
            try:
                self._episodes.count()
                health["episodic"] = True
            except Exception:
                pass
                
        if self._graph is not None:
             health["procedural"] = True
             
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("SELECT 1").fetchone()
                health["sql"] = True
        except Exception:
            pass
            
        return health


# =============================================================================
# ServiceRegistry Integration
# =============================================================================

_factory_lock = threading.Lock()

def get_memory_manager(**kwargs) -> MemoryManager:
    """Get or create the MemoryManager via ServiceRegistry (Thread-Safe)."""
    from src.core.di import ServiceRegistry
    
    memory = ServiceRegistry.get("memory")
    if memory:
        return memory

    with _factory_lock:
        memory = ServiceRegistry.get("memory")
        if memory is None:
            memory = MemoryManager(**kwargs)
            ServiceRegistry.register("memory", memory)
            logger.info("MemoryManager registered in ServiceRegistry")
        return memory


__all__ = ["MemoryManager", "get_memory_manager"]
