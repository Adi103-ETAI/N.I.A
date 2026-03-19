"""NIA Graph Nodes — Package.

Re-exports every public symbol from the nodes submodule so that all
existing ``from src.agents.nia.graph.nodes import X`` calls keep working
unchanged after the split.

Submodule layout:
    helpers.py  — config loaders (vision, prompts) + summarize_oldest
    agents.py   — supervisor_node, iris_node, general_assistant
    docker.py   — call_tara_2, docker_node, sandbox_node
    planner.py  — planner_node (Sprint 2 entry point)
    coordinator_node.py — coordinator_node (Sprint 4 multi-step dispatch)
"""
from src.agents.nia.graph.nodes.planner import planner_node
from src.agents.nia.graph.nodes.coordinator_node import coordinator_node
from src.agents.nia.graph.nodes.helpers import (
    get_vision_keywords,
    get_prompts,
    summarize_oldest,
    asummarize_oldest,
)
from src.agents.nia.graph.nodes.agents import (
    supervisor_node,
    iris_node,
    general_assistant,
)
from src.agents.nia.graph.nodes.docker import (
    call_tara_2,
    docker_node,
    sandbox_node,
    _HAS_TARA_2,
)

__all__ = [
    # Helpers
    "summarize_oldest",
    "asummarize_oldest",
    "get_vision_keywords",
    "get_prompts",
    # Agent nodes
    "supervisor_node",
    "iris_node",
    "general_assistant",
    # Execution nodes
    "call_tara_2",
    "docker_node",
    "sandbox_node",
    "_HAS_TARA_2",
    # Sprint 2: Planner
    "planner_node",
    # Sprint 4: Coordinator
    "coordinator_node",
]
