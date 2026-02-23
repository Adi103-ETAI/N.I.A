"""NIA Graph Nodes — Package.

Re-exports every public symbol from the nodes submodule so that all
existing ``from src.agents.nia.graph.nodes import X`` calls keep working
unchanged after the split.

Submodule layout:
    helpers.py  — config loaders (vision, prompts) + summarize_oldest
    agents.py   — supervisor_node, iris_node, general_assistant
    docker.py   — call_tara_2, docker_node
    routing.py  — router_node, route_from_router, route_from_tara
"""
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
    _HAS_TARA_2,
)
from src.agents.nia.graph.nodes.routing import (
    router_node,
    route_from_router,
    route_from_tara,
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
    "_HAS_TARA_2",
    # Routing
    "router_node",
    "route_from_router",
    "route_from_tara",
]
