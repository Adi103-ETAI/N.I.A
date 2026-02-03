"""Detect potential circular dependencies before they happen."""

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List


class ImportAnalyzer(ast.NodeVisitor):
    """Extract imports from Python file."""
    
    def __init__(self):
        self.imports: Set[str] = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
    
    def visit_ImportFrom(self, node):
        if node.module:
            # Get full module path
            self.imports.add(node.module)


def analyze_file(file_path: Path) -> Set[str]:
    """Get all imports from a file."""
    try:
        with open(file_path, encoding='utf-8') as f:
            tree = ast.parse(f.read())
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return analyzer.imports
    except Exception as e:
        print(f"[WARN] Error analyzing {file_path}: {e}")
        return set()


def get_module_name(file_path: Path, src_dir: Path) -> str:
    """Convert file path to module name."""
    rel_path = file_path.relative_to(src_dir)
    parts = list(rel_path.with_suffix('').parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(['src'] + parts)


def build_dependency_graph(src_dir: Path) -> Dict[str, Set[str]]:
    """Build module dependency graph."""
    graph = defaultdict(set)
    
    # Internal package names to track
    internal_packages = {'src', 'core', 'agents', 'capabilities', 'models', 'persona', 'interface', 'extensions'}
    
    for py_file in src_dir.rglob("*.py"):
        module_name = get_module_name(py_file, src_dir.parent)
        imports = analyze_file(py_file)
        
        for imp in imports:
            # Check if this is an internal import
            top_level = imp.split('.')[0]
            if top_level in internal_packages:
                graph[module_name].add(imp)
    
    return graph


def detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Detect circular dependencies using DFS."""
    
    cycles = []
    visited = set()
    rec_stack = []
    
    def dfs(node: str):
        visited.add(node)
        rec_stack.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                # Found cycle
                idx = rec_stack.index(neighbor)
                cycle = rec_stack[idx:] + [neighbor]
                cycles.append(cycle)
        
        rec_stack.pop()
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return cycles


def main():
    """Detect circular dependencies in src/."""
    src_dir = Path("src")
    
    if not src_dir.exists():
        print("[WARN] src/ directory not found")
        return
    
    print("=" * 60)
    print("N.I.A. v4.0.0 Circular Dependency Detector")
    print("=" * 60)
    print()
    print("Analyzing dependency graph...")
    
    graph = build_dependency_graph(src_dir)
    
    print(f"Found {len(graph)} modules with dependencies")
    
    cycles = detect_cycles(graph)
    
    if cycles:
        print(f"\n[FAIL] Found {len(cycles)} circular dependencies:\n")
        for i, cycle in enumerate(cycles, 1):
            print(f"{i}. {' -> '.join(cycle)}")
        print("\n[WARN] Fix these before proceeding with migration!")
        sys.exit(1)
    else:
        print("\n[OK] No circular dependencies detected!")
        print("=" * 60)


if __name__ == "__main__":
    main()
