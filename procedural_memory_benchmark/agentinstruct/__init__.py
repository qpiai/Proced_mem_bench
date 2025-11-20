"""
AgentInstruct corpus and embedding components.

Note: Baseline retrieval systems require optional dependencies.
Install with: pip install procedural-memory-benchmark[baselines]
"""

# Core corpus loader (no optional dependencies)
from .corpus_loader import AgentInstructCorpusLoader, AgentInstructTrajectory

# Lazy imports for optional baseline components
def _get_embedder():
    """Lazy import of embedder (requires sentence-transformers)."""
    from .embedder import AgentInstructEmbedder
    return AgentInstructEmbedder

def _get_database():
    """Lazy import of database (requires chromadb)."""
    from .database import AgentInstructDatabaseManager
    return AgentInstructDatabaseManager

def _get_actions_only_database():
    """Lazy import of actions-only database (requires chromadb)."""
    from .database_actions_only import AgentInstructActionOnlyDatabaseManager
    return AgentInstructActionOnlyDatabaseManager

__all__ = [
    'AgentInstructCorpusLoader',
    'AgentInstructTrajectory',
]
