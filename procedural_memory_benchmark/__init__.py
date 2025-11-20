"""
Procedural Memory Benchmark

A benchmark for evaluating procedural memory retrieval systems on task-oriented trajectories.
"""

__version__ = "0.1.0"

# Core interfaces
from .benchmark.retrieval_interface import RetrievalSystem, RetrievedTrajectory
from .benchmark.benchmark_runner import BenchmarkRunner
from .benchmark.query_bank import QueryBank

# Baseline implementations (require optional dependencies)
from .benchmark.retrieval_interface import (
    AgentInstructEmbeddingRetrieval,
    AgentInstructActionOnlyRetrieval
)

__all__ = [
    'RetrievalSystem',
    'RetrievedTrajectory',
    'BenchmarkRunner',
    'QueryBank',
    'AgentInstructEmbeddingRetrieval',
    'AgentInstructActionOnlyRetrieval',
]
