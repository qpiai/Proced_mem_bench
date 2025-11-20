"""
Retrieval Interface for Procedural Memory Benchmark

Provides abstract base class for pluggable retrieval systems, allowing
researchers to evaluate custom retrieval methods with standardized benchmarking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)


@dataclass
class RetrievedTrajectory:
    """
    Standardized format for retrieved trajectories.

    This ensures consistent interface between retrieval systems and evaluation.
    """
    trajectory_id: str
    task_instance_id: str
    task_description: str
    similarity_score: float
    total_steps: int
    document_text: str  # Full trajectory content for LLM evaluation

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trajectory_id": self.trajectory_id,
            "task_instance_id": self.task_instance_id,
            "task_description": self.task_description,
            "similarity": self.similarity_score,
            "total_steps": self.total_steps,
            "document": self.document_text
        }


class RetrievalSystem(ABC):
    """
    Abstract base class for pluggable retrieval systems.

    Researchers implement this interface to evaluate custom retrieval methods
    on the procedural memory benchmark.

    Example:
        ```python
        class MyRetrieval(RetrievalSystem):
            def retrieve(self, query: str, k: int = 5):
                # Your retrieval logic
                results = self.my_search(query, k)
                return [RetrievedTrajectory(...) for r in results]

            def get_system_name(self):
                return "my_custom_retrieval"

            def get_system_info(self):
                return {"method": "hybrid", "params": {...}}
        ```
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        """
        Retrieve top-k most relevant trajectories for query.

        Args:
            query: Task description to search for
            k: Number of results to return

        Returns:
            List of RetrievedTrajectory objects sorted by relevance (highest first)

        Note:
            Results should be sorted by similarity_score in descending order.
        """
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """
        Return unique identifier for this retrieval system.

        Returns:
            System name (e.g., "my_bm25_retrieval", "my_embedding_system")
        """
        pass

    @abstractmethod
    def get_system_info(self) -> Dict:
        """
        Return system configuration and metadata.

        Returns:
            Dictionary with system information like:
            - method: Retrieval approach (e.g., "vector_embeddings", "bm25", "hybrid")
            - model: Model name if applicable
            - params: Key parameters
            - corpus_size: Number of trajectories indexed
        """
        pass

    def validate_corpus_compatibility(self) -> bool:
        """
        Optional: Validate that retrieval system is compatible with AgentInstruct corpus.

        Returns:
            True if compatible, False otherwise
        """
        return True


class AgentInstructEmbeddingRetrieval(RetrievalSystem):
    """
    Default baseline retrieval using AgentInstruct embedding database.

    Uses the existing task_description + state-action pairs embedding approach
    with all-MiniLM-L6-v2 model.
    """

    def __init__(self):
        """Initialize embedding-based retrieval using existing database."""
        try:
            from ..agentinstruct.database import AgentInstructDatabaseManager
            self.db_manager = AgentInstructDatabaseManager()
            self._is_available = True
        except ImportError as e:
            print(f"⚠️  AgentInstruct database not available: {e}")
            self._is_available = False
            self.db_manager = None

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        """
        Retrieve using vector embedding similarity.

        Args:
            query: Task description to search for
            k: Number of results to return

        Returns:
            List of RetrievedTrajectory objects
        """
        if not self._is_available:
            raise RuntimeError("AgentInstruct database not available. Run setup first.")

        # Search using existing database
        search_results = self.db_manager.search(query, k=k)

        # Convert to standardized format
        retrieved = []
        for result in search_results['results']:
            retrieved.append(RetrievedTrajectory(
                trajectory_id=result.get('trajectory_id', result.get('task_instance_id')),
                task_instance_id=result.get('task_instance_id', ''),
                task_description=result.get('task_description', ''),
                similarity_score=result.get('similarity', 0.0),
                total_steps=result.get('total_steps', 0),
                document_text=result.get('document', '')
            ))

        return retrieved

    def get_system_name(self) -> str:
        """Return system identifier."""
        return "agentinstruct_embedding"

    def get_system_info(self) -> Dict:
        """Return system configuration."""
        if not self._is_available:
            return {
                "method": "vector_embeddings",
                "status": "unavailable"
            }

        stats = self.db_manager.get_database_stats()

        return {
            "method": "vector_embeddings",
            "model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "embedding_format": "task_description_plus_state_action_pairs",
            "corpus_size": stats.get('total_trajectories', 336),
            "distance_metric": "cosine_similarity"
        }

    def setup(self):
        """Setup database if needed."""
        if not self._is_available:
            raise RuntimeError("AgentInstruct components not available")

        stats = self.db_manager.get_database_stats()
        if stats['total_trajectories'] == 0:
            print("🔄 Building AgentInstruct embedding database...")
            self.db_manager.build_database(batch_size=32)
            print("✅ Database build complete!")
        else:
            print(f"✅ Database ready with {stats['total_trajectories']} trajectories")


class AgentInstructActionOnlyRetrieval(RetrievalSystem):
    """
    Action-only retrieval using PURE ACTION SEQUENCE embeddings.

    This is the most minimal procedural representation - raw action patterns
    without any task descriptions or state information.

    Research Question: Can pure action sequences alone support effective procedural retrieval?
    """

    def __init__(self):
        """Initialize action-only retrieval using pure action sequence database."""
        try:
            from ..agentinstruct.database_actions_only import AgentInstructActionOnlyDatabaseManager
            self.db_manager = AgentInstructActionOnlyDatabaseManager()
            self._is_available = True
        except ImportError as e:
            print(f"⚠️  AgentInstruct action-only database not available: {e}")
            self._is_available = False
            self.db_manager = None

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        """
        Retrieve using pure action sequence similarity.

        Note: Even though the query contains task descriptions, it will be matched
        against pure action sequences in the corpus to test procedural similarity.

        Args:
            query: Task description to search for
            k: Number of results to return

        Returns:
            List of RetrievedTrajectory objects
        """
        if not self._is_available:
            raise RuntimeError("AgentInstruct action-only database not available. Run build_agentinstruct_actions_only_db.py first.")

        # Search using action-only database
        search_results = self.db_manager.search(query, k=k)

        # Convert to standardized format
        retrieved = []
        for result in search_results['results']:
            # Use action sequence as document text instead of full state-action pairs
            doc_text = result.get('action_sequence', result.get('document', ''))

            retrieved.append(RetrievedTrajectory(
                trajectory_id=result.get('trajectory_id', result.get('task_instance_id')),
                task_instance_id=result.get('task_instance_id', ''),
                task_description=result.get('task_description', ''),
                similarity_score=result.get('similarity', 0.0),
                total_steps=result.get('total_steps', 0),
                document_text=doc_text
            ))

        return retrieved

    def get_system_name(self) -> str:
        """Return system identifier."""
        return "agentinstruct_actions_only"

    def get_system_info(self) -> Dict:
        """Return system configuration."""
        if not self._is_available:
            return {
                "method": "vector_embeddings",
                "embedding_format": "pure_action_sequences_only",
                "status": "unavailable"
            }

        stats = self.db_manager.get_database_stats()

        return {
            "method": "vector_embeddings",
            "model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "embedding_format": "pure_action_sequences_only",
            "corpus_size": stats.get('total_trajectories', 336),
            "distance_metric": "cosine_similarity",
            "description": "Minimal procedural representation - action patterns only"
        }

    def setup(self):
        """Setup database if needed."""
        if not self._is_available:
            raise RuntimeError("AgentInstruct action-only components not available")

        stats = self.db_manager.get_database_stats()
        if stats['total_trajectories'] == 0:
            print("🔄 Building AgentInstruct action-only database...")
            self.db_manager.build_database(batch_size=32)
            print("✅ Action-only database build complete!")
        else:
            print(f"✅ Action-only database ready with {stats['total_trajectories']} trajectories")


def test_retrieval_interface():
    """Test the retrieval interface with baseline system."""
    print("🧪 Testing Retrieval Interface")
    print("=" * 60)

    # Test baseline retrieval
    print("\n📋 Testing AgentInstructEmbeddingRetrieval...")
    retrieval = AgentInstructEmbeddingRetrieval()

    # Check system info
    info = retrieval.get_system_info()
    print(f"   System: {retrieval.get_system_name()}")
    print(f"   Method: {info.get('method')}")
    print(f"   Model: {info.get('model', 'N/A')}")
    print(f"   Corpus size: {info.get('corpus_size', 'N/A')}")

    # Test retrieval
    if retrieval._is_available:
        print("\n🔍 Testing retrieval...")
        results = retrieval.retrieve("Put two books on the table", k=3)
        print(f"   Retrieved {len(results)} trajectories")

        if results:
            print(f"\n   Top result:")
            print(f"     Task: {results[0].task_description}")
            print(f"     Similarity: {results[0].similarity_score:.3f}")
            print(f"     Steps: {results[0].total_steps}")

    # Test action-only retrieval
    print("\n📋 Testing AgentInstructActionOnlyRetrieval...")
    action_retrieval = AgentInstructActionOnlyRetrieval()

    # Check system info
    action_info = action_retrieval.get_system_info()
    print(f"   System: {action_retrieval.get_system_name()}")
    print(f"   Method: {action_info.get('method')}")
    print(f"   Embedding format: {action_info.get('embedding_format', 'N/A')}")
    print(f"   Corpus size: {action_info.get('corpus_size', 'N/A')}")

    print("\n✅ Interface test completed!")


if __name__ == "__main__":
    test_retrieval_interface()
