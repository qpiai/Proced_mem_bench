"""
Minimal Custom Retrieval Example

This is a pedagogical example showing how to implement the RetrievalSystem interface.
Uses simple keyword matching for demonstration - you should replace this with your
own retrieval method (embeddings, BM25, hybrid, neural, etc.).

The goal is to show:
1. How to implement the 3 required methods
2. How to format results as RetrievedTrajectory objects
3. How to integrate with BenchmarkRunner

Run: python examples/custom_retrieval_minimal.py
"""

from typing import List, Dict
from procedural_memory_benchmark import (
    RetrievalSystem,
    RetrievedTrajectory,
    BenchmarkRunner
)
from procedural_memory_benchmark.agentinstruct import AgentInstructCorpusLoader


class MinimalKeywordRetrieval(RetrievalSystem):
    """
    Minimal retrieval system using simple keyword matching.

    This is for DEMONSTRATION ONLY - replace with your own method!
    """

    def __init__(self):
        """Initialize by loading the corpus."""
        print("🔄 Loading AgentInstruct corpus...")
        self.corpus_loader = AgentInstructCorpusLoader()
        self.trajectories = self.corpus_loader.get_all_trajectories()
        print(f"✅ Loaded {len(self.trajectories)} trajectories")

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        """
        Retrieve top-k trajectories using simple keyword overlap.

        Args:
            query: Task description to search for
            k: Number of results to return

        Returns:
            List of RetrievedTrajectory sorted by relevance
        """
        # Simple keyword matching (REPLACE WITH YOUR METHOD)
        query_words = set(query.lower().split())

        scored_trajectories = []
        for traj in self.trajectories:
            # Score based on keyword overlap with task description
            traj_words = set(traj.task_description.lower().split())
            overlap = len(query_words & traj_words)
            similarity_score = overlap / max(len(query_words), 1)

            scored_trajectories.append((similarity_score, traj))

        # Sort by score (descending) and take top-k
        scored_trajectories.sort(reverse=True, key=lambda x: x[0])
        top_k = scored_trajectories[:k]

        # Convert to RetrievedTrajectory format
        results = []
        for score, traj in top_k:
            # Format document text for LLM evaluation
            document_text = self._format_trajectory(traj)

            results.append(RetrievedTrajectory(
                trajectory_id=traj.task_instance_id,
                task_instance_id=traj.task_instance_id,
                task_description=traj.task_description,
                similarity_score=score,
                total_steps=traj.total_steps,
                document_text=document_text  # Full trajectory for LLM judge
            ))

        return results

    def _format_trajectory(self, traj) -> str:
        """Format trajectory for LLM evaluation."""
        # Include task description and actions
        doc = f"Task: {traj.task_description}\n\nSteps:\n"
        for pair in traj.state_action_pairs:
            doc += f"{pair['step_id']}. {pair['action']}\n"
        return doc

    def get_system_name(self) -> str:
        """Return unique identifier."""
        return "minimal_keyword_matching"

    def get_system_info(self) -> Dict:
        """Return system configuration."""
        return {
            "method": "keyword_overlap",
            "corpus_size": len(self.trajectories),
            "description": "Simple keyword matching (demo only)"
        }


def main():
    print("=" * 70)
    print("CUSTOM RETRIEVAL EXAMPLE - Minimal Implementation")
    print("=" * 70)

    # Initialize custom retrieval
    print("\n🔧 Initializing custom retrieval system...")
    retrieval = MinimalKeywordRetrieval()

    # Test on a single query first
    print("\n🔍 Testing on sample query...")
    test_query = "Put a mug on the coffee maker"
    results = retrieval.retrieve(test_query, k=3)

    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.task_description} (score: {r.similarity_score:.3f})")

    # Run benchmark
    print("\n🎯 Running benchmark evaluation...")
    print("   Tier: EASY")
    print("   Queries: 3")

    runner = BenchmarkRunner(retrieval, llm_model="gpt-4")
    result = runner.run_benchmark(
        complexity_tiers=["EASY"],
        max_queries_per_tier=3,
        save_results=True
    )

    # Print results
    print("\n📊 Results:")
    print("=" * 70)
    runner.print_summary(result)

    print("\n✅ Example complete!")
    print("\nTo implement your own retrieval:")
    print("  1. Replace keyword matching with your method")
    print("  2. Ensure retrieve() returns List[RetrievedTrajectory]")
    print("  3. Provide document_text for LLM evaluation")
    print("  4. Run on full benchmark with all tiers")


if __name__ == "__main__":
    main()
