"""
Benchmark Runner for Procedural Memory Retrieval

Main orchestrator that integrates retrieval, evaluation, and metrics calculation
for standardized benchmarking of procedural memory retrieval systems.
"""

import time
import os
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from .retrieval_interface import RetrievalSystem, RetrievedTrajectory
from .query_bank import QueryBank, BenchmarkQuery
from .metrics_calculator import MetricsCalculator, QueryMetrics
from ..llm.llm_reasoner import LLMReasoner


@dataclass
class BenchmarkResult:
    """Complete benchmark evaluation result."""
    system_name: str
    system_info: Dict
    query_results: List[Dict]  # Per-query results with metrics
    overall_metrics: Dict
    complexity_stratified_metrics: Dict
    execution_time_seconds: float
    timestamp: str


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Integrates:
    - Query bank loading and filtering
    - Retrieval system execution
    - LLM-as-judge evaluation
    - Comprehensive metrics calculation
    - Complexity-stratified analysis
    """

    def __init__(
        self,
        retrieval_system: RetrievalSystem,
        query_bank_path: str = None,
        llm_model: str = "gpt-5",
        k_values: List[int] = None
    ):
        """
        Initialize benchmark runner.

        Args:
            retrieval_system: Retrieval system to evaluate (implements RetrievalSystem interface)
            query_bank_path: Path to query bank JSON (default: benchmark/data/query_bank.json)
            llm_model: LLM model for evaluation (default: "gpt-5")
            k_values: K values for metrics (default: [1, 3, 5, 10])
        """
        self.retrieval_system = retrieval_system
        self.query_bank = QueryBank(query_bank_path)
        self.llm_reasoner = LLMReasoner(model_name=llm_model)
        self.k_values = k_values if k_values is not None else [1, 3, 5, 10]
        self.metrics_calculator = MetricsCalculator(k_values=self.k_values)

        print(f"🎯 Benchmark Runner initialized")
        print(f"   Retrieval system: {retrieval_system.get_system_name()}")
        print(f"   LLM model: {llm_model}")
        print(f"   K values: {self.k_values}")

    def run_benchmark(
        self,
        complexity_tiers: List[str] = None,
        max_queries_per_tier: Optional[int] = None,
        save_results: bool = True
    ) -> BenchmarkResult:
        """
        Run complete benchmark evaluation.

        Args:
            complexity_tiers: List of tiers to evaluate (default: ["HARD", "MEDIUM", "EASY"])
            max_queries_per_tier: Optional limit on queries per tier
            save_results: Whether to save results to file

        Returns:
            BenchmarkResult with complete evaluation
        """
        if complexity_tiers is None:
            complexity_tiers = ["HARD", "MEDIUM", "EASY"]

        print(f"\n{'='*70}")
        print(f"🚀 STARTING BENCHMARK EVALUATION")
        print(f"{'='*70}")

        start_time = time.time()

        # Load query bank
        print(f"\n📋 Loading query bank...")
        self.query_bank.load()
        stats = self.query_bank.get_statistics()
        print(f"   Loaded {stats['total_queries']} queries")
        print(f"   Distribution: {stats['queries_by_tier']}")

        # Select queries
        selected_queries = self._select_queries(complexity_tiers, max_queries_per_tier)
        print(f"\n✅ Selected {len(selected_queries)} queries for evaluation")
        for tier in complexity_tiers:
            tier_queries = [q for q in selected_queries if q.complexity_tier == tier]
            print(f"   {tier}: {len(tier_queries)} queries")

        # Run evaluation
        print(f"\n🔄 Running retrieval and evaluation...")
        query_results = []

        for i, query in enumerate(selected_queries):
            if i % 5 == 0:
                print(f"   Progress: {i}/{len(selected_queries)} queries...")

            result = self._evaluate_query(query)
            query_results.append(result)

        # Calculate metrics
        print(f"\n📊 Calculating metrics...")
        overall_metrics, stratified_metrics = self._calculate_all_metrics(
            query_results,
            complexity_tiers
        )

        execution_time = time.time() - start_time

        # Create result object
        result = BenchmarkResult(
            system_name=self.retrieval_system.get_system_name(),
            system_info=self.retrieval_system.get_system_info(),
            query_results=query_results,
            overall_metrics=overall_metrics,
            complexity_stratified_metrics=stratified_metrics,
            execution_time_seconds=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Save if requested
        if save_results:
            self._save_results(result)

        print(f"\n✅ Benchmark evaluation complete!")
        print(f"   Total time: {execution_time:.2f} seconds")
        print(f"   Queries/second: {len(selected_queries)/execution_time:.2f}")

        return result

    def _select_queries(
        self,
        complexity_tiers: List[str],
        max_per_tier: Optional[int]
    ) -> List[BenchmarkQuery]:
        """Select queries based on complexity tiers and limits."""
        selected = []

        for tier in complexity_tiers:
            tier_queries = self.query_bank.get_by_complexity(tier)

            if max_per_tier is not None and len(tier_queries) > max_per_tier:
                tier_queries = tier_queries[:max_per_tier]

            selected.extend(tier_queries)

        return selected

    def _evaluate_query(self, query: BenchmarkQuery) -> Dict:
        """
        Evaluate a single query: retrieve + LLM evaluation + metrics.

        Args:
            query: BenchmarkQuery object

        Returns:
            Dictionary with query results and metrics
        """
        # 1. Retrieve trajectories
        retrieved = self.retrieval_system.retrieve(
            query.task_description,
            k=max(self.k_values)
        )

        # 2. Convert to format expected by LLM evaluator
        retrieved_for_llm = [r.to_dict() for r in retrieved]

        # 3. LLM evaluation
        try:
            llm_scores = self.llm_reasoner.batch_evaluate_with_prompt(
                prompt_template=self.llm_reasoner.UNIVERSAL_PROMPT,
                query=query.task_description,
                retrieved_results=retrieved_for_llm,
                include_raw_response=False
            )
        except Exception as e:
            print(f"   ⚠️ LLM evaluation failed for query: {query.query_id}")
            print(f"      Error: {str(e)}")
            llm_scores = {}

        # 4. Extract relevance judgments
        relevance_judgments = {}
        relevance_scores = {}

        for traj in retrieved:
            traj_id = traj.trajectory_id
            if traj_id in llm_scores:
                score = llm_scores[traj_id].get('relevance_score', 0)
                relevance_scores[traj_id] = score
                relevance_judgments[traj_id] = score >= 6.0  # Threshold for binary relevance
            else:
                relevance_scores[traj_id] = 0.0
                relevance_judgments[traj_id] = False

        # 5. Calculate metrics
        retrieved_ids = [r.trajectory_id for r in retrieved]
        metrics = self.metrics_calculator.calculate_query_metrics(
            query_id=query.query_id,
            relevance_judgments=relevance_judgments,
            retrieved_ids=retrieved_ids,
            relevance_scores=relevance_scores
        )

        # 6. Return comprehensive result
        return {
            "query": query.to_dict(),
            "retrieved_count": len(retrieved),
            "relevance_scores": relevance_scores,
            "relevance_judgments": relevance_judgments,
            "metrics": {
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "f1_at_k": metrics.f1_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "average_precision": metrics.average_precision,
                "num_relevant": metrics.num_relevant
            }
        }

    def _calculate_all_metrics(
        self,
        query_results: List[Dict],
        complexity_tiers: List[str]
    ) -> tuple[Dict, Dict]:
        """
        Calculate overall and complexity-stratified metrics.

        Args:
            query_results: List of query result dictionaries
            complexity_tiers: List of complexity tiers to analyze

        Returns:
            Tuple of (overall_metrics, stratified_metrics)
        """
        # Extract QueryMetrics objects for calculator
        all_metrics = []
        tier_metrics = {tier: [] for tier in complexity_tiers}

        for result in query_results:
            query_id = result['query']['query_id']
            tier = result['query']['complexity_tier']

            # Create QueryMetrics from result
            metrics = QueryMetrics(
                query_id=query_id,
                precision_at_k=result['metrics']['precision_at_k'],
                recall_at_k=result['metrics']['recall_at_k'],
                f1_at_k=result['metrics']['f1_at_k'],
                ndcg_at_k=result['metrics']['ndcg_at_k'],
                average_precision=result['metrics']['average_precision'],
                num_retrieved=result['retrieved_count'],
                num_relevant=result['metrics']['num_relevant']
            )

            all_metrics.append(metrics)
            if tier in tier_metrics:
                tier_metrics[tier].append(metrics)

        # Calculate overall metrics
        overall = self.metrics_calculator.calculate_aggregate_metrics(all_metrics)

        # Calculate per-tier metrics
        stratified = {}
        for tier, tier_query_metrics in tier_metrics.items():
            if tier_query_metrics:
                stratified[tier] = self.metrics_calculator.calculate_aggregate_metrics(
                    tier_query_metrics
                )

        return overall, stratified

    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results to file."""
        # Create results directory (configurable via utils)
        from ..utils.paths import get_results_dir
        results_dir = get_results_dir()

        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{result.system_name}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        # Convert to serializable format
        import json
        result_dict = {
            "system_name": result.system_name,
            "system_info": result.system_info,
            "overall_metrics": result.overall_metrics,
            "complexity_stratified_metrics": result.complexity_stratified_metrics,
            "execution_time_seconds": result.execution_time_seconds,
            "timestamp": result.timestamp,
            "query_results": result.query_results
        }

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"\n💾 Results saved to: {filepath}")

    def print_summary(self, result: BenchmarkResult):
        """Print formatted summary of benchmark results."""
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\nSystem: {result.system_name}")
        print(f"Method: {result.system_info.get('method', 'N/A')}")
        print(f"Execution time: {result.execution_time_seconds:.2f} seconds")

        print(f"\n🎯 Overall Metrics:")
        print(f"   MAP: {result.overall_metrics['map']:.3f}")
        for k in self.k_values:
            if k in result.overall_metrics['precision_at_k']:
                p = result.overall_metrics['precision_at_k'][k]
                r = result.overall_metrics['recall_at_k'][k]
                f1 = result.overall_metrics['f1_at_k'][k]
                print(f"   P@{k}: {p:.3f} | R@{k}: {r:.3f} | F1@{k}: {f1:.3f}")

        print(f"\n📈 Complexity-Stratified Metrics:")
        for tier in ["HARD", "MEDIUM", "EASY"]:
            if tier in result.complexity_stratified_metrics:
                metrics = result.complexity_stratified_metrics[tier]
                print(f"\n   {tier} Tier:")
                print(f"      MAP: {metrics['map']:.3f}")
                print(f"      P@1: {metrics['precision_at_k'].get(1, 0):.3f}")
                print(f"      Queries: {metrics['total_queries']}")


def test_benchmark_runner():
    """Test benchmark runner with default retrieval system."""
    print("🧪 Testing Benchmark Runner")
    print("=" * 70)

    from benchmark.retrieval_interface import AgentInstructEmbeddingRetrieval

    # Initialize
    retrieval = AgentInstructEmbeddingRetrieval()
    runner = BenchmarkRunner(retrieval, llm_model="gpt-5")

    # Run on small subset for testing
    print("\n🔄 Running benchmark on 5 queries (HARD tier only)...")
    result = runner.run_benchmark(
        complexity_tiers=["HARD"],
        max_queries_per_tier=5,
        save_results=False
    )

    # Print summary
    runner.print_summary(result)

    print("\n✅ Benchmark runner test completed!")


if __name__ == "__main__":
    test_benchmark_runner()
