"""
Complexity Analyzer for Procedural Memory Benchmark

Analyzes benchmark results across complexity tiers and task types,
providing insights into retrieval system performance patterns.
"""

from typing import Dict, List, Any
from collections import defaultdict
import statistics


class ComplexityAnalyzer:
    """
    Analyze benchmark results by complexity dimensions.

    Provides:
    - Per-tier performance breakdown
    - Per-task-type analysis
    - Complexity factor correlation
    - Statistical comparisons
    """

    def __init__(self):
        """Initialize complexity analyzer."""
        self.task_type_names = {
            1: "pick_and_place_simple",
            2: "look_at_obj_in_light",
            3: "pick_clean_then_place_in_recep",
            4: "pick_heat_then_place_in_recep",
            5: "pick_cool_then_place_in_recep",
            6: "pick_two_obj_and_place"
        }

    def analyze(self, benchmark_result: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive complexity analysis on benchmark results.

        Args:
            benchmark_result: Result dictionary from BenchmarkRunner

        Returns:
            Dictionary with analysis results
        """
        query_results = benchmark_result.get('query_results', [])

        if not query_results:
            return {"error": "No query results to analyze"}

        analysis = {
            "tier_comparison": self._analyze_tiers(query_results),
            "task_type_analysis": self._analyze_task_types(query_results),
            "complexity_factor_correlation": self._analyze_factors(query_results),
            "difficulty_ranking": self._rank_by_difficulty(query_results),
            "corpus_coverage_insights": self._analyze_corpus_coverage(query_results)
        }

        return analysis

    def _analyze_tiers(self, query_results: List[Dict]) -> Dict:
        """Analyze performance across complexity tiers."""
        tier_data = defaultdict(list)

        for result in query_results:
            tier = result['query']['complexity_tier']
            ap = result['metrics']['average_precision']
            tier_data[tier].append(ap)

        tier_stats = {}
        for tier in ["HARD", "MEDIUM", "EASY"]:
            if tier in tier_data and tier_data[tier]:
                aps = tier_data[tier]
                tier_stats[tier] = {
                    "count": len(aps),
                    "mean_ap": statistics.mean(aps),
                    "median_ap": statistics.median(aps),
                    "std_ap": statistics.stdev(aps) if len(aps) > 1 else 0.0,
                    "min_ap": min(aps),
                    "max_ap": max(aps)
                }

        return tier_stats

    def _analyze_task_types(self, query_results: List[Dict]) -> Dict:
        """Analyze performance by task type."""
        type_data = defaultdict(list)

        for result in query_results:
            task_type = result['query']['task_type']
            ap = result['metrics']['average_precision']
            p1 = result['metrics']['precision_at_k'].get(1, 0)
            type_data[task_type].append({
                'ap': ap,
                'p1': p1
            })

        type_stats = {}
        for task_type in sorted(type_data.keys()):
            data = type_data[task_type]
            if data:
                aps = [d['ap'] for d in data]
                p1s = [d['p1'] for d in data]

                type_stats[task_type] = {
                    "name": self.task_type_names.get(task_type, f"Unknown-{task_type}"),
                    "count": len(data),
                    "mean_ap": statistics.mean(aps),
                    "mean_p1": statistics.mean(p1s),
                    "std_ap": statistics.stdev(aps) if len(aps) > 1 else 0.0
                }

        # Sort by mean_ap (hardest first)
        sorted_types = sorted(
            type_stats.items(),
            key=lambda x: x[1]['mean_ap']
        )

        return {
            "by_type": type_stats,
            "difficulty_ranking": [
                {
                    "task_type": t[0],
                    "name": t[1]["name"],
                    "mean_ap": t[1]["mean_ap"],
                    "count": t[1]["count"]
                }
                for t in sorted_types
            ]
        }

    def _analyze_factors(self, query_results: List[Dict]) -> Dict:
        """Analyze correlation between complexity factors and performance."""
        factor_data = defaultdict(list)

        for result in query_results:
            factors = result['query']['complexity_factors']
            ap = result['metrics']['average_precision']

            for factor in factors:
                factor_data[factor].append(ap)

        factor_stats = {}
        for factor, aps in factor_data.items():
            if aps:
                factor_stats[factor] = {
                    "count": len(aps),
                    "mean_ap": statistics.mean(aps),
                    "median_ap": statistics.median(aps)
                }

        # Sort by mean_ap (factors that make queries harder)
        sorted_factors = sorted(
            factor_stats.items(),
            key=lambda x: x[1]['mean_ap']
        )

        return {
            "by_factor": factor_stats,
            "hardest_factors": [
                {
                    "factor": f[0],
                    "mean_ap": f[1]["mean_ap"],
                    "count": f[1]["count"]
                }
                for f in sorted_factors[:5]  # Top 5 hardest
            ]
        }

    def _rank_by_difficulty(self, query_results: List[Dict]) -> List[Dict]:
        """Rank queries by difficulty (lowest AP = hardest)."""
        ranked = []

        for result in query_results:
            ranked.append({
                "query_id": result['query']['query_id'],
                "task_description": result['query']['task_description'],
                "complexity_tier": result['query']['complexity_tier'],
                "task_type": result['query']['task_type'],
                "average_precision": result['metrics']['average_precision'],
                "precision_at_1": result['metrics']['precision_at_k'].get(1, 0),
                "num_relevant": result['metrics']['num_relevant']
            })

        # Sort by AP (lowest = hardest)
        ranked.sort(key=lambda x: x['average_precision'])

        return ranked

    def _analyze_corpus_coverage(self, query_results: List[Dict]) -> Dict:
        """
        Analyze corpus coverage patterns.

        Identifies queries with poor performance that might indicate
        limited relevant trajectories in the corpus.
        """
        low_coverage = []  # Queries with no relevant trajectories found
        partial_coverage = []  # Queries with some but poor retrieval

        for result in query_results:
            ap = result['metrics']['average_precision']
            p1 = result['metrics']['precision_at_k'].get(1, 0)
            num_relevant = result['metrics']['num_relevant']

            if num_relevant == 0:
                low_coverage.append({
                    "query": result['query']['task_description'],
                    "task_type": result['query']['task_type'],
                    "complexity_tier": result['query']['complexity_tier']
                })
            elif ap < 0.3:  # Poor performance threshold
                partial_coverage.append({
                    "query": result['query']['task_description'],
                    "task_type": result['query']['task_type'],
                    "ap": ap,
                    "p1": p1
                })

        return {
            "zero_relevant_count": len(low_coverage),
            "low_coverage_queries": low_coverage[:5],  # Top 5 examples
            "poor_performance_count": len(partial_coverage),
            "poor_performance_examples": partial_coverage[:5],
            "coverage_note": "Difficulty influenced by trajectory availability in corpus"
        }

    def print_analysis(self, analysis: Dict):
        """Print formatted analysis report."""
        print("\n" + "="*70)
        print("📈 COMPLEXITY ANALYSIS REPORT")
        print("="*70)

        # Tier comparison
        if "tier_comparison" in analysis:
            print("\n🎯 Performance by Complexity Tier:")
            for tier in ["HARD", "MEDIUM", "EASY"]:
                if tier in analysis["tier_comparison"]:
                    stats = analysis["tier_comparison"][tier]
                    print(f"\n   {tier}:")
                    print(f"      Queries: {stats['count']}")
                    print(f"      Mean AP: {stats['mean_ap']:.3f} ± {stats['std_ap']:.3f}")
                    print(f"      Range: [{stats['min_ap']:.3f}, {stats['max_ap']:.3f}]")

        # Task type ranking
        if "task_type_analysis" in analysis:
            print("\n📋 Task Type Difficulty Ranking (Hardest → Easiest):")
            for i, task in enumerate(analysis["task_type_analysis"]["difficulty_ranking"], 1):
                print(f"   {i}. Type {task['task_type']}: {task['name']}")
                print(f"      Mean AP: {task['mean_ap']:.3f} ({task['count']} queries)")

        # Complexity factors
        if "complexity_factor_correlation" in analysis:
            print("\n🔍 Hardest Complexity Factors:")
            for i, factor in enumerate(analysis["complexity_factor_correlation"]["hardest_factors"], 1):
                print(f"   {i}. {factor['factor']}: Mean AP = {factor['mean_ap']:.3f} ({factor['count']} queries)")

        # Corpus coverage
        if "corpus_coverage_insights" in analysis:
            coverage = analysis["corpus_coverage_insights"]
            print(f"\n💡 Corpus Coverage Insights:")
            print(f"   Queries with zero relevant: {coverage['zero_relevant_count']}")
            print(f"   Queries with poor performance: {coverage['poor_performance_count']}")
            print(f"   Note: {coverage['coverage_note']}")


def test_complexity_analyzer():
    """Test complexity analyzer with sample data."""
    print("🧪 Testing Complexity Analyzer")
    print("=" * 70)

    # Create mock benchmark result
    mock_result = {
        "query_results": [
            {
                "query": {
                    "query_id": "test_1",
                    "task_description": "Multi-object task",
                    "complexity_tier": "HARD",
                    "task_type": 6,
                    "complexity_factors": ["multi_object"]
                },
                "metrics": {
                    "average_precision": 0.3,
                    "precision_at_k": {1: 0.0, 3: 0.33},
                    "num_relevant": 1
                }
            },
            {
                "query": {
                    "query_id": "test_2",
                    "task_description": "Simple placement",
                    "complexity_tier": "MEDIUM",
                    "task_type": 1,
                    "complexity_factors": []
                },
                "metrics": {
                    "average_precision": 0.7,
                    "precision_at_k": {1: 1.0, 3: 0.67},
                    "num_relevant": 2
                }
            }
        ]
    }

    # Analyze
    analyzer = ComplexityAnalyzer()
    analysis = analyzer.analyze(mock_result)

    # Print
    analyzer.print_analysis(analysis)

    print("\n✅ Complexity analyzer test completed!")


if __name__ == "__main__":
    test_complexity_analyzer()
