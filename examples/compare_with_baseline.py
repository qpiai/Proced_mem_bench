"""
Compare Custom System with Baseline

This script demonstrates:
1. Running multiple retrieval systems on the same queries
2. Comparing performance metrics side-by-side
3. Analyzing performance differences

Run: python examples/compare_with_baseline.py
"""

import json
from procedural_memory_benchmark import (
    AgentInstructEmbeddingRetrieval,
    AgentInstructActionOnlyRetrieval,
    BenchmarkRunner
)


def run_system(system, system_name, complexity_tiers, max_queries):
    """Run benchmark for a single system."""
    print(f"\n{'='*70}")
    print(f"Running: {system_name}")
    print(f"{'='*70}")

    runner = BenchmarkRunner(system, llm_model="gpt-4")
    result = runner.run_benchmark(
        complexity_tiers=complexity_tiers,
        max_queries_per_tier=max_queries,
        save_results=True
    )

    return result


def compare_results(results):
    """Print side-by-side comparison."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    # Overall metrics comparison
    print("\nOverall Metrics:")
    print(f"{'System':<40} {'MAP':>8} {'P@1':>8} {'P@5':>8} {'NDCG@10':>10}")
    print("-" * 70)

    for system_name, result in results.items():
        overall = result.overall_metrics
        print(f"{system_name:<40} "
              f"{overall.get('MAP', 0):.4f}   "
              f"{overall.get('P@1', 0):.4f}   "
              f"{overall.get('P@5', 0):.4f}   "
              f"{overall.get('NDCG@10', 0):.4f}")

    # Per-tier breakdown
    print("\nPer-Tier Performance (MAP):")

    # Get all tiers
    all_tiers = set()
    for result in results.values():
        all_tiers.update(result.complexity_stratified_metrics.keys())

    for tier in sorted(all_tiers):
        print(f"\n{tier}:")
        print(f"{'System':<40} {'MAP':>8} {'P@1':>8}")
        print("-" * 50)

        for system_name, result in results.items():
            tier_metrics = result.complexity_stratified_metrics.get(tier, {})
            print(f"{system_name:<40} "
                  f"{tier_metrics.get('MAP', 0):.4f}   "
                  f"{tier_metrics.get('P@1', 0):.4f}")


def main():
    print("="*70)
    print("SYSTEM COMPARISON - Baseline vs Custom")
    print("="*70)

    # Configuration
    complexity_tiers = ["EASY"]  # Start with EASY tier
    max_queries = 3  # Small subset for quick comparison

    print(f"\nConfiguration:")
    print(f"  Tiers: {complexity_tiers}")
    print(f"  Max queries per tier: {max_queries}")

    # Initialize systems
    print("\n🔧 Initializing systems...")

    systems = {
        "Baseline (State-aware)": AgentInstructEmbeddingRetrieval(),
        "Baseline (Action-only)": AgentInstructActionOnlyRetrieval(),
        # Add your custom system here:
        # "My Custom System": MyCustomRetrieval(),
    }

    # Setup systems
    for name, system in systems.items():
        if hasattr(system, 'setup'):
            print(f"  Setting up {name}...")
            system.setup()

    # Run benchmarks
    results = {}
    for name, system in systems.items():
        result = run_system(system, name, complexity_tiers, max_queries)
        results[name] = result

    # Compare results
    compare_results(results)

    print("\n✅ Comparison complete!")
    print("\nTo add your custom system:")
    print("  1. Implement your RetrievalSystem")
    print("  2. Add to 'systems' dictionary above")
    print("  3. Run this script to compare")

    print("\nTo run full evaluation:")
    print("  - Set complexity_tiers=['HARD', 'MEDIUM', 'EASY']")
    print("  - Set max_queries=None (use all queries)")


if __name__ == "__main__":
    main()
