"""
Quickstart Example - Run baseline evaluation in 5 minutes

This script demonstrates:
1. Setting up a baseline retrieval system
2. Running benchmark on a small subset
3. Viewing results

Run: python examples/quickstart.py
"""

from procedural_memory_benchmark import (
    AgentInstructEmbeddingRetrieval,
    BenchmarkRunner
)

def main():
    print("=" * 70)
    print("PROCEDURAL MEMORY BENCHMARK - QUICK START")
    print("=" * 70)

    # Step 1: Initialize baseline retrieval system
    print("\n🔧 Step 1: Setting up baseline retrieval system...")
    print("   This uses state-aware embeddings (task + state-action pairs)")

    retrieval = AgentInstructEmbeddingRetrieval()
    retrieval.setup()  # Builds database on first use (~1-2 minutes)

    print("✅ Retrieval system ready!")

    # Step 2: Create benchmark runner
    print("\n🎯 Step 2: Creating benchmark runner...")
    print("   Using GPT-4 for LLM-as-judge evaluation")

    runner = BenchmarkRunner(
        retrieval_system=retrieval,
        llm_model="gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper
    )

    print("✅ Runner initialized!")

    # Step 3: Run benchmark on EASY tier (quick test)
    print("\n🚀 Step 3: Running benchmark...")
    print("   Tier: EASY")
    print("   Queries: 3 (quick test)")
    print("   This will take ~2-3 minutes...")

    result = runner.run_benchmark(
        complexity_tiers=["EASY"],      # Start with easy tier
        max_queries_per_tier=3,          # Just 3 queries for quick test
        save_results=True
    )

    # Step 4: View results
    print("\n📊 Step 4: Results Summary")
    print("=" * 70)
    runner.print_summary(result)

    print("\n✅ QUICKSTART COMPLETE!")
    print("\nNext steps:")
    print("  1. Try more queries: max_queries_per_tier=None")
    print("  2. Test other tiers: complexity_tiers=['HARD', 'MEDIUM', 'EASY']")
    print("  3. Implement your custom retrieval: see custom_retrieval_minimal.py")
    print(f"\nResults saved to: results/benchmark_{result.system_name}_*.json")


if __name__ == "__main__":
    main()
