# Procedural Memory Benchmark

A standardized benchmark for evaluating **procedural memory retrieval systems** on task-oriented trajectories. Test how well your retrieval method captures procedural similarity across different task representations.

## 📋 Overview

This benchmark evaluates retrieval systems on their ability to identify procedurally similar trajectories across:
- **40 complexity-stratified queries** (11 HARD, 14 MEDIUM, 15 EASY)
- **336 AgentInstruct trajectories** with state-action pairs
- **6 ALFWorld task types** (placement, heating, cooling, cleaning, examination, multi-object)
- **LLM-as-judge evaluation** with procedural similarity scoring
- **Standard IR metrics** (P@k, R@k, F1@k, NDCG@k, MAP)

### Key Features

✅ **Plug-and-play interface** - Implement 3 methods, get comprehensive evaluation
✅ **Complexity-stratified analysis** - Performance breakdown by query difficulty
✅ **Two baseline implementations** - State-aware and action-only embeddings
✅ **Method-neutral design** - No hints toward specific retrieval approaches
✅ **Reproducible evaluation** - Standardized queries and ground truth annotations

## 🚀 Quick Start (5 minutes)

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM-as-judge evaluation)

### Installation

```bash
# Install core package
pip install .

# Or install with baseline implementations
pip install .[baselines]

# Or install everything (baselines + LLM judge)
pip install .[all]
```

### Set up API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run baseline evaluation

```python
from procedural_memory_benchmark import (
    AgentInstructEmbeddingRetrieval,
    BenchmarkRunner
)

# Initialize baseline retrieval system (state-aware embeddings)
print("🔧 Setting up baseline retrieval system...")
retrieval = AgentInstructEmbeddingRetrieval()
retrieval.setup()  # Builds database on first use (~1-2 minutes)

# Run benchmark on EASY tier (quick test)
print("🎯 Running benchmark evaluation...")
runner = BenchmarkRunner(retrieval, llm_model="gpt-4")
result = runner.run_benchmark(
    complexity_tiers=["EASY"],
    max_queries_per_tier=3,  # Just 3 queries for quick test
    save_results=True
)

# View results
runner.print_summary(result)
```

**Expected output:**
```
📊 BENCHMARK RESULTS
System: agentinstruct_embedding
Queries evaluated: 3

Overall Metrics:
  MAP: 0.84
  P@1: 0.87
  P@5: 0.73
  NDCG@10: 0.86

Results saved to: results/benchmark_agentinstruct_embedding_20251120_195423.json
```

## 🔧 Adding Your Custom Retrieval System

### Step 1: Implement the interface (3 required methods)

```python
from procedural_memory_benchmark import RetrievalSystem, RetrievedTrajectory

class MyCustomRetrieval(RetrievalSystem):
    """Your custom retrieval implementation."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedTrajectory]:
        """
        Retrieve top-k most relevant trajectories.

        Args:
            query: Task description (e.g., "Put a mug on the coffee maker")
            k: Number of results to return

        Returns:
            List of RetrievedTrajectory objects sorted by relevance
        """
        # YOUR RETRIEVAL LOGIC HERE
        results = your_search_function(query, k)

        # Convert to standardized format
        return [
            RetrievedTrajectory(
                trajectory_id=r['id'],
                task_instance_id=r['task_id'],
                task_description=r['description'],
                similarity_score=r['score'],  # Your similarity score
                total_steps=r['num_steps'],
                document_text=r['full_trajectory']  # Full trajectory for LLM evaluation
            )
            for r in results
        ]

    def get_system_name(self) -> str:
        """Return unique identifier for this system."""
        return "my_custom_retrieval"

    def get_system_info(self) -> dict:
        """Return system configuration metadata."""
        return {
            "method": "your_method_name",
            "model": "your_model_if_applicable",
            "corpus_size": 336,
            "description": "Brief description of your approach"
        }
```

### Step 2: Run the benchmark

```python
from procedural_memory_benchmark import BenchmarkRunner

# Initialize your system
retrieval = MyCustomRetrieval()

# Run full benchmark
runner = BenchmarkRunner(retrieval, llm_model="gpt-4")
result = runner.run_benchmark(
    complexity_tiers=["HARD", "MEDIUM", "EASY"],  # All tiers
    save_results=True
)

# Analyze results
runner.print_summary(result)
```

### Step 3: Validate your implementation

```bash
python tests/validate_retrieval.py MyCustomRetrieval
```

## 📊 Benchmark Structure

### Query Bank (40 queries)

- **HARD (11 queries)**: Multi-object tasks, composite procedures, temperature transformations
- **MEDIUM (14 queries)**: Heating tasks, standard placement with constraints
- **EASY (15 queries)**: Simple placement, basic examination tasks

### Corpus (336 trajectories)

- **Source**: AgentInstruct dataset (ALFWorld-compatible tasks)
- **Format**: State-action pairs with task descriptions
- **Coverage**: 6 task types across various complexity levels
- **Size**: ~900KB bundled with package

### Evaluation

- **LLM-as-judge**: GPT-4 scores procedural similarity (0-10 scale)
- **Relevance threshold**: Score ≥6 = relevant
- **Metrics**: Precision@k, Recall@k, F1@k, NDCG@k, MAP
- **Analysis**: Overall + complexity-stratified performance

## 📖 Key Concepts

### What is procedural memory?

**Procedural memory** captures how to perform tasks through action sequences. This benchmark evaluates whether retrieval systems can identify **procedurally similar** trajectories even when:
- Objects differ (apple vs mug vs plate)
- Locations differ (cabinet vs shelf vs drawer)
- Surface details vary but core procedure matches

### Example: Procedural similarity

**Query**: "Put a mug on the coffee maker"

**Highly relevant** (score 9-10):
- "Take cup from shelf, navigate to coffee maker, place on coffee maker"
- Objects differ (cup vs mug) but procedure identical ✅

**Partially relevant** (score 6-7):
- "Take mug from counter, heat in microwave"
- Shares mug-handling but different end goal ⚠️

**Not relevant** (score 0-3):
- "Open cabinet and examine lamp"
- No procedural overlap ❌

## 📁 Repository Structure

```
procedural-memory-benchmark/
├── procedural_memory_benchmark/     # Main package
│   ├── benchmark/                   # Core benchmark components
│   │   ├── retrieval_interface.py  # Abstract interface + baselines
│   │   ├── benchmark_runner.py     # Main orchestrator
│   │   ├── query_bank.py           # Query management
│   │   ├── metrics_calculator.py   # IR metrics
│   │   ├── complexity_analyzer.py  # Complexity analysis
│   │   └── data/
│   │       └── query_bank.json     # 40 stratified queries
│   ├── agentinstruct/               # AgentInstruct components
│   │   ├── corpus_loader.py        # Trajectory loading
│   │   ├── embedder.py             # Embedding generation
│   │   ├── database.py             # State-aware baseline
│   │   └── database_actions_only.py # Action-only baseline
│   ├── llm/                         # LLM evaluation
│   │   └── llm_reasoner.py         # GPT-4 judge
│   ├── utils/                       # Utilities
│   │   └── paths.py                # Path resolution
│   └── data/
│       └── corpus/
│           └── agentinstruct_trajectories.json  # 336 trajectories
├── examples/                        # Example implementations
├── docs/                            # Documentation
├── tests/                           # Validation tools
└── README.md                        # This file
```

## 🎯 Baseline Systems

### AgentInstructEmbeddingRetrieval (State-aware)

**Method**: Embeddings of task description + state-action pairs
**Model**: all-MiniLM-L6-v2 (384-dim)
**Performance**: MAP ~0.79, P@1 ~0.78 (full 40-query evaluation)

```python
from procedural_memory_benchmark.benchmark import AgentInstructEmbeddingRetrieval

retrieval = AgentInstructEmbeddingRetrieval()
retrieval.setup()  # First-time database build
```

### AgentInstructActionOnlyRetrieval (Action-only)

**Method**: Embeddings of pure action sequences (no states/descriptions)
**Model**: all-MiniLM-L6-v2 (384-dim)
**Performance**: MAP ~0.72, P@1 ~0.65 (full 40-query evaluation)

```python
from procedural_memory_benchmark.benchmark import AgentInstructActionOnlyRetrieval

retrieval = AgentInstructActionOnlyRetrieval()
retrieval.setup()
```

## 🔍 Accessing Query Bank Data

```python
from procedural_memory_benchmark import QueryBank

bank = QueryBank()
bank.load()

# Get all queries
all_queries = bank.queries  # 40 queries

# Filter by complexity
hard_queries = bank.get_by_complexity("HARD")  # 11 queries
medium_queries = bank.get_by_complexity("MEDIUM")  # 14 queries
easy_queries = bank.get_by_complexity("EASY")  # 15 queries

# Filter by task type
examination_queries = bank.get_by_task_type(2)  # Examination tasks

# View query structure
query = all_queries[0]
print(f"Query: {query.task_description}")
print(f"Tier: {query.complexity_tier}")
print(f"Task type: {query.task_type}")
print(f"Factors: {query.complexity_factors}")
```

## 🔍 Accessing Corpus Data

```python
from procedural_memory_benchmark.agentinstruct import AgentInstructCorpusLoader

loader = AgentInstructCorpusLoader()
trajectories = loader.get_all_trajectories()  # 336 trajectories

# View trajectory structure
traj = trajectories[0]
print(f"Task: {traj.task_description}")
print(f"Steps: {traj.total_steps}")
print(f"State-action pairs: {len(traj.state_action_pairs)}")

# Get formatted text for your retrieval system
embedding_text = traj.get_embedding_text()  # Task + states + actions
action_sequence = traj.get_pure_action_sequence_text()  # Actions only
```

## ❓ Troubleshooting

### "OpenAI API key not found"

```bash
# Set environment variable
export OPENAI_API_KEY='your-key-here'

# Or create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### "ChromaDB database not available"

```python
# For baseline systems, build database first
retrieval = AgentInstructEmbeddingRetrieval()
retrieval.setup()  # This builds the database (~1-2 minutes)
```

### Database location

Databases are stored in user cache by default:
- **Linux/Mac**: `~/.cache/procedural_memory_benchmark/databases/`
- **Windows**: `%LOCALAPPDATA%/procedural_memory_benchmark/databases/`

Override with environment variable:
```bash
export PROCEDURAL_MEMORY_DB_PATH=/custom/path
```

### Results location

Results are saved to `./results/` in your current working directory.

## 📚 Further Documentation

- **API Reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete interface documentation
- **Examples**: See [examples/](examples/) for working implementations
- **Validation**: Use [tests/validate_retrieval.py](tests/validate_retrieval.py) to test your implementation



## 📄 License

Apache-2.0 License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or pull request.

---

**Questions?** Open an issue on GitHub or check the [documentation](docs/).
