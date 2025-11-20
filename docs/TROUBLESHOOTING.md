# Troubleshooting Guide

Common issues and solutions when using the Procedural Memory Benchmark.

## Table of Contents

- [Installation Issues](#installation-issues)
- [OpenAI API Setup](#openai-api-setup)
- [Database Issues](#database-issues)
- [Import Errors](#import-errors)
- [Custom Retrieval Implementation](#custom-retrieval-implementation)
- [Performance Issues](#performance-issues)
- [Evaluation Problems](#evaluation-problems)

---

## Installation Issues

### Pip Install Fails

**Problem**: `pip install -e .` fails with dependency errors

**Solution**:
```bash
# Install in order (PyTorch first for compatibility)
pip install torch>=1.9.0
pip install -e .[all]

# Or install groups separately
pip install -e .                  # Core only
pip install -e .[baselines]       # Add baseline systems
pip install -e .[llm_judge]       # Add LLM evaluation
```

### ImportError: No module named 'procedural_memory_benchmark'

**Problem**: Package not found after installation

**Solution**:
```bash
# Option 1: Install in editable mode
pip install -e .

# Option 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/procedural-memory-benchmark"

# Option 3: Install from parent directory
cd /path/to/procedural-memory-benchmark
pip install -e .
```

### Python Version Compatibility

**Problem**: Package requires Python 3.8+

**Solution**:
```bash
# Check version
python --version

# If < 3.8, use a newer Python
python3.8 -m pip install -e .
python3.9 -m pip install -e .
```

---

## OpenAI API Setup

### Missing API Key Error

**Problem**: `openai.AuthenticationError` or `OPENAI_API_KEY not set`

**Solution**:
```bash
# Option 1: Environment variable
export OPENAI_API_KEY='your-api-key-here'

# Option 2: .env file (recommended)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Option 3: Direct in code (not recommended for production)
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

### Verify API Key Setup

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f"✓ API key loaded: {api_key[:8]}...")
else:
    print("✗ API key not found!")
```

### Rate Limit Errors

**Problem**: `RateLimitError: You exceeded your current quota`

**Solutions**:
1. **Check OpenAI billing**: Ensure you have credits
2. **Reduce batch size**: Use fewer queries per tier
3. **Use cheaper model**: Switch from `gpt-4` to `gpt-3.5-turbo`

```python
runner = BenchmarkRunner(
    retrieval,
    llm_model="gpt-3.5-turbo"  # Cheaper and faster
)
```

### API Timeout Errors

**Problem**: Requests timing out for large batches

**Solution**: The LLM interface includes retry logic, but you can reduce load:
```python
# Test with fewer queries first
result = runner.run_benchmark(
    complexity_tiers=["EASY"],
    max_queries_per_tier=3  # Start small
)
```

---

## Database Issues

### ChromaDB Build Fails

**Problem**: Database creation fails or corpus not loading

**Solution**:
```bash
# Clear cache and rebuild
rm -rf ~/.cache/procedural_memory_benchmark/databases/
python -c "
from procedural_memory_benchmark import AgentInstructEmbeddingRetrieval
retrieval = AgentInstructEmbeddingRetrieval()
retrieval.setup()  # Forces rebuild
"
```

### Database Location

**Problem**: Want to use custom database location

**Solution**:
```bash
# Set environment variable
export PROCEDURAL_MEMORY_DB_PATH=/custom/path/to/databases

# Or specify in code
from procedural_memory_benchmark.agentinstruct import AgentInstructDatabase

db = AgentInstructDatabase(db_path="/custom/path")
```

### Permission Errors

**Problem**: Cannot write to cache directory

**Solution**:
```bash
# Linux/Mac: Fix permissions
chmod -R u+w ~/.cache/procedural_memory_benchmark/

# Or use custom path with write permissions
export PROCEDURAL_MEMORY_DB_PATH=./local_databases
```

### Database Corruption

**Problem**: ChromaDB errors after interruption

**Solution**:
```bash
# Delete and rebuild
rm -rf ~/.cache/procedural_memory_benchmark/databases/
# Then rebuild by running setup() again
```

---

## Import Errors

### Cannot Import RetrievalSystem

**Problem**: `ImportError: cannot import name 'RetrievalSystem'`

**Solution**:
```python
# Correct import
from procedural_memory_benchmark import RetrievalSystem, RetrievedTrajectory

# OR
from procedural_memory_benchmark.benchmark.retrieval_interface import (
    RetrievalSystem,
    RetrievedTrajectory
)
```

### Module Not Found After Installation

**Problem**: Installed but imports fail

**Solution**:
```bash
# Verify installation
pip list | grep procedural

# Reinstall if needed
pip uninstall procedural-memory-benchmark
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

---

## Custom Retrieval Implementation

### "Missing required method" Error

**Problem**: ValidationChecker reports missing methods

**Solution**: Implement all 3 required methods:
```python
from procedural_memory_benchmark import RetrievalSystem, RetrievedTrajectory
from typing import List, Dict

class MyRetrieval(RetrievalSystem):
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        # REQUIRED: Return list of RetrievedTrajectory
        ...

    def get_system_name(self) -> str:
        # REQUIRED: Return unique identifier
        return "my_retrieval_system"

    def get_system_info(self) -> Dict:
        # REQUIRED: Return configuration metadata
        return {"method": "custom", "version": "1.0"}
```

### Results Not Properly Formatted

**Problem**: "Result is not RetrievedTrajectory" error

**Solution**: Ensure you return `RetrievedTrajectory` objects with ALL required fields:
```python
from procedural_memory_benchmark import RetrievedTrajectory

def retrieve(self, query: str, k: int = 5):
    results = []
    for traj in self.corpus:
        score = self._calculate_similarity(query, traj)

        # CRITICAL: Include all 6 required fields
        results.append(RetrievedTrajectory(
            trajectory_id=traj.id,              # str
            task_instance_id=traj.id,           # str
            task_description=traj.description,  # str
            similarity_score=score,             # float
            total_steps=len(traj.actions),      # int
            document_text=traj.full_text        # str - MUST NOT BE EMPTY
        ))

    # Sort by score (highest first)
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results[:k]
```

### Empty document_text Error

**Problem**: "document_text is empty - LLM judge needs content!"

**Solution**: `document_text` must contain the full trajectory for LLM evaluation:
```python
# BAD: Empty document_text
document_text = ""

# GOOD: Include enough information for LLM to judge similarity
document_text = f"""Task: {traj.task_description}

Steps:
{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(traj.actions))}
"""
```

### Results Not Sorted

**Problem**: Warning about "Results not sorted by similarity"

**Solution**: Always sort by `similarity_score` in descending order:
```python
# Before returning
results.sort(key=lambda x: x.similarity_score, reverse=True)
return results[:k]
```

---

## Performance Issues

### Slow Corpus Loading

**Problem**: Initial setup takes too long

**Solution**: Database is built once and cached. After first run:
```python
# First run: ~1-2 minutes (builds database)
retrieval.setup()

# Subsequent runs: ~1-2 seconds (loads from cache)
retrieval.setup()
```

### Slow Evaluation

**Problem**: Benchmark takes too long to run

**Solutions**:
```python
# 1. Test on subset first
result = runner.run_benchmark(
    complexity_tiers=["EASY"],
    max_queries_per_tier=3
)

# 2. Use faster LLM model
runner = BenchmarkRunner(
    retrieval,
    llm_model="gpt-3.5-turbo"  # 10x faster than gpt-4
)

# 3. Reduce k values
runner = BenchmarkRunner(
    retrieval,
    k_values=[1, 3, 5]  # Instead of [1, 3, 5, 10]
)
```

### Memory Issues

**Problem**: Out of memory during embedding generation

**Solution**:
```python
# For sentence-transformers, reduce batch size
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(
    texts,
    batch_size=16,  # Reduce from default 32
    show_progress_bar=True
)
```

---

## Evaluation Problems

### Zero Results Retrieved

**Problem**: `retrieve()` returns empty list

**Checklist**:
1. ✓ Corpus loaded correctly?
2. ✓ Database built successfully?
3. ✓ Query is not empty?
4. ✓ Similarity calculation working?

**Debug**:
```python
# Check corpus
print(f"Corpus size: {len(retrieval.corpus)}")

# Test retrieval
results = retrieval.retrieve("test query", k=5)
print(f"Retrieved {len(results)} results")

# Check first result
if results:
    print(f"Top result: {results[0].task_description}")
    print(f"Score: {results[0].similarity_score}")
```

### All Similarity Scores are Zero

**Problem**: Warning "All similarity scores are 0"

**Solution**: Check your similarity calculation:
```python
def _calculate_similarity(self, query, trajectory):
    # BAD: Always returns 0
    return 0

    # GOOD: Actual similarity measure
    from sklearn.metrics.pairwise import cosine_similarity
    query_emb = self.embed(query)
    traj_emb = self.embed(trajectory.text)
    return cosine_similarity([query_emb], [traj_emb])[0][0]
```

### LLM Evaluation Fails

**Problem**: Errors during relevance scoring

**Checklist**:
1. ✓ OpenAI API key set?
2. ✓ Internet connection active?
3. ✓ `document_text` not empty?
4. ✓ API quota available?

**Debug**:
```python
# Test LLM directly
from procedural_memory_benchmark.llm.llm_reasoner import LLMReasoner

reasoner = LLMReasoner(model="gpt-3.5-turbo")
result = reasoner.analyze_retrieval(
    query="test query",
    retrieved_results=[...],
    overlap_analysis={...}
)
print(result)
```

### Results File Not Saved

**Problem**: Benchmark runs but no results file created

**Solution**:
```python
# Ensure save_results=True (default)
result = runner.run_benchmark(save_results=True)

# Check results directory
from procedural_memory_benchmark.utils.paths import get_results_dir
print(f"Results dir: {get_results_dir()}")

# Manually save if needed
import json
results_path = get_results_dir() / "my_results.json"
with open(results_path, 'w') as f:
    json.dump(result.to_dict(), f, indent=2)
```

---

## Validation

### Run Validation Before Benchmark

Always validate your implementation first:
```python
from tests.validate_retrieval import ValidationChecker

checker = ValidationChecker(your_retrieval)
success = checker.run_all_checks()

if success:
    # Safe to run benchmark
    runner = BenchmarkRunner(your_retrieval)
    result = runner.run_benchmark()
```

---

## Still Having Issues?

1. **Check examples**: See `examples/` directory for working code
2. **Read API docs**: See `docs/API_REFERENCE.md` for specifications
3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Create minimal reproduction**: Test with simplest possible case
5. **Check versions**:
   ```bash
   pip list | grep -E "chromadb|sentence-transformers|openai"
   ```

## Quick Reference

### Minimal Working Example

```python
from procedural_memory_benchmark import (
    RetrievalSystem,
    RetrievedTrajectory,
    BenchmarkRunner
)

class SimpleRetrieval(RetrievalSystem):
    def __init__(self):
        from procedural_memory_benchmark.agentinstruct import AgentInstructCorpusLoader
        loader = AgentInstructCorpusLoader()
        self.trajectories = loader.get_all_trajectories()

    def retrieve(self, query: str, k: int = 5):
        # Your retrieval logic here
        results = []
        for traj in self.trajectories[:k]:
            results.append(RetrievedTrajectory(
                trajectory_id=traj.task_instance_id,
                task_instance_id=traj.task_instance_id,
                task_description=traj.task_description,
                similarity_score=1.0,  # Your score
                total_steps=traj.total_steps,
                document_text=traj.get_embedding_text()
            ))
        return results

    def get_system_name(self):
        return "simple_retrieval"

    def get_system_info(self):
        return {"method": "simple"}

# Test it
retrieval = SimpleRetrieval()
runner = BenchmarkRunner(retrieval)
result = runner.run_benchmark(
    complexity_tiers=["EASY"],
    max_queries_per_tier=3
)
```
