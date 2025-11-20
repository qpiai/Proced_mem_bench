# API Reference

Complete reference for the Procedural Memory Benchmark API.

## Core Interfaces

### RetrievalSystem (Abstract Base Class)

**Location**: `procedural_memory_benchmark.benchmark.retrieval_interface`

Abstract interface that all custom retrieval systems must implement.

#### Required Methods

```python
class RetrievalSystem(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedTrajectory]:
        """
        Retrieve top-k most relevant trajectories for a query.

        Args:
            query (str): Task description to search for
                Example: "Put a mug on the coffee maker"
            k (int): Number of results to return (default: 5)

        Returns:
            List[RetrievedTrajectory]: Top-k results sorted by relevance (highest first)

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """
        Return unique identifier for this retrieval system.

        Returns:
            str: System name (used in result filenames and reports)

        Example:
            "bm25_retrieval"
            "hybrid_semantic_lexical"
        """
        pass

    @abstractmethod
    def get_system_info(self) -> Dict:
        """
        Return system configuration metadata.

        Returns:
            Dict: Configuration information

        Example:
            {
                "method": "hybrid",
                "model": "all-MiniLM-L6-v2",
                "corpus_size": 336,
                "params": {"alpha": 0.5}
            }
        """
        pass
```

### RetrievedTrajectory (Data Class)

**Location**: `procedural_memory_benchmark.benchmark.retrieval_interface`

Standardized format for retrieval results.

#### Required Fields

```python
@dataclass
class RetrievedTrajectory:
    trajectory_id: str              # Unique trajectory identifier
    task_instance_id: str           # Task instance identifier
    task_description: str           # Human-readable task description
    similarity_score: float         # Your similarity score (higher = more similar)
    total_steps: int                # Number of steps in trajectory
    document_text: str              # Full trajectory text for LLM evaluation

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
```

#### Field Descriptions

- **trajectory_id**: Unique ID for this trajectory (e.g., "alfworld_42")
- **task_instance_id**: Instance ID (often same as trajectory_id)
- **task_description**: Task description (e.g., "Put a mug on the coffee maker")
- **similarity_score**: Your retrieval score (any range, used for ranking)
- **total_steps**: Number of steps in the trajectory
- **document_text**: **CRITICAL** - Full trajectory content for LLM to evaluate
  - Should contain enough information for LLM to judge procedural similarity
  - Can be formatted however makes sense for your method
  - Examples: state-action pairs, pure actions, natural language description

## Benchmark Execution

### BenchmarkRunner

**Location**: `procedural_memory_benchmark.benchmark.benchmark_runner`

Main orchestrator for running benchmarks.

#### Constructor

```python
BenchmarkRunner(
    retrieval_system: RetrievalSystem,
    query_bank_path: str = None,
    llm_model: str = "gpt-4",
    k_values: List[int] = None
)
```

**Parameters:**
- `retrieval_system`: Your RetrievalSystem implementation
- `query_bank_path`: Path to custom query bank (default: package data)
- `llm_model`: OpenAI model for evaluation ("gpt-4", "gpt-3.5-turbo", "gpt-5")
- `k_values`: K values for metrics (default: [1, 3, 5, 10])

#### Methods

##### run_benchmark()

```python
def run_benchmark(
    self,
    complexity_tiers: List[str] = None,
    max_queries_per_tier: Optional[int] = None,
    save_results: bool = True
) -> BenchmarkResult:
    """
    Run complete benchmark evaluation.

    Args:
        complexity_tiers: List of tiers to evaluate
            Options: ["HARD", "MEDIUM", "EASY"]
            Default: ["HARD", "MEDIUM", "EASY"]

        max_queries_per_tier: Optional limit on queries per tier
            None = use all queries (recommended for final evaluation)
            3-5 = quick testing
            Default: None

        save_results: Whether to save results to file
            Default: True

    Returns:
        BenchmarkResult: Complete evaluation results
    """
```

##### print_summary()

```python
def print_summary(self, result: BenchmarkResult):
    """
    Print formatted summary of results.

    Args:
        result: BenchmarkResult from run_benchmark()
    """
```

### BenchmarkResult (Data Class)

```python
@dataclass
class BenchmarkResult:
    system_name: str                      # Your system name
    system_info: Dict                     # Your system configuration
    query_results: List[Dict]             # Per-query results with metrics
    overall_metrics: Dict                 # Aggregate metrics
    complexity_stratified_metrics: Dict   # Per-tier metrics
    execution_time_seconds: float         # Total runtime
    timestamp: str                        # Evaluation timestamp
```

## Query Management

### QueryBank

**Location**: `procedural_memory_benchmark.benchmark.query_bank`

Manages complexity-stratified queries.

#### Constructor

```python
QueryBank(query_bank_path: str = None)
```

**Parameters:**
- `query_bank_path`: Path to query bank JSON (default: package data with 40 queries)

#### Methods

```python
def load(self) -> List[BenchmarkQuery]:
    """Load queries from file."""

def get_by_complexity(self, tier: str) -> List[BenchmarkQuery]:
    """
    Get queries by complexity tier.

    Args:
        tier: "HARD", "MEDIUM", or "EASY"

    Returns:
        List of queries in that tier
    """

def get_by_task_type(self, task_type: int) -> List[BenchmarkQuery]:
    """
    Get queries by ALFWorld task type.

    Args:
        task_type: Integer 1-6
            1: pick_and_place_simple
            2: look_at_obj_in_light
            3: pick_clean_then_place
            4: pick_heat_then_place
            5: pick_cool_then_place
            6: pick_two_obj_and_place

    Returns:
        List of queries of that type
    """

def get_statistics(self) -> Dict:
    """Get statistics about the query bank."""
```

### BenchmarkQuery (Data Class)

```python
@dataclass
class BenchmarkQuery:
    query_id: str                     # Unique query ID (e.g., "hard_1")
    task_description: str             # Task description
    complexity_tier: str              # "HARD", "MEDIUM", or "EASY"
    task_type: int                    # ALFWorld task type (1-6)
    complexity_factors: List[str]     # Complexity factors
    source: str                       # "valid_seen" or "valid_unseen"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
```

## Metrics

### MetricsCalculator

**Location**: `procedural_memory_benchmark.benchmark.metrics_calculator`

Calculates standard IR metrics.

#### Metrics Computed

**Precision@k**: Fraction of top-k results that are relevant
```
P@k = (# relevant in top-k) / k
```

**Recall@k**: Fraction of all relevant items found in top-k
```
R@k = (# relevant in top-k) / (# total relevant)
```

**F1@k**: Harmonic mean of precision and recall
```
F1@k = 2 * (P@k * R@k) / (P@k + R@k)
```

**NDCG@k**: Normalized Discounted Cumulative Gain
- Accounts for position of relevant results
- Weighted by relevance scores (0-10 from LLM)

**MAP**: Mean Average Precision
- Average of precision values at each relevant result
- Single-number summary of retrieval quality

**AP**: Average Precision (per query)

#### Constructor

```python
MetricsCalculator(k_values: List[int] = [1, 3, 5, 10])
```

#### Methods

```python
def calculate_query_metrics(
    self,
    query_id: str,
    relevance_judgments: Dict[str, bool],
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float] = None
) -> QueryMetrics:
    """
    Calculate metrics for a single query.

    Args:
        query_id: Query identifier
        relevance_judgments: {trajectory_id: is_relevant} (score >= 6)
        retrieved_ids: Ordered list of retrieved trajectory IDs
        relevance_scores: {trajectory_id: score_0_to_10}

    Returns:
        QueryMetrics with P@k, R@k, F1@k, NDCG@k, AP
    """

def calculate_aggregate_metrics(
    self,
    query_metrics_list: List[QueryMetrics]
) -> Dict:
    """
    Calculate aggregate metrics across queries.

    Args:
        query_metrics_list: List of QueryMetrics from multiple queries

    Returns:
        Dict with mean metrics across all queries
    """
```

## Corpus Access

### AgentInstructCorpusLoader

**Location**: `procedural_memory_benchmark.agentinstruct.corpus_loader`

Loads AgentInstruct trajectory corpus.

#### Constructor

```python
AgentInstructCorpusLoader(corpus_path: str = None)
```

**Parameters:**
- `corpus_path`: Path to corpus JSON (default: package data)

#### Methods

```python
def load_corpus(self) -> Dict:
    """Load raw corpus data."""

def get_all_trajectories(self) -> List[AgentInstructTrajectory]:
    """Get all 336 trajectories as structured objects."""

def get_trajectory_by_id(self, task_instance_id: str) -> Optional[AgentInstructTrajectory]:
    """Get specific trajectory by ID."""

def get_corpus_statistics(self) -> Dict:
    """Get corpus statistics (counts, patterns, etc.)."""
```

### AgentInstructTrajectory (Data Class)

```python
@dataclass
class AgentInstructTrajectory:
    task_instance_id: str           # Unique ID (e.g., "alfworld_0")
    task_description: str           # Task description
    state_action_pairs: List[Dict]  # List of {step_id, state, action}
    total_steps: int                # Number of steps
    source: str                     # "agentinstruct"

    def get_embedding_text(self) -> str:
        """
        Get formatted text for embedding generation.
        Includes task description + state-action pairs.
        """

    def get_action_sequence(self) -> List[str]:
        """Get list of actions only."""

    def get_pure_action_sequence_text(self) -> str:
        """
        Get action sequence as text (no states/descriptions).
        Format: "action1 | action2 | action3"
        """
```

## Analysis Tools

### ComplexityAnalyzer

**Location**: `procedural_memory_benchmark.benchmark.complexity_analyzer`

Analyzes performance across complexity dimensions.

#### Methods

```python
def analyze_by_tier(
    self,
    query_results: List[Dict],
    tiers: List[str]
) -> Dict:
    """Analyze performance by complexity tier."""

def analyze_by_task_type(
    self,
    query_results: List[Dict]
) -> Dict:
    """Analyze performance by task type."""

def identify_difficulty_patterns(
    self,
    query_results: List[Dict]
) -> Dict:
    """Identify which complexity factors correlate with poor performance."""
```

### ReportGenerator

**Location**: `procedural_memory_benchmark.benchmark.report_generator`

Generates formatted reports.

#### Methods

```python
def generate_report(
    self,
    result: BenchmarkResult,
    output_format: str = "markdown",
    output_path: str = None
) -> str:
    """
    Generate formatted report.

    Args:
        result: BenchmarkResult to report on
        output_format: "markdown", "json", or "console"
        output_path: Optional file path to save

    Returns:
        Formatted report string
    """
```

## Environment Configuration

### Path Utilities

**Location**: `procedural_memory_benchmark.utils.paths`

```python
def get_query_bank_path(custom_path: Path = None) -> Path:
    """Get path to query bank (package data or custom)."""

def get_corpus_path(custom_path: Path = None) -> Path:
    """Get path to corpus (package data or custom)."""

def get_default_db_path(db_name: str = "databases") -> Path:
    """
    Get default database path in user cache.
    - Linux/Mac: ~/.cache/procedural_memory_benchmark/
    - Windows: %LOCALAPPDATA%/procedural_memory_benchmark/
    """

def get_results_dir(custom_dir: Path = None) -> Path:
    """Get directory for saving results (default: ./results)."""
```

### Environment Variables

- **OPENAI_API_KEY**: Required for LLM-as-judge evaluation
- **PROCEDURAL_MEMORY_DB_PATH**: Optional custom database location

## Usage Examples

See [examples/](../examples/) directory for complete working examples:
- `quickstart.py` - 5-minute baseline evaluation
- `custom_retrieval_minimal.py` - Minimal custom implementation
- `compare_with_baseline.py` - Multi-system comparison
