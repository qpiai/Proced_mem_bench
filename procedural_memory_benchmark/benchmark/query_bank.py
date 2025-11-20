"""
Query Bank for Procedural Memory Benchmark

Manages complexity-stratified queries for benchmark evaluation.
Provides classification, loading, and filtering capabilities.
"""

import json
import os
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum


class ComplexityTier(Enum):
    """Empirically-validated complexity tiers."""
    HARD = "HARD"
    MEDIUM = "MEDIUM"
    EASY = "EASY"


@dataclass
class BenchmarkQuery:
    """
    A benchmark query with intrinsic complexity properties.

    No historical performance metrics included - only objective properties
    used for complexity classification.
    """
    query_id: str
    task_description: str
    complexity_tier: str  # "HARD", "MEDIUM", or "EASY"
    task_type: int  # ALFWorld task type (1-6)
    complexity_factors: List[str]  # Objective linguistic/semantic features
    source: str  # "valid_seen" or "valid_unseen"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class TaskTypeClassifier:
    """
    Classify ALFWorld task types based on intrinsic query properties.

    Uses empirically-validated complexity mapping:
    - HARD: Types 6, 3, and Type 1 with composite objects
    - MEDIUM: Types 4, 1 (standard)
    - EASY: Types 2, 5
    """

    # ALFWorld task type definitions
    TASK_TYPE_NAMES = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place"
    }

    @staticmethod
    def classify_task_type(query: str) -> int:
        """
        Classify task type based on query content.

        Args:
            query: Task description string

        Returns:
            Task type integer (1-6)
        """
        query_lower = query.lower()

        # Type 2: look_at_obj_in_light
        if any(phrase in query_lower for phrase in ['examine', 'look at', 'light', 'lamp']):
            return 2

        # Type 6: pick_two_obj_and_place (multi-object)
        if any(phrase in query_lower for phrase in ['two', 'both', 'pair of', '2 ']):
            return 6

        # Type 3: pick_clean_then_place_in_recep
        if any(phrase in query_lower for phrase in ['clean', 'wash', 'washed']):
            return 3

        # Type 4: pick_heat_then_place_in_recep
        if any(phrase in query_lower for phrase in ['heat', 'cook', 'microwave', 'heated', 'cooked', 'microwaved', 'hot ']):
            # But not if it's about placing chilled/cold items IN microwave
            if not any(phrase in query_lower for phrase in ['chilled', 'cold', 'cool']):
                return 4

        # Type 5: pick_cool_then_place_in_recep
        if any(phrase in query_lower for phrase in ['chill', 'cool', 'cold', 'chilled']):
            return 5

        # Default: Type 1: pick_and_place_simple
        return 1

    @staticmethod
    def extract_complexity_factors(query: str) -> List[str]:
        """
        Extract objective complexity factors from query.

        Returns list of complexity factors present in the query.
        """
        query_lower = query.lower()
        factors = []

        # Multi-object coordination
        if any(word in query_lower for word in ['two', 'both', 'pair', '2 ']):
            factors.append('multi_object')

        # Composite objects (X containing Y, X with Y on top)
        if any(phrase in query_lower for phrase in ['containing', 'with', ' in it', 'on top']):
            factors.append('composite')

        # Temperature transformation
        if any(word in query_lower for word in ['heat', 'cool', 'chill', 'cook', 'microwave', 'cold', 'hot']):
            factors.append('temperature')

        # Unusual/ambiguous destination
        if any(word in query_lower for word in ['gold bin', 'black bin']):
            factors.append('unusual_destination')

        # Sequential actions
        if ' and ' in query_lower or ' then ' in query_lower:
            factors.append('sequential')

        # Disposal action
        if any(word in query_lower for word in ['throw', 'trash', 'garbage']):
            factors.append('disposal')

        # Cleaning action
        if any(word in query_lower for word in ['clean', 'wash']):
            factors.append('cleaning')

        # Examination action
        if any(word in query_lower for word in ['examine', 'look at']):
            factors.append('examination')

        return factors

    @staticmethod
    def classify_complexity(query: str, task_type: int) -> str:
        """
        Classify query complexity based on empirically-validated tiers.

        Args:
            query: Task description
            task_type: ALFWorld task type (1-6)

        Returns:
            Complexity tier: "HARD", "MEDIUM", or "EASY"
        """
        query_lower = query.lower()

        # HARD tier: Structural complexity
        if task_type == 6:  # Multi-object
            return ComplexityTier.HARD.value

        if task_type == 3:  # Cleaning (sequential steps)
            return ComplexityTier.HARD.value

        # Type 1 with composite objects is HARD
        if task_type == 1:
            if any(phrase in query_lower for phrase in ['containing', 'with', ' in it', 'on top']):
                return ComplexityTier.HARD.value
            # Check for unusual destinations
            if any(word in query_lower for word in ['gold bin', 'black bin']):
                return ComplexityTier.HARD.value

        # EASY tier: Simple, clear semantics
        if task_type == 2:  # Examination
            return ComplexityTier.EASY.value

        if task_type == 5:  # Cooling (simpler than heating)
            return ComplexityTier.EASY.value

        # MEDIUM tier: Everything else
        return ComplexityTier.MEDIUM.value


class QueryBank:
    """
    Manages benchmark query bank with complexity stratification.

    Provides loading, filtering, and access to queries organized by
    empirically-validated complexity tiers.
    """

    def __init__(self, query_bank_path: str = None):
        """
        Initialize query bank.

        Args:
            query_bank_path: Path to query bank JSON file
        """
        if query_bank_path is None:
            # Use package data via path utility
            from ..utils.paths import get_query_bank_path
            query_bank_path = str(get_query_bank_path())

        self.query_bank_path = query_bank_path
        self.queries: List[BenchmarkQuery] = []
        self.metadata: Dict = {}

    def load(self) -> List[BenchmarkQuery]:
        """
        Load query bank from JSON file.

        Returns:
            List of BenchmarkQuery objects
        """
        if not os.path.exists(self.query_bank_path):
            raise FileNotFoundError(f"Query bank not found: {self.query_bank_path}")

        with open(self.query_bank_path, 'r') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})
        self.queries = []

        for query_data in data.get('queries', []):
            # Map JSON fields to BenchmarkQuery fields
            # Handle both old format (tier, query_text) and new format (complexity_tier, task_description)
            mapped_data = {
                'query_id': query_data.get('query_id'),
                'task_description': query_data.get('task_description', query_data.get('query_text')),
                'complexity_tier': query_data.get('complexity_tier', query_data.get('tier')),
                'task_type': query_data.get('task_type', 1),  # Default to type 1 if not specified
                'complexity_factors': query_data.get('complexity_factors', []),
                'source': query_data.get('source', 'unknown')
            }
            query = BenchmarkQuery(**mapped_data)
            self.queries.append(query)

        print(f"✅ Loaded {len(self.queries)} queries from query bank")
        return self.queries

    def save(self, queries: List[BenchmarkQuery], metadata: Dict = None):
        """
        Save query bank to JSON file.

        Args:
            queries: List of BenchmarkQuery objects
            metadata: Optional metadata dictionary
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.query_bank_path), exist_ok=True)

        data = {
            "metadata": metadata if metadata is not None else self.metadata,
            "queries": [q.to_dict() for q in queries]
        }

        with open(self.query_bank_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved {len(queries)} queries to: {self.query_bank_path}")

    def get_by_complexity(self, tier: str) -> List[BenchmarkQuery]:
        """
        Get queries filtered by complexity tier.

        Args:
            tier: "HARD", "MEDIUM", or "EASY"

        Returns:
            List of queries in the specified tier
        """
        return [q for q in self.queries if q.complexity_tier == tier]

    def get_by_task_type(self, task_type: int) -> List[BenchmarkQuery]:
        """
        Get queries filtered by task type.

        Args:
            task_type: Task type integer (1-6)

        Returns:
            List of queries of the specified type
        """
        return [q for q in self.queries if q.task_type == task_type]

    def get_statistics(self) -> Dict:
        """
        Get statistics about the query bank.

        Returns:
            Dictionary with query bank statistics
        """
        if not self.queries:
            return {}

        tier_counts = {}
        task_type_counts = {}
        factor_counts = {}

        for query in self.queries:
            # Count by tier
            tier_counts[query.complexity_tier] = tier_counts.get(query.complexity_tier, 0) + 1

            # Count by task type
            task_type_counts[query.task_type] = task_type_counts.get(query.task_type, 0) + 1

            # Count complexity factors
            for factor in query.complexity_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1

        return {
            "total_queries": len(self.queries),
            "queries_by_tier": tier_counts,
            "queries_by_task_type": task_type_counts,
            "complexity_factors": factor_counts,
            "metadata": self.metadata
        }


def create_benchmark_query(
    task_description: str,
    source: str = "valid_unseen",
    query_id_prefix: str = "query"
) -> BenchmarkQuery:
    """
    Create a benchmark query from a task description.

    Automatically classifies task type, complexity tier, and extracts factors.

    Args:
        task_description: Task description string
        source: Data source ("valid_seen" or "valid_unseen")
        query_id_prefix: Prefix for query ID

    Returns:
        BenchmarkQuery object
    """
    classifier = TaskTypeClassifier()

    # Classify task type
    task_type = classifier.classify_task_type(task_description)

    # Extract complexity factors
    factors = classifier.extract_complexity_factors(task_description)

    # Classify complexity tier
    complexity_tier = classifier.classify_complexity(task_description, task_type)

    # Generate query ID
    tier_abbrev = complexity_tier[0].lower()  # 'h', 'm', or 'e'
    query_id = f"{query_id_prefix}_{tier_abbrev}_{task_type}"

    return BenchmarkQuery(
        query_id=query_id,
        task_description=task_description,
        complexity_tier=complexity_tier,
        task_type=task_type,
        complexity_factors=factors,
        source=source
    )


def test_query_bank():
    """Test query bank functionality."""
    print("🧪 Testing Query Bank")
    print("=" * 60)

    # Test classification
    print("\n📋 Testing Query Classification:")
    test_queries = [
        "To move two bars of soap to the gold bin.",  # HARD - multi-object + unusual dest
        "Put a clean egg in the microwave.",  # HARD - cleaning
        "Heat a tomato and throw it away.",  # MEDIUM - heating
        "Examine the bat with the light on the desk.",  # EASY - examination
        "Put a chilled potato in the microwave.",  # EASY - cooling
    ]

    queries = []
    for i, desc in enumerate(test_queries):
        query = create_benchmark_query(desc, query_id_prefix=f"test_{i}")
        queries.append(query)
        print(f"\n   Query: {desc[:50]}...")
        print(f"     Type: {query.task_type} ({TaskTypeClassifier.TASK_TYPE_NAMES[query.task_type]})")
        print(f"     Tier: {query.complexity_tier}")
        print(f"     Factors: {', '.join(query.complexity_factors)}")

    # Test query bank operations
    print("\n\n📋 Testing Query Bank Operations:")
    bank = QueryBank()
    bank.queries = queries

    stats = bank.get_statistics()
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   By tier: {stats['queries_by_tier']}")
    print(f"   By type: {stats['queries_by_task_type']}")

    print("\n✅ Query bank test completed!")


if __name__ == "__main__":
    test_query_bank()
