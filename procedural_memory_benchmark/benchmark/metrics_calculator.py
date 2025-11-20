"""
Metrics Calculator for Procedural Memory Benchmark

Provides comprehensive metrics calculation including precision, recall, F1,
NDCG, and mean average precision for retrieval evaluation.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""
    query_id: str
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    average_precision: float
    num_retrieved: int
    num_relevant: int


class MetricsCalculator:
    """
    Calculate comprehensive retrieval metrics for benchmark evaluation.

    Supports standard IR metrics:
    - Precision@k
    - Recall@k
    - F1@k
    - NDCG@k (Normalized Discounted Cumulative Gain)
    - AP (Average Precision)
    - MAP (Mean Average Precision)
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics calculator.

        Args:
            k_values: List of k values for @k metrics (default: [1, 3, 5, 10])
        """
        self.k_values = k_values if k_values is not None else [1, 3, 5, 10]

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
            relevance_judgments: Dict mapping trajectory_id to binary relevance (True/False)
            retrieved_ids: List of retrieved trajectory IDs in rank order
            relevance_scores: Optional dict mapping trajectory_id to relevance score (for NDCG)

        Returns:
            QueryMetrics object with all calculated metrics
        """
        # Calculate metrics for each k value
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}

        total_relevant = sum(1 for is_rel in relevance_judgments.values() if is_rel)

        for k in self.k_values:
            # Get top-k retrieved
            top_k = retrieved_ids[:k]

            # Count relevant in top-k
            relevant_retrieved = sum(
                1 for traj_id in top_k
                if relevance_judgments.get(traj_id, False)
            )

            # Precision@k
            precision_k = relevant_retrieved / k if k > 0 else 0.0

            # Recall@k
            recall_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

            # F1@k
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0.0

            precision_at_k[k] = precision_k
            recall_at_k[k] = recall_k
            f1_at_k[k] = f1_k

            # NDCG@k
            ndcg_at_k[k] = self._calculate_ndcg_at_k(
                retrieved_ids=top_k,
                relevance_judgments=relevance_judgments,
                relevance_scores=relevance_scores
            )

        # Average Precision
        ap = self._calculate_average_precision(
            retrieved_ids=retrieved_ids,
            relevance_judgments=relevance_judgments
        )

        return QueryMetrics(
            query_id=query_id,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            ndcg_at_k=ndcg_at_k,
            average_precision=ap,
            num_retrieved=len(retrieved_ids),
            num_relevant=total_relevant
        )

    def _calculate_ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_judgments: Dict[str, bool],
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).

        Args:
            retrieved_ids: List of retrieved trajectory IDs in rank order
            relevance_judgments: Binary relevance judgments
            relevance_scores: Optional continuous relevance scores (0-10 scale)

        Returns:
            NDCG@k score
        """
        if not retrieved_ids:
            return 0.0

        # Get relevance values
        if relevance_scores is not None:
            # Use continuous scores, normalize to 0-3 scale for NDCG
            relevance_values = [
                min(3, max(0, int(relevance_scores.get(traj_id, 0) / 3.33)))
                for traj_id in retrieved_ids
            ]
        else:
            # Use binary relevance (0 or 1)
            relevance_values = [
                1 if relevance_judgments.get(traj_id, False) else 0
                for traj_id in retrieved_ids
            ]

        # Calculate DCG@k
        dcg = 0.0
        for i, relevance in enumerate(relevance_values):
            if relevance > 0:
                dcg += relevance / math.log2(i + 2)  # i+2 because rank starts from 1

        # Calculate IDCG@k (ideal ranking)
        ideal_relevance = sorted(relevance_values, reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            if relevance > 0:
                idcg += relevance / math.log2(i + 2)

        # Return NDCG@k
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_average_precision(
        self,
        retrieved_ids: List[str],
        relevance_judgments: Dict[str, bool]
    ) -> float:
        """
        Calculate Average Precision for a single query.

        Args:
            retrieved_ids: List of retrieved trajectory IDs in rank order
            relevance_judgments: Binary relevance judgments

        Returns:
            Average Precision score
        """
        if not retrieved_ids:
            return 0.0

        # Find relevant positions
        relevant_positions = []
        for i, traj_id in enumerate(retrieved_ids):
            if relevance_judgments.get(traj_id, False):
                relevant_positions.append(i + 1)  # 1-indexed position

        if not relevant_positions:
            return 0.0

        # Calculate AP = (1/|relevant|) * Σ(precision@k for each relevant position)
        precision_sum = 0.0
        for pos in relevant_positions:
            # Precision@pos = number of relevant items up to position pos / pos
            relevant_up_to_pos = sum(1 for p in relevant_positions if p <= pos)
            precision_at_pos = relevant_up_to_pos / pos
            precision_sum += precision_at_pos

        total_relevant = len(relevant_positions)
        return precision_sum / total_relevant

    def calculate_aggregate_metrics(
        self,
        query_metrics_list: List[QueryMetrics]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across multiple queries.

        Args:
            query_metrics_list: List of QueryMetrics objects

        Returns:
            Dictionary with aggregate metrics including:
            - precision_at_k: Mean precision for each k
            - recall_at_k: Mean recall for each k
            - f1_at_k: Mean F1 for each k
            - ndcg_at_k: Mean NDCG for each k
            - map: Mean Average Precision
            - total_queries: Number of queries
        """
        if not query_metrics_list:
            return {
                "precision_at_k": {k: 0.0 for k in self.k_values},
                "recall_at_k": {k: 0.0 for k in self.k_values},
                "f1_at_k": {k: 0.0 for k in self.k_values},
                "ndcg_at_k": {k: 0.0 for k in self.k_values},
                "map": 0.0,
                "total_queries": 0
            }

        aggregate = {
            "precision_at_k": {},
            "recall_at_k": {},
            "f1_at_k": {},
            "ndcg_at_k": {},
            "map": 0.0,
            "total_queries": len(query_metrics_list)
        }

        # Average metrics for each k
        for k in self.k_values:
            precisions = [qm.precision_at_k.get(k, 0.0) for qm in query_metrics_list]
            recalls = [qm.recall_at_k.get(k, 0.0) for qm in query_metrics_list]
            f1_scores = [qm.f1_at_k.get(k, 0.0) for qm in query_metrics_list]
            ndcg_scores = [qm.ndcg_at_k.get(k, 0.0) for qm in query_metrics_list]

            aggregate["precision_at_k"][k] = sum(precisions) / len(precisions)
            aggregate["recall_at_k"][k] = sum(recalls) / len(recalls)
            aggregate["f1_at_k"][k] = sum(f1_scores) / len(f1_scores)
            aggregate["ndcg_at_k"][k] = sum(ndcg_scores) / len(ndcg_scores)

        # Mean Average Precision
        ap_scores = [qm.average_precision for qm in query_metrics_list]
        aggregate["map"] = sum(ap_scores) / len(ap_scores)

        return aggregate


def test_metrics_calculator():
    """Test metrics calculator with sample data."""
    print("🧪 Testing Metrics Calculator")
    print("=" * 60)

    calculator = MetricsCalculator(k_values=[1, 3, 5])

    # Sample data: 5 retrieved items, 2 are relevant
    retrieved_ids = ["traj_1", "traj_2", "traj_3", "traj_4", "traj_5"]
    relevance_judgments = {
        "traj_1": False,
        "traj_2": True,   # Relevant
        "traj_3": False,
        "traj_4": True,   # Relevant
        "traj_5": False
    }
    relevance_scores = {
        "traj_1": 3.0,
        "traj_2": 8.0,
        "traj_3": 2.0,
        "traj_4": 7.0,
        "traj_5": 1.0
    }

    # Calculate metrics
    metrics = calculator.calculate_query_metrics(
        query_id="test_query_1",
        relevance_judgments=relevance_judgments,
        retrieved_ids=retrieved_ids,
        relevance_scores=relevance_scores
    )

    print("\n📊 Query Metrics:")
    print(f"   Precision@1: {metrics.precision_at_k[1]:.3f}")
    print(f"   Precision@3: {metrics.precision_at_k[3]:.3f}")
    print(f"   Precision@5: {metrics.precision_at_k[5]:.3f}")
    print(f"   Recall@5: {metrics.recall_at_k[5]:.3f}")
    print(f"   F1@5: {metrics.f1_at_k[5]:.3f}")
    print(f"   NDCG@5: {metrics.ndcg_at_k[5]:.3f}")
    print(f"   Average Precision: {metrics.average_precision:.3f}")

    # Test aggregate metrics
    query_metrics_list = [metrics]  # Single query for test
    aggregate = calculator.calculate_aggregate_metrics(query_metrics_list)

    print("\n📊 Aggregate Metrics:")
    print(f"   MAP: {aggregate['map']:.3f}")
    print(f"   Total queries: {aggregate['total_queries']}")

    print("\n✅ Metrics calculator test completed!")


if __name__ == "__main__":
    test_metrics_calculator()
