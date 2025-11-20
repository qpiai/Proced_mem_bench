"""
Report Generator for Procedural Memory Benchmark

Generates comprehensive reports from benchmark results in multiple formats.
"""

import json
import os
from typing import Dict, Any
from .complexity_analyzer import ComplexityAnalyzer


class ReportGenerator:
    """
    Generate comprehensive benchmark reports.

    Supports:
    - JSON export
    - Markdown summary
    - Console output
    """

    def __init__(self):
        """Initialize report generator."""
        self.analyzer = ComplexityAnalyzer()

    def generate_report(
        self,
        benchmark_result: Dict,
        output_format: str = "markdown",
        output_path: str = None
    ) -> str:
        """
        Generate comprehensive benchmark report.

        Args:
            benchmark_result: Result from BenchmarkRunner
            output_format: "markdown", "json", or "console"
            output_path: Optional file path to save report

        Returns:
            Report string
        """
        if output_format == "json":
            report = self._generate_json_report(benchmark_result)
        elif output_format == "markdown":
            report = self._generate_markdown_report(benchmark_result)
        else:  # console
            report = self._generate_console_report(benchmark_result)

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"💾 Report saved to: {output_path}")

        return report

    def _generate_markdown_report(self, result: Dict) -> str:
        """Generate Markdown-formatted report."""
        lines = []
        lines.append("# Procedural Memory Retrieval Benchmark Report")
        lines.append("")
        lines.append(f"**System**: {result['system_name']}")
        lines.append(f"**Timestamp**: {result['timestamp']}")
        lines.append(f"**Execution Time**: {result['execution_time_seconds']:.2f} seconds")
        lines.append("")

        # System info
        lines.append("## System Configuration")
        lines.append("")
        for key, value in result['system_info'].items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Performance")
        lines.append("")
        overall = result['overall_metrics']
        lines.append(f"- **MAP**: {overall['map']:.3f}")
        lines.append(f"- **Precision@1**: {overall['precision_at_k'].get(1, 0):.3f}")
        lines.append(f"- **Precision@3**: {overall['precision_at_k'].get(3, 0):.3f}")
        lines.append(f"- **Precision@5**: {overall['precision_at_k'].get(5, 0):.3f}")
        lines.append(f"- **Total Queries**: {overall['total_queries']}")
        lines.append("")

        # Complexity-stratified metrics
        lines.append("## Complexity-Stratified Performance")
        lines.append("")
        for tier in ["HARD", "MEDIUM", "EASY"]:
            if tier in result['complexity_stratified_metrics']:
                metrics = result['complexity_stratified_metrics'][tier]
                lines.append(f"### {tier} Tier")
                lines.append("")
                lines.append(f"- **Queries**: {metrics['total_queries']}")
                lines.append(f"- **MAP**: {metrics['map']:.3f}")
                lines.append(f"- **Precision@1**: {metrics['precision_at_k'].get(1, 0):.3f}")
                lines.append(f"- **Precision@3**: {metrics['precision_at_k'].get(3, 0):.3f}")
                lines.append("")

        # Analysis
        analysis = self.analyzer.analyze(result)

        if "task_type_analysis" in analysis:
            lines.append("## Task Type Difficulty Ranking")
            lines.append("")
            lines.append("| Rank | Task Type | Name | Mean AP | Queries |")
            lines.append("|------|-----------|------|---------|---------|")
            for i, task in enumerate(analysis["task_type_analysis"]["difficulty_ranking"], 1):
                lines.append(f"| {i} | {task['task_type']} | {task['name']} | {task['mean_ap']:.3f} | {task['count']} |")
            lines.append("")

        if "corpus_coverage_insights" in analysis:
            coverage = analysis["corpus_coverage_insights"]
            lines.append("## Corpus Coverage Insights")
            lines.append("")
            lines.append(f"- **Queries with zero relevant**: {coverage['zero_relevant_count']}")
            lines.append(f"- **Queries with poor performance**: {coverage['poor_performance_count']}")
            lines.append(f"- **Note**: {coverage['coverage_note']}")
            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self, result: Dict) -> str:
        """Generate JSON-formatted report."""
        # Add analysis to result
        analysis = self.analyzer.analyze(result)
        result_with_analysis = result.copy()
        result_with_analysis['complexity_analysis'] = analysis

        return json.dumps(result_with_analysis, indent=2)

    def _generate_console_report(self, result: Dict) -> str:
        """Generate console-formatted report."""
        lines = []
        lines.append("="*70)
        lines.append("PROCEDURAL MEMORY RETRIEVAL BENCHMARK REPORT")
        lines.append("="*70)
        lines.append(f"\nSystem: {result['system_name']}")
        lines.append(f"Timestamp: {result['timestamp']}")
        lines.append(f"Execution: {result['execution_time_seconds']:.2f}s")

        lines.append("\n" + "-"*70)
        lines.append("OVERALL PERFORMANCE")
        lines.append("-"*70)
        overall = result['overall_metrics']
        lines.append(f"MAP: {overall['map']:.3f}")
        lines.append(f"P@1: {overall['precision_at_k'].get(1, 0):.3f}")
        lines.append(f"P@3: {overall['precision_at_k'].get(3, 0):.3f}")
        lines.append(f"P@5: {overall['precision_at_k'].get(5, 0):.3f}")

        lines.append("\n" + "-"*70)
        lines.append("COMPLEXITY-STRATIFIED PERFORMANCE")
        lines.append("-"*70)
        for tier in ["HARD", "MEDIUM", "EASY"]:
            if tier in result['complexity_stratified_metrics']:
                metrics = result['complexity_stratified_metrics'][tier]
                lines.append(f"\n{tier}: MAP={metrics['map']:.3f}, P@1={metrics['precision_at_k'].get(1, 0):.3f} ({metrics['total_queries']} queries)")

        lines.append("\n" + "="*70)
        return "\n".join(lines)


def test_report_generator():
    """Test report generator."""
    print("🧪 Testing Report Generator")
    print("=" * 70)

    # Mock result
    mock_result = {
        "system_name": "test_system",
        "system_info": {"method": "test", "model": "test-model"},
        "timestamp": "2025-10-13 12:00:00",
        "execution_time_seconds": 120.5,
        "overall_metrics": {
            "map": 0.545,
            "precision_at_k": {1: 0.4, 3: 0.43, 5: 0.3},
            "recall_at_k": {1: 0.4, 3: 0.85, 5: 0.9},
            "total_queries": 20
        },
        "complexity_stratified_metrics": {
            "HARD": {
                "map": 0.493,
                "precision_at_k": {1: 0.2, 3: 0.4},
                "total_queries": 7
            },
            "MEDIUM": {
                "map": 0.541,
                "precision_at_k": {1: 0.4, 3: 0.43},
                "total_queries": 8
            },
            "EASY": {
                "map": 0.700,
                "precision_at_k": {1: 0.6, 3: 0.58},
                "total_queries": 5
            }
        },
        "query_results": []
    }

    generator = ReportGenerator()

    # Generate markdown report
    print("\n📄 Generating Markdown report...")
    md_report = generator.generate_report(mock_result, output_format="markdown")
    print(md_report[:500] + "...")

    # Generate console report
    print("\n📺 Generating Console report...")
    console_report = generator.generate_report(mock_result, output_format="console")
    print(console_report)

    print("\n✅ Report generator test completed!")


if __name__ == "__main__":
    test_report_generator()
