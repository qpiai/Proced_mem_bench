"""
LLM Reasoning Interface for Procedural Memory Evaluation

Provides LLM-based evaluation of retrieved trajectories for benchmark evaluation.
Uses OpenAI API to assess procedural similarity and relevance.
"""

import os
import re
from typing import Dict, List, Optional

# Lazy imports for optional dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


class LLMReasoner:
    """
    LLM-based reasoning system for procedural trajectory evaluation.

    Analyzes retrieved trajectories and provides relevance scoring with
    explanations for procedural similarity.
    """

    UNIVERSAL_PROMPT = """You are evaluating whether retrieved trajectories would be useful for solving the same task as the query.

OUTPUT FORMAT:
TRAJECTORY: [X]
RELEVANCE_SCORE: [0-10]
REASONING: [Brief explanation]

SCORING (aligned with 6+ threshold for relevance):
- 8-10: HIGHLY USEFUL - Same goal with complete procedure
- 6-7: USEFUL - Same goal partial procedure OR significant procedural overlap with different goal
- 4-5: SOMEWHAT USEFUL - Minor procedural overlap or related sub-tasks
- 0-3: NOT USEFUL - No procedural overlap or opposite actions

KEY RULES:
- Ignore object names entirely (apple=mug=plate for procedural purposes)
- Ignore location specifics (cabinet=shelf=drawer for storage purposes)
- Reward procedural overlap even if goals differ
- Value useful sub-procedures that could be extracted and applied
- Punish only when trajectories provide no transferable procedural knowledge

QUERY: "{query}"
RETRIEVED TRAJECTORIES: {trajectory_batch}"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize LLM reasoner.

        Args:
            model_name: OpenAI model to use for reasoning
            api_key: OpenAI API key (if not in environment)
        """
        self.model_name = model_name
        self.api_key = api_key

        # Lazy initialization
        self._client = None
        self._corpus_loader = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package is required for LLM evaluation. "
                    "Install with: pip install openai>=1.12.0"
                )

            # Set up OpenAI client
            if self.api_key:
                openai.api_key = self.api_key
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self._client = openai.OpenAI()

        return self._client

    @property
    def corpus_loader(self):
        """Lazy initialization of corpus loader."""
        if self._corpus_loader is None:
            from ..agentinstruct.corpus_loader import AgentInstructCorpusLoader
            self._corpus_loader = AgentInstructCorpusLoader()
        return self._corpus_loader

    def batch_evaluate_with_prompt(
        self,
        prompt_template: str,
        query: str,
        retrieved_results: List[Dict],
        include_raw_response: bool = False
    ) -> Dict:
        """
        Evaluate all k retrieved trajectories with a specific prompt in single API call.

        Args:
            prompt_template: Prompt template (typically UNIVERSAL_PROMPT)
            query: Search query string
            retrieved_results: List of retrieved trajectory dictionaries
            include_raw_response: If True, include raw LLM response in results

        Returns:
            Dictionary mapping trajectory_id to {relevance_score, reasoning} data
        """
        if not retrieved_results:
            return {}

        # Format all k trajectories for single API call
        trajectory_batch = self._format_trajectories_batch(retrieved_results)
        formatted_prompt = prompt_template.format(query=query, trajectory_batch=trajectory_batch)

        # Single API call
        llm_response = self._query_llm(formatted_prompt)
        parsed_results = self._parse_batch_response(llm_response, retrieved_results)

        # Optionally include raw response for debugging
        if include_raw_response:
            for traj_id in parsed_results:
                parsed_results[traj_id]["raw_llm_response"] = llm_response

        return parsed_results

    def _format_trajectories_batch(self, results: List[Dict]) -> str:
        """
        Format all k retrieved trajectories into single prompt text block.

        Args:
            results: List of retrieval result dictionaries

        Returns:
            Formatted string with all trajectories for batch processing
        """
        formatted_batch = []

        for i, result in enumerate(results):
            trajectory_id = result.get('trajectory_id', result.get('task_instance_id'))

            # Try to get full trajectory from corpus
            formatted_traj = self._format_agentinstruct_trajectory(i + 1, trajectory_id)

            # Fallback to document text if trajectory not found
            if not formatted_traj:
                document_text = result.get('document_text', result.get('document', 'No trajectory data available'))
                formatted_traj = f"TRAJECTORY {i + 1}: {document_text}"

            formatted_batch.append(formatted_traj)

        return "\n\n".join(formatted_batch)

    def _format_agentinstruct_trajectory(self, traj_number: int, trajectory_id: str) -> str:
        """
        Format AgentInstruct trajectory with full state-action pairs.

        Args:
            traj_number: Trajectory number for display (1-indexed)
            trajectory_id: ID of the trajectory to format

        Returns:
            Formatted trajectory string, or empty string if not found
        """
        if not trajectory_id:
            return ""

        try:
            # Get trajectory from corpus
            traj = self.corpus_loader.get_trajectory_by_id(trajectory_id)
            if not traj:
                return ""

            # Format with state-action pairs
            formatted = f"TRAJECTORY {traj_number}:\n"
            formatted += f"Task: {traj.task_description}\n"
            formatted += "Steps:\n"

            for pair in traj.state_action_pairs:
                step_id = pair['step_id']
                state = pair['state']
                action = pair['action']
                formatted += f"{step_id}. State: {state} -> Action: {action}\n"

            return formatted.strip()

        except Exception:
            return ""

    def _query_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response."""
        try:
            # Configure API parameters
            api_params = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in procedural memory evaluation and task similarity analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
            }

            # Model-specific configuration
            if self.model_name in ["gpt-5", "o1-preview", "o1-mini"]:
                # Reasoning models only support default temperature
                api_params["reasoning_effort"] = "low"
            else:
                # Other models support temperature and max_tokens
                api_params["temperature"] = 0.1  # Low temperature for consistent reasoning
                api_params["max_tokens"] = 2000

            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ Error querying LLM: {e}")
            return f"Error: Could not get LLM response - {str(e)}"

    def _parse_batch_response(self, response: str, original_results: List[Dict]) -> Dict:
        """
        Parse structured LLM response containing evaluations for all k trajectories.

        Args:
            response: Raw LLM response string
            original_results: Original retrieval results for trajectory mapping

        Returns:
            Dictionary mapping trajectory_id to {relevance_score, reasoning} data
        """
        trajectory_scores = {}
        lines = response.split('\n')

        current_trajectory_idx = None
        current_score = None
        current_reasoning = None

        for line in lines:
            line = line.strip()

            # Handle TRAJECTORY line
            if line.upper().startswith("TRAJECTORY"):
                # Save previous trajectory if exists
                if current_trajectory_idx is not None and current_score is not None:
                    if current_trajectory_idx < len(original_results):
                        traj_data = original_results[current_trajectory_idx]
                        traj_id = (
                            traj_data.get("trajectory_id") or
                            traj_data.get("task_instance_id") or
                            f"trajectory_{current_trajectory_idx}"
                        )
                        trajectory_scores[traj_id] = {
                            "relevance_score": current_score,
                            "reasoning": current_reasoning or "No reasoning provided"
                        }

                # Extract trajectory number
                match = re.search(r'TRAJECTORY[:\s]+\[?(\d+)\]?', line, re.IGNORECASE)
                if match:
                    current_trajectory_idx = int(match.group(1)) - 1  # Convert to 0-indexed
                    current_score = None
                    current_reasoning = None

            # Handle RELEVANCE_SCORE or SCORE line
            elif "RELEVANCE_SCORE:" in line.upper() or line.upper().startswith("SCORE:"):
                try:
                    # Extract numeric score
                    score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if score_match:
                        current_score = float(score_match.group(1))
                except (ValueError, IndexError):
                    current_score = 0.0

            # Handle REASONING line
            elif line.upper().startswith("REASONING:"):
                # Extract reasoning text (everything after "REASONING:")
                reasoning_parts = line.split(":", 1)
                if len(reasoning_parts) > 1:
                    current_reasoning = reasoning_parts[1].strip()

        # Save last trajectory
        if current_trajectory_idx is not None and current_score is not None:
            if current_trajectory_idx < len(original_results):
                traj_data = original_results[current_trajectory_idx]
                traj_id = (
                    traj_data.get("trajectory_id") or
                    traj_data.get("task_instance_id") or
                    f"trajectory_{current_trajectory_idx}"
                )
                trajectory_scores[traj_id] = {
                    "relevance_score": current_score,
                    "reasoning": current_reasoning or "No reasoning provided"
                }

        return trajectory_scores
