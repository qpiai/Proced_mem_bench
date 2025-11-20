"""
AgentInstruct Corpus Loader

Loads the 336 AgentInstruct ALFWorld trajectories for procedural memory evaluation.
Provides structured access to task descriptions and state-action pairs.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AgentInstructTrajectory:
    """Container for AgentInstruct trajectory data."""
    task_instance_id: str
    task_description: str
    state_action_pairs: List[Dict]
    total_steps: int
    source: str = "agentinstruct"

    def get_embedding_text(self) -> str:
        """
        Create embedding text combining task description and state-action pairs.

        Returns:
            Formatted text for embedding generation
        """
        text = f"Task: {self.task_description}\n"
        text += "Steps:\n"

        for pair in self.state_action_pairs:
            step_id = pair['step_id']
            state = pair['state']
            action = pair['action']
            text += f"{step_id}. State: {state} -> Action: {action}\n"

        return text.strip()

    def get_action_sequence(self) -> List[str]:
        """Get just the action sequence for comparison purposes."""
        return [pair['action'] for pair in self.state_action_pairs]

    def get_pure_action_sequence_text(self) -> str:
        """
        Get ONLY the action sequence as text for embedding (no task description, no states).

        This creates the most minimal procedural representation - just the raw action pattern.
        Mirrors ALFWorld's "pure trajectories" approach focusing on action-level similarity.

        Returns:
            Pure action sequence joined with " | " separator

        Example:
            "go to diningtable 1 | take laptop 1 from diningtable 1 | go to bed 1 | put laptop 1 in/on bed 1"
        """
        actions = [pair['action'] for pair in self.state_action_pairs]
        return " | ".join(actions)


class AgentInstructCorpusLoader:
    """
    Loader for AgentInstruct ALFWorld procedural memory corpus.

    Loads all 336 trajectories as the retrieval corpus for procedural memory evaluation.
    """

    def __init__(self, corpus_path: str = None):
        """
        Initialize AgentInstruct corpus loader.

        Args:
            corpus_path: Path to AgentInstruct trajectories JSON file.
                        If None, uses package data location.
        """
        if corpus_path is None:
            # Use package data via path utility
            from ..utils.paths import get_corpus_path
            corpus_path = str(get_corpus_path())

        self.corpus_path = corpus_path
        self.corpus_data = None
        self.trajectories = None

    def load_corpus(self) -> Dict:
        """Load the full AgentInstruct corpus from JSON file."""
        if self.corpus_data is None:
            print(f"🔄 Loading AgentInstruct corpus from: {self.corpus_path}")

            if not os.path.exists(self.corpus_path):
                raise FileNotFoundError(f"AgentInstruct corpus not found: {self.corpus_path}")

            with open(self.corpus_path, 'r') as f:
                self.corpus_data = json.load(f)

            print(f"✅ Loaded {len(self.corpus_data['trajectories'])} AgentInstruct trajectories")
            print(f"📊 Average steps per trajectory: {self.corpus_data['metadata']['average_steps']:.1f}")

        return self.corpus_data

    def get_all_trajectories(self) -> List[AgentInstructTrajectory]:
        """
        Get all trajectories as structured data objects.

        Returns:
            List of AgentInstructTrajectory objects
        """
        if self.trajectories is None:
            corpus = self.load_corpus()
            self.trajectories = []

            for traj_raw in corpus['trajectories']:
                trajectory = AgentInstructTrajectory(
                    task_instance_id=traj_raw['task_instance_id'],
                    task_description=traj_raw['task_description'],
                    state_action_pairs=traj_raw['state_action_pairs'],
                    total_steps=traj_raw['total_steps'],
                    source=traj_raw.get('source', 'agentinstruct')
                )
                self.trajectories.append(trajectory)

            print(f"📋 Processed {len(self.trajectories)} trajectory objects")

        return self.trajectories

    def get_trajectory_by_id(self, task_instance_id: str) -> Optional[AgentInstructTrajectory]:
        """
        Get a specific trajectory by its instance ID.

        Args:
            task_instance_id: The trajectory ID (e.g., "alfworld_0")

        Returns:
            AgentInstructTrajectory object or None if not found
        """
        trajectories = self.get_all_trajectories()

        for traj in trajectories:
            if traj.task_instance_id == task_instance_id:
                return traj

        return None

    def get_trajectories_by_task_type(self, task_keywords: List[str]) -> List[AgentInstructTrajectory]:
        """
        Get trajectories that contain specific keywords in their task description.

        Args:
            task_keywords: Keywords to search for in task descriptions

        Returns:
            List of matching trajectories
        """
        trajectories = self.get_all_trajectories()
        matches = []

        for traj in trajectories:
            task_desc_lower = traj.task_description.lower()
            if any(keyword.lower() in task_desc_lower for keyword in task_keywords):
                matches.append(traj)

        return matches

    def get_corpus_statistics(self) -> Dict:
        """
        Get statistics about the AgentInstruct corpus.

        Returns:
            Dictionary with corpus statistics
        """
        trajectories = self.get_all_trajectories()

        # Task description analysis
        task_descriptions = [traj.task_description for traj in trajectories]
        unique_tasks = set(task_descriptions)

        # Step length analysis
        step_lengths = [traj.total_steps for traj in trajectories]

        # Action analysis
        all_actions = []
        for traj in trajectories:
            all_actions.extend(traj.get_action_sequence())
        unique_actions = set(all_actions)

        # Task type patterns (based on task descriptions)
        task_patterns = {}
        for desc in task_descriptions:
            # Simple pattern detection
            if "heat" in desc:
                task_patterns.setdefault("heating", 0)
                task_patterns["heating"] += 1
            elif "cool" in desc:
                task_patterns.setdefault("cooling", 0)
                task_patterns["cooling"] += 1
            elif "clean" in desc:
                task_patterns.setdefault("cleaning", 0)
                task_patterns["cleaning"] += 1
            elif "examine" in desc or "look at" in desc:
                task_patterns.setdefault("examination", 0)
                task_patterns["examination"] += 1
            elif "two" in desc or "2" in desc:
                task_patterns.setdefault("multi_object", 0)
                task_patterns["multi_object"] += 1
            else:
                task_patterns.setdefault("placement", 0)
                task_patterns["placement"] += 1

        return {
            "total_trajectories": len(trajectories),
            "unique_task_descriptions": len(unique_tasks),
            "average_steps": sum(step_lengths) / len(step_lengths),
            "min_steps": min(step_lengths),
            "max_steps": max(step_lengths),
            "total_state_action_pairs": sum(step_lengths),
            "unique_actions": len(unique_actions),
            "task_patterns": task_patterns,
            "sample_tasks": list(unique_tasks)[:10]
        }
