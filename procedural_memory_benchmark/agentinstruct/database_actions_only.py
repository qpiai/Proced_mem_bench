"""
AgentInstruct Action-Only Database Manager

Manages ChromaDB vector database for PURE ACTION SEQUENCE embeddings only.
This is the most minimal procedural representation - raw action patterns without
any task descriptions or state information.

Research Question: Can pure action sequences alone support effective procedural retrieval?
"""

import os
from typing import List, Dict, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

from .corpus_loader import AgentInstructCorpusLoader, AgentInstructTrajectory
from .embedder import AgentInstructEmbedder


class AgentInstructActionOnlyDatabaseManager:
    """
    Manages ChromaDB database for PURE ACTION SEQUENCE embeddings.

    Unlike the standard database which embeds task descriptions + state-action pairs,
    this database ONLY embeds raw action sequences like:
    "go to table 1 | take item 2 | go to shelf 3 | put item in/on shelf 3"

    This tests whether procedural similarity can be detected from action patterns alone.
    """

    def __init__(self, db_path: str = None, collection_name: str = "agentinstruct_actions_only"):
        """
        Initialize action-only database manager.

        Args:
            db_path: Path to ChromaDB database directory.
                    If None, uses user cache directory with actions_only subdirectory.
            collection_name: Name of the ChromaDB collection
        """
        if db_path is None:
            # Use user cache directory via path utility with actions_only subdirectory
            from ..utils.paths import get_default_db_path
            db_path = str(get_default_db_path("databases") / "actions_only")

        self.db_path = db_path
        self.collection_name = collection_name

        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"📂 Connected to existing action-only collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=collection_name)
            print(f"📂 Created new action-only collection: {collection_name}")

        # Initialize components
        self.corpus_loader = None
        self.embedder = None

    def initialize_components(self, embedder_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize corpus loader and embedder.

        Args:
            embedder_model: HuggingFace model name for embeddings (same as standard for fair comparison)
        """
        if self.corpus_loader is None:
            self.corpus_loader = AgentInstructCorpusLoader()

        if self.embedder is None:
            self.embedder = AgentInstructEmbedder(model_name=embedder_model)

    def build_database(self, batch_size: int = 32, force_rebuild: bool = False):
        """
        Build the database using PURE ACTION SEQUENCES ONLY.

        This method generates embeddings from raw action patterns without any semantic context.

        Args:
            batch_size: Batch size for embedding generation
            force_rebuild: Whether to rebuild even if database exists
        """
        # Check if database already exists
        current_count = self.collection.count()
        if current_count > 0 and not force_rebuild:
            print(f"📊 Action-only database already contains {current_count} trajectories")
            print("   Use force_rebuild=True to rebuild from scratch")
            return

        if force_rebuild and current_count > 0:
            print(f"🗑️  Clearing existing action-only database with {current_count} trajectories...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)

        # Initialize components
        self.initialize_components()

        # Load trajectories
        print("🔄 Loading AgentInstruct trajectories for action-only embedding...")
        trajectories = self.corpus_loader.get_all_trajectories()
        print(f"✅ Loaded {len(trajectories)} trajectories")

        # Extract PURE action sequences for embedding
        print("🔄 Extracting pure action sequences (no task descriptions, no states)...")
        action_sequences = []
        for traj in trajectories:
            action_seq = traj.get_pure_action_sequence_text()
            action_sequences.append(action_seq)

        print(f"📊 Sample action sequence (first trajectory):")
        print(f"   {action_sequences[0][:200]}...")

        # Generate embeddings from pure action sequences
        print("🔄 Generating embeddings from pure action sequences...")
        embeddings = self.embedder.embed_texts_batch(
            action_sequences,
            batch_size=batch_size,
            show_progress=True
        )

        # Prepare data for ChromaDB
        print("🔄 Preparing data for action-only database...")
        documents = []
        metadatas = []
        ids = []

        for i, (traj, embedding, action_seq) in enumerate(zip(trajectories, embeddings, action_sequences)):
            # Store pure action sequence as document
            documents.append(action_seq)

            # Create metadata (still track task info for analysis, but NOT embedded)
            metadata = {
                "task_instance_id": traj.task_instance_id,
                "task_description": traj.task_description,  # For reference, not embedded
                "total_steps": traj.total_steps,
                "source": traj.source,
                "embedding_model": self.embedder.model_name,
                "embedding_format": "pure_action_sequences_only"
            }
            metadatas.append(metadata)

            # Create unique ID
            ids.append(f"agentinstruct_actions_{traj.task_instance_id}")

        # Add to database
        print(f"💾 Adding {len(trajectories)} pure action embeddings to database...")
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Action-only database built successfully!")
        print(f"   Total trajectories: {len(trajectories)}")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Embedding format: Pure action sequences only (no task descriptions, no states)")
        print(f"   Database location: {self.db_path}")

    def search(self, query_text: str, k: int = 10) -> Dict:
        """
        Search for similar trajectories using query text.

        Note: Query will be embedded as-is. For pure action matching, query should
        ideally be formatted as action sequence, but this supports any text query.

        Args:
            query_text: Query text to search for
            k: Number of results to return

        Returns:
            Dictionary with search results
        """
        if self.embedder is None:
            self.initialize_components()

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query_text)

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    "trajectory_id": results['ids'][0][i],
                    "task_instance_id": results['metadatas'][0][i]['task_instance_id'],
                    "task_description": results['metadatas'][0][i]['task_description'],
                    "total_steps": results['metadatas'][0][i]['total_steps'],
                    "distance": results['distances'][0][i],
                    "similarity": max(0, 1 - abs(results['distances'][0][i])),
                    "document": results['documents'][0][i][:200] + "..." if len(results['documents'][0][i]) > 200 else results['documents'][0][i]
                }
                formatted_results.append(result)

        return {
            "query": query_text,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

    def get_database_stats(self) -> Dict:
        """Get statistics about the action-only database."""
        count = self.collection.count()

        # Get sample of metadata to analyze
        sample_size = min(100, count) if count > 0 else 0
        sample = self.collection.peek(sample_size) if sample_size > 0 else {"metadatas": []}

        # Analyze task patterns
        task_patterns = {}
        step_lengths = []

        for metadata in sample["metadatas"]:
            task_desc = metadata.get("task_description", "").lower()
            steps = metadata.get("total_steps", 0)
            step_lengths.append(steps)

            # Categorize tasks
            if "heat" in task_desc:
                task_patterns.setdefault("heating", 0)
                task_patterns["heating"] += 1
            elif "cool" in task_desc:
                task_patterns.setdefault("cooling", 0)
                task_patterns["cooling"] += 1
            elif "clean" in task_desc:
                task_patterns.setdefault("cleaning", 0)
                task_patterns["cleaning"] += 1
            elif "examine" in task_desc or "look at" in task_desc:
                task_patterns.setdefault("examination", 0)
                task_patterns["examination"] += 1
            elif "two" in task_desc:
                task_patterns.setdefault("multi_object", 0)
                task_patterns["multi_object"] += 1
            else:
                task_patterns.setdefault("placement", 0)
                task_patterns["placement"] += 1

        return {
            "total_trajectories": count,
            "database_path": self.db_path,
            "collection_name": self.collection_name,
            "average_steps": sum(step_lengths) / len(step_lengths) if step_lengths else 0,
            "task_patterns": task_patterns,
            "embedding_model": self.embedder.model_name if self.embedder else "Not initialized",
            "embedding_format": "pure_action_sequences_only"
        }
