"""
AgentInstruct Database Manager

Manages ChromaDB vector database for AgentInstruct trajectory embeddings.
Provides storage, retrieval, and similarity search capabilities.
"""

import os
import json
from typing import List, Dict, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

from .corpus_loader import AgentInstructCorpusLoader, AgentInstructTrajectory
from .embedder import AgentInstructEmbedder


class AgentInstructDatabaseManager:
    """
    Manages ChromaDB database for AgentInstruct procedural memory evaluation.

    Handles embedding storage, retrieval queries, and similarity search
    for the AgentInstruct corpus of 336 trajectories.
    """

    def __init__(self, db_path: str = None, collection_name: str = "agentinstruct_procedural"):
        """
        Initialize database manager.

        Args:
            db_path: Path to ChromaDB database directory.
                    If None, uses user cache directory.
            collection_name: Name of the ChromaDB collection
        """
        if db_path is None:
            # Use user cache directory via path utility
            from ..utils.paths import get_default_db_path
            db_path = str(get_default_db_path("databases"))

        self.db_path = db_path

        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)

        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"📂 Connected to existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=collection_name)
            print(f"📂 Created new collection: {collection_name}")

        # Initialize components
        self.corpus_loader = None
        self.embedder = None

    def initialize_components(self, embedder_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize corpus loader and embedder.

        Args:
            embedder_model: HuggingFace model name for embeddings
        """
        if self.corpus_loader is None:
            self.corpus_loader = AgentInstructCorpusLoader()

        if self.embedder is None:
            self.embedder = AgentInstructEmbedder(model_name=embedder_model)

    def build_database(self, batch_size: int = 32, force_rebuild: bool = False):
        """
        Build the database by loading trajectories and generating embeddings.

        Args:
            batch_size: Batch size for embedding generation
            force_rebuild: Whether to rebuild even if database exists
        """
        # Check if database already exists
        current_count = self.collection.count()
        if current_count > 0 and not force_rebuild:
            print(f"📊 Database already contains {current_count} trajectories")
            print("   Use force_rebuild=True to rebuild from scratch")
            return

        if force_rebuild and current_count > 0:
            print(f"🗑️  Clearing existing database with {current_count} trajectories...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)

        # Initialize components
        self.initialize_components()

        # Load trajectories
        print("🔄 Loading AgentInstruct trajectories...")
        trajectories = self.corpus_loader.get_all_trajectories()
        print(f"✅ Loaded {len(trajectories)} trajectories")

        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.embedder.embed_trajectories_batch(
            trajectories,
            batch_size=batch_size,
            show_progress=True
        )

        # Prepare data for ChromaDB
        print("🔄 Preparing data for database...")
        documents = []
        metadatas = []
        ids = []

        for i, (traj, embedding) in enumerate(zip(trajectories, embeddings)):
            # Create document text (for ChromaDB's built-in search if needed)
            documents.append(traj.get_embedding_text())

            # Create metadata
            metadata = {
                "task_instance_id": traj.task_instance_id,
                "task_description": traj.task_description,
                "total_steps": traj.total_steps,
                "source": traj.source,
                "embedding_model": self.embedder.model_name
            }
            metadatas.append(metadata)

            # Create unique ID
            ids.append(f"agentinstruct_{traj.task_instance_id}")

        # Add to database
        print(f"💾 Adding {len(trajectories)} trajectories to database...")
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Database built successfully!")
        print(f"   Total trajectories: {len(trajectories)}")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Database location: {self.db_path}")

    def search(self, query_text: str, k: int = 10) -> Dict:
        """
        Search for similar trajectories using query text.

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
        """Get statistics about the database."""
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
            "embedding_model": self.embedder.model_name if self.embedder else "Not initialized"
        }
