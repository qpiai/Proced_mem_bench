"""
AgentInstruct Embedder

Creates embeddings for AgentInstruct trajectories using combined task description
and state-action pairs. Uses sentence transformers for embedding generation.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .corpus_loader import AgentInstructTrajectory, AgentInstructCorpusLoader


class AgentInstructEmbedder:
    """
    Creates embeddings for AgentInstruct trajectories.

    Combines task descriptions with state-action pairs to create rich
    procedural representations for similarity search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name for sentence transformer
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"🔄 Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed_trajectory(self, trajectory: AgentInstructTrajectory) -> np.ndarray:
        """
        Create embedding for a single trajectory.

        Args:
            trajectory: AgentInstructTrajectory object

        Returns:
            Numpy array with embedding vector
        """
        embedding_text = trajectory.get_embedding_text()
        embedding = self.model.encode([embedding_text], convert_to_tensor=False)[0]
        return embedding.astype(np.float32)

    def embed_trajectories_batch(self, trajectories: List[AgentInstructTrajectory],
                                 batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        Create embeddings for multiple trajectories in batches.

        Args:
            trajectories: List of trajectory objects
            batch_size: Number of trajectories to process at once
            show_progress: Whether to show progress information

        Returns:
            List of embedding vectors
        """
        if show_progress:
            print(f"🔄 Generating embeddings for {len(trajectories)} trajectories...")
            print(f"   Batch size: {batch_size}, Device: {self.device}")

        # Extract embedding texts
        embedding_texts = [traj.get_embedding_text() for traj in trajectories]

        # Generate embeddings in batches
        embeddings = []
        total_batches = (len(embedding_texts) + batch_size - 1) // batch_size

        for i in range(0, len(embedding_texts), batch_size):
            if show_progress and i // batch_size % max(1, total_batches // 10) == 0:
                batch_num = i // batch_size + 1
                print(f"   Processing batch {batch_num}/{total_batches}...")

            batch_texts = embedding_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=False
            )

            embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])

        if show_progress:
            print(f"✅ Generated {len(embeddings)} embeddings")

        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Create embedding for a query text.

        Args:
            query_text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode([query_text], convert_to_tensor=False)[0]
        return embedding.astype(np.float32)

    def embed_query_batch(self, query_texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Create embeddings for multiple query texts.

        Args:
            query_texts: List of query texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            query_texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return [emb.astype(np.float32) for emb in embeddings]

    def embed_texts_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        Create embeddings for arbitrary text strings in batches.

        This is a generic method that can embed any text, including pure action sequences.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress information

        Returns:
            List of embedding vectors
        """
        if show_progress:
            print(f"🔄 Generating embeddings for {len(texts)} text strings...")
            print(f"   Batch size: {batch_size}, Device: {self.device}")

        # Generate embeddings in batches
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            if show_progress and i // batch_size % max(1, total_batches // 10) == 0:
                batch_num = i // batch_size + 1
                print(f"   Processing batch {batch_num}/{total_batches}...")

            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=False
            )

            embeddings.extend([emb.astype(np.float32) for emb in batch_embeddings])

        if show_progress:
            print(f"✅ Generated {len(embeddings)} embeddings")

        return embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2

        # Compute cosine similarity
        similarity = np.dot(normalized1, normalized2)
        return float(similarity)

    def get_model_info(self) -> Dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": self.model.max_seq_length,
            "model_type": "sentence_transformer"
        }
