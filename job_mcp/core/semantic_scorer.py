"""Zero-Cost CPU Semantic Scorer using fastembed ONNX runtime."""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class SemanticScorer:
    """Zero-cost CPU semantic similarity scorer using fastembed ONNX embeddings.

    Provides lightweight, CPU-optimized semantic similarity scoring with graceful
    fallback to zero scores when fastembed is unavailable or when runtime errors occur.
    """

    _instance: ClassVar[Optional[SemanticScorer]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _available: ClassVar[Optional[bool]] = None

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._warned_error = False

    @classmethod
    def is_available(cls) -> bool:
        """Dynamically check if fastembed and numpy are available in the environment (cached)."""
        if cls._available is None:
            try:
                import fastembed  # noqa: F401
                import numpy  # noqa: F401

                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL) -> SemanticScorer:
        """Singleton factory returning the shared SemanticScorer instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None
            cls._available = None

    def _get_model(self) -> Any:
        """Lazy load the fastembed TextEmbedding model under a lock."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def score_similarity(self, query: str, documents: list[str]) -> list[float]:
        """Compute cosine similarity scores between query and documents.

        Args:
            query: The search or target text query.
            documents: A list of candidate document texts to score.

        Returns:
            A list of float similarity scores in [0.0, 1.0], matching documents order.
        """
        if not documents:
            return []

        if not query or not query.strip() or not self.is_available():
            return [0.0] * len(documents)

        # Identify documents that have non-empty text
        valid_indices = [i for i, doc in enumerate(documents) if doc and doc.strip()]
        if not valid_indices:
            return [0.0] * len(documents)

        try:
            import numpy as np

            model = self._get_model()
            valid_docs = [documents[i] for i in valid_indices]
            all_texts = [query] + valid_docs
            embeddings = list(model.embed(all_texts))

            # Normalize vectors using numpy L2 norm
            q_vec = np.array(embeddings[0], dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm > 0:
                q_vec = q_vec / q_norm

            doc_vecs = np.array(embeddings[1:], dtype=np.float32)
            doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            doc_norms[doc_norms == 0] = 1.0
            norm_doc_vecs = doc_vecs / doc_norms

            # Dot product between normalized query vector and each document vector
            similarities = np.dot(norm_doc_vecs, q_vec)
            clamped = np.clip(similarities, 0.0, 1.0)

            result = [0.0] * len(documents)
            for idx, score in zip(valid_indices, clamped):
                result[idx] = float(score)

            return result
        except Exception as e:
            if not self._warned_error:
                logger.warning("Semantic scoring failed, falling back to 0.0: %s", e)
                self._warned_error = True
            return [0.0] * len(documents)

    def score_single(self, query: str, document: str) -> float:
        """Compute cosine similarity score between a query and a single document."""
        scores = self.score_similarity(query, [document])
        return scores[0] if scores else 0.0
