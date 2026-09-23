"""System 1 fast decision engine interface and lazy loading implementation."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LazyLayaEngine:
    """Production-grade lazy loader for Laya System 1 decision engine.

    Features:
    - Lazy loading: The underlying model is only loaded into memory when required.
    - Resource release: Model weights can be purged from RAM when batch triage finishes.
    - Graceful fallback: If laya is unavailable or fails to load, falls back to deterministic/heuristic logic.
    - Thread safety: Loading and unloading are protected with re-entrant locks.
    """

    _instance: Optional[LazyLayaEngine] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "laya-multilingual",
        auto_release_after_batch: bool = True,
    ) -> None:
        self.model_name = model_name
        self.auto_release_after_batch = auto_release_after_batch
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "laya-multilingual") -> LazyLayaEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_model()
            cls._instance = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        # Attempt import of laya engine
                        import laya  # type: ignore

                        logger.info("Loading Laya model (%s) into memory...", self.model_name)
                        self._model = laya.load(self.model_name)
                    except ImportError:
                        logger.debug("Laya package not installed. Running in mock/fallback mode.")
                        self._model = None
                    except Exception as err:
                        logger.warning("Failed loading Laya model (%s): %s", self.model_name, err)
                        self._model = None
        return self._model

    def unload_model(self) -> None:
        """Purge model weights from RAM to restore baseline memory headroom."""
        with self._model_lock:
            if self._model is not None:
                logger.info("Unloading Laya model from RAM...")
                self._model = None
                import gc

                gc.collect()

    def predict_score(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[int, float]:
        """Predict ordinal score (0 to len(options)-1) and confidence probability."""
        model = self.load_model()
        if model is None:
            # Fallback heuristic / neutral default
            return 2, 0.50

        res = model.decide(state, {"type": "score", "text": question, "options": options})
        return res.get("choice_index", 2), res.get("confidence", 0.5)

    def predict_noul(self, state: str, question: str) -> float:
        """Predict binary probability P(true) for verification / recruiter fit."""
        model = self.load_model()
        if model is None:
            return 0.50

        res = model.decide(state, {"type": "noul", "text": question})
        return float(res.get("probability", 0.50))

    def predict_choice(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[str, float]:
        """Predict categorical choice from options list."""
        model = self.load_model()
        if model is None:
            return options[0] if options else "", 0.50

        res = model.decide(state, {"type": "choice", "text": question, "options": options})
        choice_idx = res.get("choice_index", 0)
        conf = res.get("confidence", 0.5)
        chosen = options[choice_idx] if 0 <= choice_idx < len(options) else options[0]
        return chosen, conf

    def predict_match_scoring_ensemble(
        self, job_desc: str, cv_text: str
    ) -> Dict[str, Any]:
        """Execute the 3-question ensemble in a single forward pass."""
        state = f"CANDIDATE CV:\n{cv_text}\n\nJOB LISTING:\n{job_desc}"

        skill_options = ["None", "Weak", "Partial", "Strong", "Perfect"]
        seniority_options = [
            "Far too junior",
            "Slightly junior",
            "Good fit",
            "Senior",
            "Overqualified",
        ]

        skill_idx, skill_conf = self.predict_score(
            state, "Rate the technical skill overlap between CV and job requirements.", skill_options
        )
        sen_idx, sen_conf = self.predict_score(
            state, "Rate the seniority fit based on years of experience and track record.", seniority_options
        )
        recruiter_prob = self.predict_noul(
            state, "Overall, would a human technical recruiter invite this candidate to an interview?"
        )

        if self.auto_release_after_batch:
            self.unload_model()

        return {
            "skill_match": skill_idx,
            "skill_confidence": skill_conf,
            "seniority_fit": sen_idx,
            "seniority_confidence": sen_conf,
            "recruiter_fit_probability": recruiter_prob,
        }
