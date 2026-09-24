import sys
from typing import Any, Dict, List, Tuple

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class System1Engine(Protocol):
    """Protocol defining the interface for the System 1 decision engine.
    
    This abstraction allows for dependency injection and A/B testing between
    the fine-tuned local LAYA model and a zero-shot generative LLM baseline.
    """

    def is_loaded(self) -> bool:
        """Check if the engine is ready/loaded."""
        ...

    def predict_score(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[int, float]:
        """Predict ordinal score (0 to len(options)-1) and confidence probability."""
        ...

    def predict_noul(self, state: str, question: str) -> float:
        """Predict binary probability P(true) for verification / recruiter fit."""
        ...

    def predict_choice(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[str, float]:
        """Predict categorical choice from options list."""
        ...

    def predict_match_scoring_ensemble(
        self, job_desc: str = "", cv_text: str = "", job_title: str = "Unknown Role"
    ) -> Dict[str, Any]:
        """Execute the 3-question ensemble for a single candidate/job pair."""
        ...

    def predict_match_scoring_batch(
        self, items: List[Dict[str, str]], chunk_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Vectorized batched inference evaluating all 3 ensemble questions in minimal forward passes."""
        ...
