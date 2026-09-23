"""System 1 fast decision engine interface and lazy loading implementation."""

from __future__ import annotations

import logging
import os
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
        model_name: str = "data/models/laya-techjob",
        auto_release_after_batch: bool = False,
    ) -> None:
        self.model_name = model_name
        self.auto_release_after_batch = auto_release_after_batch
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self.temperature: float = 0.75
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str = "data/models/laya-techjob") -> LazyLayaEngine:
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
                    import json
                    from pathlib import Path

                    model_path = Path(self.model_name)
                    # Support legacy alias
                    if not model_path.exists() and self.model_name == "laya-multilingual":
                        candidate = Path("data/models/laya-techjob")
                        if candidate.exists():
                            model_path = candidate

                    if not model_path.exists():
                        logger.debug("Model path %s not found. Running in fallback mode.", self.model_name)
                        return None

                    try:
                        import torch
                        from transformers import AutoModelForSequenceClassification, AutoTokenizer

                        logger.info("Loading fine-tuned Laya System 1 model from %s...", model_path)
                        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                        self._model.eval()

                        # Configure CPU optimization threads
                        num_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
                        try:
                            torch.set_num_threads(num_threads)
                        except Exception:
                            pass

                        # Optional INT8 Dynamic Quantization for 2x CPU speedup
                        if os.getenv("ENABLE_INT8_QUANTIZATION", "false").strip().lower() in ("true", "1", "yes"):
                            try:
                                self._model = torch.quantization.quantize_dynamic(
                                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                                )
                                logger.info("Applied dynamic INT8 quantization to Laya System 1 model.")
                            except Exception as q_err:
                                logger.warning("Could not apply dynamic quantization: %s", q_err)

                        # Read calibrated temperature from config if present
                        cfg_file = model_path / "rl_agent_config.json"
                        if cfg_file.exists():
                            try:
                                with open(cfg_file, "r") as f:
                                    cfg = json.load(f)
                                    self.temperature = float(cfg.get("calibrated_temperature", 0.75))
                                logger.info("Using calibrated temperature T = %.2f", self.temperature)
                            except Exception as err:
                                logger.warning("Could not parse rl_agent_config.json: %s", err)
                    except ImportError:
                        logger.debug("PyTorch or Transformers not installed. Running in fallback mode.")
                        self._model = None
                        self._tokenizer = None
                    except Exception as err:
                        logger.warning("Failed loading Laya model (%s): %s", self.model_name, err)
                        self._model = None
                        self._tokenizer = None
        return self._model

    def warmup(self) -> bool:
        """Pre-warm model and tokenizer in memory to guarantee sub-50ms P99 queries."""
        model = self.load_model()
        if model is None or self._tokenizer is None:
            return False
        try:
            import torch

            with torch.inference_mode():
                dummy_enc = self._tokenizer("Warmup check", return_tensors="pt")
                model(**dummy_enc)
            logger.info("Laya System 1 model successfully pre-warmed in RAM.")
            return True
        except Exception as err:
            logger.warning("Warmup failed on Laya model: %s", err)
            return False

    def unload_model(self) -> None:
        """Purge model weights from RAM to restore baseline memory headroom."""
        with self._model_lock:
            if self._model is not None:
                logger.info("Unloading Laya model from RAM...")
                self._model = None
                self._tokenizer = None
                import gc

                gc.collect()

    def predict_score(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[int, float]:
        """Predict ordinal score (0 to len(options)-1) and confidence probability."""
        model = self.load_model()
        if model is None or self._tokenizer is None:
            return min(2, len(options) - 1) if options else 0, 0.50

        try:
            import torch

            opts = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(options))
            prompt = f"Context:\n{state}\n\nInstruction: {question}\nOptions:\n{opts}"

            enc = self._tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**enc).logits[0, : len(options)]
                scaled = logits / max(1e-4, self.temperature)
                probs = torch.softmax(scaled, dim=-1)
                choice_idx = int(torch.argmax(probs).item())
                conf = float(probs[choice_idx].item())
            return choice_idx, conf
        except Exception as err:
            logger.warning("Inference error in predict_score: %s. Using neutral fallback.", err)
            return min(2, len(options) - 1) if options else 0, 0.50

    def predict_noul(self, state: str, question: str) -> float:
        """Predict binary probability P(true) for verification / recruiter fit."""
        model = self.load_model()
        if model is None or self._tokenizer is None:
            return 0.50

        try:
            import torch

            options = ["Reject / No", "Advance / Yes"]
            opts = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(options))
            prompt = f"Context:\n{state}\n\nInstruction: {question}\nOptions:\n{opts}"

            enc = self._tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**enc).logits[0, :2]
                scaled = logits / max(1e-4, self.temperature)
                probs = torch.softmax(scaled, dim=-1)
                p_true = float(probs[1].item())
            return p_true
        except Exception as err:
            logger.warning("Inference error in predict_noul: %s. Using neutral fallback.", err)
            return 0.50

    def predict_choice(
        self, state: str, question: str, options: List[str]
    ) -> Tuple[str, float]:
        """Predict categorical choice from options list."""
        if not options:
            return "", 0.50
        model = self.load_model()
        if model is None or self._tokenizer is None:
            return options[0], 0.50
        choice_idx, conf = self.predict_score(state, question, options)
        chosen = options[choice_idx] if 0 <= choice_idx < len(options) else options[0]
        return chosen, conf

    def predict_match_scoring_ensemble(
        self, job_title: str, job_desc: str, cv_text: str
    ) -> Dict[str, Any]:
        """Execute the 3-question ensemble for a single candidate/job pair."""
        results = self.predict_match_scoring_batch([{"job_title": job_title, "job_desc": job_desc, "cv_text": cv_text}])
        return results[0] if results else {
            "skill_match": 2,
            "skill_confidence": 0.50,
            "seniority_fit": 2,
            "seniority_confidence": 0.50,
            "recruiter_fit_probability": 0.50,
        }

    def predict_match_scoring_batch(
        self, items: List[Dict[str, str]], chunk_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Vectorized batched inference evaluating all 3 ensemble questions in minimal forward passes.

        For N jobs, evaluates:
          - Question 1: Technical skill overlap (5 options)
          - Question 2: Seniority fit (5 options)
          - Question 3: Recruiter interview probability (2 options)
        All 3N prompts are dynamically padded and executed in batched chunks to achieve consistent low latency on CPU.
        """
        if not items:
            return []

        model = self.load_model()
        if model is None or self._tokenizer is None:
            return [
                {
                    "skill_match": 2,
                    "skill_confidence": 0.50,
                    "seniority_fit": 2,
                    "seniority_confidence": 0.50,
                    "recruiter_fit_probability": 0.50,
                }
                for _ in items
            ]

        skill_options = ["None (0-20%)", "Weak (20-40%)", "Partial (40-60%)", "Strong (60-80%)", "Perfect (80-100%)"]
        sen_options = ["Far too junior", "Slightly junior", "Good fit", "Senior", "Overqualified"]
        rec_options = ["No", "Yes"]

        skill_opts_str = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(skill_options))
        sen_opts_str = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(sen_options))
        rec_opts_str = " | ".join(f"[{i}] {opt}" for i, opt in enumerate(rec_options))

        all_results: List[Dict[str, Any]] = []

        try:
            import torch

            # Process in chunks of 4 jobs (12 sequences) to optimize CPU cache & AVX throughput
            for chunk_start in range(0, len(items), chunk_size):
                chunk = items[chunk_start : chunk_start + chunk_size]
                prompts: List[str] = []

                for item in chunk:
                    job_title = item.get("job_title", "Unknown Role")
                    state = f"Job Title: {job_title}\nJob Description:\n{item['job_desc']}\nCandidate CV:\n{item['cv_text']}"
                    p_skill = f"Context:\n{state}\n\nInstruction: Evaluate the technical and professional skill match of the candidate for this role.\nOptions:\n{skill_opts_str}"
                    p_sen = f"Context:\n{state}\n\nInstruction: Assess the seniority alignment of the candidate relative to the requirements.\nOptions:\n{sen_opts_str}"
                    p_rec = f"Context:\n{state}\n\nInstruction: Would a human technical recruiter recommend advancing this candidate to an interview?\nOptions:\n{rec_opts_str}"
                    prompts.extend([p_skill, p_sen, p_rec])

                # Batched dynamic padding
                enc = self._tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    logits = model(**enc).logits
                    scaled = logits / max(1e-4, self.temperature)
                    probs = torch.softmax(scaled, dim=-1)

                    # Extract Question 1: Skill overlap (stride 3, offset 0)
                    skill_probs = probs[0::3, :5]
                    skill_idx = torch.argmax(skill_probs, dim=-1).tolist()
                    skill_confs = skill_probs.gather(1, torch.tensor(skill_idx, device=probs.device).unsqueeze(1)).squeeze(1).tolist()

                    # Extract Question 2: Seniority fit (stride 3, offset 1)
                    sen_probs = probs[1::3, :5]
                    sen_idx = torch.argmax(sen_probs, dim=-1).tolist()
                    sen_confs = sen_probs.gather(1, torch.tensor(sen_idx, device=probs.device).unsqueeze(1)).squeeze(1).tolist()

                    # Extract Question 3: Recruiter fit probability (stride 3, offset 2)
                    rec_probs = probs[2::3, :2]
                    rec_p_true = rec_probs[:, 1].tolist()

                for i in range(len(chunk)):
                    all_results.append({
                        "skill_match": skill_idx[i],
                        "skill_confidence": float(skill_confs[i]),
                        "seniority_fit": sen_idx[i],
                        "seniority_confidence": float(sen_confs[i]),
                        "recruiter_fit_probability": float(rec_p_true[i]),
                    })

        except Exception as exc:
            logger.warning("Batched inference error in predict_match_scoring_batch: %s. Using neutral fallback.", exc, exc_info=True)
            while len(all_results) < len(items):
                all_results.append({
                    "skill_match": 2,
                    "skill_confidence": 0.50,
                    "seniority_fit": 2,
                    "seniority_confidence": 0.50,
                    "recruiter_fit_probability": 0.50,
                })

        if self.auto_release_after_batch:
            self.unload_model()

        return all_results
