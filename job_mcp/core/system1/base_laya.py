"""System 1 Base LAYA Engine implementation using un-fine-tuned official laya package."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseLayaEngine:
    """System 1 decision engine using the base pre-trained Laya multilingual model.

    Utilizes the native `laya` python package and `convaiinnovations/laya-multilingual`
    checkpoint without domain fine-tuning for empirical zero-shot evaluation and A/B benchmarking.
    """

    _instance: Optional[BaseLayaEngine] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_id: str = "convaiinnovations/laya-multilingual") -> None:
        self.model_id = model_id
        self._agent: Optional[Any] = None
        self._agent_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_id: str = "convaiinnovations/laya-multilingual") -> BaseLayaEngine:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_id=model_id)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_model()
            cls._instance = None

    def is_loaded(self) -> bool:
        return self._agent is not None

    def load_model(self) -> Any:
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    try:
                        import laya
                        logger.info("Loading base pre-trained Laya agent from %s...", self.model_id)
                        self._agent = laya.load(self.model_id)
                    except Exception as err:
                        logger.warning("Failed to load base laya agent (%s). Running in fallback mode.", err)
                        return None
        return self._agent

    def unload_model(self) -> None:
        with self._agent_lock:
            self._agent = None

    def predict_score(self, state: str, question: str, options: List[str]) -> Tuple[int, float]:
        agent = self.load_model()
        if agent is None:
            return 1, 0.5

        try:
            questions = {
                "score_q": {
                    "type": "score",
                    "instructions": question,
                    "criteria": options,
                }
            }
            res = agent.predict(state, questions)
            ans = res.get("answers", {}).get("score_q", {})
            probs = ans.get("probabilities", {})
            if probs:
                best_idx = int(max(probs.items(), key=lambda kv: float(kv[1]))[0])
                conf = float(probs[str(best_idx)])
                return best_idx, conf
            return int(round(ans.get("score", 1.0))), float(ans.get("confidence", 0.5))
        except Exception as err:
            logger.warning("Base Laya score prediction failed: %s", err)
            return 1, 0.5

    def predict_choice(self, state: str, question: str, options: List[str]) -> Tuple[str, float]:
        agent = self.load_model()
        if agent is None:
            return options[0] if options else "", 0.5

        try:
            criteria = {f"opt_{i}": opt for i, opt in enumerate(options)}
            questions = {
                "choice_q": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": criteria,
                }
            }
            res = agent.predict(state, questions)
            ans = res.get("answers", {}).get("choice_q", {})
            choice_key = ans.get("choice", "opt_0")
            idx = int(choice_key.split("_")[-1]) if "_" in choice_key else 0
            idx = max(0, min(idx, len(options) - 1))
            conf = float(ans.get("confidence", 0.5))
            return options[idx], conf
        except Exception as err:
            logger.warning("Base Laya choice prediction failed: %s", err)
            return options[0] if options else "", 0.5

    def predict_noul(self, state: str, question: str) -> float:
        agent = self.load_model()
        if agent is None:
            return 0.5

        try:
            questions = {
                "noul_q": {
                    "type": "noul",
                    "instructions": question,
                }
            }
            res = agent.predict(state, questions)
            ans = res.get("answers", {}).get("noul_q", {})
            return float(ans.get("noul", 0.5))
        except Exception as err:
            logger.warning("Base Laya noul prediction failed: %s", err)
            return 0.5

    def predict_match_scoring_ensemble(
        self, job_desc: str = "", cv_text: str = "", job_title: str = "Unknown Role"
    ) -> Dict[str, Any]:
        agent = self.load_model()
        if agent is None:
            return {
                "skill_match": 2,
                "skill_confidence": 0.5,
                "seniority_fit": 1,
                "seniority_confidence": 0.5,
                "recruiter_fit_probability": 0.5,
            }

        state = f"Job Title: {job_title}\nJob Description:\n{job_desc}\n\nCandidate CV:\n{cv_text}"
        questions = {
            "skill_match": {
                "type": "score",
                "instructions": "Evaluate how well the candidate technical skills match the job requirements.",
                "criteria": [
                    "Completely unqualified, zero skill match",
                    "Missing most core skills",
                    "Moderate match with partial requirements",
                    "Strong match with most core technologies",
                    "Exceptional match covering all requirements",
                ],
            },
            "seniority_fit": {
                "type": "score",
                "instructions": "Evaluate candidate seniority fit against the job level requirements.",
                "criteria": [
                    "Candidate is far too junior or underqualified for this role",
                    "Candidate seniority roughly matches the required level",
                    "Candidate is overqualified or exceeds seniority requirements",
                ],
            },
            "recruiter_fit": {
                "type": "noul",
                "instructions": "Will a tech recruiter screen in and invite this candidate for an interview for this role?",
            },
        }

        try:
            res = agent.predict(state, questions)
            answers = res.get("answers", {})

            skill_ans = answers.get("skill_match", {})
            skill_probs = skill_ans.get("probabilities", {})
            if skill_probs:
                skill_val = int(max(skill_probs.items(), key=lambda kv: float(kv[1]))[0])
                skill_conf = float(skill_probs[str(skill_val)])
            else:
                skill_val = int(round(skill_ans.get("score", 2.0)))
                skill_conf = float(skill_ans.get("confidence", 0.5))

            sen_ans = answers.get("seniority_fit", {})
            sen_probs = sen_ans.get("probabilities", {})
            if sen_probs:
                sen_val = int(max(sen_probs.items(), key=lambda kv: float(kv[1]))[0])
                sen_conf = float(sen_probs[str(sen_val)])
            else:
                sen_val = int(round(sen_ans.get("score", 1.0)))
                sen_conf = float(sen_ans.get("confidence", 0.5))

            rec_ans = answers.get("recruiter_fit", {})
            rec_prob = float(rec_ans.get("noul", 0.5))

            return {
                "skill_match": max(0, min(4, skill_val)),
                "skill_confidence": skill_conf,
                "seniority_fit": max(0, min(2, sen_val)),
                "seniority_confidence": sen_conf,
                "recruiter_fit_probability": max(0.0, min(1.0, rec_prob)),
            }
        except Exception as err:
            logger.warning("Base Laya ensemble prediction failed: %s", err)
            return {
                "skill_match": 2,
                "skill_confidence": 0.5,
                "seniority_fit": 1,
                "seniority_confidence": 0.5,
                "recruiter_fit_probability": 0.5,
            }

    def predict_match_scoring_batch(
        self, items: List[Dict[str, str]], chunk_size: int = 4
    ) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            res = self.predict_match_scoring_ensemble(
                job_desc=item.get("job_desc", ""),
                cv_text=item.get("cv_text", ""),
                job_title=item.get("job_title", "Unknown Role"),
            )
            results.append(res)
        return results
