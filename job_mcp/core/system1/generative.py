import asyncio
import logging
from typing import Any, Dict, List, Tuple

from job_mcp.core.llm.gateway import ResilientLLMGateway
from job_mcp.core.system1.interface import System1Engine

logger = logging.getLogger(__name__)


class GenerativeBaselineEngine:
    """Generative LLM baseline implementing the System1Engine interface.
    
    This simulates the System 1 logic using zero-shot prompting via the
    ResilientLLMGateway. Used for A/B testing against the fine-tuned local LAYA model.
    """

    def __init__(self, gateway: ResilientLLMGateway = None):
        self.gateway = gateway or ResilientLLMGateway()

    def is_loaded(self) -> bool:
        return True

    def predict_score(self, state: str, question: str, options: List[str]) -> Tuple[int, float]:
        if not options:
            return 0, 0.50
        opts = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
        prompt = f"Context:\n{state}\n\nQuestion: {question}\nOptions:\n{opts}\n\nRespond with only the integer ID (0 to {len(options)-1}) of the best option."
        
        try:
            # We run this synchronously since System1Engine interface is sync
            ans = asyncio.run(self.gateway.ask_question(prompt, system_prompt="You are a strict text classification evaluator. Only output the integer ID."))
            ans = ans.strip()
            # extract first digit
            import re
            match = re.search(r'\d+', ans)
            if match:
                idx = int(match.group())
                if 0 <= idx < len(options):
                    return idx, 0.75 # Arbitrary high confidence
            return min(2, len(options) - 1), 0.50
        except Exception as e:
            logger.warning("Generative baseline failed predict_score: %s", e)
            return min(2, len(options) - 1), 0.50

    def predict_noul(self, state: str, question: str) -> float:
        try:
            prompt = f"Context:\n{state}\n\nQuestion: {question}\nRespond strictly with YES or NO."
            ans = asyncio.run(self.gateway.ask_question(prompt, system_prompt="You are a strict boolean classifier. Only output YES or NO."))
            if "yes" in ans.strip().lower():
                return 0.90
            return 0.10
        except Exception as e:
            logger.warning("Generative baseline failed predict_noul: %s", e)
            return 0.50

    def predict_choice(self, state: str, question: str, options: List[str]) -> Tuple[str, float]:
        if not options:
            return "", 0.50
        idx, conf = self.predict_score(state, question, options)
        return options[idx], conf

    def predict_match_scoring_ensemble(self, job_desc: str = "", cv_text: str = "", job_title: str = "Unknown Role") -> Dict[str, Any]:
        results = self.predict_match_scoring_batch([{"job_desc": job_desc, "cv_text": cv_text, "job_title": "Unknown"}])
        return results[0] if results else {
            "skill_match": 2,
            "skill_confidence": 0.50,
            "seniority_fit": 2,
            "seniority_confidence": 0.50,
            "recruiter_fit_probability": 0.50,
        }

    def predict_match_scoring_batch(self, items: List[Dict[str, str]], chunk_size: int = 4) -> List[Dict[str, Any]]:
        """Run the generative LLM evaluating the 3 questions sequentially or combined."""
        all_results = []
        for item in items:
            job_title = item.get("job_title", "Unknown Role")
            state = f"Job Title: {job_title}\nJob Description:\n{item['job_desc']}\nCandidate CV:\n{item['cv_text']}"
            
            # 1. Skill
            skill_opts = ["None (0-20%)", "Weak (20-40%)", "Partial (40-60%)", "Strong (60-80%)", "Perfect (80-100%)"]
            skill_idx, skill_conf = self.predict_score(state, "Evaluate the technical and professional skill match of the candidate for this role.", skill_opts)
            
            # 2. Seniority
            sen_opts = ["Far too junior", "Slightly junior", "Good fit", "Senior", "Overqualified"]
            sen_idx, sen_conf = self.predict_score(state, "Assess the seniority alignment of the candidate relative to the requirements.", sen_opts)
            
            # 3. Recruiter Fit
            rec_prob = self.predict_noul(state, "Would a human technical recruiter recommend advancing this candidate to an interview?")
            
            all_results.append({
                "skill_match": skill_idx,
                "skill_confidence": skill_conf,
                "seniority_fit": sen_idx,
                "seniority_confidence": sen_conf,
                "recruiter_fit_probability": rec_prob,
            })
            
        return all_results
