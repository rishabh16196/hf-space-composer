"""
Expert Reviewer (Snorkel sub-theme).

Simulated subject-matter expert that scores final submissions.

Has 3 personas (one active per episode, can shift mid-episode):
  - speed_first:    rewards minimal pipeline length
  - accuracy_first: rewards thorough, multi-source verification
  - cost_first:     rewards minimal Space invocations

Provides:
  - persona_hint: noisy NL clue about active persona (in observation)
  - score: final answer quality 0.0-1.0 (used by rubric)
  - feedback: optional NL critique (in observation)

Scoring approach:
  - Mock mode (default): compares submitted answer to task's expert_rubric
    using string similarity / keyword presence / field structure
  - Live mode: optionally calls LLM judge (with caching for reproducibility)
"""

import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _fuzzy_similarity(a: str, b: str) -> float:
    """0.0 to 1.0 string similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _keyword_coverage(text: str, keywords: List[str]) -> float:
    """Fraction of keywords present in text."""
    if not keywords:
        return 1.0
    if not text:
        return 0.0
    norm = _normalize(text)
    hits = sum(1 for kw in keywords if _normalize(kw) in norm)
    return hits / len(keywords)


class ExpertReviewer:
    """Stateful expert reviewer."""

    def __init__(self, persona: str = "accuracy_first") -> None:
        self.persona = persona
        self.last_feedback = ""
        self.last_score: Optional[float] = None

    def reset(self, persona: str) -> None:
        self.persona = persona
        self.last_feedback = ""
        self.last_score = None

    def shift_persona(self, new_persona: str) -> None:
        """Mid-episode persona shift (Snorkel evolving requirements)."""
        self.persona = new_persona

    def score_submission(
        self,
        submitted: Optional[Dict[str, Any]],
        expected_schema: Dict[str, Any],
        expert_rubric: Dict[str, Any],
        space_call_history: List[Dict[str, Any]],
    ) -> Tuple[float, str]:
        """
        Score the submission. Returns (score, feedback).

        expert_rubric is task-defined, e.g.:
            {
                "transcript": {"weight": 0.4, "method": "fuzzy", "target": "..."},
                "summary":    {"weight": 0.4, "method": "keywords", "keywords": ["..."]},
                "sentiment":  {"weight": 0.2, "method": "exact", "target": "positive"}
            }
        """
        if submitted is None or not isinstance(submitted, dict):
            self.last_score = 0.0
            self.last_feedback = "No valid submission to review."
            return 0.0, self.last_feedback

        if not expert_rubric:
            # Fallback: presence-based scoring against expected schema
            score = self._fallback_score(submitted, expected_schema)
            self.last_score = score
            self.last_feedback = self._format_feedback(
                score,
                f"No expert rubric defined; using field presence ({score:.2f})",
            )
            return score, self.last_feedback

        component_scores: List[Tuple[str, float, float]] = []
        for field, rubric in expert_rubric.items():
            weight = float(rubric.get("weight", 1.0))
            method = rubric.get("method", "fuzzy")
            value = submitted.get(field, "")

            if method == "fuzzy":
                target = rubric.get("target", "")
                s = _fuzzy_similarity(str(value), str(target))
            elif method == "keywords":
                keywords = rubric.get("keywords", [])
                s = _keyword_coverage(str(value), keywords)
            elif method == "exact":
                target = rubric.get("target", "")
                s = 1.0 if _normalize(str(value)) == _normalize(str(target)) else 0.0
            elif method == "non_empty":
                s = 1.0 if value not in (None, "", [], {}) else 0.0
            elif method == "min_length":
                min_len = int(rubric.get("min_length", 1))
                s = 1.0 if len(str(value)) >= min_len else len(str(value)) / min_len
            else:
                s = 0.5  # unknown method

            component_scores.append((field, weight, s))

        total_weight = sum(w for _, w, _ in component_scores)
        if total_weight == 0:
            base_score = 0.0
        else:
            base_score = sum(w * s for _, w, s in component_scores) / total_weight

        # Apply persona modifier
        score = self._apply_persona_modifier(
            base_score, space_call_history, len(component_scores)
        )

        # Generate feedback
        weakest = min(component_scores, key=lambda x: x[2]) if component_scores else None
        feedback_str = (
            f"Score {score:.2f} (persona={self.persona}). "
            f"Weakest field: {weakest[0]} ({weakest[2]:.2f})."
            if weakest else f"Score {score:.2f}"
        )

        self.last_score = score
        self.last_feedback = feedback_str
        return score, feedback_str

    def _apply_persona_modifier(
        self,
        base_score: float,
        space_call_history: List[Dict[str, Any]],
        num_components: int,
    ) -> float:
        """Modify base score based on persona preferences (time-aware)."""
        n_calls = len(space_call_history)
        n_successful = sum(1 for c in space_call_history if c.get("success"))
        total_latency = sum(c.get("latency_s", 0.0) for c in space_call_history)

        if self.persona == "speed_first":
            # Bonus for fast total wall-clock; penalty for slow Spaces
            if total_latency < 15 and n_calls <= 3:
                return min(1.0, base_score + 0.15)
            elif total_latency > 60 or n_calls > 6:
                return max(0.0, base_score - 0.20)
            return base_score

        elif self.persona == "cost_first":
            # Penalty per successful call + small bonus for picking fast Spaces
            penalty = 0.05 * max(0, n_successful - 2)
            if total_latency < 20 and n_successful >= 1:
                penalty -= 0.05
            return max(0.0, base_score - penalty)

        elif self.persona == "accuracy_first":
            # Bonus for multi-Space verification; time less important
            if n_successful >= 2:
                return min(1.0, base_score + 0.05)
            return base_score

        return base_score

    def _fallback_score(
        self,
        submitted: Dict[str, Any],
        expected_schema: Dict[str, Any],
    ) -> float:
        """Used when no expert_rubric is defined for the task."""
        if not expected_schema:
            return 0.5
        present = sum(
            1 for k in expected_schema
            if k in submitted and submitted[k] not in (None, "", [], {})
        )
        return present / len(expected_schema)

    def _format_feedback(self, score: float, base_message: str) -> str:
        return f"[{self.persona}] {base_message}"


# ---------------------------------------------------------------------------
# Persona definitions (could be loaded from fixture)
# ---------------------------------------------------------------------------

PERSONA_HINTS = {
    "speed_first": [
        "Reviewer mentions a tight deadline.",
        "Reviewer wants results fast — quality is secondary.",
        "Reviewer says 'just get me something quickly'.",
    ],
    "accuracy_first": [
        "Reviewer wants thorough, well-verified output.",
        "Reviewer values cross-checking and multiple sources.",
        "Reviewer says 'take your time, I want this right'.",
    ],
    "cost_first": [
        "Reviewer is budget-conscious; minimize compute.",
        "Reviewer mentions cost overruns last quarter.",
        "Reviewer says 'use the cheapest option that works'.",
    ],
}
