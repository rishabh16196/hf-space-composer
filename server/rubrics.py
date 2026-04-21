"""
Grading rubric for the Spaces Pipeline Pro Environment.

Combines signals from:
  - Expert Reviewer (Snorkel) — final answer quality, persona-weighted
  - Auditor (Fleet AI) — pipeline correctness, severity-weighted flag count
  - Efficiency — actions used vs. budget
  - Cost — Space invocations made
  - Format — output schema compliance

Day 2: stub implementation. Auditor and Expert score components plug in Day 3.

Score weights:
  expert      0.50
  auditor     0.20
  efficiency  0.15
  cost        0.10
  format      0.05
"""

from typing import Any, Dict, List, Tuple

from openenv.core.rubrics.trajectory import TrajectoryRubric


# Component weights (must sum to 1.0)
WEIGHT_EXPERT = 0.45
WEIGHT_AUDITOR = 0.18
WEIGHT_EFFICIENCY = 0.12
WEIGHT_COST = 0.08
WEIGHT_TIME = 0.12     # NEW: wall-clock time score
WEIGHT_FORMAT = 0.05

# Auditor severity penalties
SEVERITY_PENALTY = {
    "info": 0.0,
    "warning": 0.10,
    "error": 0.30,
    "critical": 0.80,
}

# Persona reward modifiers (multiplicative on each component)
PERSONA_MODIFIERS: Dict[str, Dict[str, float]] = {
    "speed_first":    {"expert": 1.0, "auditor": 1.0, "efficiency": 2.0, "cost": 1.0, "time": 2.5, "format": 1.0},
    "accuracy_first": {"expert": 1.5, "auditor": 1.2, "efficiency": 0.8, "cost": 0.8, "time": 0.6, "format": 1.0},
    "cost_first":     {"expert": 1.0, "auditor": 1.0, "efficiency": 1.0, "cost": 2.0, "time": 1.0, "format": 1.0},
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_efficiency_score(actions_used: int, max_actions: int) -> float:
    """1.0 if used <= 1/3 budget, decays linearly to 0.0 at full budget."""
    if max_actions <= 0:
        return 0.0
    ratio = actions_used / max_actions
    if ratio <= 0.33:
        return 1.0
    return _clamp01(1.0 - (ratio - 0.33) / 0.67)


def compute_cost_score(spaces_called: int, max_space_calls: int) -> float:
    """1.0 if called <= 1/3 budget, decays linearly."""
    if max_space_calls <= 0:
        return 0.0
    ratio = spaces_called / max_space_calls
    if ratio <= 0.33:
        return 1.0
    return _clamp01(1.0 - (ratio - 0.33) / 0.67)


def compute_time_score(time_used_s: float, time_budget_s: float) -> float:
    """1.0 if time_used <= 1/2 budget, decays linearly to 0 at full budget.

    Exceeding the budget returns 0 (the env separately force-terminates).
    """
    if time_budget_s <= 0:
        return 0.0
    ratio = time_used_s / time_budget_s
    if ratio <= 0.5:
        return 1.0
    return _clamp01(1.0 - (ratio - 0.5) / 0.5)


def compute_format_score(
    submitted: Dict[str, Any], expected_schema: Dict[str, Any]
) -> float:
    """Fraction of expected fields present and non-empty in submission."""
    if not expected_schema:
        return 1.0
    if not submitted or not isinstance(submitted, dict):
        return 0.0
    expected_keys = list(expected_schema.keys())
    if not expected_keys:
        return 1.0
    present = sum(
        1 for k in expected_keys
        if k in submitted and submitted[k] not in (None, "", [], {})
    )
    return present / len(expected_keys)


def compute_auditor_score(
    flags: List[Dict[str, Any]], max_flags_for_zero: int = 10
) -> float:
    """Inverse of severity-weighted flag count.

    No flags → 1.0. As severity-weighted total grows, score drops.
    """
    if not flags:
        return 1.0
    total_penalty = sum(
        SEVERITY_PENALTY.get(f.get("severity", "info"), 0.0) for f in flags
    )
    score = 1.0 - (total_penalty / max_flags_for_zero)
    return _clamp01(score)


def compute_intermediate_reward(
    action_type: str,
    success: bool,
    is_redundant: bool = False,
) -> float:
    """Per-step reward shaping for training warmup.

    Toggleable via environment variable DENSE_REWARDS (in env class).
    Anneals to 0 in late training (handled by env scheduler).
    """
    if not success:
        return -0.20  # Fail penalty
    if is_redundant:
        return -0.10
    if action_type == "search_spaces":
        return 0.05
    if action_type == "read_card":
        return 0.10
    if action_type == "call_space":
        return 0.20
    if action_type == "submit":
        return 0.0  # Final reward is computed by rubric
    return 0.0


class SpacesPipelineRubric(TrajectoryRubric):
    """Grades agent performance on Spaces Pipeline tasks.

    On submit, computes:
      final_score = (
          weight_expert * persona_mult_expert * expert_score
          + weight_auditor * persona_mult_auditor * auditor_score
          + weight_efficiency * persona_mult_efficiency * efficiency_score
          + weight_cost * persona_mult_cost * cost_score
          + weight_format * persona_mult_format * format_score
      ) normalized to [0, 1]
    """

    def __init__(self) -> None:
        super().__init__(intermediate_reward=0.0)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Compute final grade from full trajectory."""
        if not trajectory:
            return 0.0

        _, final_obs = trajectory[-1]

        # Pull metrics off the final observation
        actions_used = getattr(final_obs, "step_number", 0)
        max_actions = getattr(final_obs, "max_steps", 1)
        spaces_called = getattr(final_obs, "spaces_called", 0)

        # Submitted answer
        submitted = getattr(final_obs, "submitted_answer", None) or {}
        expected_schema = getattr(final_obs, "expected_output_schema", {}) or {}

        # Auditor flags
        flags = getattr(final_obs, "auditor_flags", []) or []

        # Persona (from metadata or default to accuracy_first)
        meta = getattr(final_obs, "metadata", {}) or {}
        persona = meta.get("expert_persona", "accuracy_first")
        modifiers = PERSONA_MODIFIERS.get(persona, PERSONA_MODIFIERS["accuracy_first"])

        # Expert score: pulled from final_obs metadata if Expert Reviewer ran
        # Day 3 will populate this; Day 2 stub uses format compliance as proxy.
        expert_score = float(meta.get("expert_score", -1.0))
        if expert_score < 0.0:
            # Stub fallback: if no expert score, use format completeness
            expert_score = compute_format_score(submitted, expected_schema)

        # Component scores
        max_space_calls = max(1, max_actions // 2)  # heuristic if not provided
        meta_max_space = meta.get("max_space_calls")
        if meta_max_space:
            max_space_calls = meta_max_space

        auditor_score = compute_auditor_score(flags)
        efficiency_score = compute_efficiency_score(actions_used, max_actions)
        cost_score = compute_cost_score(spaces_called, max_space_calls)
        format_score = compute_format_score(submitted, expected_schema)

        # Time score (NEW)
        time_used = float(getattr(final_obs, "time_used_s", 0.0))
        time_budget = float(meta.get("time_budget_s", 120.0))
        time_score = compute_time_score(time_used, time_budget)

        # Weighted combination with persona modifiers
        raw_score = (
            WEIGHT_EXPERT * modifiers["expert"] * expert_score
            + WEIGHT_AUDITOR * modifiers["auditor"] * auditor_score
            + WEIGHT_EFFICIENCY * modifiers["efficiency"] * efficiency_score
            + WEIGHT_COST * modifiers["cost"] * cost_score
            + WEIGHT_TIME * modifiers.get("time", 1.0) * time_score
            + WEIGHT_FORMAT * modifiers["format"] * format_score
        )
        max_possible = (
            WEIGHT_EXPERT * modifiers["expert"]
            + WEIGHT_AUDITOR * modifiers["auditor"]
            + WEIGHT_EFFICIENCY * modifiers["efficiency"]
            + WEIGHT_COST * modifiers["cost"]
            + WEIGHT_TIME * modifiers.get("time", 1.0)
            + WEIGHT_FORMAT * modifiers["format"]
        )
        final_score = raw_score / max_possible if max_possible > 0 else 0.0
        final_score = _clamp01(final_score)

        # Engagement gating — close the "lazy submit" reward-hacking loophole.
        # An agent can otherwise submit with zero Space calls, collect
        # efficiency + cost + time + format points, and land at 0.4-0.6
        # without actually solving anything. We hard-cap such runs.
        n_successful_calls = sum(
            1 for _, o in trajectory
            if o.recent_outputs and o.recent_outputs[-1].get("success") and
               o.recent_actions and o.recent_actions[-1].get("action_type") == "call_space"
        )
        task_requires_tools = bool(expected_schema) and len(expected_schema) >= 1
        engagement_gate_applied = False
        if task_requires_tools and n_successful_calls == 0:
            # Cap at 0.15 — well below pass threshold
            if final_score > 0.15:
                final_score = 0.15
                engagement_gate_applied = True

        # Save details
        self._grade_details = {
            "task_id": getattr(final_obs, "task_id", ""),
            "score": round(final_score, 4),
            "components": {
                "expert": round(expert_score, 4),
                "auditor": round(auditor_score, 4),
                "efficiency": round(efficiency_score, 4),
                "cost": round(cost_score, 4),
                "time": round(time_score, 4),
                "format": round(format_score, 4),
            },
            "persona": persona,
            "actions_used": actions_used,
            "spaces_called": spaces_called,
            "successful_space_calls": n_successful_calls,
            "time_used_s": round(time_used, 2),
            "time_budget_s": round(time_budget, 2),
            "flags_count": len(flags),
            "engagement_gate_applied": engagement_gate_applied,
            "passed": final_score >= 0.5,
        }

        return final_score

    def compute_step_rewards(self) -> List[float]:
        """Uniform credit assignment — every step shares the final score."""
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)

    @property
    def grade_details(self) -> Dict[str, Any]:
        return getattr(self, "_grade_details", {})

    def reset(self) -> None:
        super().reset()
        self._grade_details = {}
