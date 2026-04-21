"""
Spaces Pipeline Environment.

Day 2: Adds rubric-based grading, reward shaping, and improved validation.
Day 3 will add Auditor + Expert Reviewer.
Day 4 will add schema drift.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DEFAULT_MAX_ACTIONS,
        DEFAULT_MAX_SPACE_CALLS,
        DEFAULT_TIME_BUDGET_S,
        HISTORY_WINDOW,
        ActionType,
        SpacesPipelineAction,
        SpacesPipelineObservation,
    )
    from .auditor import Auditor
    from .expert_reviewer import ExpertReviewer, PERSONA_HINTS
    from .rubrics import (
        SpacesPipelineRubric,
        compute_intermediate_reward,
    )
    from .schema_drift import SchemaDriftManager
    from .space_catalog import SpaceCatalog, get_mode
    from .space_caller import SpaceCaller
except ImportError:
    from models import (
        DEFAULT_MAX_ACTIONS,
        DEFAULT_MAX_SPACE_CALLS,
        DEFAULT_TIME_BUDGET_S,
        HISTORY_WINDOW,
        ActionType,
        SpacesPipelineAction,
        SpacesPipelineObservation,
    )
    from server.auditor import Auditor
    from server.expert_reviewer import ExpertReviewer, PERSONA_HINTS
    from server.rubrics import (
        SpacesPipelineRubric,
        compute_intermediate_reward,
    )
    from server.schema_drift import SchemaDriftManager
    from server.space_catalog import SpaceCatalog, get_mode
    from server.space_caller import SpaceCaller


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
TASKS_FIXTURE = FIXTURES_DIR / "tasks.json"


def _load_tasks() -> List[Dict[str, Any]]:
    if not TASKS_FIXTURE.exists():
        return []
    with open(TASKS_FIXTURE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reward shaping config (env vars)
# ---------------------------------------------------------------------------

def _dense_rewards_enabled() -> bool:
    """DENSE_REWARDS env var: 'true' (default) or 'false'."""
    return os.getenv("DENSE_REWARDS", "true").lower() == "true"


def _shaping_coefficient() -> float:
    """SHAPING_COEFF env var: scales intermediate rewards (0.0 to 1.0)."""
    try:
        return float(os.getenv("SHAPING_COEFF", "1.0"))
    except ValueError:
        return 1.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SpacesPipelineEnvironment(Environment):
    """
    Day 2 environment with rubric grading and reward shaping.

    Reward signal:
      - Dense (training warmup): small per-step rewards for valid actions
      - Sparse (training final): only final reward on submit (rubric-based)
      - Toggle via DENSE_REWARDS env var; scale via SHAPING_COEFF
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(rubric=SpacesPipelineRubric())
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()

        # Service interfaces
        self._catalog = SpaceCatalog()
        self._caller = SpaceCaller()

        # Task pool
        self._all_tasks = _load_tasks()
        self._current_task: Dict[str, Any] = {}

        # Episode state
        self._actions_remaining = DEFAULT_MAX_ACTIONS
        self._spaces_called = 0
        self._spaces_budget_remaining = DEFAULT_MAX_SPACE_CALLS
        self._time_used_s: float = 0.0
        self._time_budget_s: float = DEFAULT_TIME_BUDGET_S
        self._last_action_latency_s: float = 0.0
        self._recent_actions: List[Dict[str, Any]] = []
        self._recent_outputs: List[Dict[str, Any]] = []
        self._all_actions_log: List[Dict[str, Any]] = []  # full log for redundancy
        self._last_search_results: List[Dict[str, Any]] = []
        self._last_card_read: Optional[Dict[str, Any]] = None
        self._cards_read: Dict[str, Dict[str, Any]] = {}  # cache
        self._space_call_history: List[Dict[str, Any]] = []  # for redundancy
        self._submitted_answer: Optional[Dict[str, Any]] = None
        self._is_done = False
        self._auditor_flags: List[Dict[str, Any]] = []
        self._expert_persona = "accuracy_first"
        self._expert_persona_hint = ""
        self._expert_recent_feedback = ""

        # Multi-actor agents (Day 3)
        self._auditor = Auditor()
        self._expert = ExpertReviewer()
        self._expert_score: float = -1.0  # -1 = not yet scored

        # Schema drift manager (Day 4)
        self._drift = SchemaDriftManager()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> SpacesPipelineObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Pick task
        if not self._all_tasks:
            self._current_task = self._stub_task()
        elif task and task != "random":
            matches = [t for t in self._all_tasks if t.get("task_id") == task]
            self._current_task = matches[0] if matches else self._all_tasks[0]
        else:
            self._current_task = self._rng.choice(self._all_tasks)

        # Reset episode state
        self._actions_remaining = self._current_task.get(
            "max_actions", DEFAULT_MAX_ACTIONS
        )
        self._spaces_called = 0
        self._spaces_budget_remaining = self._current_task.get(
            "max_space_calls", DEFAULT_MAX_SPACE_CALLS
        )
        self._time_used_s = 0.0
        self._time_budget_s = float(
            self._current_task.get("time_budget_s", DEFAULT_TIME_BUDGET_S)
        )
        self._last_action_latency_s = 0.0
        self._recent_actions = []
        self._recent_outputs = []
        self._all_actions_log = []
        self._last_search_results = []
        self._last_card_read = None
        self._cards_read = {}
        self._space_call_history = []
        self._submitted_answer = None
        self._is_done = False
        self._auditor_flags = []
        self._expert_score = -1.0

        # Set Expert persona for this episode (rotated; can be overridden by task)
        self._expert_persona = self._current_task.get("expert_persona") or self._rng.choice(
            ["speed_first", "accuracy_first", "cost_first"]
        )
        self._expert_persona_hint = self._persona_hint(self._expert_persona)
        self._expert_recent_feedback = ""

        # Reset multi-actor agents
        self._auditor.reset()
        self._expert.reset(self._expert_persona)

        # Reset schema drift
        self._drift.reset(self._current_task.get("drift_events", []))

        # Reset rubric
        self._reset_rubric()

        return self._build_observation(reward=0.0)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: SpacesPipelineAction
    ) -> SpacesPipelineObservation:  # type: ignore[override]
        self._state.step_count += 1

        # Fire any pending drift events (Patronus)
        self._drift.maybe_fire(self._state.step_count)

        action_type_str = action.action_type
        payload = action.payload or {}

        # Validate action type
        try:
            atype = ActionType(action_type_str)
            type_valid = True
        except ValueError:
            atype = ActionType.NOOP
            type_valid = False

        # Decrement action budget
        self._actions_remaining = max(0, self._actions_remaining - 1)

        # Default small latency for non-call actions (reset each step)
        self._last_action_latency_s = 0.0
        if atype in (ActionType.SEARCH_SPACES, ActionType.READ_CARD):
            non_call_latency = 0.5
            self._last_action_latency_s = non_call_latency
            self._time_used_s += non_call_latency

        success = True
        output_snippet = ""
        error_msg: Optional[str] = None
        is_redundant = False

        # Dispatch
        if not type_valid:
            success = False
            error_msg = f"Unknown action_type: {action_type_str}"

        elif atype == ActionType.SEARCH_SPACES:
            output_snippet, error_msg = self._handle_search(payload)
            success = error_msg is None

        elif atype == ActionType.READ_CARD:
            output_snippet, error_msg = self._handle_read_card(payload)
            success = error_msg is None

        elif atype == ActionType.CALL_SPACE:
            if self._spaces_budget_remaining <= 0:
                success = False
                error_msg = "Space call budget exhausted"
            else:
                # Check redundancy BEFORE invoking
                is_redundant = self._is_redundant_call(payload)
                output_snippet, error_msg = self._handle_call_space(payload)
                success = error_msg is None
                # Charge simulated latency for this Space call
                space_id = payload.get("space_id", "")
                latency = self._get_space_latency(space_id)
                self._last_action_latency_s = latency
                self._time_used_s += latency
                if success or error_msg:
                    self._space_call_history.append({
                        "space_id": space_id,
                        "inputs": payload.get("inputs", {}),
                        "success": success,
                        "latency_s": latency,
                    })
                    self._spaces_called += 1
                    self._spaces_budget_remaining -= 1

        elif atype == ActionType.SUBMIT:
            output_snippet, error_msg = self._handle_submit(payload)
            success = error_msg is None
            self._is_done = True

        elif atype == ActionType.NOOP:
            success = False
            error_msg = "NOOP action"

        # Record history
        self._record_history(
            atype.value, payload, success, output_snippet, error_msg
        )

        # Auditor review (Fleet AI) — pass catalog for faster-equivalent detection
        new_flags = self._auditor.review_action(
            step=self._state.step_count,
            action_type=atype.value,
            payload=payload,
            success=success,
            error=error_msg,
            cards_read=self._cards_read,
            space_call_history=self._space_call_history,
            catalog=self._catalog,
        )
        self._auditor_flags = list(self._auditor.flags)

        # Expert reviews submission
        if atype == ActionType.SUBMIT and success:
            sub_flags = self._auditor.review_submission(
                step=self._state.step_count,
                submitted=self._submitted_answer,
                expected_schema=self._current_task.get("expected_output_schema", {}),
                space_call_history=self._space_call_history,
            )
            self._auditor_flags = list(self._auditor.flags)

            expert_score, expert_feedback = self._expert.score_submission(
                submitted=self._submitted_answer,
                expected_schema=self._current_task.get("expected_output_schema", {}),
                expert_rubric=self._current_task.get("expert_rubric", {}),
                space_call_history=self._space_call_history,
            )
            self._expert_recent_feedback = expert_feedback
            # Stash expert score for rubric to find
            self._expert_score = expert_score

        # Compute step reward
        step_reward = self._compute_step_reward(
            atype.value, success, is_redundant
        )

        # Apply auditor flag penalties to step reward (small immediate signal)
        if new_flags:
            for f in new_flags:
                sev = f.get("severity", "info")
                if sev == "warning":
                    step_reward -= 0.05
                elif sev == "error":
                    step_reward -= 0.15
                elif sev == "critical":
                    step_reward -= 0.40

        # Check forced termination
        if self._actions_remaining <= 0 and not self._is_done:
            self._is_done = True
        # Time budget exhausted
        if self._time_used_s >= self._time_budget_s and not self._is_done:
            self._is_done = True

        # On terminal step, compute final grade reward
        final_reward_bonus = 0.0
        obs = self._build_observation(reward=step_reward)
        grade_score: Optional[float] = None
        if self._is_done and self.rubric is not None:
            # Apply rubric to record this step
            grade_reward = self._apply_rubric(action, obs)
            grade_score = float(grade_reward)
            final_reward_bonus = grade_score
            obs.grade_score = round(grade_score, 4)
            obs.grade_details = self.rubric.grade_details
            # Final reward = step shaping + rubric grade
            obs.reward = round(step_reward + final_reward_bonus, 4)
        else:
            # Still apply rubric mid-trajectory so it can accumulate
            self._apply_rubric(action, obs)

        return obs

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_step_reward(
        self, action_type: str, success: bool, is_redundant: bool
    ) -> float:
        if not _dense_rewards_enabled():
            return 0.0
        coeff = _shaping_coefficient()
        if coeff <= 0:
            return 0.0
        return round(
            coeff * compute_intermediate_reward(action_type, success, is_redundant),
            4,
        )

    def _get_space_latency(self, space_id: str) -> float:
        """Fetch simulated latency for a Space, from its cached card."""
        if not space_id:
            return 5.0
        card = self._cards_read.get(space_id)
        if card is None:
            # Also check catalog for pre-loaded latency info
            card = self._catalog.read_card(space_id)
            if card is not None:
                self._cards_read[space_id] = card
        if card is None:
            return 5.0  # default
        latency = card.get("estimated_latency_s", 5.0)
        try:
            return float(latency)
        except (TypeError, ValueError):
            return 5.0

    def _is_redundant_call(self, payload: Dict[str, Any]) -> bool:
        """Detect if this Space call is identical to a previous successful one."""
        space_id = payload.get("space_id", "")
        inputs = payload.get("inputs", {})
        for prev in self._space_call_history:
            if (
                prev["space_id"] == space_id
                and prev["inputs"] == inputs
                and prev["success"]
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_search(self, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        query = payload.get("query", "")
        try:
            top_k = int(payload.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(20, top_k))
        if not query or not isinstance(query, str):
            return "", "Missing or invalid 'query' (must be non-empty string)"
        try:
            results = self._catalog.search(query, top_k=top_k)
            self._last_search_results = results
            return f"Found {len(results)} results", None
        except Exception as e:
            return "", f"Search failed: {e}"

    def _handle_read_card(self, payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        space_id = payload.get("space_id", "")
        if not space_id or not isinstance(space_id, str):
            return "", "Missing or invalid 'space_id'"
        try:
            card = self._catalog.read_card(space_id)
            if card is None:
                return "", f"No card available for {space_id}"
            # Apply schema drift if active
            card = self._drift.transform_card(space_id, card)
            self._last_card_read = card
            self._cards_read[space_id] = card
            desc = (card.get("description") or "")[:120]
            note = " [DRIFTED]" if card.get("_drifted") else ""
            return f"Card for {space_id}{note}: {desc}", None
        except Exception as e:
            return "", f"Read card failed: {e}"

    def _handle_call_space(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        space_id = payload.get("space_id", "")
        inputs = payload.get("inputs", {})
        if not space_id or not isinstance(space_id, str):
            return "", "Missing or invalid 'space_id'"
        if not isinstance(inputs, dict):
            return "", "'inputs' must be a dict"

        # Validate inputs against schema if card was read
        card = self._cards_read.get(space_id)
        if card:
            schema = card.get("input_schema", {}) or {}
            missing = []
            for field, spec in schema.items():
                if isinstance(spec, dict) and spec.get("required") and field not in inputs:
                    missing.append(field)
            if missing:
                return "", f"Missing required input fields: {missing}"

        # Validate against drift contract (Patronus)
        drift_error = self._drift.validates_drift(space_id, inputs)
        if drift_error:
            self._drift.record_detection(self._state.step_count, space_id, drift_error)
            return "", f"DRIFT: {drift_error}"

        try:
            response = self._caller.call(
                task_id=self._current_task.get("task_id", "unknown"),
                space_id=space_id,
                inputs=inputs,
            )
            if not response.get("success"):
                return "", response.get("error") or "Space call failed"
            output = response.get("output")
            snippet = str(output)[:200] if output else "(empty output)"
            return snippet, None
        except Exception as e:
            return "", f"Call space failed: {e}"

    def _handle_submit(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        answer = payload.get("answer")
        if answer is None:
            return "", "Missing 'answer' in submit payload"
        if not isinstance(answer, dict):
            return "", "'answer' must be a dict"
        self._submitted_answer = answer
        snippet = f"Submitted answer with keys: {list(answer.keys())}"
        return snippet, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persona_hint(self, persona: str) -> str:
        """Generate a noisy NL hint about active persona (sampled from variants)."""
        variants = PERSONA_HINTS.get(persona, ["Reviewer expectations are unclear."])
        return self._rng.choice(variants)

    def _record_history(
        self,
        action_type: str,
        payload: Dict[str, Any],
        success: bool,
        output_snippet: str,
        error: Optional[str],
    ) -> None:
        compact_payload = self._compact_payload(payload)
        action_record = {
            "step": self._state.step_count,
            "action_type": action_type,
            "payload": compact_payload,
        }
        output_record = {
            "step": self._state.step_count,
            "success": success,
            "output_snippet": output_snippet,
            "error": error,
        }
        self._recent_actions.append(action_record)
        self._recent_outputs.append(output_record)
        self._all_actions_log.append({**action_record, **output_record})
        if len(self._recent_actions) > HISTORY_WINDOW:
            self._recent_actions.pop(0)
            self._recent_outputs.pop(0)

    def _compact_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compress payload values for history storage."""
        result = {}
        for k, v in payload.items():
            if isinstance(v, str) and len(v) > 100:
                result[k] = v[:100] + "..."
            elif isinstance(v, dict):
                result[k] = {
                    sk: (sv if not isinstance(sv, str) or len(sv) < 100 else sv[:100] + "...")
                    for sk, sv in v.items()
                }
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float) -> SpacesPipelineObservation:
        flag_count = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for f in self._auditor_flags:
            sev = f.get("severity", "info")
            if sev in flag_count:
                flag_count[sev] += 1

        time_remaining = max(0.0, self._time_budget_s - self._time_used_s)

        return SpacesPipelineObservation(
            task_id=self._current_task.get("task_id", ""),
            task_description=self._current_task.get("description", ""),
            task_input=self._current_task.get("input", {}),
            expected_output_schema=self._current_task.get("expected_output_schema", {}),
            actions_remaining=self._actions_remaining,
            spaces_called=self._spaces_called,
            spaces_budget_remaining=self._spaces_budget_remaining,
            time_used_s=round(self._time_used_s, 2),
            time_remaining_s=round(time_remaining, 2),
            last_action_latency_s=round(self._last_action_latency_s, 2),
            recent_actions=list(self._recent_actions),
            recent_outputs=list(self._recent_outputs),
            last_search_results=list(self._last_search_results),
            last_card_read=self._last_card_read,
            auditor_flags=list(self._auditor_flags),
            flag_count_by_severity=flag_count,
            expert_persona_hint=self._expert_persona_hint,
            expert_recent_feedback=self._expert_recent_feedback,
            detected_drift=list(self._drift.detected),
            step_number=self._state.step_count,
            max_steps=self._current_task.get("max_actions", DEFAULT_MAX_ACTIONS),
            submitted_answer=self._submitted_answer,
            grade_score=None,
            grade_details=None,
            done=self._is_done,
            reward=reward,
            metadata={
                "mode": get_mode(),
                "task_id": self._current_task.get("task_id", ""),
                "expert_persona": self._expert_persona,
                "expert_score": self._expert_score,
                "max_space_calls": self._current_task.get(
                    "max_space_calls", DEFAULT_MAX_SPACE_CALLS
                ),
                "time_budget_s": self._time_budget_s,
                "time_used_s": round(self._time_used_s, 2),
            },
        )

    # ------------------------------------------------------------------
    # Stub task
    # ------------------------------------------------------------------

    def _stub_task(self) -> Dict[str, Any]:
        return {
            "task_id": "stub_001",
            "description": "Stub task for smoke testing.",
            "input": {},
            "expected_output_schema": {"answer": "string"},
            "max_actions": 10,
            "max_space_calls": 3,
            "expert_persona": "accuracy_first",
        }

    @property
    def state(self) -> State:
        return self._state
