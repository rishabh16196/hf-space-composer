"""Spaces Pipeline Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DEFAULT_MAX_ACTIONS,
    DEFAULT_MAX_SPACE_CALLS,
    DEFAULT_TIME_BUDGET_S,
    SpacesPipelineAction,
    SpacesPipelineObservation,
)


class SpacesPipelineEnv(
    EnvClient[SpacesPipelineAction, SpacesPipelineObservation, State]
):
    """
    Client for the Spaces Pipeline Pro Environment.

    The agent learns to discover, compose, and adapt to live HF Spaces.

    Use reset(task="task_id") to select a task scenario, or task="random".

    Example:
        >>> async with SpacesPipelineEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="random")
        ...     obs = result.observation
        ...
        ...     result = await env.step(SpacesPipelineAction(
        ...         action_type="search_spaces",
        ...         payload={"query": "audio transcription", "top_k": 5}
        ...     ))
        ...     print(result.observation.last_search_results)
    """

    def _step_payload(self, action: SpacesPipelineAction) -> Dict:
        return {
            "action_type": action.action_type,
            "payload": action.payload,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SpacesPipelineObservation]:
        obs_data = payload.get("observation", {})
        observation = SpacesPipelineObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            task_input=obs_data.get("task_input", {}),
            expected_output_schema=obs_data.get("expected_output_schema", {}),
            actions_remaining=obs_data.get("actions_remaining", DEFAULT_MAX_ACTIONS),
            spaces_called=obs_data.get("spaces_called", 0),
            spaces_budget_remaining=obs_data.get(
                "spaces_budget_remaining", DEFAULT_MAX_SPACE_CALLS
            ),
            time_used_s=obs_data.get("time_used_s", 0.0),
            time_remaining_s=obs_data.get("time_remaining_s", DEFAULT_TIME_BUDGET_S),
            last_action_latency_s=obs_data.get("last_action_latency_s", 0.0),
            recent_actions=obs_data.get("recent_actions", []),
            recent_outputs=obs_data.get("recent_outputs", []),
            last_search_results=obs_data.get("last_search_results", []),
            last_card_read=obs_data.get("last_card_read"),
            auditor_flags=obs_data.get("auditor_flags", []),
            flag_count_by_severity=obs_data.get(
                "flag_count_by_severity",
                {"info": 0, "warning": 0, "error": 0, "critical": 0},
            ),
            expert_persona_hint=obs_data.get("expert_persona_hint", ""),
            expert_recent_feedback=obs_data.get("expert_recent_feedback", ""),
            detected_drift=obs_data.get("detected_drift", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", DEFAULT_MAX_ACTIONS),
            submitted_answer=obs_data.get("submitted_answer"),
            grade_score=obs_data.get("grade_score"),
            grade_details=obs_data.get("grade_details"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
