"""
Data models for the Spaces Pipeline Pro Environment.

An RL environment where agents learn to discover, compose, and adapt to
HuggingFace Spaces under multi-actor oversight (Auditor, Expert Reviewer)
with schema drift on Space contracts.

Stacks 4 sub-themes:
  - Halluminate: multi-actor (Spaces + Auditor + Expert)
  - Fleet AI: oversight Auditor agent
  - Patronus: schema drift on tool contracts
  - Snorkel: Expert Reviewer with evolving personas
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action types
class ActionType(str, Enum):
    SEARCH_SPACES = "search_spaces"
    READ_CARD = "read_card"
    CALL_SPACE = "call_space"
    SUBMIT = "submit"
    NOOP = "noop"  # Reserved for invalid actions during training


ALL_ACTION_TYPES = [a.value for a in ActionType]

# Expert Reviewer personas (Snorkel)
class ExpertPersona(str, Enum):
    SPEED_FIRST = "speed_first"        # Reward fewer pipeline steps
    ACCURACY_FIRST = "accuracy_first"  # Reward correctness above efficiency
    COST_FIRST = "cost_first"          # Reward minimal Space invocations


ALL_PERSONAS = [p.value for p in ExpertPersona]

# Auditor flag severity levels (Fleet AI)
class FlagSeverity(str, Enum):
    INFO = "info"           # Informational, no penalty
    WARNING = "warning"     # Small penalty
    ERROR = "error"         # Significant penalty
    CRITICAL = "critical"   # Episode-ending in some configs


# Schema drift event types (Patronus)
class DriftType(str, Enum):
    FIELD_RENAME = "field_rename"          # Input field renamed
    TYPE_CHANGE = "type_change"            # Field type changed
    NEW_REQUIRED_FIELD = "new_required"    # New field required
    OUTPUT_FORMAT_CHANGE = "output_change" # Output format changed
    DEPRECATION = "deprecation"            # Space deprecated, points to successor


# Default budgets
DEFAULT_MAX_ACTIONS = 25       # Per-episode action budget
DEFAULT_MAX_SPACE_CALLS = 10   # Hard cap on actual Space invocations
DEFAULT_TIME_BUDGET_S = 120.0  # Per-episode wall-clock budget in simulated seconds
HISTORY_WINDOW = 5             # Last N actions/outputs in observation

# Speed tier (matches speed_tier field on cards)
SPEED_FAST = "fast"           # < 5s
SPEED_MEDIUM = "medium"       # 5-20s
SPEED_SLOW = "slow"           # 20-60s
SPEED_VERY_SLOW = "very_slow" # 60s+
ALL_SPEED_TIERS = [SPEED_FAST, SPEED_MEDIUM, SPEED_SLOW, SPEED_VERY_SLOW]

# Modes (env var SPACES_MODE)
MODE_MOCK = "mock"
MODE_LIVE = "live"
MODE_RECORD = "record"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SpacesPipelineAction(Action):
    """Action: structured tool-use action over the HF Space catalog.

    The agent picks one of 4 action types each step:

    - SEARCH_SPACES: search the catalog by keywords
        payload: {"query": "audio transcription", "top_k": 5}

    - READ_CARD: read a Space's full card (schema, description, examples)
        payload: {"space_id": "openai/whisper-large-v3"}

    - CALL_SPACE: invoke a Space with structured inputs
        payload: {"space_id": "openai/whisper-large-v3",
                  "inputs": {"audio_url": "...", "language": "hi"}}

    - SUBMIT: submit final answer (ends episode)
        payload: {"answer": {"transcript": "...", "summary": "..."}}
    """

    action_type: str = Field(
        ...,
        description=(
            "Action type: 'search_spaces', 'read_card', 'call_space', "
            "or 'submit' (ends episode)"
        ),
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. See action_type for required fields."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-objects in Observation
# ---------------------------------------------------------------------------

class SpaceSearchResult(dict):
    """One entry in a search result list. Backed by dict for JSON-serializability."""
    # Expected keys: space_id, name, downloads, likes, tags, summary, sdk
    pass


class SpaceCard(dict):
    """A Space's card data. Backed by dict for JSON-serializability."""
    # Expected keys: space_id, description, input_schema, output_schema,
    #                example_inputs, license, hardware, sdk, last_modified
    pass


class AuditorFlag(dict):
    """One Auditor flag. Backed by dict."""
    # Expected keys: step, severity, message, action_type, space_id
    pass


class HistoryEntry(dict):
    """One past action+result in history. Backed by dict."""
    # Expected keys: step, action_type, payload, output_snippet, success
    pass


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SpacesPipelineObservation(Observation):
    """Observation from the Spaces Pipeline environment."""

    # Task info
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(
        default="", description="Natural-language task brief"
    )
    task_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input artifacts (URLs, text, etc.) for the task",
    )
    expected_output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema describing what the submit answer must contain",
    )

    # Budget tracking
    actions_remaining: int = Field(
        default=DEFAULT_MAX_ACTIONS,
        description="Number of actions still available before forced termination",
    )
    spaces_called: int = Field(
        default=0,
        description="Total number of Space invocations made so far",
    )
    spaces_budget_remaining: int = Field(
        default=DEFAULT_MAX_SPACE_CALLS,
        description="Number of Space calls still available",
    )

    # Time budget (simulated wall-clock)
    time_used_s: float = Field(
        default=0.0,
        description="Cumulative simulated wall-clock time used so far (seconds)",
    )
    time_remaining_s: float = Field(
        default=DEFAULT_TIME_BUDGET_S,
        description="Remaining simulated wall-clock budget (seconds). Exceeding is penalized.",
    )
    last_action_latency_s: float = Field(
        default=0.0,
        description="Simulated latency of the most recent action (seconds)",
    )

    # History (last HISTORY_WINDOW events)
    recent_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last N actions: [{step, action_type, payload}]",
    )
    recent_outputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Last N outputs: [{step, success, output_snippet, error}]",
    )

    # Search results (most recent search)
    last_search_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from the last search_spaces action",
    )

    # Last card read
    last_card_read: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Card data from the last read_card action (full card)",
    )

    # Auditor (Fleet AI) signals
    auditor_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All Auditor flags raised during episode",
    )
    flag_count_by_severity: Dict[str, int] = Field(
        default_factory=lambda: {"info": 0, "warning": 0, "error": 0, "critical": 0},
        description="Aggregate flag counts by severity",
    )

    # Expert Reviewer (Snorkel) hints
    expert_persona_hint: str = Field(
        default="",
        description=(
            "Noisy hint about the active Expert persona, e.g., "
            "'Reviewer mentions a tight deadline' (Speed-First)"
        ),
    )
    expert_recent_feedback: str = Field(
        default="",
        description="Free-form feedback from Expert on most recent submission attempt",
    )

    # Schema drift signals (Patronus)
    detected_drift: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Drift events the agent has noticed via errors. "
            "[{step, space_id, drift_type, hint}]"
        ),
    )

    # Step info
    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(
        default=DEFAULT_MAX_ACTIONS, description="Total step budget for this task"
    )

    # Submission state
    submitted_answer: Optional[Dict[str, Any]] = Field(
        default=None, description="Agent's submitted answer (set on submit action)"
    )

    # Grading (terminal only)
    grade_score: Optional[float] = Field(
        default=None,
        description="Final grade 0.0-1.0 (only set on terminal observation)",
    )
    grade_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Breakdown of grading components",
    )
