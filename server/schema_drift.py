"""
Schema Drift Mechanic (Patronus sub-theme).

Mid-episode, Space contracts can drift:
  - field_rename:  input field renamed (e.g., 'text' -> 'input_text')
  - type_change:   field type changes (e.g., string -> list)
  - new_required:  new required field added
  - output_change: output format changed
  - deprecation:   Space replaced by successor

Drift events are defined per-task in fixtures/tasks.json under "drift_events".
The drift manager applies them when their trigger step is reached.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional


class DriftType:
    FIELD_RENAME = "field_rename"
    TYPE_CHANGE = "type_change"
    NEW_REQUIRED = "new_required"
    OUTPUT_CHANGE = "output_change"
    DEPRECATION = "deprecation"


class SchemaDriftManager:
    """Tracks pending drift events for the current task and applies them.

    The manager owns:
      - A list of pending drift events (consumed as triggered)
      - A set of currently-active drifts (modify Space behavior)
      - A history of fired events (for observation)
    """

    def __init__(self) -> None:
        self.pending: List[Dict[str, Any]] = []
        self.active: Dict[str, List[Dict[str, Any]]] = {}  # space_id -> list of drifts
        self.fired: List[Dict[str, Any]] = []  # for observation
        self.detected: List[Dict[str, Any]] = []  # drifts the agent has experienced

    def reset(self, drift_events: Optional[List[Dict[str, Any]]] = None) -> None:
        self.pending = list(drift_events or [])
        self.active = {}
        self.fired = []
        self.detected = []

    def maybe_fire(self, current_step: int) -> List[Dict[str, Any]]:
        """Fire any drift events whose trigger step <= current_step."""
        newly_fired: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for event in self.pending:
            if event.get("trigger_step", 0) <= current_step:
                self._activate(event)
                self.fired.append({**event, "fired_at_step": current_step})
                newly_fired.append(event)
            else:
                remaining.append(event)
        self.pending = remaining
        return newly_fired

    def _activate(self, event: Dict[str, Any]) -> None:
        """Mark this drift as active on its Space."""
        space_id = event.get("space_id", "")
        if not space_id:
            return
        self.active.setdefault(space_id, []).append(event)

    def is_drifted(self, space_id: str) -> bool:
        return space_id in self.active and len(self.active[space_id]) > 0

    def transform_inputs(
        self, space_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply active drifts to inputs (simulating that the Space now expects them differently).

        Returns a new dict with drifts applied. The agent's call will succeed
        only if its inputs ALREADY match the drifted contract.
        """
        if not self.is_drifted(space_id):
            return inputs

        # The agent's inputs come in as-is; we check if they match the drifted schema.
        # For this method we DON'T modify; we just inform the caller whether the
        # inputs satisfy the drifted contract. Validation happens in `validates_drift`.
        return inputs

    def validates_drift(
        self, space_id: str, inputs: Dict[str, Any]
    ) -> Optional[str]:
        """Check if agent's inputs satisfy the drifted contract.

        Returns None if inputs are valid post-drift, or an error message
        describing why they fail (which the agent must use to detect drift).
        """
        if not self.is_drifted(space_id):
            return None

        for event in self.active[space_id]:
            etype = event.get("type", "")

            if etype == DriftType.FIELD_RENAME:
                # change: {"old_field": "new_field"}
                change = event.get("change", {})
                for old_field, new_field in change.items():
                    if old_field in inputs and new_field not in inputs:
                        return (
                            f"Field '{old_field}' has been renamed to '{new_field}'. "
                            f"Update inputs and re-read the card."
                        )

            elif etype == DriftType.TYPE_CHANGE:
                # change: {"field": "expected_type"}
                change = event.get("change", {})
                for field, expected_type in change.items():
                    val = inputs.get(field)
                    if val is None:
                        continue
                    if expected_type == "list" and not isinstance(val, list):
                        return f"Field '{field}' now expects a list, got {type(val).__name__}"
                    if expected_type == "string" and not isinstance(val, str):
                        return f"Field '{field}' now expects a string, got {type(val).__name__}"

            elif etype == DriftType.NEW_REQUIRED:
                # change: {"new_field": "default_or_description"}
                change = event.get("change", {})
                for new_field in change:
                    if new_field not in inputs:
                        return (
                            f"New required field '{new_field}' must be provided. "
                            f"Re-read the card for details."
                        )

            elif etype == DriftType.DEPRECATION:
                # change: {"successor": "new/space_id"}
                successor = event.get("change", {}).get("successor", "")
                return (
                    f"Space '{space_id}' is deprecated. "
                    f"Use '{successor}' instead."
                )

        return None

    def transform_card(
        self, space_id: str, card: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Apply active drifts to a card so re-reading shows updated schema."""
        if card is None or not self.is_drifted(space_id):
            return card

        new_card = deepcopy(card)
        input_schema = new_card.get("input_schema", {}) or {}

        for event in self.active[space_id]:
            etype = event.get("type", "")
            change = event.get("change", {})

            if etype == DriftType.FIELD_RENAME:
                for old_field, new_field in change.items():
                    if old_field in input_schema:
                        input_schema[new_field] = input_schema.pop(old_field)

            elif etype == DriftType.NEW_REQUIRED:
                for new_field, desc in change.items():
                    input_schema[new_field] = {
                        "type": "string",
                        "description": str(desc),
                        "required": True,
                    }

            elif etype == DriftType.TYPE_CHANGE:
                for field, expected_type in change.items():
                    if field in input_schema and isinstance(input_schema[field], dict):
                        input_schema[field]["type"] = expected_type

            elif etype == DriftType.DEPRECATION:
                new_card["description"] = (
                    f"DEPRECATED. Use successor '{change.get('successor', '')}' instead. "
                    + (new_card.get("description", "") or "")
                )

        new_card["input_schema"] = input_schema
        new_card["_drifted"] = True
        return new_card

    def record_detection(
        self, step: int, space_id: str, error: str
    ) -> None:
        """Record that the agent encountered drift via an error."""
        active_types = [e.get("type") for e in self.active.get(space_id, [])]
        self.detected.append({
            "step": step,
            "space_id": space_id,
            "drift_types": active_types,
            "hint": error,
        })
